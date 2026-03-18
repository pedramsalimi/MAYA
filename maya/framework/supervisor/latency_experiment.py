from __future__ import annotations

import argparse
import os
import queue
import sys
import time
import warnings
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import sounddevice as sd
import soundfile as sf

# Keep experiment output clean from dependency deprecation noise.
warnings.filterwarnings(
    "ignore",
    message="pkg_resources is deprecated as an API.*",
    category=UserWarning,
)
import webrtcvad
from dotenv import load_dotenv
from langgraph.types import Command

ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)

if __package__ in {None, ""}:
    sys.path.append(str(ROOT))
    import maya.framework.supervisor.factory as sup_factory
    from maya.framework.supervisor import run_with_vad as rv
    from maya.framework.supervisor.utils import strip_citations_and_references, strip_markdown
else:
    from . import factory as sup_factory
    from . import run_with_vad as rv
    from .utils import strip_citations_and_references, strip_markdown

load_dotenv()

VAD_AGGRESSIVENESS = rv.VAD_AGGRESSIVENESS
VAD_FRAME_MS = rv.VAD_FRAME_MS
VAD_PADDING_MS = rv.VAD_PADDING_MS
VAD_SILENCE_MS = rv.VAD_SILENCE_MS
VAD_RMS_THRESHOLD = rv.VAD_RMS_THRESHOLD
VOICE_SECONDS = rv.VOICE_SECONDS
VOICE_RATE = rv.VOICE_RATE

ROUTE_TO_NODE = {
    "health_rag": "health_rag",
    "user_profile": "update_profile",
    "general_memory": "update_memory",
    "scan_portal": "scan_portal",
}


@dataclass
class AsrTiming:
    capture_seconds: float
    asr_seconds: float

    @property
    def speech_to_supervisor_seconds(self) -> float:
        return self.capture_seconds + self.asr_seconds


@dataclass
class SupervisorTiming:
    final_text: str | None
    route_type: str | None
    supervisor_route_seconds: float | None
    routed_node_name: str | None
    route_to_node_handoff_seconds: float | None
    routed_node_seconds: float | None
    routed_node_total_seconds: float | None
    routed_node_call_count: int
    health_rag_seconds: float | None
    supervisor_after_tool_seconds: float | None
    reasoning_total_seconds: float | None
    clarification_wait_seconds: float


@dataclass
class TtsTiming:
    synth_until_audio_seconds: float | None
    playback_seconds: float | None
    total_seconds: float | None


@dataclass
class MemoryStoreTiming:
    search_total_seconds: float = 0.0
    put_total_seconds: float = 0.0
    search_profile_seconds: float = 0.0
    search_memory_seconds: float = 0.0
    search_health_memory_seconds: float = 0.0
    search_other_seconds: float = 0.0
    put_profile_seconds: float = 0.0
    put_memory_seconds: float = 0.0
    put_health_memory_seconds: float = 0.0
    put_other_seconds: float = 0.0


_ACTIVE_MEMORY_TIMING: MemoryStoreTiming | None = None


class TimedStoreProxy:
    def __init__(self, wrapped: object) -> None:
        self._wrapped = wrapped

    def __getattr__(self, item: str):
        return getattr(self._wrapped, item)

    @staticmethod
    def _namespace_bucket(namespace: Any) -> str:
        if isinstance(namespace, tuple) and namespace:
            if namespace[0] in {"profile", "memory", "health_memory"}:
                return str(namespace[0])
        return "other"

    def search(self, namespace, *args, **kwargs):
        t0 = time.perf_counter()
        result = self._wrapped.search(namespace, *args, **kwargs)
        dt = time.perf_counter() - t0

        rec = _ACTIVE_MEMORY_TIMING
        if rec is not None:
            rec.search_total_seconds += dt
            bucket = self._namespace_bucket(namespace)
            if bucket == "profile":
                rec.search_profile_seconds += dt
            elif bucket == "memory":
                rec.search_memory_seconds += dt
            elif bucket == "health_memory":
                rec.search_health_memory_seconds += dt
            else:
                rec.search_other_seconds += dt
        return result

    def put(self, namespace, *args, **kwargs):
        t0 = time.perf_counter()
        result = self._wrapped.put(namespace, *args, **kwargs)
        dt = time.perf_counter() - t0

        rec = _ACTIVE_MEMORY_TIMING
        if rec is not None:
            rec.put_total_seconds += dt
            bucket = self._namespace_bucket(namespace)
            if bucket == "profile":
                rec.put_profile_seconds += dt
            elif bucket == "memory":
                rec.put_memory_seconds += dt
            elif bucket == "health_memory":
                rec.put_health_memory_seconds += dt
            else:
                rec.put_other_seconds += dt
        return result


def speak_and_time(tts: object | None, text: str) -> TtsTiming:
    if not tts or not text:
        return TtsTiming(
            synth_until_audio_seconds=None,
            playback_seconds=None,
            total_seconds=None,
        )

    t0 = time.perf_counter()
    try:
        result = tts.speak_text_async(text).get()
        t_audio_ready = time.perf_counter()
        audio_bytes = getattr(result, "audio_data", None)
        if audio_bytes:
            buffer = BytesIO(audio_bytes)
            data, rate = sf.read(buffer, dtype="float32")
            sd.play(data, rate)
            sd.wait()
        t_done = time.perf_counter()
        synth = t_audio_ready - t0
        total = t_done - t0
        return TtsTiming(
            synth_until_audio_seconds=synth,
            playback_seconds=max(0.0, total - synth),
            total_seconds=total,
        )
    except Exception as exc:
        print(f"[tts] {exc}")
        return TtsTiming(
            synth_until_audio_seconds=None,
            playback_seconds=None,
            total_seconds=None,
        )


def _extract_message_parts(message: Any) -> tuple[str | None, str | None, str | None, list[dict[str, Any]]]:
    if hasattr(message, "get"):
        role = message.get("role") or message.get("type")
        name = message.get("name")
        content = message.get("content")
        tool_calls = message.get("tool_calls") or []
        return role, name, content, tool_calls

    role = getattr(message, "type", None)
    name = getattr(message, "name", None)
    content = getattr(message, "content", None)
    tool_calls = getattr(message, "tool_calls", None) or []
    return role, name, content, tool_calls


def _clean_token(value: Any) -> str:
    text = str(value).strip()
    if ":" in text:
        text = text.split(":", 1)[0]
    return text


def _chunk_tokens(chunk: Any, node_name: str) -> set[str]:
    tokens = {_clean_token(node_name)}
    if isinstance(chunk, tuple):
        namespace = chunk[0]
        if isinstance(namespace, (list, tuple)):
            for part in namespace:
                tokens.add(_clean_token(part))
        else:
            tokens.add(_clean_token(namespace))
    return {t for t in tokens if t}


def capture_voice_and_transcribe(
    asr_client: object,
    asr_model: str,
    voice: rv.VoiceConfig,
) -> tuple[str, AsrTiming]:
    print(f"Recording... speak and pause to stop (max {voice.seconds:.1f}s)")

    t_capture_start = time.perf_counter()
    vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)
    frame_ms = VAD_FRAME_MS
    padding_ms = VAD_PADDING_MS
    silence_ms = VAD_SILENCE_MS
    rms_threshold = VAD_RMS_THRESHOLD

    frame_samples = int(voice.rate * frame_ms / 1000)
    max_frames = int(voice.seconds * voice.rate)
    padding_frames = max(1, int(padding_ms / frame_ms))
    silence_frames = max(1, int(silence_ms / frame_ms))

    audio_queue: queue.Queue[np.ndarray] = queue.Queue()
    recorded_blocks: list[np.ndarray] = []
    ring = deque(maxlen=padding_frames)
    silence_run = 0
    total_frames = 0
    triggered = False

    def callback(indata: np.ndarray, _frames: int, _time, status) -> None:
        if status:
            print(f"[audio] {status}", file=sys.stderr)
        audio_queue.put(indata.copy())

    with sd.InputStream(
        samplerate=voice.rate,
        channels=1,
        dtype="int16",
        blocksize=frame_samples,
        callback=callback,
    ):
        while True:
            block = audio_queue.get().reshape(-1)
            total_frames += len(block)
            pcm = block.tobytes()
            rms = float(np.sqrt(np.mean(block.astype(np.float32) ** 2)) / 32768.0)
            is_speech = rms > rms_threshold and vad.is_speech(pcm, voice.rate)

            if not triggered:
                ring.append(block)
                if is_speech:
                    triggered = True
                    recorded_blocks.extend(ring)
                    ring.clear()
                if total_frames >= max_frames:
                    break
                continue

            recorded_blocks.append(block)
            if is_speech:
                silence_run = 0
            else:
                silence_run += 1

            if silence_run >= silence_frames or total_frames >= max_frames:
                break

    capture_seconds = time.perf_counter() - t_capture_start

    if not recorded_blocks:
        return "", AsrTiming(capture_seconds=capture_seconds, asr_seconds=0.0)

    frames = np.concatenate(recorded_blocks, axis=0)
    buffer = BytesIO()
    with sf.SoundFile(
        buffer,
        mode="w",
        samplerate=voice.rate,
        channels=1,
        format="WAV",
    ) as wav_file:
        wav_file.write(frames)
    buffer.seek(0)

    t_asr_start = time.perf_counter()
    transcription = asr_client.audio.transcriptions.create(
        model=asr_model,
        file=("input.wav", buffer, "audio/wav"),
    )
    asr_seconds = time.perf_counter() - t_asr_start
    text = (getattr(transcription, "text", None) or "").strip()

    return text, AsrTiming(capture_seconds=capture_seconds, asr_seconds=asr_seconds)


def run_supervisor_with_timing(app, payload: Any, base_config: dict[str, Any]) -> SupervisorTiming:
    final_text: str | None = None
    final_text_any_at: float | None = None
    route_type: str | None = None
    expected_route_node: str | None = None
    supervisor_route_at: float | None = None
    routed_node_name: str | None = None
    routed_node_start_at: float | None = None
    routed_node_done_at: float | None = None
    route_to_node_handoff_seconds: float | None = None
    routed_node_durations: list[float] = []
    supervisor_final_at: float | None = None
    clarification_wait_seconds = 0.0

    current_payload: Any = payload
    logical_elapsed = 0.0

    while True:
        seg_start = time.perf_counter()
        interrupts = []
        seg_node_start: float | None = None
        seg_node_end: float | None = None

        for chunk in app.stream(
            current_payload,
            config=base_config,
            stream_mode="updates",
            subgraphs=True,
        ):
            now = time.perf_counter()
            now_logical = logical_elapsed + (now - seg_start)

            update = chunk[1] if isinstance(chunk, tuple) else chunk

            if "__interrupt__" in update:
                interrupts.extend(update["__interrupt__"])
                continue

            for node, node_payload in update.items():
                node_name = str(node)
                tokens = _chunk_tokens(chunk, node_name)
                messages = node_payload.get("messages", [])
                if isinstance(messages, dict):
                    messages = list(messages.values())
                if not messages:
                    continue

                last = messages[-1]
                role, _name, content, tool_calls = _extract_message_parts(last)

                if "supervisor" in node_name and tool_calls and supervisor_route_at is None:
                    first_tool = tool_calls[0] if tool_calls else {}
                    args = first_tool.get("args", {}) if hasattr(first_tool, "get") else {}
                    route_type = args.get("route_type") if hasattr(args, "get") else None
                    expected_route_node = ROUTE_TO_NODE.get(route_type or "")
                    supervisor_route_at = now_logical

                if (
                    expected_route_node
                    and supervisor_route_at is not None
                    and expected_route_node in tokens
                ):
                    routed_node_name = expected_route_node
                    if seg_node_start is None:
                        seg_node_start = now_logical
                        if routed_node_start_at is None:
                            routed_node_start_at = now_logical
                            route_to_node_handoff_seconds = max(
                                0.0, routed_node_start_at - supervisor_route_at
                            )
                    seg_node_end = now_logical

                if role in {"assistant", "ai"} and content:
                    final_text_any_at = now_logical
                    if "supervisor" in tokens:
                        final_text = content
                        supervisor_final_at = now_logical

        if seg_node_start is not None and seg_node_end is not None:
            routed_node_done_at = seg_node_end
            routed_node_durations.append(max(0.0, seg_node_end - seg_node_start))

        logical_elapsed += time.perf_counter() - seg_start

        if not interrupts:
            break

        answers: list[str] = []
        wait_start = time.perf_counter()
        for interrupt_ in interrupts:
            prompt = str(interrupt_.value)
            answer = input(f"MAYA: {prompt}\nYou: ").strip()
            answers.append(answer)
        clarification_wait_seconds += time.perf_counter() - wait_start

        resume_value: str | list[str] = answers[0] if len(answers) == 1 else answers
        current_payload = Command(resume=resume_value)

    health_rag_seconds = None
    routed_node_seconds = None
    routed_node_total_seconds = None
    routed_node_call_count = len(routed_node_durations)
    supervisor_after_tool_seconds = None

    if routed_node_durations:
        routed_node_seconds = routed_node_durations[-1]
        routed_node_total_seconds = sum(routed_node_durations)

    if route_type == "health_rag" and routed_node_seconds is not None:
        health_rag_seconds = routed_node_seconds

    if routed_node_done_at is not None and supervisor_final_at is not None:
        supervisor_after_tool_seconds = max(0.0, supervisor_final_at - routed_node_done_at)

    return SupervisorTiming(
        final_text=final_text,
        route_type=route_type,
        supervisor_route_seconds=supervisor_route_at,
        routed_node_name=routed_node_name,
        route_to_node_handoff_seconds=route_to_node_handoff_seconds,
        routed_node_seconds=routed_node_seconds,
        routed_node_total_seconds=routed_node_total_seconds,
        routed_node_call_count=routed_node_call_count,
        health_rag_seconds=health_rag_seconds,
        supervisor_after_tool_seconds=supervisor_after_tool_seconds,
        reasoning_total_seconds=supervisor_final_at or final_text_any_at,
        clarification_wait_seconds=clarification_wait_seconds,
    )


def append_result_row(output_path: Path, row: dict[str, Any]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    header = [
        "utc_ts",
        "run_id",
        "sample_id",
        "input_mode",
        "user_text",
        "route_type",
        "routed_node_name",
        "capture_s",
        "asr_s",
        "speech_to_supervisor_s",
        "supervisor_route_s",
        "route_to_node_handoff_s",
        "routed_node_s",
        "routed_node_total_s",
        "routed_node_calls",
        "health_rag_s",
        "supervisor_after_tool_s",
        "reasoning_total_s",
        "supervisor_response_s",
        "clarification_wait_s",
        "mem_search_total_s",
        "mem_search_profile_s",
        "mem_search_memory_s",
        "mem_search_health_memory_s",
        "mem_put_total_s",
        "mem_put_profile_s",
        "mem_put_memory_s",
        "mem_put_health_memory_s",
        "tts_synth_until_audio_s",
        "tts_playback_s",
        "tts_total_s",
        "response_ready_s",
        "end_to_end_complete_s",
    ]

    if not output_path.exists():
        output_path.write_text("\t".join(header) + "\n", encoding="utf-8")

    values = [str(row.get(col, "")) for col in header]
    with output_path.open("a", encoding="utf-8") as f:
        f.write("\t".join(values) + "\n")


def fmt(value: float | None) -> str:
    return "" if value is None else f"{value:.3f}"


def with_unit(seconds_text: str) -> str:
    return f"{seconds_text}s" if seconds_text else "n/a"


def run_sample(
    *,
    app,
    base_config: dict[str, Any],
    tts: object | None,
    asr_client: object | None,
    asr_model: str | None,
    voice: rv.VoiceConfig,
    user_text: str | None,
) -> tuple[dict[str, Any], str | None]:
    global _ACTIVE_MEMORY_TIMING

    mem_timing = MemoryStoreTiming()
    _ACTIVE_MEMORY_TIMING = mem_timing

    try:
        asr_timing = AsrTiming(capture_seconds=0.0, asr_seconds=0.0)
        input_mode = "text"

        start_t = time.perf_counter()

        if user_text is None:
            if not asr_client or not asr_model:
                raise RuntimeError("ASR is not configured. Set Whisper/Azure Whisper env vars.")

            input_mode = "voice"
            user_text, asr_timing = capture_voice_and_transcribe(asr_client, asr_model, voice)
            print(f"You (voice): {user_text}")

        payload = {"messages": [{"role": "user", "content": user_text}]}
        sup_timing = run_supervisor_with_timing(app, payload, base_config)
    finally:
        _ACTIVE_MEMORY_TIMING = None

    final_text = sup_timing.final_text
    if final_text:
        final_text = strip_citations_and_references(strip_markdown(final_text))

    tts_timing = speak_and_time(tts, final_text or "")
    end_to_end_seconds = time.perf_counter() - start_t
    response_ready_seconds = (
        (asr_timing.speech_to_supervisor_seconds if input_mode == "voice" else 0.0)
        + (sup_timing.reasoning_total_seconds or 0.0)
    )
    supervisor_response_seconds = (
        sup_timing.supervisor_after_tool_seconds
        if sup_timing.route_type == "health_rag" and sup_timing.supervisor_after_tool_seconds is not None
        else sup_timing.reasoning_total_seconds
    )

    row = {
        "utc_ts": datetime.now(timezone.utc).isoformat(),
        "input_mode": input_mode,
        "user_text": (user_text or "").replace("\t", " ").replace("\n", " "),
        "route_type": sup_timing.route_type or "",
        "routed_node_name": sup_timing.routed_node_name or "",
        "capture_s": fmt(asr_timing.capture_seconds if input_mode == "voice" else None),
        "asr_s": fmt(asr_timing.asr_seconds if input_mode == "voice" else None),
        "speech_to_supervisor_s": fmt(
            asr_timing.speech_to_supervisor_seconds if input_mode == "voice" else None
        ),
        "supervisor_route_s": fmt(sup_timing.supervisor_route_seconds),
        "route_to_node_handoff_s": fmt(sup_timing.route_to_node_handoff_seconds),
        "routed_node_s": fmt(sup_timing.routed_node_seconds),
        "routed_node_total_s": fmt(sup_timing.routed_node_total_seconds),
        "routed_node_calls": str(sup_timing.routed_node_call_count),
        "health_rag_s": fmt(sup_timing.health_rag_seconds),
        "supervisor_after_tool_s": fmt(sup_timing.supervisor_after_tool_seconds),
        "reasoning_total_s": fmt(sup_timing.reasoning_total_seconds),
        "supervisor_response_s": fmt(supervisor_response_seconds),
        "clarification_wait_s": fmt(sup_timing.clarification_wait_seconds),
        "mem_search_total_s": fmt(mem_timing.search_total_seconds),
        "mem_search_profile_s": fmt(mem_timing.search_profile_seconds),
        "mem_search_memory_s": fmt(mem_timing.search_memory_seconds),
        "mem_search_health_memory_s": fmt(mem_timing.search_health_memory_seconds),
        "mem_put_total_s": fmt(mem_timing.put_total_seconds),
        "mem_put_profile_s": fmt(mem_timing.put_profile_seconds),
        "mem_put_memory_s": fmt(mem_timing.put_memory_seconds),
        "mem_put_health_memory_s": fmt(mem_timing.put_health_memory_seconds),
        "tts_synth_until_audio_s": fmt(tts_timing.synth_until_audio_seconds),
        "tts_playback_s": fmt(tts_timing.playback_seconds),
        "tts_total_s": fmt(tts_timing.total_seconds),
        "response_ready_s": fmt(response_ready_seconds),
        "end_to_end_complete_s": fmt(end_to_end_seconds),
    }
    return row, final_text


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Standalone latency experiment for ASR -> supervisor -> health_rag -> TTS",
    )
    parser.add_argument(
        "--output",
        default="artifacts/latency_probe/latency_results.txt",
        help="Append-only TSV text file for timing rows.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=3,
        help="Number of samples in interactive mode.",
    )
    parser.add_argument(
        "--auto-text",
        action="store_true",
        help="Run built-in text examples instead of voice/manual input.",
    )
    args = parser.parse_args()

    user_id = os.getenv("MAYA_RUN_USER_ID", "test")
    thread_id = os.getenv("MAYA_RUN_THREAD_ID", "latency_probe")
    base_config = {
        "configurable": {"thread_id": thread_id, "user_id": user_id},
        "recursion_limit": 40,
    }

    original_get_postgres_memory = sup_factory.get_postgres_memory

    def timed_get_postgres_memory():
        store, checkpointer = original_get_postgres_memory()
        return TimedStoreProxy(store), checkpointer

    sup_factory.get_postgres_memory = timed_get_postgres_memory
    try:
        app = rv.build_supervisor()
    finally:
        sup_factory.get_postgres_memory = original_get_postgres_memory
    asr_client, asr_model = rv.init_asr()
    tts = rv.init_tts()
    voice = rv.VoiceConfig(seconds=VOICE_SECONDS, rate=VOICE_RATE)

    examples = [
        "What are early signs of insulin resistance?",
        "How does metformin work?",
        "Can high stress increase blood pressure?",
    ]

    run_id = datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%SZ")
    output_path = Path(args.output)

    if args.auto_text:
        samples = examples[: max(1, args.max_samples)]
        print(f"Running {len(samples)} text samples in auto mode...")
        for i, sample in enumerate(samples, start=1):
            row, final_text = run_sample(
                app=app,
                base_config=base_config,
                tts=tts,
                asr_client=asr_client,
                asr_model=asr_model,
                voice=voice,
                user_text=sample,
            )
            row["run_id"] = run_id
            row["sample_id"] = str(i)
            append_result_row(output_path, row)
            print(
                f"[{i}] route={row['route_type'] or 'n/a'} "
                f"routed_node={row['routed_node_name'] or 'n/a'} "
                f"asr_total={with_unit(row['speech_to_supervisor_s'])} "
                f"routing={with_unit(row['supervisor_route_s'])} "
                f"route_to_node_handoff={with_unit(row['route_to_node_handoff_s'])} "
                f"routed_node_delay(final_call)={with_unit(row['routed_node_s'])} "
                f"routed_node_total={with_unit(row['routed_node_total_s'])} "
                f"routed_node_calls={row['routed_node_calls'] or '0'} "
                f"health_rag={with_unit(row['health_rag_s'])} "
                f"supervisor_response={with_unit(row['supervisor_response_s'])} "
                f"clarification_wait={with_unit(row['clarification_wait_s'])} "
                f"mem_search={with_unit(row['mem_search_total_s'])} "
                f"mem_put={with_unit(row['mem_put_total_s'])} "
                f"response_ready={with_unit(row['response_ready_s'])} "
                f"tts_synth={with_unit(row['tts_synth_until_audio_s'])} "
                f"tts_total={with_unit(row['tts_total_s'])} "
                f"end_to_end_complete={with_unit(row['end_to_end_complete_s'])}"
            )
            if final_text:
                print(f"MAYA: {final_text}\n")
        print(f"Saved rows to: {output_path}")
        return

    if not asr_client or not asr_model:
        print("ASR is not configured. Use --auto-text for text-only timing, or set Whisper env vars.")
        return

    print("Latency probe mode: press Enter to start voice capture, type /quit to stop.")
    for i in range(1, max(1, args.max_samples) + 1):
        cmd = input(f"\nSample {i}/{args.max_samples} - Enter to record, /quit to exit: ").strip()
        if cmd in {"/quit", "/exit"}:
            break

        row, final_text = run_sample(
            app=app,
            base_config=base_config,
            tts=tts,
            asr_client=asr_client,
            asr_model=asr_model,
            voice=voice,
            user_text=None,
        )
        row["run_id"] = run_id
        row["sample_id"] = str(i)
        append_result_row(output_path, row)

        print(
            " | ".join(
                [
                    f"route={row['route_type'] or 'n/a'}",
                    f"routed_node={row['routed_node_name'] or 'n/a'}",
                    f"capture={with_unit(row['capture_s'])}",
                    f"asr={with_unit(row['asr_s'])}",
                    f"asr_total={with_unit(row['speech_to_supervisor_s'])}",
                    f"routing={with_unit(row['supervisor_route_s'])}",
                    f"route_to_node_handoff={with_unit(row['route_to_node_handoff_s'])}",
                    f"routed_node_delay(final_call)={with_unit(row['routed_node_s'])}",
                    f"routed_node_total={with_unit(row['routed_node_total_s'])}",
                    f"routed_node_calls={row['routed_node_calls'] or '0'}",
                    f"health_rag={with_unit(row['health_rag_s'])}",
                    f"supervisor_response={with_unit(row['supervisor_response_s'])}",
                    f"clarification_wait={with_unit(row['clarification_wait_s'])}",
                    f"mem_search={with_unit(row['mem_search_total_s'])}",
                    f"mem_search(profile)={with_unit(row['mem_search_profile_s'])}",
                    f"mem_search(memory)={with_unit(row['mem_search_memory_s'])}",
                    f"mem_put={with_unit(row['mem_put_total_s'])}",
                    f"response_ready={with_unit(row['response_ready_s'])}",
                    f"tts_synth={with_unit(row['tts_synth_until_audio_s'])}",
                    f"tts_playback={with_unit(row['tts_playback_s'])}",
                    f"tts_total={with_unit(row['tts_total_s'])}",
                    f"end_to_end_complete={with_unit(row['end_to_end_complete_s'])}",
                ]
            )
        )
        if final_text:
            print(f"MAYA: {final_text}")

    print(f"Saved rows to: {output_path}")


if __name__ == "__main__":
    main()
