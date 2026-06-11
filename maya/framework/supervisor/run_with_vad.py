from __future__ import annotations

import os
import sys
import queue
import select
import uuid
import atexit
import fcntl
from collections import deque
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
import time
import numpy as np
import sounddevice as sd
import soundfile as sf
import webrtcvad
from dotenv import load_dotenv
from openai import OpenAI, AzureOpenAI, BadRequestError
import azure.cognitiveservices.speech as speechsdk
from langgraph.types import Command

ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)

if __package__ in {None, ""}:
    sys.path.append(str(ROOT))
    from utils import strip_markdown, strip_citations_and_references
    from maya.framework.supervisor.factory import build_supervisor
else:
    from .utils import strip_markdown, strip_citations_and_references
    from .factory import build_supervisor

load_dotenv()


@dataclass(frozen=True)
class VoiceConfig:
    seconds: float
    rate: int


VAD_AGGRESSIVENESS = int(os.getenv("MAYA_VAD_AGGRESSIVENESS", "3"))
VAD_FRAME_MS = int(os.getenv("MAYA_VAD_FRAME_MS", "10"))
VAD_PADDING_MS = int(os.getenv("MAYA_VAD_PADDING_MS", "100"))
VAD_SILENCE_MS = int(os.getenv("MAYA_VAD_SILENCE_MS", "200"))
VAD_RMS_THRESHOLD = float(os.getenv("MAYA_VAD_RMS_THRESHOLD", "0.006"))
VOICE_SECONDS = float(os.getenv("MAYA_VOICE_SECONDS", "20"))
VOICE_RATE = int(os.getenv("MAYA_VOICE_SAMPLE_RATE", "16000"))
PHYXIO_END_INTERACTION_DELAY = float(os.getenv("MAYA_PHYXIO_END_INTERACTION_DELAY", "4.5"))


def init_asr() -> tuple[object | None, str | None]:
    whisper_key = os.getenv("OPENAI_WHISPER_API_KEY")
    if whisper_key:
        whisper_model = os.getenv("OPENAI_WHISPER_MODEL", "whisper-1")
        return OpenAI(api_key=whisper_key), whisper_model

    azure_key = os.getenv("AZURE_OPENAI_WHISPER_API_KEY")
    azure_endpoint = os.getenv("AZURE_OPENAI_WHISPER_ENDPOINT")
    azure_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_WHISPER")
    azure_version = (
        os.getenv("AZURE_OPENAI_WHISPER_API_VERSION")
        or os.getenv("AZURE_OPENAI_API_VERSION")
        or "2024-06-01"
    )
    if azure_key and azure_endpoint and azure_deployment:
        base_endpoint = (
            azure_endpoint.split("/openai/")[0]
            if "/openai/" in azure_endpoint
            else azure_endpoint
        )
        return (
            AzureOpenAI(
                api_key=azure_key,
                api_version=azure_version,
                azure_endpoint=base_endpoint,
            ),
            azure_deployment,
        )

    return None, None


def init_tts() -> speechsdk.SpeechSynthesizer | None:
    speech_key = os.getenv("AZURE_SPEECH_KEY")
    speech_region = os.getenv("AZURE_SPEECH_REGION")
    if not speech_key or not speech_region:
        return None

    speech_voice = os.getenv("AZURE_SPEECH_VOICE", "en-GB-RyanNeural")
    speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=speech_region)
    speech_config.speech_synthesis_voice_name = speech_voice
    speech_config.set_speech_synthesis_output_format(
        speechsdk.SpeechSynthesisOutputFormat.Riff24Khz16BitMonoPcm
    )
    return speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=None)


def speak(tts: speechsdk.SpeechSynthesizer | None, text: str) -> None:
    if not tts or not text:
        return
    try:
        result = tts.speak_text_async(text).get()
        audio_bytes = getattr(result, "audio_data", None)
        if audio_bytes:
            buffer = BytesIO(audio_bytes)
            data, rate = sf.read(buffer, dtype="float32")
            sd.play(data, rate)
            sd.wait()
    except Exception as tts_err:
        print(f"[tts] {tts_err}")


def init_phyxio_io():
    if os.getenv("MAYA_ENABLE_PHYXIO_IO", "1").strip().lower() not in {"1", "true", "yes"}:
        return None
    try:
        from maya.agents.phyxio_exercise_agent.bridge import get_phyxio_service

        return get_phyxio_service()
    except Exception as phyxio_err:
        print(f"[phyxio] disabled: {phyxio_err}")
        return None


def speak_phyxio(phyxio_io, text: str) -> None:
    if not phyxio_io or not text:
        return
    try:
        phyxio_io.show_text(text)
        set_phyxio_state(phyxio_io, "talking")
        phyxio_io.speak_text(text)
    except Exception as phyxio_err:
        print(f"[phyxio][tts] {phyxio_err}")


def set_phyxio_state(phyxio_io, state: str) -> None:
    if not phyxio_io:
        return
    try:
        set_state = getattr(phyxio_io, "set_agent_state", None)
        if callable(set_state):
            set_state(state)
    except Exception as phyxio_err:
        print(f"[phyxio][state] {phyxio_err}")


def end_phyxio_interaction(phyxio_io, delay_seconds: float = PHYXIO_END_INTERACTION_DELAY) -> None:
    if not phyxio_io:
        return
    try:
        end_interaction = getattr(phyxio_io, "end_interaction", None)
        if callable(end_interaction):
            end_interaction(delay_seconds)
    except Exception as phyxio_err:
        print(f"[phyxio][end_interaction] {phyxio_err}")


def speak_output(
    tts: speechsdk.SpeechSynthesizer | None,
    phyxio_io,
    text: str,
    *,
    end_interaction: bool = False,
) -> None:
    if phyxio_io is not None:
        speak_phyxio(phyxio_io, text)
        if end_interaction:
            end_phyxio_interaction(phyxio_io)
    else:
        speak(tts, text)


def read_text(
    prompt_text: str | None,
    tts: speechsdk.SpeechSynthesizer | None,
    asr_client: object | None,
    asr_model: str | None,
    voice: VoiceConfig,
    phyxio_io=None,
    *,
    allow_exit: bool = False,
    blank_returns_none: bool = False,
) -> str | None:
    if prompt_text:
        speak_output(tts, phyxio_io, prompt_text)
        input_prompt = f"MAYA: {prompt_text}\nYou: "
    else:
        input_prompt = "You: "

    def capture_voice() -> str:
        print(f"Recording... speak and pause to stop (max {voice.seconds:.1f}s)")

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

        if not recorded_blocks:
            return ""

        frames = np.concatenate(recorded_blocks, axis=0)
        buffer = BytesIO()
        with sf.SoundFile(
            buffer, mode="w", samplerate=voice.rate, channels=1, format="WAV"
        ) as wav_file:
            wav_file.write(frames)
        buffer.seek(0)

        transcription = asr_client.audio.transcriptions.create(
            model=asr_model,
            file=("input.wav", buffer, "audio/wav"),
        )
        return (getattr(transcription, "text", None) or "").strip()

    while True:
        if phyxio_io is not None:
            print(input_prompt, end="", flush=True)
            while True:
                try:
                    mirror_text = phyxio_io.asr_queue.get_nowait()
                except queue.Empty:
                    mirror_text = ""
                except Exception:
                    mirror_text = ""
                if isinstance(mirror_text, str) and mirror_text.strip():
                    spoken = mirror_text.strip()
                    print()
                    print(f"You (phyxio): {spoken}")
                    return spoken
                
                if not sys.stdin.isatty():
                    time.sleep(0.1)
                    continue
                try:
                    ready, _, _ = select.select([sys.stdin], [], [], 0.2)
                except (ValueError, OSError):
                    ready = []
                if ready:
                    raw = sys.stdin.readline()
                    if raw == "":
                        print()
                        return None
                    user_input = raw.strip()
                    break
        else:
            try:
                user_input = input(input_prompt).strip()
            except (EOFError, KeyboardInterrupt):
                print()
                return None

        if allow_exit and user_input in {"/exit", "/quit"}:
            return None

        wants_voice = user_input == "/voice" or (user_input == "" and asr_client and phyxio_io is None)
        if wants_voice:
            if not asr_client or not asr_model:
                if user_input == "/voice":
                    print("Whisper key not configured. Please provide text input instead.")
                    continue
                return None if blank_returns_none else ""
            spoken_text = capture_voice()
            if not spoken_text:
                print("No speech detected. Try again.")
                continue
            print(f"You (voice): {spoken_text}")
            return spoken_text

        if user_input:
            return user_input

        if phyxio_io is not None:
            continue

        if blank_returns_none:
            return None
        return ""


def stream_updates(app, payload, base_config, phyxio_io=None):
    final_text = None
    interrupts = []
    seen_health_rag_traces: set[tuple[bool, str, bool, bool]] = set()

    set_phyxio_state(phyxio_io, "thinking")
    try:
        for chunk in app.stream(
            payload,
            config=base_config,
            stream_mode="updates",
            subgraphs=True,
        ):
            update = chunk[1] if isinstance(chunk, tuple) else chunk
            if not isinstance(update, dict):
                continue

            if "__interrupt__" in update:
                interrupts.extend(update.get("__interrupt__", []))
                continue

            for node, node_payload in update.items():
                if node == "__interrupt__" or node_payload is None:
                    continue

                if isinstance(node_payload, dict):
                    messages = node_payload.get("messages", [])
                elif isinstance(node_payload, list):
                    messages = node_payload
                else:
                    messages = getattr(node_payload, "messages", [])

                if messages is None:
                    continue
                if isinstance(messages, dict):
                    messages = list(messages.values())
                elif not isinstance(messages, list):
                    messages = [messages]
                if not messages:
                    continue
                last = messages[-1]
                if isinstance(last, dict):
                    role = last.get("role") or last.get("type")
                    name = last.get("name")
                    content = last.get("content")
                    additional_kwargs = last.get("additional_kwargs") or {}
                else:
                    role = getattr(last, "type", None)
                    name = getattr(last, "name", None)
                    content = getattr(last, "content", None)
                    additional_kwargs = getattr(last, "additional_kwargs", {}) or {}

                if name:
                    print(f"[{node}] {role} :: {name}")

                workflow_trace = additional_kwargs.get("workflow_trace")
                if isinstance(workflow_trace, list):
                    for item in workflow_trace:
                        if not isinstance(item, dict):
                            continue
                        stage = str(item.get("stage", "")).strip()
                        if stage != "health_rag":
                            continue
                        memory_hit = bool(item.get("memory_hit"))
                        memory_lookup = str(item.get("memory_lookup", ""))
                        retrieval_used = bool(item.get("retrieval_used"))
                        summary_stored = bool(item.get("summary_stored"))
                        trace_key = (memory_hit, memory_lookup, retrieval_used, summary_stored)
                        if trace_key in seen_health_rag_traces:
                            continue
                        seen_health_rag_traces.add(trace_key)
                        print(
                            "[health_rag] trace :: "
                            f"memory_hit={memory_hit} "
                            f"memory_lookup={memory_lookup or 'unknown'} "
                            f"retrieval_used={retrieval_used} "
                            f"summary_stored={summary_stored}"
                        )

                if role in {"assistant", "ai"} and content:
                    final_text = content
    finally:
        set_phyxio_state(phyxio_io, "idle")

    return final_text, interrupts


def is_tool_call_history_error(err: Exception) -> bool:
    if not isinstance(err, BadRequestError):
        return False
    text = str(err)
    return (
        "assistant message with 'tool_calls' must be followed by tool messages" in text
        and "tool_call_id" in text
    )


def acquire_single_instance_lock():
    lock_path = os.getenv("MAYA_RUN_LOCK", "/tmp/maya_run_with_vad.lock")
    lock_file = open(lock_path, "w", encoding="utf-8")
    try:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        owner = "unknown"
        try:
            with open(lock_path, "r", encoding="utf-8") as existing:
                owner = (existing.read() or "").strip() or owner
        except Exception:
            pass
        print(
            f"[runner] Another run_with_vad instance is active (pid={owner}). "
            "Stop it before starting a new one."
        )
        lock_file.close()
        return None

    lock_file.seek(0)
    lock_file.truncate()
    lock_file.write(str(os.getpid()))
    lock_file.flush()
    return lock_file


def main() -> None:
    lock_file = acquire_single_instance_lock()
    if lock_file is None:
        return
    atexit.register(lock_file.close)

    user_id = os.getenv("MAYA_RUN_USER_ID", "test1")
    thread_id = os.getenv("MAYA_RUN_THREAD_ID") or f"run-{uuid.uuid4().hex[:10]}"
    base_config = {
        "configurable": {"thread_id": thread_id, "user_id": user_id},
        "recursion_limit": 40,
    }

    app = build_supervisor()

    asr_client, asr_model = init_asr()
    voice = VoiceConfig(
        seconds=VOICE_SECONDS,
        rate=VOICE_RATE,
    )
    tts = init_tts()
    phyxio_io = init_phyxio_io()

    if phyxio_io is not None:
        print("Type a message or speak from the Phyxio mirror. (Ctrl+C to exit.)")
    elif asr_client:
        print("Type a message or press Enter for voice input. (Ctrl+C to exit.)")
    else:
        print("Type your message (blank line to exit).")

    while True:
        user_message = read_text(
            None,
            tts,
            asr_client,
            asr_model,
            voice,
            phyxio_io,
            allow_exit=True,
            blank_returns_none=True,
        )
        if user_message is None:
            break

        payload = {"messages": [{"role": "user", "content": user_message}]}

        while True:
            try:
                final_text, interrupts = stream_updates(app, payload, base_config, phyxio_io)
            except Exception as err:
                if is_tool_call_history_error(err):
                    new_thread = f"run-{uuid.uuid4().hex[:10]}"
                    base_config["configurable"]["thread_id"] = new_thread
                    print(
                        "[runner] recovered from invalid stored tool-call history; "
                        f"switched to thread_id={new_thread} and retrying."
                    )
                    final_text, interrupts = stream_updates(app, payload, base_config, phyxio_io)
                else:
                    raise
            if interrupts:
                answers: list[str] = []
                for interrupt_ in interrupts:
                    prompt = str(interrupt_.value)
                    answer = read_text(prompt, tts, asr_client, asr_model, voice, phyxio_io)
                    answers.append(answer)
                resume_value = answers[0] if len(answers) == 1 else answers
                payload = Command(resume=resume_value)
                continue
            break

        if final_text:
            final_text = strip_citations_and_references(strip_markdown(final_text))
            print(f"\n***************\n\nMAYA: {final_text}")
            speak_output(tts, phyxio_io, final_text, end_interaction=True)


if __name__ == "__main__":
    main()
