from __future__ import annotations

import os
import sys
import queue
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf
from dotenv import load_dotenv
from openai import OpenAI, AzureOpenAI
import azure.cognitiveservices.speech as speechsdk
from langgraph.types import Command
from utils import strip_markdown, strip_citations_and_references

ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)

if __package__ in {None, ""}:
    sys.path.append(str(ROOT))
    from maya.framework.supervisor.factory import build_supervisor
else:
    from .factory import build_supervisor


@dataclass(frozen=True)
class VoiceConfig:
    seconds: float
    rate: int
    silence_seconds: float
    silence_threshold: float


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


def read_text(
    prompt_text: str | None,
    tts: speechsdk.SpeechSynthesizer | None,
    asr_client: object | None,
    asr_model: str | None,
    voice: VoiceConfig,
    *,
    allow_exit: bool = False,
    blank_returns_none: bool = False,
) -> str | None:
    if prompt_text:
        speak(tts, prompt_text)
        input_prompt = f"MAYA: {prompt_text}\nYou: "
    else:
        input_prompt = "You: "

    def capture_voice() -> str:
        print(f"Recording... speak and pause to stop (max {voice.seconds:.1f}s)")

        audio_queue: queue.Queue[np.ndarray] = queue.Queue()
        recorded_blocks: list[np.ndarray] = []
        silence_frames_target = int(voice.silence_seconds * voice.rate)
        max_frames = int(voice.seconds * voice.rate)
        silence_run = 0
        total_frames = 0
        noise_floor = voice.silence_threshold
        max_level = 0.0

        def callback(indata: np.ndarray, _frames: int, _time, status) -> None:
            if status:
                print(f"[audio] {status}", file=sys.stderr)
            audio_queue.put(indata.copy())

        with sd.InputStream(
            samplerate=voice.rate,
            channels=1,
            dtype="float32",
            callback=callback,
        ):
            while True:
                block = audio_queue.get()
                recorded_blocks.append(block)
                total_frames += len(block)

                block_level = float(np.sqrt(np.mean(np.square(block))))
                max_level = max(max_level, block_level)

                if block_level < voice.silence_threshold * 4:
                    noise_floor = 0.9 * noise_floor + 0.1 * block_level

                effective_threshold = max(
                    voice.silence_threshold,
                    noise_floor * 3.0,
                    max_level * 0.05,
                )

                silence_run = silence_run + len(block) if block_level < effective_threshold else 0

                if silence_run >= silence_frames_target or total_frames >= max_frames:
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
        try:
            user_input = input(input_prompt).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return None

        if allow_exit and user_input in {"/exit", "/quit"}:
            return None

        wants_voice = user_input == "/voice" or (user_input == "" and asr_client)
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

        if blank_returns_none:
            return None
        return ""


def stream_updates(app, payload, base_config):
    final_text = None
    interrupts = []

    for chunk in app.stream(
        payload,
        config=base_config,
        stream_mode="updates",
        subgraphs=True,
    ):
        update = chunk[1] if isinstance(chunk, tuple) else chunk

        if "__interrupt__" in update:
            interrupts.extend(update["__interrupt__"])
            continue

        for node, node_payload in update.items():
            last = node_payload["messages"][-1]
            if hasattr(last, "get"):
                role = last.get("role") or last.get("type")
                name = last.get("name")
                content = last.get("content")
            else:
                role = last.type
                name = last.name
                content = last.content

            if name:
                print(f"[{node}] {role} :: {name}")

            if role in {"assistant", "ai"} and content:
                final_text = content

    return final_text, interrupts


def main() -> None:
    load_dotenv()

    user_id = os.getenv("MAYA_RUN_USER_ID", "test")
    thread_id = os.getenv("MAYA_RUN_THREAD_ID", "test17")
    base_config = {
        "configurable": {"thread_id": thread_id, "user_id": user_id},
        "recursion_limit": 40,
    }

    app = build_supervisor()

    asr_client, asr_model = init_asr()
    voice = VoiceConfig(
        seconds=float(os.getenv("MAYA_VOICE_SECONDS", "20")),
        rate=int(os.getenv("MAYA_VOICE_SAMPLE_RATE", "16000")),
        silence_seconds=float(os.getenv("MAYA_VOICE_SILENCE_SECONDS", "1.5")),
        silence_threshold=float(os.getenv("MAYA_VOICE_SILENCE_THRESHOLD", "0.015")),
    )
    tts = init_tts()

    if asr_client:
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
            allow_exit=True,
            blank_returns_none=True,
        )
        if user_message is None:
            break

        payload = {"messages": [{"role": "user", "content": user_message}]}

        while True:
            final_text, interrupts = stream_updates(app, payload, base_config)
            if interrupts:
                answers: list[str] = []
                for interrupt_ in interrupts:
                    prompt = str(interrupt_.value)
                    answer = read_text(prompt, tts, asr_client, asr_model, voice)
                    answers.append(answer)
                resume_value = answers[0] if len(answers) == 1 else answers
                payload = Command(resume=resume_value)
                continue
            break

        if final_text:
            final_text = strip_citations_and_references(strip_markdown(final_text))
            print(f"\n***************\n\nMAYA: {final_text}")
            speak(tts, final_text)


if __name__ == "__main__":
    main()
