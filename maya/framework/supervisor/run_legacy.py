from __future__ import annotations

import os
import sys
import queue
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


def init_whisper_client() -> tuple[object | None, str | None]:
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


def init_speech_synthesizer() -> speechsdk.SpeechSynthesizer | None:
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


def speak_text(speech_synthesizer: speechsdk.SpeechSynthesizer | None, text: str) -> None:
    if not speech_synthesizer or not text:
        return
    try:
        result = speech_synthesizer.speak_text_async(text).get()
        audio_bytes = getattr(result, "audio_data", None)
        if audio_bytes:
            buffer = BytesIO(audio_bytes)
            data, rate = sf.read(buffer, dtype="float32")
            sd.play(data, rate)
            sd.wait()
    except Exception as tts_err:
        print(f"[tts] {tts_err}")


def record_voice(
    voice_seconds: float,
    voice_rate: int,
    voice_silence_seconds: float,
    voice_silence_threshold: float,
) -> np.ndarray | None:
    print(f"Recording... speak and pause to stop (max {voice_seconds:.1f}s)")

    audio_queue: queue.Queue[np.ndarray] = queue.Queue()
    recorded_blocks: list[np.ndarray] = []
    silence_frames_target = int(voice_silence_seconds * voice_rate)
    max_frames = int(voice_seconds * voice_rate)
    silence_run = 0
    total_frames = 0
    noise_floor = voice_silence_threshold
    max_level = 0.0

    def callback(indata: np.ndarray, _frames: int, _time, status) -> None:
        if status:
            print(f"[audio] {status}", file=sys.stderr)
        audio_queue.put(indata.copy())

    with sd.InputStream(
        samplerate=voice_rate,
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

            if block_level < voice_silence_threshold * 4:
                noise_floor = 0.9 * noise_floor + 0.1 * block_level

            effective_threshold = max(
                voice_silence_threshold,
                noise_floor * 3.0,
                max_level * 0.05,
            )

            silence_run = silence_run + len(block) if block_level < effective_threshold else 0

            if silence_run >= silence_frames_target or total_frames >= max_frames:
                break

    if not recorded_blocks:
        return None

    return np.concatenate(recorded_blocks, axis=0)


def transcribe_audio(
    whisper_client: object,
    whisper_model: str,
    frames: np.ndarray,
    voice_rate: int,
) -> str:
    buffer = BytesIO()
    with sf.SoundFile(buffer, mode="w", samplerate=voice_rate, channels=1, format="WAV") as wav_file:
        wav_file.write(frames)
    buffer.seek(0)

    transcription = whisper_client.audio.transcriptions.create(
        model=whisper_model,
        file=("input.wav", buffer, "audio/wav"),
    )
    return (getattr(transcription, "text", None) or "").strip()


def get_user_message(
    whisper_client: object | None,
    whisper_model: str | None,
    voice_seconds: float,
    voice_rate: int,
    voice_silence_seconds: float,
    voice_silence_threshold: float,
) -> str | None:
    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return None

        if user_input in {"/exit", "/quit"}:
            return None

        if not user_input:
            if whisper_client:
                user_input = "/voice"
            else:
                return None

        if user_input != "/voice":
            return user_input

        if whisper_client is None or whisper_model is None:
            print("Whisper key not configured. Please provide text input instead.")
            continue

        frames = record_voice(
            voice_seconds,
            voice_rate,
            voice_silence_seconds,
            voice_silence_threshold,
        )
        if frames is None:
            print("No speech detected. Try again.")
            continue

        spoken_text = transcribe_audio(whisper_client, whisper_model, frames, voice_rate)
        if not spoken_text:
            print("No speech detected. Try again.")
            continue

        print(f"You (voice): {spoken_text}")
        return spoken_text


def message_fields(message) -> tuple[str | None, str | None, str | None]:
    role = getattr(message, "type", None)
    name = getattr(message, "name", None)
    content = getattr(message, "content", None)
    getter = getattr(message, "get", None)
    if getter:
        role = getter("role") or getter("type") or role
        name = getter("name") or name
        content = getter("content") or content
    return role, name, content


def normalize_update(chunk):
    return chunk[1] if isinstance(chunk, tuple) else chunk


def stream_until_interrupt(app, payload, base_config):
    final_text = None
    interrupts = []

    for chunk in app.stream(
        payload,
        config=base_config,
        stream_mode="updates",
        subgraphs=True,
    ):
        update = normalize_update(chunk)

        if "__interrupt__" in update:
            interrupts.extend(update["__interrupt__"])
            continue

        for node, node_payload in update.items():
            messages = node_payload["messages"]
            last = messages[-1]
            role, name, content = message_fields(last)

            if name:
                print(f"[{node}] {role} :: {name}")

            if role in {"assistant", "ai"} and content:
                final_text = content

    return final_text, interrupts


def prompt_interrupt_response(
    prompt: str,
    speech_synthesizer: speechsdk.SpeechSynthesizer | None,
    whisper_client: object | None,
    whisper_model: str | None,
    voice_seconds: float,
    voice_rate: int,
    voice_silence_seconds: float,
    voice_silence_threshold: float,
) -> str:
    speak_text(speech_synthesizer, prompt)
    while True:
        answer = input(f"MAYA: {prompt}\nYou: ").strip()
        if answer:
            return answer
        if whisper_client and whisper_model:
            frames = record_voice(
                voice_seconds,
                voice_rate,
                voice_silence_seconds,
                voice_silence_threshold,
            )
            if frames is None:
                print("No speech detected. Try again.")
                continue
            spoken_text = transcribe_audio(whisper_client, whisper_model, frames, voice_rate)
            if not spoken_text:
                print("No speech detected. Try again.")
                continue
            print(f"You (voice): {spoken_text}")
            return spoken_text
        return answer


def collect_interrupt_answers(
    interrupts,
    speech_synthesizer: speechsdk.SpeechSynthesizer | None,
    whisper_client: object | None,
    whisper_model: str | None,
    voice_seconds: float,
    voice_rate: int,
    voice_silence_seconds: float,
    voice_silence_threshold: float,
) -> str | list[str]:
    answers: list[str] = []
    for interrupt_ in interrupts:
        prompt = str(interrupt_.value)
        answer = prompt_interrupt_response(
            prompt,
            speech_synthesizer,
            whisper_client,
            whisper_model,
            voice_seconds,
            voice_rate,
            voice_silence_seconds,
            voice_silence_threshold,
        )
        answers.append(answer)
    return answers[0] if len(answers) == 1 else answers


def main() -> None:
    load_dotenv()

    user_id = os.getenv("MAYA_RUN_USER_ID", "test")
    thread_id = os.getenv("MAYA_RUN_THREAD_ID", "test17")
    base_config = {
        "configurable": {"thread_id": thread_id, "user_id": user_id},
        "recursion_limit": 40,
    }

    app = build_supervisor()

    whisper_client, whisper_model = init_whisper_client()

    voice_seconds = float(os.getenv("MAYA_VOICE_SECONDS", "20"))
    voice_rate = int(os.getenv("MAYA_VOICE_SAMPLE_RATE", "16000"))
    voice_silence_seconds = float(os.getenv("MAYA_VOICE_SILENCE_SECONDS", "1.5"))
    voice_silence_threshold = float(os.getenv("MAYA_VOICE_SILENCE_THRESHOLD", "0.015"))

    speech_synthesizer = init_speech_synthesizer()

    if whisper_client:
        print("Type a message or press Enter for voice input. (Ctrl+C to exit.)")
    else:
        print("Type your message (blank line to exit).")

    while True:
        user_message = get_user_message(
            whisper_client,
            whisper_model,
            voice_seconds,
            voice_rate,
            voice_silence_seconds,
            voice_silence_threshold,
        )
        if user_message is None:
            break

        payload = {"messages": [{"role": "user", "content": user_message}]}

        while True:
            final_text, interrupts = stream_until_interrupt(app, payload, base_config)
            if interrupts:
                resume_value = collect_interrupt_answers(
                    interrupts,
                    speech_synthesizer,
                    whisper_client,
                    whisper_model,
                    voice_seconds,
                    voice_rate,
                    voice_silence_seconds,
                    voice_silence_threshold,
                )
                payload = Command(resume=resume_value)
                continue
            break

        if final_text:
            final_text = strip_markdown(final_text)
            final_text = strip_citations_and_references(final_text)
            print(f"\n***************\n\nMAYA: {final_text}")
            speak_text(speech_synthesizer, final_text)


if __name__ == "__main__":
    main()
