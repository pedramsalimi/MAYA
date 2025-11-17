from __future__ import annotations

import os
import sys
from io import BytesIO
from pathlib import Path

import sounddevice as sd
import soundfile as sf
from dotenv import load_dotenv
from openai import OpenAI, AzureOpenAI
import azure.cognitiveservices.speech as speechsdk
from utils import strip_markdown, strip_citations_and_references

ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)

if __package__ in {None, ""}:
    sys.path.append(str(ROOT))
    from maya.framework.supervisor.factory import build_supervisor
else:
    from .factory import build_supervisor

def main() -> None:
    load_dotenv()

    user_id = os.getenv("MAYA_RUN_USER_ID", "test")
    thread_id = os.getenv("MAYA_RUN_THREAD_ID", "test")
    base_config = {
        "configurable": {"thread_id": thread_id, "user_id": user_id},
        "recursion_limit": 40,
    }

    app = build_supervisor()

    whisper_model = None
    whisper_client = None

    whisper_key = os.getenv("OPENAI_WHISPER_API_KEY")
    if whisper_key:
        whisper_model = os.getenv("OPENAI_WHISPER_MODEL", "whisper-1")
        whisper_client = OpenAI(api_key=whisper_key)
    else:
        azure_key = os.getenv("AZURE_OPENAI_WHISPER_API_KEY")
        azure_endpoint = os.getenv("AZURE_OPENAI_WHISPER_ENDPOINT")
        azure_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_WHISPER")
        azure_version = os.getenv("AZURE_OPENAI_WHISPER_API_VERSION") or os.getenv("AZURE_OPENAI_API_VERSION") or "2024-06-01"
        if azure_key and azure_endpoint and azure_deployment:
            base_endpoint = azure_endpoint.split("/openai/")[0] if "/openai/" in azure_endpoint else azure_endpoint
            whisper_model = azure_deployment
            whisper_client = AzureOpenAI(
                api_key=azure_key,
                api_version=azure_version,
                azure_endpoint=base_endpoint,
            )

    voice_seconds = float(os.getenv("MAYA_VOICE_SECONDS", "6"))
    voice_rate = int(os.getenv("MAYA_VOICE_SAMPLE_RATE", "16000"))

    speech_synthesizer = None
    speech_voice = os.getenv("AZURE_SPEECH_VOICE", "en-GB-RyanNeural")
    speech_key = os.getenv("AZURE_SPEECH_KEY")
    speech_region = os.getenv("AZURE_SPEECH_REGION")
    if speech_key and speech_region:
        speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=speech_region)
        speech_config.speech_synthesis_voice_name = speech_voice
        speech_config.set_speech_synthesis_output_format(
            speechsdk.SpeechSynthesisOutputFormat.Riff24Khz16BitMonoPcm
        )
        speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=None)

    if whisper_client:
        print("Type a message or press Enter for voice input. (Ctrl+C to exit.)")
    else:
        print("Type your message (blank line to exit).")

    while True:
        try:
            user_input = input("You: ")
        except (EOFError, KeyboardInterrupt):
            print()
            break

        user_input = user_input.strip()
        if user_input in {"/exit", "/quit"}:
            break

        if not user_input:
            if whisper_client:
                user_input = "/voice"
            else:
                break

        # # ONLY VOICE
        # if whisper_client:
        #     user_input = "/voice"
        # # # ONLY VOICE   
        if user_input == "/voice":
            if whisper_client is None:
                print("Whisper key not configured. Please provide text input instead.")
                continue

            print(f"Recording for {voice_seconds:.1f}s...")
            frames = sd.rec(int(voice_seconds * voice_rate), samplerate=voice_rate, channels=1, dtype="float32")
            sd.wait()

            buffer = BytesIO()
            with sf.SoundFile(buffer, mode="w", samplerate=voice_rate, channels=1, format="WAV") as wav_file:
                wav_file.write(frames)
            buffer.seek(0)

            transcription = whisper_client.audio.transcriptions.create(
                model=whisper_model,
                file=("input.wav", buffer, "audio/wav"),
            )
            spoken_text = (getattr(transcription, "text", None) or "").strip()
            if not spoken_text:
                print("No speech detected. Try again.")
                continue

            print(f"You (voice): {spoken_text}")
            payload = {"messages": [{"role": "user", "content": spoken_text}]}
        else:
            payload = {"messages": [{"role": "user", "content": user_input}]}

        final_text = None

        for chunk in app.stream(
            payload,
            config=base_config,
            stream_mode="updates",  
            subgraphs=True,          
        ):
            # normalize (namespace, data) -> data
            update = chunk[1] if (isinstance(chunk, tuple) and len(chunk) == 2) else chunk

            for node, payload in update.items():
                msgs = []
                if isinstance(payload, list):
                    msgs = payload
                elif isinstance(payload, dict):
                    msgs = payload.get("messages")
                    if isinstance(msgs, dict):
                        msgs = list(msgs.values())
                    elif msgs is None:
                        msgs = []
                    elif not isinstance(msgs, list):
                        msgs = [msgs]

                if not msgs:
                    continue

                last = msgs[-1]
                if isinstance(last, dict):
                    role = last.get("role") or last.get("type")
                    name = last.get("name")
                    content = last.get("content")
                else:
                    role = getattr(last, "type", None)
                    name = getattr(last, "name", None)
                    content = getattr(last, "content", None)

                # Tool calls (incl. transfer_to_* / transfer_back_to_supervisor)
                if name:
                    print(f"[{node}] {role} :: {name}")

                # Capture final reply
                if role in {"assistant", "ai"} and isinstance(content, str) and content:
                    final_text = content

        if final_text:
            final_text = strip_markdown(final_text)
            final_text = strip_citations_and_references(final_text)
            print(f"\n\n***************\n\nMAYA: {final_text}")
            if speech_synthesizer:
                try:
                    result = speech_synthesizer.speak_text_async(final_text).get()
                    audio_bytes = getattr(result, "audio_data", None)
                    if audio_bytes:
                        buffer = BytesIO(audio_bytes)
                        data, rate = sf.read(buffer, dtype="float32")
                        sd.play(data, rate)
                        sd.wait()
                except Exception as tts_err:
                    print(f"[tts] {tts_err}")

if __name__ == "__main__":
    main()
