from __future__ import annotations

import os
import sys
import time
import queue
import logging as log
import types
from dataclasses import dataclass
from io import BytesIO
from threading import Thread, Event, Lock
from typing import Optional, Any
import json
import numpy as np
import sounddevice as sd
import soundfile as sf
from dotenv import load_dotenv
from openai import OpenAI, AzureOpenAI
import azure.cognitiveservices.speech as speechsdk

import Ice  # from zeroc-ice


def _install_ice_compat() -> None:
    # -----------------------------
    # 1) Ice.loadSlice: new Ice expects list[str]
    # -----------------------------
    orig_loadSlice = Ice.loadSlice

    def loadSlice_compat(cmd, *args, **kwargs):
        if isinstance(cmd, str):
            incs = []
            for d in ("/usr/share/slice", "/usr/share/slice/phyxio"):
                if os.path.isdir(d):
                    incs.append(f"-I{d}")
            return orig_loadSlice([*incs, cmd], *args, **kwargs)
        return orig_loadSlice(cmd, *args, **kwargs)

    Ice.loadSlice = loadSlice_compat  # type: ignore

    # -----------------------------
    # 2) Communicator.stringToIdentity: old code calls communicator.stringToIdentity(...)
    #    new Ice provides Ice.stringToIdentity(...)
    # -----------------------------
    def _ensure_stringToIdentity(communicator) -> None:
        if hasattr(communicator, "stringToIdentity"):
            return
        if not hasattr(Ice, "stringToIdentity"):
            return

        def _sti(self, s: str):
            return Ice.stringToIdentity(s)

        # Try patching the class (best: keeps type == Communicator)
        try:
            setattr(communicator.__class__, "stringToIdentity", _sti)
            return
        except Exception:
            pass

        # Fallback: patch instance (may fail if Communicator has no __dict__)
        try:
            communicator.stringToIdentity = types.MethodType(_sti, communicator)  # type: ignore
        except Exception:
            pass

    # -----------------------------
    # 3) Ice.initialize: old code uses data=..., new Ice expects initData=...
    # -----------------------------
    orig_initialize = Ice.initialize

    def initialize_compat(*args, **kwargs):
        if "data" in kwargs and "initData" not in kwargs:
            data = kwargs.pop("data")
            try:
                comm = orig_initialize(*args, initData=data, **kwargs)
            except TypeError:
                comm = orig_initialize(*args, **kwargs)
        else:
            comm = orig_initialize(*args, **kwargs)

        _ensure_stringToIdentity(comm)
        return comm

    Ice.initialize = initialize_compat  # type: ignore

# _install_ice_compat()


from phyxio import connect

load_dotenv()
log.basicConfig(level=log.INFO)


# -----------------------------
# Config
# -----------------------------
@dataclass(frozen=True)
class VoiceConfig:
    max_seconds: float = float(os.getenv("MAYA_VOICE_SECONDS", "20"))
    rate: int = int(os.getenv("MAYA_VOICE_SAMPLE_RATE", "16000"))
    silence_seconds: float = float(os.getenv("MAYA_VOICE_SILENCE_SECONDS", "1.5"))
    silence_threshold: float = float(os.getenv("MAYA_VOICE_SILENCE_THRESHOLD", "0.015"))
    min_seconds: float = float(os.getenv("MAYA_VOICE_MIN_SECONDS", "0.6"))
    mic_device: Optional[int] = None  # MAYA_MIC_DEVICE=int index


def _mic_device_from_env() -> Optional[int]:
    v = os.getenv("MAYA_MIC_DEVICE")
    return int(v) if v and v.isdigit() else None


def _init_whisper() -> tuple[Any, str]:
    k = os.getenv("OPENAI_WHISPER_API_KEY")
    if k:
        return OpenAI(api_key=k), os.getenv("OPENAI_WHISPER_MODEL", "whisper-1")

    azure_key = os.getenv("AZURE_OPENAI_WHISPER_API_KEY")
    azure_endpoint = os.getenv("AZURE_OPENAI_WHISPER_ENDPOINT")
    azure_deploy = os.getenv("AZURE_OPENAI_DEPLOYMENT_WHISPER")
    azure_ver = (
        os.getenv("AZURE_OPENAI_WHISPER_API_VERSION")
        or os.getenv("AZURE_OPENAI_API_VERSION")
        or "2024-06-01"
    )
    if not (azure_key and azure_endpoint and azure_deploy):
        raise RuntimeError("Whisper not configured (OpenAI or Azure OpenAI).")

    base = azure_endpoint.split("/openai/")[0] if "/openai/" in azure_endpoint else azure_endpoint
    return AzureOpenAI(api_key=azure_key, api_version=azure_ver, azure_endpoint=base), azure_deploy


def _init_tts() -> speechsdk.SpeechSynthesizer:
    speech_key = os.getenv("AZURE_SPEECH_KEY")
    speech_region = os.getenv("AZURE_SPEECH_REGION")
    if not (speech_key and speech_region):
        raise RuntimeError("Missing AZURE_SPEECH_KEY / AZURE_SPEECH_REGION for TTS")

    voice = os.getenv("AZURE_SPEECH_VOICE", "en-GB-LibbyNeural")
    cfg = speechsdk.SpeechConfig(subscription=speech_key, region=speech_region)
    cfg.speech_synthesis_voice_name = voice
    cfg.set_speech_synthesis_output_format(speechsdk.SpeechSynthesisOutputFormat.Riff24Khz16BitMonoPcm)
    log.info("[TTS] voice=%s", voice)
    return speechsdk.SpeechSynthesizer(speech_config=cfg, audio_config=None)


def _list_audio_devices_once() -> None:
    try:
        devs = sd.query_devices()
        log.info("---- Audio Devices ----")
        for i, d in enumerate(devs):
            log.info("%d | %s | in=%s out=%s", i, d["name"], d["max_input_channels"], d["max_output_channels"])
        log.info("sd.default.device=%s", sd.default.device)
        log.info("-----------------------")
    except Exception as e:
        log.warning("Could not list audio devices: %s", e)


# -----------------------------
# Phyxio Connector
# -----------------------------
class MyPhyxioConnector(connect.PhyxioConnector):
    """
    Phyxio ASR is UI only.
    - UI tap -> change_interaction_state_request(True) on new Phyxio builds
      or asr_change_state_request(True) on older builds
    - We record mic until pause (max seconds), transcribe with Whisper,
      then push text into asr_queue for the agent.
    - Agent replies -> speak_text() (Azure TTS) + agent/asr UI updates
    """

    def __init__(self, config_path: str | None = None):
        super().__init__(config_path)

        # add this patch to avoid 'address already in use from ICE'
        # Ensure Ice adapter is created only once
        if not hasattr(self, "_adapter_created"):
            self._adapter_created = True
        else:
            log.warning("Adapter already created; skipping duplicate creation")
        # add this patch to avoid 'address already in use from ICE'

        self.client_connected = False
        self.asr_queue: queue.Queue[str] = queue.Queue()

        self.voice = VoiceConfig(mic_device=_mic_device_from_env())
        self.cooldown_s = float(os.getenv("ASR_TTS_COOLDOWN", "1.2"))
        self._cooldown_until = 0.0

        self._lock = Lock()
        self._recording = False
        self._cancel_recording = Event()
        self._record_thread: Optional[Thread] = None

        self._tts_running = Event()
        self._tts_queue: queue.Queue[str] = queue.Queue()

        self._asr_client, self._asr_model = _init_whisper()
        self._tts = _init_tts()

        _list_audio_devices_once()
        Thread(target=self._tts_loop, daemon=True).start()

        log.info("✅ Phyxio connector ready (Whisper ASR + Azure TTS)")

    def on_client_connected(self):
        log.info("🪞 Phyxio Mirror connected")
        self.client_connected = True

    def change_interaction_state_request(self, active: bool):
        return self._handle_interaction_state_request(active, "change_interaction_state_request")

    def asr_change_state_request(self, active: bool):
        return self._handle_interaction_state_request(active, "asr_change_state_request")

    def _handle_interaction_state_request(self, active: bool, source: str):
        """
        Must return True/(True, "") to accept, or (False, reason) to deny.
        """
        log.info("[PHYXIO] %s(active=%s)", source, active)

        if not self.client_connected:
            # Fallback: if a tap reaches us but connect callback was missed, treat this
            # as a late handshake and proceed.
            try:
                self._agent_set_text("Connecting…")
                self.client_connected = True
                log.warning("Late client handshake via ASR tap; continuing.")
            except Exception:
                return False, "I am not ready yet"

        now = time.monotonic()
        if now < self._cooldown_until:
            return False, "One moment…"

        with self._lock:
            if active:
                if self._tts_running.is_set():
                    return False, "I'm speaking. Tap again when I finish."
                if self._recording:
                    return False, "Already listening"

                self._cancel_recording.clear()
                self._recording = True

                try:
                    self._start_interaction()
                    self._asr_set_state("listening")
                    self._user_set_text("Listening…")
                except Exception as e:
                    log.warning("UI listening state failed: %s", e)

                self._record_thread = Thread(target=self._record_transcribe_once, daemon=True)
                self._record_thread.start()
                return True

            # active == False -> cancel current recording (if any)
            self._cancel_recording.set()
            self._recording = False
            self._ui_off()
            return True

    # -----------------------------
    # Recording + Whisper ASR
    # -----------------------------
    def _record_transcribe_once(self) -> None:
        try:
            wav, stats = self._record_until_pause()
            log.info("[mic] %s", stats)

            if self._cancel_recording.is_set():
                self._ui_off()
                return

            if stats["peak"] < 0.005 or not stats["speech_seen"]:
                self._user_set_text("No speech detected. Tap and try again.")
                self._ui_off()
                return

            self._user_set_text("Transcribing…")
            text = self._transcribe_wav(wav)
            log.info("[ASR] %r", text)

            if not text:
                self._user_set_text("No speech detected. Tap and try again.")
                self._ui_off()
                return

            # show transcript then push to agent
            self._user_set_text(text)
            self._ui_off()
            self.asr_queue.put(text)

        except Exception:
            log.exception("ASR failed")
            self._user_set_text("ASR error (see logs).")
            self._ui_off()
        finally:
            with self._lock:
                self._recording = False

    def _record_until_pause(self) -> tuple[BytesIO, dict]:
        v = self.voice
        audio_q: queue.Queue[np.ndarray] = queue.Queue()
        blocks: list[np.ndarray] = []

        silence_target = int(v.silence_seconds * v.rate)
        max_frames = int(v.max_seconds * v.rate)
        min_frames = int(v.min_seconds * v.rate)

        silence_run = 0
        total_frames = 0
        noise_floor = v.silence_threshold
        max_level = 0.0
        speech_seen = False
        last_eff = v.silence_threshold

        def cb(indata: np.ndarray, _frames: int, _time, status) -> None:
            if status:
                print(f"[audio] {status}", file=sys.stderr)
            if self._cancel_recording.is_set() or self._tts_running.is_set():
                raise sd.CallbackStop()
            audio_q.put(indata.copy())

        with sd.InputStream(
            samplerate=v.rate,
            channels=1,
            dtype="float32",
            device=v.mic_device,
            callback=cb,
        ):
            while True:
                if self._cancel_recording.is_set() or self._tts_running.is_set():
                    break

                block = audio_q.get()
                blocks.append(block)
                total_frames += len(block)

                level = float(np.sqrt(np.mean(np.square(block))) + 1e-12)
                max_level = max(max_level, level)

                if level < v.silence_threshold * 4:
                    noise_floor = 0.9 * noise_floor + 0.1 * level

                eff = max(v.silence_threshold, noise_floor * 3.0, max_level * 0.05)
                last_eff = eff

                if level >= eff:
                    speech_seen = True
                    silence_run = 0
                else:
                    silence_run = silence_run + len(block) if speech_seen else 0

                if total_frames >= max_frames:
                    break
                if speech_seen and total_frames >= min_frames and silence_run >= silence_target:
                    break

        frames = np.concatenate(blocks, axis=0) if blocks else np.zeros((0, 1), dtype=np.float32)
        peak = float(np.max(np.abs(frames))) if frames.size else 0.0
        rms = float(np.sqrt(np.mean(np.square(frames))) + 1e-12) if frames.size else 0.0

        buf = BytesIO()
        with sf.SoundFile(buf, mode="w", samplerate=v.rate, channels=1, format="WAV") as f:
            f.write(frames)
        buf.seek(0)

        stats = {
            "seconds_captured": total_frames / v.rate,
            "peak": peak,
            "rms": rms,
            "speech_seen": speech_seen,
            "noise_floor": float(noise_floor),
            "max_level": float(max_level),
            "last_effective_threshold": float(last_eff),
            "device": v.mic_device,
        }
        return buf, stats

    def _transcribe_wav(self, wav: BytesIO) -> str:
        tr = self._asr_client.audio.transcriptions.create(
            model=self._asr_model,
            file=("input.wav", wav, "audio/wav"),
        )
        return (getattr(tr, "text", None) or "").strip()

    # -----------------------------
    # Azure TTS (worker thread)
    # -----------------------------
    def speak_text(self, text: str) -> None:
        text = (text or "").strip()
        if text:
            self._tts_queue.put(text)

    def _tts_loop(self) -> None:
        while True:
            try:
                text = self._tts_queue.get(timeout=1)
            except queue.Empty:
                continue

            text = (text or "").strip()
            if not text:
                continue

            self._cancel_recording.set()
            self._tts_running.set()
            try:
                self._asr_set_state("stop")
                self._agent_set_state("talking")

                res = self._tts.speak_text_async(text).get()
                audio = getattr(res, "audio_data", None)
                if not audio:
                    log.error("TTS returned no audio_data")
                    continue

                data, rate = sf.read(BytesIO(audio), dtype="float32")
                sd.play(data, rate)
                sd.wait()
            except Exception:
                log.exception("TTS failed")
            finally:
                self._agent_set_state("idle")
                self._tts_running.clear()
                self._cooldown_until = time.monotonic() + self.cooldown_s

    # -----------------------------
    # UI helpers
    # -----------------------------
    @staticmethod
    def _call_ui(obj: object, method_name: str, *args) -> bool:
        method = getattr(obj, method_name, None)
        if not callable(method):
            return False
        try:
            method(*args)
            return True
        except Exception as err:
            log.debug("UI call %s failed: %s", method_name, err)
            return False

    def _start_interaction(self) -> None:
        agent = getattr(self, "agent", None)
        if agent is not None:
            self._call_ui(agent, "start_interaction")

    def _end_interaction(self) -> None:
        agent = getattr(self, "agent", None)
        if agent is not None:
            self._call_ui(agent, "end_interaction")

    def start_interaction(self) -> None:
        self._start_interaction()

    def end_interaction(self, delay_seconds: float = 0.0) -> None:
        delay = max(float(delay_seconds or 0.0), 0.0)
        if delay <= 0:
            self._end_interaction()
            return

        def delayed_end() -> None:
            wait_started = time.monotonic()
            while self._tts_running.is_set() or not self._tts_queue.empty():
                if time.monotonic() - wait_started > 120:
                    break
                time.sleep(0.1)
            time.sleep(delay)
            self._end_interaction()

        Thread(target=delayed_end, daemon=True).start()

    def _asr_activate(self, active: bool) -> None:
        asr = getattr(self, "asr", None)
        if asr is not None:
            self._call_ui(asr, "activate", active)

    def _asr_set_state(self, state: str) -> None:
        asr = getattr(self, "asr", None)
        if asr is not None and self._call_ui(asr, "set_state", state):
            return
        if state == "listening":
            self._asr_activate(True)
        elif state == "stop":
            self._asr_activate(False)

    def _user_set_text(self, text: str) -> None:
        asr = getattr(self, "asr", None)
        if asr is not None:
            self._call_ui(asr, "set_text", text)

    def _agent_set_text(self, text: str) -> None:
        agent = getattr(self, "agent", None)
        if agent is not None and self._call_ui(agent, "set_text", text):
            return
        self._user_set_text(text)

    def _agent_set_state(self, state: str) -> None:
        agent = getattr(self, "agent", None)
        if agent is not None:
            self._call_ui(agent, "set_state", state)

    def _ui_off(self) -> None:
        self._asr_set_state("stop")


# -----------------------------
# Thin service wrapper
# -----------------------------
class PhyxioService:
    def __init__(self):
        self._phyxio = MyPhyxioConnector()

        r = self._phyxio.routine
        log.info("routine type: %s", type(r))
        log.info("routine dir: %s", [a for a in dir(r) if not a.startswith("_")])
        log.info("routine dict: %s", getattr(r, "__dict__", None))


        data = self._phyxio.routine
        log.info(f"Routine data:\n {data}")
        log.info("✅ PhyxioService ready")

    @property
    def asr_queue(self) -> queue.Queue[str]:
        return self._phyxio.asr_queue

    # def stop(self) -> None:
    #     self._phyxio.stop()
    def stop(self) -> None:
        try:
            self._phyxio.stop()
        except Exception:
            pass
        try:
            if hasattr(self._phyxio, "_ic"):
                self._phyxio._ic.destroy()
                log.info("Destroyed Ice communicator")
        except Exception:
            pass

    def show_text(self, text: str) -> None:
        self._phyxio._agent_set_text(text)

    def set_agent_state(self, state: str) -> None:
        self._phyxio._agent_set_state(state)

    def start_interaction(self) -> None:
        self._phyxio.start_interaction()

    def end_interaction(self, delay_seconds: float = 0.0) -> None:
        self._phyxio.end_interaction(delay_seconds)

    def speak_text(self, text: str) -> None:
        self._phyxio.speak_text(text)

    def get_routine_list(self) -> Any:
        return self._phyxio.routine.get_list()

    def get_sessions(self, page: int = 1) -> Any:
        get_sessions = getattr(self._phyxio.routine, "get_sessions", None)
        if not callable(get_sessions):
            raise RuntimeError("Phyxio routine.get_sessions is not available in this connector version.")
        return get_sessions(page=page)

    @staticmethod
    def _routine_id_candidates(rid: str) -> list[Any]:
        text = str(rid or "").strip()
        candidates: list[Any] = []
        if text:
            try:
                candidates.append(int(text))
            except ValueError:
                pass
            if text not in candidates:
                candidates.append(text)
        return candidates

    def _perform_routine_action(self, action: str, rid: str) -> str:
        method = getattr(self._phyxio.routine, action)
        candidates = self._routine_id_candidates(rid)
        if not candidates:
            return "Could not do that: no routine id was provided"

        last_error: Exception | None = None
        for candidate in candidates:
            try:
                method(candidate)
                return {
                    "show": "Showing routine",
                    "run": "Started routine",
                    "calibrate": "Calibrated routine",
                }[action]
            except connect.UnableToPerform as err:
                return f"Could not do that: {err.reason}"
            except (TypeError, ValueError) as err:
                last_error = err

        return f"Could not do that: {last_error}"

    def show_routine(self, rid: str) -> str:
        return self._perform_routine_action("show", rid)

    def run_routine(self, rid: str) -> str:
        return self._perform_routine_action("run", rid)

    def calibrate_routine(self, rid: str) -> str:
        return self._perform_routine_action("calibrate", rid)
