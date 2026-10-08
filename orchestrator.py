import os
import uuid
import time
import asyncio
import logging
import threading
from typing import Dict, List, Optional
from dataclasses import dataclass, field
import httpx
import wave
import io
import json
from pydantic import BaseModel, Field

AUDIT_PHRASES = [
    "repetitive defects", "recurring faults", "thrust reverser",
    "flap slat lever", "speed governor", "obstruction survey",
    "unserviceable ground equipment", "pre flight medical",
    "SOP deviation", "abnormal runway contact", "worn tyres",
    "runway markings", "work order", "snag rectification"
]

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("AviationVoiceOrchestrator")

# ==========================================
# 1. SESSION MANAGEMENT MODULE
# ==========================================
@dataclass
class UserSession:
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    start_time: float = field(default_factory=time.time)
    transcript_history: List[str] = field(default_factory=list)
    extracted_data: Dict = field(default_factory=dict)
    transaction_count: int = 0
    metadata: Dict = field(default_factory=dict)

class SessionManager:
    """
    Manages user sessions, authenticates tokens, and preserves conversation
    history across multi-turn interactions.
    """
    def __init__(self):
        self._sessions: Dict[str, UserSession] = {}
        self._lock = threading.Lock()

    def create_session(self, metadata: Optional[Dict] = None) -> UserSession:
        session = UserSession(metadata=metadata or {})
        with self._lock:
            self._sessions[session.session_id] = session
        logger.info(f"🔑 Created new session: {session.session_id}")
        return session

    def get_session(self, session_id: str) -> Optional[UserSession]:
        with self._lock:
            return self._sessions.get(session_id)

    def append_transcript(self, session_id: str, text: str):
        session = self.get_session(session_id)
        if session:
            with self._lock:
                session.transcript_history.append(text)
                session.transaction_count += 1
            logger.info(f"📝 Appended transcript to session {session_id}. Turn count: {session.transaction_count}")

    def terminate_session(self, session_id: str):
        with self._lock:
            if session_id in self._sessions:
                del self._sessions[session_id]
                logger.info(f"🚫 Terminated session: {session_id}")


# ==========================================
# 2. STATE MANAGEMENT MODULE
# ==========================================
class VoiceState:
    IDLE = "IDLE"
    LISTENING = "LISTENING"
    TRANSCRIBING = "TRANSCRIBING"
    THINKING = "THINKING"
    SPEAKING = "SPEAKING"

class StateManager:
    """
    Thread-safe finite state machine. Handles interruptions by monitoring VAD
    and halting ongoing speaker playback if pilot barge-in is detected.
    """
    def __init__(self, initial_state: str = VoiceState.IDLE):
        self._current_state = initial_state
        self._lock = threading.Lock()
        self._interruption_event = asyncio.Event()

    def transition_to(self, new_state: str):
        with self._lock:
            old_state = self._current_state
            self._current_state = new_state
            logger.info(f"🔄 State Transition: {old_state} ➡️ {new_state}")
            
            # If user starts speaking while agent is speaking, trigger interruption
            if old_state == VoiceState.SPEAKING and new_state == VoiceState.LISTENING:
                self.trigger_interruption()

    @property
    def current_state(self) -> str:
        with self._lock:
            return self._current_state

    def trigger_interruption(self):
        logger.warning("🚨 Interruption detected! Halting active text-to-speech output...")
        self._interruption_event.set()

    def clear_interruption(self):
        self._interruption_event.clear()

    async def wait_for_interruption(self):
        await self._interruption_event.wait()


# ==========================================
# 3. CONVERSATIONAL FILLER WORD MODULE
# ==========================================
class FillerWordModule:
    """
    Generates natural, contextual filler phrases during slow LLM thinking phases
    or external API lookups (e.g. querying airport runways or weather reports).
    """
    def __init__(self):
        self._fillers = [
            "Let me check our safety checklist for you...",
            "Understood, compiling those telemetry parameters...",
            "Checking the active logs for that occurrence...",
            "Roger that, let me verify those flight coordinates...",
            "Standby captain, cross-referencing that system log..."
        ]
        self._index = 0

    def get_filler_phrase(self) -> str:
        phrase = self._fillers[self._index]
        # Rotate index to ensure phrase variety
        self._index = (self._index + 1) % len(self._fillers)
        return phrase

    async def play_filler_if_delayed(self, delay_threshold: float, task: asyncio.Task, speak_callback):
        """
        Plays a filler phrase if the main task (such as LLM generation or tool execution)
        takes longer than the delay_threshold (e.g., 0.8 seconds).
        """
        try:
            await asyncio.wait_for(asyncio.shield(task), timeout=delay_threshold)
        except asyncio.TimeoutError:
            # Task is slow, play a filler word to keep the pilot engaged
            filler = self.get_filler_phrase()
            logger.info(f"⏳ Tool call slow. Injecting filler verbal cue: '{filler}'")
            await speak_callback(filler)
            # Await the actual task to finish
            await task


# ==========================================
# 4. POST-SESSION OBSERVER (EVALUATION ENGINE)
# ==========================================
class PostSessionObserver:
    """
    Asynchronous background audit runner. Evaluates metrics (sentiment, length,
    error checking) once the call terminates, completely off the hot latency path.
    """
    def __init__(self):
        self._queue = asyncio.Queue()
        self._worker_task: Optional[asyncio.Task] = None

    def start(self):
        self._worker_task = asyncio.create_task(self._process_queue())
        logger.info("👁️ Post-Session Observer engine started.")

    async def submit_audit(self, session_id: str, transcript: List[str], latency_metrics: Dict, user_id: Optional[str] = None):
        await self._queue.put((session_id, user_id, transcript, latency_metrics))
        logger.info(f"📥 Submitted call log for Session {session_id} to the evaluation queue.")

    async def _process_queue(self):
        while True:
            try:
                session_id, user_id, transcript, metrics = await self._queue.get()
                logger.info(f"🔬 Auditing Session {session_id} (user: {user_id or 'unknown'}) in background...")
                
                # Simulate latency-free post-processing
                await asyncio.sleep(2.0)
                
                # Calculate metrics (e.g., words per turn, mock sentiment)
                full_text = " ".join(transcript)
                word_count = len(full_text.split())
                is_emergency = "mayday" in full_text.lower() or "strike" in full_text.lower()
                
                logger.info(f"📈 Audit Completed for {session_id}:")
                logger.info(f"   - Word Count: {word_count}")
                logger.info(f"   - Emergency Status: {is_emergency}")
                logger.info(f"   - Input Audio Latency: {metrics.get('stt_latency_ms', 0)}ms")
                
                self._queue.task_done()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"❌ Error inside Observer processing thread: {e}", exc_info=True)


# ==========================================
# 5. CORE AZURE SPEECH ORCHESTRATOR
# ==========================================
class AviationSpeechPipeline:
    """
    Integrates 16 kHz Mono Wave PCM audio streaming with the Azure Fast Transcription REST API.
    Audio is buffered during the stream and uploaded once the stream ends.
    """

    def __init__(self, subscription_key: str, endpoint_url: str, session_id: str):
        self.session_id = session_id
        self.subscription_key = subscription_key
        # Ensure the endpoint doesn't have a trailing slash
        self.endpoint_url = endpoint_url.rstrip("/")
        self._audio_buffer = bytearray()
        self._lock = threading.Lock()

    def start(self):
        logger.info("⚡ Starting audio buffering for REST API...")

    def ingest_audio_bytes(self, chunk: bytes):
        """
        Accepts raw 16 kHz Mono bytes streamed from the client websocket 
        and buffers them in memory.
        """
        with self._lock:
            self._audio_buffer.extend(chunk)

    def stop(self) -> str:
        """
        Constructs a WAV file from the buffered PCM bytes and sends a multipart/form-data
        POST request to the Azure Fast Transcription API.
        """
        logger.info("🛑 Stopping audio buffering and uploading to Azure...")
        
        with self._lock:
            pcm_bytes = bytes(self._audio_buffer)

        if not pcm_bytes:
            logger.warning("⚠️ No audio data received.")
            return ""

        # Create a WAV file in memory
        wav_io = io.BytesIO()
        with wave.open(wav_io, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2) # 16-bit
            wav_file.setframerate(16000)
            wav_file.writeframes(pcm_bytes)
        
        wav_io.seek(0)
        
        url = f"{self.endpoint_url}/speechtotext/transcriptions:transcribe?api-version=2025-10-15"
        
        headers = {
            "Ocp-Apim-Subscription-Key": self.subscription_key,
        }
        
        files = {
            "audio": ("audio.wav", wav_io, "audio/wav")
        }
        
        data = {
            "definition": json.dumps({
                "locales": ["en-US"],
                "diarization": {
                    "maxSpeakers": 2,
                    "enabled": True
                },
                "customPhrases": AUDIT_PHRASES
            })
        }

        try:
            with httpx.Client(timeout=30.0) as client:
                response = client.post(url, headers=headers, files=files, data=data)
                
                if response.status_code != 200:
                    logger.error(f"❌ Fast Transcription API returned {response.status_code}: {response.text}")
                    return ""
                
                resp_json = response.json()
                logger.info(f"Received JSON response from STT API: {resp_json}")
                
                if "combinedPhrases" in resp_json and len(resp_json["combinedPhrases"]) > 0:
                    return resp_json["combinedPhrases"][0].get("text", "")
                elif "text" in resp_json:
                    return resp_json["text"]
                elif "phrases" in resp_json:
                    segments = []
                    for phrase in resp_json["phrases"]:
                        segments.append(phrase.get("text", ""))
                    return " ".join(segments).strip()
                
                return ""
        except Exception as e:
            logger.error(f"❌ Error calling Azure Fast Transcription API: {e}", exc_info=True)
            return ""
