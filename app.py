import os
import json
import time
import uvicorn
import asyncio
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv
import asyncpg

from orchestrator import (
    SessionManager, StateManager, FillerWordModule,
    PostSessionObserver, AviationSpeechPipeline, VoiceState
)
from extractor_service import FormExtractor

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("AviationASRApp")

# ── Shared Singletons ─────────────────────────────────────
session_manager = SessionManager()
filler_module   = FillerWordModule()
extractor       = FormExtractor()
observer        = PostSessionObserver()

SPEECH_KEY      = os.getenv("AZURE_SPEECH_KEY", "")
SPEECH_ENDPOINT = os.getenv("AZURE_SPEECH_ENDPOINT") or os.getenv("AZURE_ENDPOINT_KEY", "")

async def mock_tts(text: str):
    logger.info(f"🔊 [TTS Filler]: '{text}'")
    await asyncio.sleep(1.0)

# ── Database & Lifespan ───────────────────────────────────
db_pool = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global db_pool
    # Initialize Postgres Connection Pool
    db_dsn = os.getenv("DATABASE_URL")
    if db_dsn:
        try:
            db_pool = await asyncpg.create_pool(db_dsn)
            # Create feedback table if not exists
            async with db_pool.acquire() as conn:
                await conn.execute('''
                    CREATE TABLE IF NOT EXISTS feedback_log (
                        id SERIAL PRIMARY KEY,
                        user_id TEXT NOT NULL,
                        original_transcript TEXT NOT NULL,
                        original_extraction JSONB,
                        corrected_fields JSONB,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
            logger.info("✅ Postgres connection pool initialized for feedback logging.")
        except Exception as e:
            logger.error(f"❌ Failed to connect to Postgres: {e}")
    else:
        logger.warning("⚠️ DATABASE_URL not set in .env. Feedback endpoint will mock DB writes.")

    observer.start()
    logger.info("🚀 Aviation ASR Voice Portal is live.")
    yield
    if db_pool:
        await db_pool.close()

# ── FastAPI App ───────────────────────────────────────────
app = FastAPI(
    title="✈️ QM Smart Tech — Air Safety Report Voice Portal",
    lifespan=lifespan
)

# ── Static Files & Root ───────────────────────────────────
static_dir = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=static_dir), name="static")

@app.get("/")
async def serve_ui():
    return FileResponse(os.path.join(static_dir, "index.html"))

@app.get("/health")
async def health():
    return {"status": "ok", "service": "aviation-asr-voice-portal"}

# ── Feedback Endpoint (Phase 3) ───────────────────────────
class FeedbackSubmission(BaseModel):
    user_id: str
    original_transcript: str
    original_extraction: dict
    corrected_fields: dict

@app.post("/api/feedback/submit")
async def submit_feedback(data: FeedbackSubmission):
    global db_pool
    if db_pool:
        try:
            async with db_pool.acquire() as conn:
                await conn.execute(
                    '''INSERT INTO feedback_log 
                       (user_id, original_transcript, original_extraction, corrected_fields) 
                       VALUES ($1, $2, $3, $4)''',
                    data.user_id,
                    data.original_transcript,
                    json.dumps(data.original_extraction),
                    json.dumps(data.corrected_fields)
                )
            return {"status": "success", "message": "Feedback recorded in Postgres"}
        except Exception as e:
            logger.error(f"Failed to record feedback: {e}")
            return {"status": "error", "message": "Database error"}
    else:
        # Mock behavior for local testing if Postgres isn't running
        logger.info(f"Mock DB write for feedback: {data.user_id}")
        return {"status": "success", "message": "Feedback recorded (mocked)"}

# ── WebSocket — Live Audio Stream ─────────────────────────
@app.websocket("/ws/audio")
async def handle_audio_stream(websocket: WebSocket):
    await websocket.accept()

    session       = session_manager.create_session(metadata={"transport": "WebSocket"})
    state_manager = StateManager(initial_state=VoiceState.IDLE)

    if not SPEECH_KEY or not SPEECH_ENDPOINT:
        await websocket.send_json({"error": "AZURE_SPEECH_KEY or AZURE_ENDPOINT_KEY not set in .env"})
        await websocket.close()
        return

    stt = AviationSpeechPipeline(SPEECH_KEY, SPEECH_ENDPOINT, session.session_id)
    state_manager.transition_to(VoiceState.LISTENING)
    await asyncio.to_thread(stt.start)

    # Notify client: listening started
    await websocket.send_json({"event": "listening_started", "session_id": session.session_id})

    start_time = time.time()
    stop_requested = False

    try:
        while True:
            message = await websocket.receive()

            if message["type"] == "websocket.disconnect":
                raise WebSocketDisconnect(message.get("code", 1000))

            # Handle binary audio data
            if "bytes" in message and message.get("bytes"):
                stt.ingest_audio_bytes(message["bytes"])
                continue

            # Handle text messages (JSON commands)
            if "text" in message and message.get("text"):
                try:
                    cmd = json.loads(message["text"])
                except (json.JSONDecodeError, TypeError):
                    continue

                if cmd.get("action") == "stop":
                    logger.info(f"🛑 Stop signal received for session {session.session_id}")
                    stop_requested = True
                    break

    except WebSocketDisconnect:
        logger.warning(f"🔌 Client disconnected — Session: {session.session_id}")

    # ── Finalise STT ──────────────────────────────────
    state_manager.transition_to(VoiceState.TRANSCRIBING)
    try:
        transcript = await asyncio.wait_for(
            asyncio.to_thread(stt.stop), timeout=35.0
        )
    except asyncio.TimeoutError:
        logger.error(
            f"⏱️ STT teardown timed out for session {session.session_id} "
            "— possible engine hang. Returning empty transcript."
        )
        transcript = ""
    logger.info(f"📝 Transcript: '{transcript}'")

    if not transcript.strip():
        try:
            await websocket.send_json({"event": "no_speech_detected"})
        except Exception:
            pass
        session_manager.terminate_session(session.session_id)
        return

    session_manager.append_transcript(session.session_id, transcript)

    # Send transcript to UI immediately
    try:
        await websocket.send_json({"event": "transcript", "text": transcript})
    except Exception:
        pass

    user_id = "pilot-001" # Mock user ID for the demo

    # ── LLM Extraction ────────────────────────────────
    state_manager.transition_to(VoiceState.THINKING)
    extraction_task = asyncio.create_task(
        asyncio.to_thread(extractor.extract_incident_fields, transcript, user_id)
    )

    await filler_module.play_filler_if_delayed(
        delay_threshold=0.8,
        task=extraction_task,
        speak_callback=mock_tts
    )

    form_result = await extraction_task
    state_manager.transition_to(VoiceState.SPEAKING)

    # Send filled form JSON to client
    try:
        payload = form_result.model_dump()
        payload["event"] = "form_filled"
        await websocket.send_json(payload)
        logger.info(f"✅ Sent form_filled event to client for session {session.session_id}")
    except Exception as e:
        logger.error(f"❌ Failed to send form_filled to client: {e}")

    # ── Background Audit ──────────────────────────────
    latency_ms = int((time.time() - start_time) * 1000)
    sess = session_manager.get_session(session.session_id)
    await observer.submit_audit(
        session_id=session.session_id,
        user_id=user_id,
        transcript=sess.transcript_history if sess else [],
        latency_metrics={"stt_latency_ms": latency_ms}
    )

    state_manager.transition_to(VoiceState.IDLE)
    session_manager.terminate_session(session.session_id)

    # Gracefully close the WebSocket
    try:
        await websocket.close()
    except Exception:
        pass


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
