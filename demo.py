import os
import asyncio
import httpx
import json
from voice_profiler import VoicePatternAnalyzer
from stt_accuracy_tracker import STTAccuracyTracker
from extractor_service import FormExtractor

async def run_demo():
    print("🚀 Starting Continuous Voice Personalization Demo...")
    
    user_id = "pilot-001"
    
    # 1. Mock 10 feedback records representing historical corrections
    print("\n--- Generating Mock Historical Feedback ---")
    mock_records = []
    
    # Let's say STT keeps transcribing "bird hit" instead of "bird strike" 
    # and keeps spelling out "runway nine" instead of "runway 09".
    for i in range(10):
        mock_records.append({
            "original_transcript": "bird hit on runway nine at 0400",
            "corrected_transcript": "bird strike on runway 09 at 0400",
            "original_extraction": {"occurrence_title": "bird hit", "time_of_occurrence_utc": "0400"},
            "corrected_fields": {"occurrence_title": "bird strike", "time_of_occurrence_utc": "04:00 UTC", "runway": "RWY 09"}
        })
    print(f"Created {len(mock_records)} historical records.")
    
    # 2. Phase 2: STT Accuracy Tracking (Acoustic Layer)
    print("\n--- Phase 2: STT Accuracy & Phrase Lists ---")
    tracker = STTAccuracyTracker()
    tracker.print_report(user_id, mock_records)
    
    # The mined errors (e.g. "strike", "09") would be sent to Azure Speech Phrase List
    recurring_errors = tracker.mine_recurring_errors(mock_records, threshold=2)
    phrase_list = tracker.generate_phrase_list(recurring_errors)
    print(f"Generated Phrase List for Azure Speech: {phrase_list}")
    
    # 3. Phase 1: Voice Pattern Profiling (Phrasing Layer)
    print("\n--- Phase 1: Voice Pattern Profiling ---")
    profiler = VoicePatternAnalyzer()
    
    transcripts = [r["original_transcript"] for r in mock_records[:3]]
    corrections = [r["corrected_fields"] for r in mock_records[:3]]
    
    print("Invoking GPT-4o-mini to analyze patterns...")
    profile = profiler.generate_profile(user_id, transcripts, corrections)
    
    if profile:
        print("\n✅ Successfully Generated Profile:")
        print(f" - Verbosity: {profile.verbosity}")
        print(f" - Omitted Defaults: {profile.omitted_defaults}")
        for q in profile.shorthand_quirks:
            print(f" - Shorthand: '{q.shorthand}' -> '{q.meaning}'")
            
    # 4. Profile-Aware Extraction
    print("\n--- Simulating Profile-Aware Extraction ---")
    extractor = FormExtractor()
    test_transcript = "bird hit on runway 9 at 0400"
    
    print(f"New Raw Transcript: '{test_transcript}'")
    print("Extracting with user profile injected...")
    
    # This will load the JSON profile we just saved to the scratch directory
    result = await asyncio.to_thread(extractor.extract_incident_fields, test_transcript, user_id)
    
    print("\n✅ Extracted Form Results:")
    print(f" - Title: {result.occurrence_title}")
    print(f" - Time UTC: {result.time_of_occurrence_utc}")
    print(f" - Runway: {result.runway}")
    
    # 5. Phase 3: Feedback Endpoint
    print("\n--- Phase 3: Submitting Feedback to Postgres ---")
    payload = {
        "user_id": user_id,
        "original_transcript": test_transcript,
        "original_extraction": result.model_dump(),
        "corrected_fields": {"occurrence_title": "bird strike"} # User corrected it
    }
    
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post("http://localhost:8000/api/feedback/submit", json=payload)
            print(f"Feedback Endpoint Response: {resp.status_code} - {resp.json()}")
    except Exception as e:
        print(f"Could not reach feedback endpoint (is FastAPI running?): {e}")

if __name__ == "__main__":
    asyncio.run(run_demo())
