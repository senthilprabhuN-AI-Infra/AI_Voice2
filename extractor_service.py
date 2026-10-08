import os
from pydantic import BaseModel, Field
from typing import Optional
from voice_profiler import VoicePatternAnalyzer, make_llm_client
from dotenv import load_dotenv

load_dotenv()

# ==========================================
# FULL ASR STRUCTURED OUTPUT PYDANTIC SCHEMA
# Covers all fields visible in QM Smart Tech
# Air Safety Report Form (ASR) screenshots
# ==========================================
class AviationIncidentForm(BaseModel):

    # ── Section 1: OCCURRENCE DETAILS ─────────────────────
    occurrence_title: Optional[str] = Field(
        None, description="Title or category of the safety occurrence (e.g., Bird Strike, Engine Failure).")
    date_of_occurrence: Optional[str] = Field(
        None, description="Date of the incident. Convert spoken dates to dd-mm-yyyy format (e.g. 14-08-2026).")
    time_of_occurrence_utc: Optional[str] = Field(
        None, description="UTC time of the incident, e.g., 09:14 UTC.")
    place_of_occurrence: Optional[str] = Field(
        None, description="Location where the incident occurred, e.g., airport name or airspace.")
    origin: Optional[str] = Field(
        None, description="Departure airport or origin location of the flight.")
    destination: Optional[str] = Field(
        None, description="Arrival airport or destination of the flight.")
    department: Optional[str] = Field(
        None, description="Department responsible for the report, e.g., Flight Operations, Maintenance.")

    # ── Section 2: AIRCRAFT AND FLIGHT INFORMATION ────────
    aircraft_registration: Optional[str] = Field(
        None, description="Aircraft tail number or registration code, e.g., VT-ANQ.")
    aircraft_type: Optional[str] = Field(
        None, description="Aircraft model or type, e.g., Boeing 737-800, Airbus A320.")
    aircraft_manufacturer: Optional[str] = Field(
        None, description="Manufacturer of the aircraft, e.g., Boeing, Airbus.")
    squawk: Optional[str] = Field(
        None, description="Transponder squawk code, a 4-digit code e.g., 7700.")
    altitude_fl: Optional[str] = Field(
        None, description="Altitude or Flight Level at time of incident, e.g., FL350 or 12000ft.")
    aircraft_speed_kts: Optional[str] = Field(
        None, description="Aircraft speed in knots at the time of the incident.")
    aircraft_to_weight: Optional[str] = Field(
        None, description="Aircraft take-off weight in tonnes or kilograms if mentioned.")
    landed_at: Optional[str] = Field(
        None, description="Airport where the aircraft landed following the incident.")
    flight_phase: Optional[str] = Field(
        None, description="Phase of flight during the incident: Takeoff, Climb, Cruise, Descent, Approach, Landing, Taxi.")
    flight_rules: Optional[str] = Field(
        None, description="Flight rules in use: IFR (Instrument) or VFR (Visual).")

    # ── Section 3: ENVIRONMENT ────────────────────────────
    wind: Optional[str] = Field(
        None, description="Wind conditions at the time, e.g., 270/15 knots, calm, gusty.")
    cloud_base: Optional[str] = Field(
        None, description="Cloud base altitude, e.g., 2500ft, overcast.")
    runway: Optional[str] = Field(
        None, description="Runway designator in use, e.g., Runway 28L, RWY 09.")

    # ── Section 4: CONSEQUENCES AND OCCURRENCE DESCRIPTION ─
    consequences: Optional[str] = Field(
        None, description="Resulting consequences, e.g., None, Aircraft Damage, Minor Injury.")
    configuration_at_event: Optional[str] = Field(
        None, description="Aircraft configuration at time of event, e.g., Flaps 15, Gear Down.")
    occurrence_description: Optional[str] = Field(
        None, description="Detailed narrative description of what happened.")
    immediate_action_taken: Optional[str] = Field(
        None, description="Any immediate corrective or procedural actions taken by the crew.")


class EnhancedAviationIncidentForm(BaseModel):
    reporter_name: Optional[str] = Field(None, description="Name of the person reporting.")
    flight_number: Optional[str] = Field(None, description="Alphanumeric flight identifier.")
    incident_time_utc: Optional[str] = Field(None, description="UTC time of the incident.")
    aircraft_part_affected: Optional[str] = Field(None, description="Part of aircraft involved.")
    is_recurring_defect: Optional[bool] = Field(None, description="True if the transcript mentions a defect reappearing, recurring, or repeating.")
    sop_compliance_deviation: Optional[str] = Field(None, description="Explicit deviations from standard operating procedures if reported.")
    ground_equipment_involved: Optional[str] = Field(None, description="Ground support equipment or airport infrastructure faults (e.g. unserviceable baggage trollies, faded markings).")
    narrative_summary: Optional[str] = Field(None, description="Clear, concise summary of the incident.")


class FormExtractor:
    """
    Uses Azure OpenAI GPT-5.6-sol Structured Outputs API to extract all
    Air Safety Report (ASR) fields from a pilot's raw voice transcript.
    """
    def __init__(self):
        self.deployment = (
            os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
            or os.getenv("AZURE_OPENAI_MODEL_DEPLOYMENT_NAME")
            or "gpt-5.6-luna"
        )
        self.client = make_llm_client()

    def extract_incident_fields(self, raw_transcript: str, user_id: Optional[str] = None) -> AviationIncidentForm:
        """
        Sends the raw pilot voice transcript to the LLM and returns a
        fully-populated AviationIncidentForm (the schema the UI form maps to).
        """
        if not raw_transcript.strip():
            return AviationIncidentForm()

        system_prompt = (
            "You are a professional aviation safety data analyst for QM Smart Technologies. "
            "Your task is to extract structured Air Safety Report (ASR) fields from the pilot's "
            "raw voice transcript. Map spoken phrases accurately to the correct fields. "
            "Use aviation conventions: convert spoken dates into dd-mm-yyyy format (e.g. '14 August 2026' becomes 14-08-2026), times as HH:MM UTC, speeds in knots. "
            "Only populate fields that are explicitly stated or clearly implied in the transcript. "
            "Do not guess or assume information not present in the text."
        )

        # Inject User Voice Profile if available
        if user_id:
            profile = VoicePatternAnalyzer.load_profile(user_id)
            if profile:
                profile_context = "\n\n--- PERSONALIZED USER PROFILE ---\nThis user has specific speech habits. Please adjust your extraction accordingly:\n"
                profile_context += f"- Verbosity: {profile.verbosity}\n"
                
                if profile.omitted_defaults:
                    profile_context += f"- Frequently Omitted (Imply these contextually): {', '.join(profile.omitted_defaults)}\n"
                    
                if profile.shorthand_quirks:
                    quirks_str = "; ".join([f"'{q.shorthand}' means '{q.meaning}'" for q in profile.shorthand_quirks])
                    profile_context += f"- Shorthand Quirks: {quirks_str}\n"
                    
                if profile.specialized_jargon:
                    profile_context += f"- Specialized Jargon used: {', '.join(profile.specialized_jargon)}\n"
                    
                system_prompt += profile_context

        try:
            response = self.client.chat.completions.parse(
                model=self.deployment,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": raw_transcript}
                ],
                response_format=AviationIncidentForm
            )
            return response.choices[0].message.parsed
        except Exception as e:
            print(f"❌ Structured output extraction failed: {e}")
            return AviationIncidentForm(
                occurrence_description=f"Extraction error: {str(e)}"
            )
