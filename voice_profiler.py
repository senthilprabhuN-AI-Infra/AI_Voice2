import os
import json
import logging
from typing import List, Dict, Optional
from pydantic import BaseModel, Field
from openai import OpenAI
from azure.identity import DefaultAzureCredential, get_bearer_token_provider

logger = logging.getLogger("VoicePatternAnalyzer")


# ==========================================
# SHARED LLM CLIENT
# ==========================================
def make_llm_client() -> OpenAI:
    """
    The Azure AI Foundry /openai/v1 endpoint rejects the api-version query
    parameter that AzureOpenAI adds, so use the standard OpenAI client.
    """
    base_url = os.getenv("AZURE_OPENAI_ENDPOINT", "").rstrip("/")
    v1_idx = base_url.find("/openai/v1")
    if v1_idx != -1:
        # Tolerate endpoints that include a trailing path such as /responses
        base_url = base_url[:v1_idx + len("/openai/v1")]

    api_key = os.getenv("AZURE_OPENAI_KEY", "").strip()
    if not api_key:
        # Keyless auth (managed identity / az login): the client accepts a token callable
        api_key = get_bearer_token_provider(
            DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default"
        )
    return OpenAI(api_key=api_key, base_url=base_url)

# ==========================================
# SCRATCH DIRECTORY SECURITY
# ==========================================
# Restrict all profile file I/O to this explicit directory
SCRATCH_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "scratch"))
os.makedirs(SCRATCH_DIR, exist_ok=True)

def _get_safe_profile_path(user_id: str) -> str:
    """Ensures the generated file path is strictly within the scratch directory."""
    # Sanitize user_id to prevent directory traversal
    safe_user_id = "".join(c for c in user_id if c.isalnum() or c in ("-", "_"))
    if not safe_user_id:
        raise ValueError("Invalid user_id provided.")
        
    file_path = os.path.abspath(os.path.join(SCRATCH_DIR, f"profile_{safe_user_id}.json"))
    
    # Final security check: ensure the resolved path starts with the scratch dir path
    if not file_path.startswith(SCRATCH_DIR):
        raise PermissionError("Path traversal attempted outside of scratch directory.")
        
    return file_path


# ==========================================
# SCHEMA DEFINITION
# ==========================================
class ShorthandQuirk(BaseModel):
    shorthand: str = Field(description="The short/quirky phrase the user said")
    meaning: str = Field(description="What they actually meant in aviation terminology")

class UserVoiceProfile(BaseModel):
    verbosity: str = Field(description="low, medium, or high")
    shorthand_quirks: List[ShorthandQuirk] = Field(description="User-specific shorthand to meaning mappings")
    omitted_defaults: List[str] = Field(description="Fields this user routinely leaves implicit (e.g., UTC offsets, AM/PM)")
    specialized_jargon: List[str] = Field(description="Role-specific aviation terms this user uses frequently")


# ==========================================
# PROFILER LOGIC
# ==========================================
class VoicePatternAnalyzer:
    """
    Analyzes raw transcripts + corrected fields to build a specific user's
    phrasing profile (verbosity, shorthand, jargon, omissions).
    """
    def __init__(self):
        deployment = (
            os.getenv("AZURE_OPENAI_MINI_DEPLOYMENT_NAME")
            or os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
            or "gpt-5.6-luna"
        )

        try:
            self.client = make_llm_client()
            self.deployment = deployment
            self.enabled = True
        except Exception as e:
            logger.warning(f"⚠️ Could not initialize VoicePatternAnalyzer (Missing credentials/endpoint?): {e}")
            self.enabled = False

    def generate_profile(self, user_id: str, transcript_history: List[str], corrected_fields_history: List[Dict]) -> Optional[UserVoiceProfile]:
        """
        Takes historical transcripts and the actual corrected fields to deduce the user's speech habits.
        """
        if not self.enabled:
            logger.error("VoicePatternAnalyzer is disabled due to missing config.")
            return None

        # Build analysis context
        history_text = ""
        for i, (transcript, corrections) in enumerate(zip(transcript_history, corrected_fields_history)):
            history_text += f"\n--- Session {i+1} ---\nTranscript: {transcript}\nCorrected Fields: {json.dumps(corrections)}\n"

        system_prompt = (
            "You are an expert NLP profiler analyzing an aviation professional's speech patterns. "
            "Compare what they literally said (Transcript) with the final accepted form data (Corrected Fields). "
            "Identify their verbosity, any shorthand they use (e.g., 'bird hit' -> 'bird strike'), "
            "what they routinely omit (e.g., leaving out 'UTC'), and specialized jargon. "
            "Be concise and objective. Output strictly conforming to the requested schema."
        )

        try:
            response = self.client.chat.completions.parse(
                model=self.deployment,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": history_text}
                ],
                response_format=UserVoiceProfile,
            )
            profile = response.choices[0].message.parsed
            self._save_profile(user_id, profile)
            logger.info(f"✅ Generated and saved voice profile for user: {user_id}")
            return profile
        except Exception as e:
            logger.error(f"❌ Failed to generate voice profile: {e}")
            return None

    def _save_profile(self, user_id: str, profile: UserVoiceProfile):
        """Securely saves the profile to the scratch directory."""
        path = _get_safe_profile_path(user_id)
        with open(path, "w", encoding="utf-8") as f:
            # We strictly write a Pydantic model dump to JSON (safe text serialization)
            json.dump(profile.model_dump(), f, indent=2)

    @staticmethod
    def load_profile(user_id: str) -> Optional[UserVoiceProfile]:
        """Loads a user's profile from the scratch directory if it exists."""
        try:
            path = _get_safe_profile_path(user_id)
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    return UserVoiceProfile(**data)
            return None
        except Exception as e:
            logger.warning(f"Could not load profile for {user_id}: {e}")
            return None
