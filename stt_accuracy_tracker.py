import os
import logging
from typing import List, Dict, Tuple
from collections import Counter
import jiwer
import azure.cognitiveservices.speech as speechsdk
from azure.identity import DefaultAzureCredential

logger = logging.getLogger("STTAccuracyTracker")

# ==========================================
# PHASE 2: STT ACCURACY TRACKING
# ==========================================

class STTAccuracyTracker:
    def __init__(self):
        # We don't initialize a full DB connection here; we assume the caller
        # (e.g. demo.py or app.py) fetches the records and passes them in.
        self.speech_key = os.getenv("AZURE_SPEECH_KEY", "")
        self.speech_region = os.getenv("AZURE_SPEECH_REGION", "eastus2")
        
        # We cap phrase lists to Azure's limit (e.g., typically a few thousand, 
        # but we'll use a conservative limit of 500 for dynamic updates).
        self.MAX_PHRASES = 500

    def compute_wer(self, reference: str, hypothesis: str) -> float:
        """
        Computes Word Error Rate between the corrected transcript (reference) 
        and the original STT output (hypothesis).
        """
        # jiwer needs non-empty strings. If empty, handle edge cases.
        if not reference.strip() and not hypothesis.strip():
            return 0.0
        if not reference.strip():
            return 1.0 # 100% error if we inserted words where there were none
        if not hypothesis.strip():
            return 1.0
            
        # Basic normalization could be added here
        # jiwer 4.x: transforms must end by reducing to a list of word lists
        transformation = jiwer.Compose([
            jiwer.ToLowerCase(),
            jiwer.RemovePunctuation(),
            jiwer.ExpandCommonEnglishContractions(),
            jiwer.RemoveMultipleSpaces(),
            jiwer.Strip(),
            jiwer.ReduceToListOfListOfWords()
        ])

        try:
            return jiwer.wer(
                reference,
                hypothesis,
                reference_transform=transformation,
                hypothesis_transform=transformation
            )
        except Exception as e:
            logger.error(f"Failed to compute WER: {e}")
            return 0.0

    def calculate_rolling_wer(self, records: List[Dict]) -> float:
        """Calculates average WER across a list of correction records."""
        if not records:
            return 0.0
        total_wer = sum(self.compute_wer(r.get("corrected_transcript", ""), r.get("original_transcript", "")) for r in records)
        return total_wer / len(records)

    def mine_recurring_errors(self, records: List[Dict], threshold: int = 2) -> List[str]:
        """
        Mines the correction records for words that repeatedly appear in the 
        corrected transcript but not in the original STT (or are mismatched).
        This is a simplified approach: in a real system, we'd do strict word-level alignment.
        """
        error_words = []
        for r in records:
            ref_words = set(r.get("corrected_transcript", "").lower().split())
            hyp_words = set(r.get("original_transcript", "").lower().split())
            
            # Words present in the correction that STT missed
            missed_words = ref_words - hyp_words
            error_words.extend(list(missed_words))
            
        counter = Counter(error_words)
        # Filter by threshold and ignore tiny words
        recurring = [word for word, count in counter.items() if count >= threshold and len(word) > 3]
        return recurring

    def generate_phrase_list(self, error_terms: List[str]) -> List[str]:
        """
        Deduplicates and caps the phrase list to Azure's limit.
        """
        unique_terms = list(set(error_terms))
        return unique_terms[:self.MAX_PHRASES]

    def update_azure_phrase_list(self, recognizer: speechsdk.SpeechRecognizer, terms: List[str]):
        """
        Applies a dynamically generated phrase list to an active SpeechRecognizer instance
        to bias the acoustic model towards the corrected terms.
        """
        if not terms:
            return
            
        phrase_list_grammar = speechsdk.PhraseListGrammar.from_recognizer(recognizer)
        phrase_list_grammar.clear()
        for term in terms:
            phrase_list_grammar.addPhrase(term)
        logger.info(f"Updated Azure Speech Phrase List with {len(terms)} custom terms.")

    def print_report(self, user_id: str, records: List[Dict]):
        """CLI reporting function as requested in Phase 2."""
        wer = self.calculate_rolling_wer(records)
        recurring = self.mine_recurring_errors(records, threshold=1) # threshold=1 for demo
        
        print("\n" + "="*40)
        print(f"📊 STT ACCURACY REPORT FOR {user_id}")
        print("="*40)
        print(f"Rolling Average WER: {wer:.2%}")
        print(f"Total Utterances Evaluated: {len(records)}")
        print("\nTop Recurring Missing Terms (Candidate for Phrase List):")
        for term in recurring[:10]:
            print(f" - {term}")
        print("="*40 + "\n")
