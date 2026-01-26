"""
Pass 1: Observer
Extracts objective observations and detected evaluations.
Uses: judgment_markers_ontology
"""

from typing import Dict, Any
from passes.base import BaseLLMPass


class ObserverPass(BaseLLMPass):
    PASS_NAME = "observer"
    OUTPUT_FIELDS = [
        "ofnr.observation",
        "ofnr.evaluations_detected"
    ]
    PROMPT_FILE = "pass_observer.txt"
    REQUIRED_ONTOLOGIES = ["judgment_markers_ontology"]
    
    def _default_system_prompt(self) -> str:
        return """You are an NVC Observer. Extract objective observations only.
Apply the Camera Test: only facts a video camera could record.
Detect judgment/evaluation words and list them.
Output JSON: {"observation": [...], "evaluations_detected": [...]}"""
    
    def build_user_prompt(self, row: Dict[str, Any]) -> str:
        input_data = row.get("input", {})
        
        if input_data.get("format") == "pair":
            text = input_data.get("chosen") or input_data.get("rejected") or ""
        else:
            text = input_data.get("assistant_response") or ""
        
        prompt = input_data.get("prompt", "")
        context = input_data.get("context", "")
        
        return f"""Analyze this conversation and extract observations.

CONTEXT:
{context[:1500] if context else "(none)"}

LAST USER TURN:
{prompt}

FULL TEXT:
{text[:3000]}

Extract: (1) objective observations (camera-test), (2) any judgment/evaluation words detected."""
    
    def parse_response(self, response: str) -> Dict[str, Any]:
        data = self._extract_json(response)
        return {
            "ofnr.observation": data.get("observation", []),
            "ofnr.evaluations_detected": data.get("evaluations_detected", [])
        }
