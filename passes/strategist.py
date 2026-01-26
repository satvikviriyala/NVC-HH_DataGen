"""
Pass 3: Strategist
Generates constructive requests.
Uses: plato_strategy_filter, request_quality_ontology
"""

from typing import Dict, Any
from passes.base import BaseLLMPass


class StrategistPass(BaseLLMPass):
    PASS_NAME = "strategist"
    OUTPUT_FIELDS = [
        "ofnr.explicit_request",
        "ofnr.implicit_request",
        "ofnr.implicit_intent",
        "ofnr.strategy_leakage_detected",
        "ofnr.translation_notes"
    ]
    PROMPT_FILE = "pass_strategist.txt"
    REQUIRED_ONTOLOGIES = ["plato_strategy_filter", "request_quality_ontology"]
    
    def _default_system_prompt(self) -> str:
        return """You are an NVC Strategist. Formulate constructive requests.
Apply PLATO test to detect strategies.
Convert demands to positive requests.
Output JSON: {"explicit_request": [...], "implicit_request": [...], ...}"""
    
    def build_user_prompt(self, row: Dict[str, Any]) -> str:
        input_data = row.get("input", {})
        ofnr = row.get("ofnr", {})
        
        prompt = input_data.get("prompt", "")
        observation = ofnr.get("observation", [])
        feelings = ofnr.get("feelings", [])
        needs = ofnr.get("need", [])
        explicit_needs = ofnr.get("explicit_needs", [])
        
        return f"""Based on the analysis, formulate requests.

OBSERVATION: {observation}
FEELINGS: {feelings}
NEEDS (universal): {needs}
EXPLICIT WANTS: {explicit_needs}

ORIGINAL USER TURN:
{prompt}

Generate: explicit_request, implicit_request (NVC-style "Would you be willing to...?"), implicit_intent, strategy_leakage."""
    
    def parse_response(self, response: str) -> Dict[str, Any]:
        data = self._extract_json(response)
        return {
            "ofnr.explicit_request": data.get("explicit_request", []),
            "ofnr.implicit_request": data.get("implicit_request", []),
            "ofnr.implicit_intent": data.get("implicit_intent"),
            "ofnr.strategy_leakage_detected": data.get("strategy_leakage_detected", []),
            "ofnr.translation_notes": data.get("translation_notes")
        }
