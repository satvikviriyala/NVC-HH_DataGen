"""
NVC-HH Dataset Initializer (Streamlined)
Populates NVC_HH folder with master JSONL files.
- Processes train.jsonl files (JSONL format)
- Processes red_team_attempts.jsonl (JSON array format)
"""

import json
import os
from tqdm import tqdm

SOURCE_ROOT = "../hh-rlhf"
TARGET_ROOT = "../NVC_HH"

DATASET_META = {
    "dataset_name": "NVC-HH",
    "dataset_version": "1.1",
    "release_tier": "RAW"
}

def generate_id(folder: str, line_idx: int) -> str:
    return f"{folder.replace('-', '_')}_{line_idx:06d}"

def parse_hh_conversation(text: str):
    """Parse HH-RLHF conversation format."""
    turns = []
    parts = text.split("\n\nHuman: ")
    
    for i, part in enumerate(parts):
        if i == 0 and not part.strip():
            continue
        sub_parts = part.split("\n\nAssistant: ")
        for j, sp in enumerate(sub_parts):
            if j == 0:
                turns.append({"role": "user", "content": sp.strip()})
            else:
                turns.append({"role": "assistant", "content": sp.strip()})
    
    prompt = ""
    for t in reversed(turns):
        if t["role"] == "user":
            prompt = t["content"]
            break
    
    context = text[:text.rfind(prompt)].strip() if prompt and len(turns) > 1 else None
    return prompt, context, turns

def create_record_pair(source_data: dict, folder: str, file: str, line_idx: int) -> dict:
    """Create record for chosen/rejected pair format."""
    raw_text = source_data.get("chosen", "")
    prompt, context, turns = parse_hh_conversation(raw_text)
    
    return {
        "id": generate_id(folder, line_idx),
        "dataset": DATASET_META.copy(),
        "source": {
            "corpus": "hh-rlhf",
            "folder": folder,
            "split": "train",
            "file": file,
            "line_id": str(line_idx),
            "pair_type": "chosen_rejected"
        },
        "input": {
            "format": "pair",
            "prompt": prompt,
            "context": context,
            "chosen": source_data.get("chosen"),
            "rejected": source_data.get("rejected"),
            "assistant_response": None,
            "conversation_turns": turns,
            "target_turn_index": None,
            "notes": None
        },
        "ofnr": {"observation": None, "feelings": None, "need": None, "explicit_needs": None, "implicit_needs": None, "explicit_request": None, "implicit_request": None, "implicit_intent": None, "evaluations_detected": None, "pseudo_feelings_detected": None, "strategy_leakage_detected": None, "translation_notes": None},
        "metadata": {"somatic_markers": None, "emotion_arousal_hint": None, "emotion_valence_hint": None, "language": "en"},
        "safety": {"label": None, "policy_category": None, "reason": None, "rewrite_mode": None, "safe_alternative": None, "safety_confidence": None},
        "quality": {"ofnr_compliance": None, "observation_is_nonjudgmental": None, "pseudo_feeling_translation_quality": None, "needs_list_match": None, "strategy_leakage_score": None, "request_is_actionable": None, "request_is_noncoercive": None, "overall_confidence": None},
        "flags": {"error_flags": None, "warnings": None},
        "teacher_agreement": {"multi_teacher_enabled": False, "teachers": [], "consensus": {"enabled": False, "method": "majority_vote", "consensus_score": 0.0, "consensus_pass": False, "resolved_need": [], "resolved_feelings": [], "resolved_observation": [], "resolved_implicit_request": []}},
        "pairwise_preference": {"available": True, "preference_alignment": {"chosen_more_constructive": None, "chosen_more_safe": None, "chosen_more_helpful": None, "notes": None}, "pairwise_signals": {"toxicity_delta": None, "constructiveness_delta": None, "helpfulness_delta": None}}
    }

def create_record_redteam(source_data: dict, folder: str, file: str, line_idx: int) -> dict:
    """Create record for red-team transcript format."""
    raw_text = source_data.get("transcript", "")
    prompt, context, turns = parse_hh_conversation(raw_text)
    
    assistant_response = turns[-1]["content"] if turns and turns[-1]["role"] == "assistant" else None
    
    return {
        "id": generate_id(folder, line_idx),
        "dataset": DATASET_META.copy(),
        "source": {
            "corpus": "hh-rlhf",
            "folder": folder,
            "split": "train",
            "file": file,
            "line_id": str(line_idx),
            "pair_type": "single"
        },
        "input": {
            "format": "single",
            "prompt": prompt,
            "context": context,
            "chosen": None,
            "rejected": None,
            "assistant_response": assistant_response,
            "conversation_turns": turns,
            "target_turn_index": None,
            "notes": source_data.get("task_description")
        },
        "ofnr": {"observation": None, "feelings": None, "need": None, "explicit_needs": None, "implicit_needs": None, "explicit_request": None, "implicit_request": None, "implicit_intent": None, "evaluations_detected": None, "pseudo_feelings_detected": None, "strategy_leakage_detected": None, "translation_notes": None},
        "metadata": {"somatic_markers": None, "emotion_arousal_hint": None, "emotion_valence_hint": None, "language": "en"},
        "safety": {"label": None, "policy_category": None, "reason": None, "rewrite_mode": None, "safe_alternative": None, "safety_confidence": None},
        "quality": {"ofnr_compliance": None, "observation_is_nonjudgmental": None, "pseudo_feeling_translation_quality": None, "needs_list_match": None, "strategy_leakage_score": None, "request_is_actionable": None, "request_is_noncoercive": None, "overall_confidence": None},
        "flags": {"error_flags": None, "warnings": None},
        "teacher_agreement": {"multi_teacher_enabled": False, "teachers": [], "consensus": {"enabled": False, "method": "majority_vote", "consensus_score": 0.0, "consensus_pass": False, "resolved_need": [], "resolved_feelings": [], "resolved_observation": [], "resolved_implicit_request": []}},
        "pairwise_preference": {"available": False, "preference_alignment": {"chosen_more_constructive": None, "chosen_more_safe": None, "chosen_more_helpful": None, "notes": None}, "pairwise_signals": {"toxicity_delta": None, "constructiveness_delta": None, "helpfulness_delta": None}}
    }

def init_jsonl_file(source_path: str, target_path: str, folder: str, file: str):
    """Process JSONL file (train.jsonl)."""
    print(f"Processing JSONL: {source_path}")
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    
    with open(source_path, 'r') as infile, open(target_path, 'w') as outfile:
        lines = infile.readlines()
        for i, line in enumerate(tqdm(lines, desc=os.path.basename(source_path))):
            try:
                data = json.loads(line)
                record = create_record_pair(data, folder, file, i + 1)
                outfile.write(json.dumps(record, ensure_ascii=False) + "\n")
            except json.JSONDecodeError:
                continue

def init_json_array_file(source_path: str, target_path: str, folder: str, file: str):
    """Process JSON array file (red_team_attempts.jsonl which is actually a JSON array)."""
    print(f"Processing JSON Array: {source_path}")
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    
    with open(source_path, 'r') as infile:
        data_array = json.load(infile)
    
    with open(target_path, 'w') as outfile:
        for i, data in enumerate(tqdm(data_array, desc=os.path.basename(source_path))):
            record = create_record_redteam(data, folder, file, i + 1)
            outfile.write(json.dumps(record, ensure_ascii=False) + "\n")

def main():
    print("=" * 60)
    print("NVC-HH Dataset Initializer")
    print("=" * 60)
    
    # Process train.jsonl files
    for root, dirs, files in os.walk(SOURCE_ROOT):
        for file in files:
            if file == "train.jsonl":
                src = os.path.join(root, file)
                rel_path = os.path.relpath(src, SOURCE_ROOT)
                tgt = os.path.join(TARGET_ROOT, rel_path)
                folder = os.path.dirname(rel_path)
                init_jsonl_file(src, tgt, folder, file)
    
    # Process red-team attempts (JSON array format)
    redteam_src = os.path.join(SOURCE_ROOT, "red-team-attempts", "red_team_attempts.jsonl")
    if os.path.exists(redteam_src):
        redteam_tgt = os.path.join(TARGET_ROOT, "red-team-attempts", "red_team_attempts.jsonl")
        init_json_array_file(redteam_src, redteam_tgt, "red-team-attempts", "red_team_attempts.jsonl")

    print("=" * 60)
    print("✅ Initialization Complete.")
    print("=" * 60)

if __name__ == "__main__":
    main()
