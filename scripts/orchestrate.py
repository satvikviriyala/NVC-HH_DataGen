#!/usr/bin/env python3
"""
Orchestrate all 4 passes of the NVC-HH pipeline.

This script runs passes sequentially:
1. Observer
2. Empathizer
3. Strategist
4. Critic

Each pass reads from the previous output and writes to its own output.
The final output contains all fields filled.

Usage:
    python orchestrate.py --input data.jsonl --output final.jsonl
    python orchestrate.py --input data.jsonl --output final.jsonl --models config.yaml
"""

import argparse
import sys
import os
import json
import tempfile
import shutil

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from passes import ObserverPass, EmpathizerPass, StrategistPass, CriticPass


# Default model config (can be overridden via YAML)
DEFAULT_CONFIG = {
    "observer": {
        "model_id": "gpt-4o-mini",
        "api_base": "http://localhost:8000/v1"
    },
    "empathizer": {
        "model_id": "gpt-4o-mini",
        "api_base": "http://localhost:8000/v1"
    },
    "strategist": {
        "model_id": "gpt-4o-mini",
        "api_base": "http://localhost:8000/v1"
    },
    "critic": {
        "model_id": "gpt-4o-mini",
        "api_base": "http://localhost:8000/v1"
    }
}

PASS_ORDER = [
    ("observer", ObserverPass),
    ("empathizer", EmpathizerPass),
    ("strategist", StrategistPass),
    ("critic", CriticPass)
]


def load_config(config_path: str) -> dict:
    """Load model config from YAML or JSON."""
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            if config_path.endswith('.yaml') or config_path.endswith('.yml'):
                import yaml
                return yaml.safe_load(f)
            else:
                return json.load(f)
    return DEFAULT_CONFIG


def main():
    parser = argparse.ArgumentParser(description="Run full NVC-HH pipeline")
    parser.add_argument("--input", required=True, help="Input JSONL file")
    parser.add_argument("--output", required=True, help="Final output JSONL file")
    parser.add_argument("--config", default=None, 
                        help="Model config file (YAML/JSON)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of rows")
    parser.add_argument("--prompts-dir", default="prompts",
                        help="Directory containing pass prompts")
    parser.add_argument("--keep-intermediates", action="store_true",
                        help="Keep intermediate files")
    parser.add_argument("--start-from", type=int, default=1,
                        help="Start from pass N (1-4)")
    
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    print("=" * 60)
    print("NVC-HH FULL PIPELINE ORCHESTRATION")
    print("=" * 60)
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(f"Passes to run: {len(PASS_ORDER) - args.start_from + 1}")
    print("=" * 60)
    
    # Create temp dir for intermediates
    temp_dir = tempfile.mkdtemp(prefix="nvc_pipeline_")
    
    try:
        current_input = args.input
        
        for i, (pass_name, PassClass) in enumerate(PASS_ORDER, 1):
            if i < args.start_from:
                print(f"Skipping Pass {i}: {pass_name}")
                continue
            
            print(f"\n{'='*60}")
            print(f"PASS {i}/4: {pass_name.upper()}")
            print(f"{'='*60}")
            
            # Get pass config
            # Config structure is now config['models'][pass_name]
            pass_config = config.get("models", {}).get(pass_name, DEFAULT_CONFIG.get(pass_name, {}))
            
            # Initialize pass
            llm_pass = PassClass(
                model_id=pass_config.get("model_id", "gpt-4o-mini"),
                api_base=config.get("api", {}).get("base_url", "http://localhost:8000/v1"),
                prompts_dir=args.prompts_dir
            )
            pass_params = pass_config.get("parameters", {})
            if "temperature" in pass_params:
                llm_pass.temperature = pass_params["temperature"]
            if "max_tokens" in pass_params:
                llm_pass.max_tokens = pass_params["max_tokens"]
            
            # Determine output path
            if i == len(PASS_ORDER):
                # Final pass writes to final output
                pass_output = args.output
            else:
                pass_output = os.path.join(temp_dir, f"pass_{i}_{pass_name}.jsonl")
            
            # Run pass
            batch_size = config.get("processing", {}).get("batch_size", 64)
            llm_pass.run_file(
                input_path=current_input,
                output_path=pass_output,
                limit=args.limit,
                batch_size=batch_size
            )
            
            # Next pass reads from this output
            current_input = pass_output
            
            print(f"✅ Pass {i} ({pass_name}) complete!")
        
        print("\n" + "=" * 60)
        print("🎉 FULL PIPELINE COMPLETE!")
        print(f"Final output: {args.output}")
        print("=" * 60)
        
    finally:
        if not args.keep_intermediates:
            shutil.rmtree(temp_dir, ignore_errors=True)
        else:
            print(f"Intermediate files kept in: {temp_dir}")


if __name__ == "__main__":
    main()
