#!/usr/bin/env python3
"""
Run a single pass of the NVC-HH pipeline.

Usage:
    python run_pass.py --pass observer --input data.jsonl --output out.jsonl
    python run_pass.py --pass empathizer --input out.jsonl --output out2.jsonl
"""

import argparse
import sys
import os

# Add parent dir to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from passes import ObserverPass, EmpathizerPass, StrategistPass, CriticPass


PASS_CLASSES = {
    "observer": ObserverPass,
    "empathizer": EmpathizerPass,
    "strategist": StrategistPass,
    "critic": CriticPass
}


def main():
    parser = argparse.ArgumentParser(description="Run a single LLM pass")
    parser.add_argument("--pass", dest="pass_name", required=True, 
                        choices=list(PASS_CLASSES.keys()),
                        help="Which pass to run")
    parser.add_argument("--input", required=True, help="Input JSONL file")
    parser.add_argument("--output", required=True, help="Output JSONL file")
    parser.add_argument("--model", default="gpt-4o-mini", 
                        help="Model ID for the LLM")
    parser.add_argument("--api-base", default="http://localhost:8000/v1",
                        help="API base URL (vLLM or OpenAI-compatible)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of rows to process")
    parser.add_argument("--prompts-dir", default="prompts",
                        help="Directory containing pass prompt files")
    
    args = parser.parse_args()
    
    # Get pass class
    PassClass = PASS_CLASSES[args.pass_name]
    
    # Initialize pass
    llm_pass = PassClass(
        model_id=args.model,
        api_base=args.api_base,
        prompts_dir=args.prompts_dir
    )
    
    print(f"=" * 60)
    print(f"Running Pass: {args.pass_name.upper()}")
    print(f"Model: {args.model}")
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(f"=" * 60)
    
    # Run
    llm_pass.run_file(
        input_path=args.input,
        output_path=args.output,
        limit=args.limit
    )
    
    print(f"✅ Pass {args.pass_name} complete!")


if __name__ == "__main__":
    main()
