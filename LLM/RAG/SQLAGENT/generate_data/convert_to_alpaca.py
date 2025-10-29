import json
import os
import argparse
from copy import deepcopy
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_files", type=str, nargs="+", required=True)
    parser.add_argument("--output_file", type=str, required=True)

    args = parser.parse_args()
    base_structure = {
        "instruction": "",
        "input": "",
        "output": ""
    }
    for input_file in args.input_files:
        if input_file.endswith(".json"):
            with open(input_file, "r", encoding="utf-8") as f:
                json_content = json.load(f)
                print(f"{input_file} has {len(json_content)} samples")
                for i, data in enumerate(json_content):
                    current_base_structure = deepcopy(base_structure)
                    if len(data["prompts"]) > 1:
                        print(f"{input_file} has more than one prompt at line {i}")
                        continue
                    if len(data['generations']) > 1:
                        print(f"{input_file} has more than one generation at line {i}")
                        continue
                    current_base_structure["instruction"] = data["prompts"][0]
                    current_base_structure["input"] = 'the input question you must answer'
                    current_base_structure["output"] = data["generations"][0]
                    with open(args.output_file, "a", encoding="utf-8") as of:
                        content = json.dumps(current_base_structure, ensure_ascii=False)
                        print(content, file=of)
        if input_file.endswith(".jsonl"):
            with open(input_file, "r", encoding="utf-8") as f:
                for line in f:
                    data = json.loads(line)
                    current_base_structure = deepcopy(base_structure)
                    if "prompts" in data and len(data["prompts"]) > 1:
                        print(f"{input_file} has more than one prompt at line {i}")
                        continue
                    if "generations" in data and len(data['generations']) > 1:
                        print(f"{input_file} has more than one generation at line {i}")
                        continue
                    
                    current_base_structure["instruction"] = data["prompts"][0] if "prompts" in data else data["prompt"]
                    current_base_structure["input"] = 'the input question you must answer'
                    current_base_structure["output"] = data["generations"][0] if "generations" in data else data["response"]
                    with open(args.output_file, "a", encoding="utf-8") as of:
                        content = json.dumps(current_base_structure, ensure_ascii=False)
                        print(content, file=of)
