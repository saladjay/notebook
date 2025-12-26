

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", type=str, choices=["remote_sft", "remote_sft_advanced","local_sft"], help="Action to perform: 'remote_sft' or 'remote_sft_advanced' or 'local_sft'")
    parser.add_argument("--dataset_desp", type=str, default="", required=False, help="Additional option for the action")
    parser.add_argument("--query_template", type=str, default="", required=False, help="Additional option for the action")
    parser.add_argument("--dataset_path", type=str, default="", required=False, help="Additional option for the action")
    
    args = parser.parse_args()
    if args.action == "remote_sft":
        from generate_data.generate_sft_data import generate_sft_data
        generate_sft_data()
    if args.action == "remote_sft_advanced":
        from generate_data.generate_sft_data_advanced import generate_sft_data
        generate_sft_data()
    if args.action == "local_sft":
        from generate_data.generate_sft_data_local_qwen3 import generate_sft_data
        generate_sft_data()