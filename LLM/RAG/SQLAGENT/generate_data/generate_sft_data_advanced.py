import os
import json
import logging
import random
from sql_agent.sql_agent_demo import SQLAgentDemo
from sql_agent.sql_agent_qwen import SQLAgentQwen3
from callbacks import LLMLoggingCallback
import sqlite3
# 关闭 httpx 的 HTTP 请求日志
logging.getLogger("httpx").setLevel(logging.WARNING)
# 如果还有其他干扰日志，也可以关闭
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

callbacks = [LLMLoggingCallback(json_file="generate_data\sft_data7_20251105_part2.json", log_to_file=False, log_to_console=False, save_json=True)]

agent = SQLAgentQwen3()

KEY_WORD = ['{dataset_name}', '{top_k}', '{label}', '{top_k_cn}', '{tag}', '{project_name}', \
    '{path}', '{short_cut}', '{severe_score}', '{region_range}', '{region_size}']

def find_and_count_key_word(query_template):
    info = {}
    for key_word in KEY_WORD:
        count = query_template.count(key_word)
        info[key_word] = count
    return info

def check_key_word_exists(modified_query):
    if modified_query.find('{') != -1 or modified_query.find('}') != -1:
        return False
    else:
        return True

def get_key_words(count, choices):
    final_set = set()
    if count > len(choices):
        raise ValueError("Count exceeds the number of available choices.")
    while len(final_set)<count:
        final_set.add(random.choice(choices))
    return list(final_set)

def fill_key_word(query_template, datasets, labelClasses, tags, paths, short_cuts, 
                  project_names=['当前', '目前', '本次'], 
                  top_ks=[str(i) for i in range(1,9)],
                  top_k_cns=['一','二','三','四','五','六','七','八'],
                  severe_scores=[i for i in range(1,11)],                
                  region_ranges=[],
                  region_sizes=[],
                  ):
    info = find_and_count_key_word(query_template)
    modified_query = query_template
    # print(info)
    for key_word, count in info.items():
        if count > 0:
            if key_word == '{dataset_name}':
                choices = datasets
            elif key_word == '{label}':
                choices = labelClasses
            elif key_word == '{tag}':
                choices = tags
            elif key_word == '{project_name}':
                choices = project_names
            elif key_word == '{path}':
                choices = paths
            elif key_word == '{short_cut}':
                choices = short_cuts
            elif key_word == '{top_k}':
                choices = top_ks
            elif key_word == '{top_k_cn}':
                choices = top_k_cns
            elif key_word == '{severe_score}':
                choices = severe_scores
            elif key_word == '{region_range}':
                choices = region_ranges
            elif key_word == '{region_size}':
                choices = region_sizes
            else:
                raise ValueError(f"unknown key word: {key_word}")
            
            selected_words = get_key_words(count, choices)
            for word in selected_words:
                modified_query = modified_query.replace(key_word, str(word), 1)
    return modified_query


def generate_sft_data(dataset_desp_json:str=r'D:\github\notebook\LLM\RAG\SQLAGENT\example_dataset.json',
                      sql_query_template:str=r'generate_data\sql_query_template_example.json',
                      debug_flag=False):
    config = {}
    with open(dataset_desp_json, 'r', encoding='utf-8') as f:
        dataset_desp = json.load(f)
        
        datasets = [item['Name'] for item in dataset_desp['datasets']]
        labelClasses = [item['Name'] for item in dataset_desp['labelClasses']]
        tags = [item['Name'] for item in dataset_desp['tags']]
        paths = [str(item) for item in dataset_desp['partial_paths'] if str(item)!="None"]
        short_cuts = dataset_desp['short_cuts']
        region_ranges = list(map(str, dataset_desp['region_ranges'])) 
        region_sizes = dataset_desp["region_sizes"]
        config = {
            'datasets': datasets,
            'labelClasses': labelClasses,
            'tags': tags,
            'paths': paths,
            'short_cuts': short_cuts,
            'region_ranges': region_ranges,
            'region_sizes': region_sizes
        }
        print("dataset desp:", json.dumps(config, indent=4, ensure_ascii=False))

    query_list = []
    with open(sql_query_template, 'r', encoding='utf-8') as f:
        query_templates = json.load(f)
        for i, query_template in enumerate(query_templates):
            if debug_flag:
                print(f"Processing query template {i+1}: {query_template}")
            modified_query = fill_key_word(query_template, **config)
            if debug_flag:
                print("Modified Query:", modified_query)
            query_list.append(modified_query)
    if debug_flag:
        exit()

    for i, query in enumerate(query_list):
        result = agent.query(query, callbacks)

if __name__ == "__main__":
    generate_sft_data()

