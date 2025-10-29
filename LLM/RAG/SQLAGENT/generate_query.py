from generate_data import generate_query 
import re
import json
if __name__ == "__main__":
    # generate_query.generate_query()

    with open("generate_data\generate_query.json", 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f.readlines()]
    
    total_question_list = []

    for line in lines:
        # INSERT_YOUR_CODE
        # 提取subject后的字符串内容
        subject_pattern = r'\\?"subject\\?"\s*:\s*\\?"(.*?)\\?"'
        subject_match = re.search(subject_pattern, line)
        if subject_match:
            extracted_subject = subject_match.group(1)
            print(f"Extracted subject: {extracted_subject}")
        # 方法1: 改进的正则表达式，处理转义字符
        pattern = r'\\?"question\\?"\s*:\s*(\[(?:[^\[\]]|\[[^\]]*\])*\])'
        matches = re.findall(pattern, line, re.DOTALL)
        
        if matches:
            for idx, questions_str in enumerate(matches):
                if idx == 0:
                    continue
                # print(f"Extracted question list string #{idx+1}:")
                
                try:
                    questions_str = questions_str.replace("\\n", "").replace("\\", "")
                    questions_list = json.loads(questions_str)
                    for question in questions_list:
                        total_question_list.append(f"{extracted_subject}:{question}")
                    print(f"Parsed question list #{idx+1}:")
                    print(questions_list)
                except Exception as e:
                    print(f"Error parsing question list #{idx+1}: {e}")
    
    with open('generate_data\ai_generate_query.txt', 'w', encoding='utf-8') as f:
        for question in total_question_list:
            f.write(question + '\n')
