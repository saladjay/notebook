from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import os
os.environ["TRANSFORMERS_VERBOSITY"] = "info"
model_name = 'gpt2'
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

prompt = "Artificial intelligence can help in many area,"

inputs = tokenizer(prompt, return_tensors='pt')

output_sequences = model.generate(
    input_ids=inputs['input_ids'],
    max_length=100,
    num_return_sequences=1,
    no_repeat_ngram_size=2,
    top_k=50,
    top_p=0.7,
    temperature=0.5
)

generated_text = tokenizer.decode(output_sequences[0], skip_special_tokens=True)

print(f"输入提示语：{prompt}")
print(f"生成的文本：{generated_text} end")