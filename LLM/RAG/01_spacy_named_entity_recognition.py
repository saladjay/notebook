import spacy
# python -m spacy download en_core_web_sm 
# 使用上面的命令下载
# use above command to download model
nlp = spacy.load('en_core_web_sm')

text = "OpenAI developed the powerful language model GPT-4, which is revolutionized AI research."

doc = nlp(text)

entities = [(ent.text, ent.label_) for ent in doc.ents]
print('original text:', text)
print('named entities:', entities)