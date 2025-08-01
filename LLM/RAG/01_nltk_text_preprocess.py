import  nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('stopwords')

text = "RAG, or Retrieval-Augmented Generation, is a powerful tool in AI research"

tokens = word_tokenize(text)

filtered_tokens = [word for word in tokens if word.isalpha() and word.lower() not in stopwords.words('english')]

lemmatizer=WordNetLemmatizer()
lemmatized_tokens = [lemmatizer.lemmatize(word) for word in filtered_tokens]

print("original text:", text)
print("tokens:", tokens)
print("filtered tokens:", filtered_tokens)
print("lemmatized tokens:", lemmatized_tokens)