import re #importing re module with regular expressions
import nltk #importing natural language toolkit module
from nltk.corpus import stopwords #importing collection of stop words in NLT (you can add
#related stop words to the problem in addition to these)
nltk.download('stopwords')
from nltk.stem import WordNetLemmatizer #for lemmatization
nltk.download('wordnet')

movie_data = load_files(r"txt_sentoken")
X, y = movie_data.data, movie_data.target
