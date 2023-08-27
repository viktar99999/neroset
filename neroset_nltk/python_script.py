from collections import Counter
from collections import defaultdict
import json
import string
import gensim
import nltk
import pandas as pd
from gensim.models import Word2Vec
from nltk.corpus import abc
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
from nltk import pos_tag
from nltk import RegexpParser
from nltk.stem import PorterStemmer
from nltk.stem import 	WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from textblob import Word
nltk.download('stopwords')
nltk.download('punkt')
TEXT = ("Python99 is the site where you can find the best tutorials for Software Testing."
    "Tutorial, Courses for Beginners."
    "Python Tutorial for Beginners and much more.")
tokens = word_tokenize(TEXT)
tokenization = [word for word in tokens if not word in stopwords.words('english')]
print(tokens)
print(tokenization)
tokenizer = RegexpTokenizer(r'\w+')
filterdText = tokenizer.tokenize("Python99 is the site where you can find the best tutorials for Software Testing.")
print(filterdText)
TEXT = "Tutorial, Courses for Beginners."
print(word_tokenize(TEXT))
TEXT = "Python Tutorial for Beginners and much more."
print(sent_tokenize(TEXT))
TEXT = "Python99 is the site where you can find the best tutorials for Software Testing.".split()
print("After Split: ", TEXT)
tokens_tag = pos_tag(TEXT)
print("After Token:",tokens_tag)
PATTERNS = """mychunk:{<NN.?>*<VBD.?>*<JJ.?>*<CC>?}"""
chunker = RegexpParser(PATTERNS)
print("After Regex:",chunker)
output = chunker.parse(tokens_tag)
print("After Chunking",output)
TEXT = "Python99 is the site where you can find the best tutorials for Software Testing."
tokens = nltk.word_tokenize(TEXT)
print(tokens)
tag = nltk.pos_tag(tokens)
print(tag)
GRAMMAR = "NP: {<DT>?<JJ>*<NN>}"
cp  =nltk.RegexpParser(GRAMMAR)
result = cp.parse(tag)
print(result)
ps =PorterStemmer()
e_words = ["python", "PYTHON", "Python"]
for w in e_words:
    rootWord=ps.stem(w)
    print(rootWord)
SENTENCE = "Python99 is the site where you can find the best tutorials for Software Testing."
words = word_tokenize(SENTENCE)
ps = PorterStemmer()
for w in words:
    rootWord = ps.stem(w)
    print(rootWord)
porter_stemmer  = PorterStemmer()
TEXT = "Tutorial, Courses for Beginners."
tokenization = nltk.word_tokenize(TEXT)
for w in tokenization:
    print("Stemming for {} is {}", w, porter_stemmer.stem(w))
wordnet_lemmatizer = WordNetLemmatizer()
TEXT = "Python Tutorial for Beginners and much more."
tokenization = nltk.word_tokenize(TEXT)
for w in tokenization:
    print("Lemma for {} is {}", w, wordnet_lemmatizer.lemmatize(w))
tag_map = defaultdict(lambda : wn.NOUN)
tag_map['J'] = wn.ADJ
tag_map['V'] = wn.VERB
tag_map['R'] = wn.ADV
TEXT = "Python99 is the site where you can find the best tutorials for Software Testing."
tokens = word_tokenize(TEXT)
lemma_function = WordNetLemmatizer()
for token, tag in pos_tag(tokens):
    lemma = lemma_function.lemmatize(token, tag_map[tag[0]])
    print(token, "=>", lemma)
syns = wn.synsets("dog")
print(syns)
synonyms = []
antonyms = []
for syn in wn.synsets("active"):
    for l in syn.lemmas():
        synonyms.append(l.name())
        if l.antonyms():
            antonyms.append(l.antonyms()[0].name())
            print(set(synonyms))
            print(set(antonyms))
TEXT = "Tutorial, Courses for Beginners."
sentence = nltk.sent_tokenize(TEXT)
for sent in sentence:
    print(nltk.pos_tag(nltk.word_tokenize(sent)))
TEXT = "Python Tutorial for Beginners and much more."
LOWER_CASE = TEXT.lower()
tokens = nltk.word_tokenize(LOWER_CASE)
tags = nltk.pos_tag(tokens)
counts = Counter( tag for word,  tag in tags)
print(counts)
A = ("Python99 is the site where you can find the best tutorials for Software Testing."
    "Tutorial, Courses for Beginners."
    "Python Tutorial for Beginners and much more.")
words = nltk.tokenize.word_tokenize(A)
fd = nltk.FreqDist(words)
TEXT = "Python99 is the site where you can find the best tutorials for Software Testing."
Tokens = nltk.word_tokenize(TEXT)
output = list(nltk.bigrams(Tokens))
print(output)
output = list(nltk.trigrams(Tokens))
print(output)
vectorizer = CountVectorizer()
data_corpus = ["Python99 is the site where you can find the best tutorials for Software Testing."
    "Tutorial, Courses for Beginners."
    "Python Tutorial for Beginners and much more."]
vocabulary = vectorizer.fit(data_corpus)
vocabulary = vectorizer.get_feature_names_out()
X = vectorizer.transform(data_corpus)
print(X.toarray())
print(vocabulary)
model = gensim.models.Word2Vec(abc.sents())
data = model.wv['language']
print(data)
data = [
    {"tag1": "Python99 is the site where you can find the best tutorials for Software Testing."},
    {"tag2": "Tutorial, Courses for Beginners."},
    {"tag3": "Python Tutorial for Beginners and much more."},
    ]
with open('intents.json', 'w') as file:
    json.dump(data, file)
JSON_FILE ="intents.json"
with open("intents.json", "br") as file:
    data = json.load(file)
df = pd.DataFrame(data)
df = df.join('')
stop = stopwords.words('english')
Bigger_list=[]
for i in df:
    li = list(i.split(" "))
    Bigger_list.append(li)	
print("Data format for the overall list: ", Bigger_list)
model = Word2Vec(Bigger_list, min_count=1, vector_size=300, workers=4)
print(model)
model.save("word2vec.model")
