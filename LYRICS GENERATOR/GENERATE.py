import tensorflow
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
import tensorflow.keras.utils as ku
import numpy as np
#loading the models

new_model2 = keras.models.load_model('model location')


# loading and preparing the data
tokenizer = Tokenizer()

data = open('rock.txt').read()
corpus = data.lower().split("\n")





tokenizer.fit_on_texts(corpus)
total_words = len(tokenizer.word_index) + 1

# create input sequences using list of tokens
input_sequences = []
for line in corpus:
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        input_sequences.append(n_gram_sequence)


# pad sequences
max_sequence_len = max([len(x) for x in input_sequences])
input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))

# collecting word stoppers from lyrics
data = open('rock.txt').read()
corpus = data.lower().split("\n")
def lastWord(string):
    newstring = ""    
    length = len(string)
    for i in range(length-1, 0, -1):
        if(string[i] == " "):
            return newstring[::-1]
        else:
            newstring = newstring + string[i]
stopwords=[]
for i in corpus:
  stopwords.append(lastWord(i))
improved_stopwords = []
for i in stopwords:
  try:
    if i.lower() not in improved_stopwords:
        improved_stopwords.append(i.lower())
  except:
    if i not in improved_stopwords:
        improved_stopwords.append(i)

stopwordss = ["a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "as", "at",
              "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "could", "did", "do",
              "does", "doing", "down", "during", "each", "few", "for", "from", "further", "had", "has", "have",
              "having", "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself", "him", "himself",
              "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "it", "it's", "its",
              "itself", "let's", "me", "more", "most", "my", "myself", "nor", "of", "on", "once", "only", "or", "other",
              "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "she", "she'd", "she'll", "she's",
              "should", "so", "some", "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves",
              "then", "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this", "those",
              "through", "to", "too", "under", "until", "up", "very", "was", "we", "we'd", "we'll", "we're", "we've",
              "were", "what", "what's", "when", "when's", "where", "where's", "which", "while", "who", "who's", "whom",
              "why", "why's", "with", "would", "you", "you'd", "you'll", "you're", "you've", "your", "yours",
              "yourself", "yourselves"]


# genrating new lyrics
def generate(x):
    seed_text = x
    next_words = 50

    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len - 1, padding='pre')
        # predicting with the trained model
        predicted = new_model2.predict_classes(token_list, verbose=0)
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
        seed_text += " " + output_word
    wh = seed_text.split(' ')
    # can use stopwords from txt file as well
    for i in wh:
        if i in stopwordss:
            print(i)

            continue
        print(i, end=" ")


inn = input(' TYPE YOUR STARTING LYRIC')
generate(inn)

