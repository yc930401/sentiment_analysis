import numpy
import keras
import pickle
from keras.utils.data_utils import get_file
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.metrics import accuracy_score


numpy.random.seed(7)
# load the dataset but only keep the top n words, zero the rest
path = get_file('imdb_full.pkl',
               origin='https://s3.amazonaws.com/text-datasets/imdb_full.pkl',
                md5_hash='d091312047c43cf9e4e38fef92437263')
f = open(path, 'rb')
(x_train, y_train), (x_test, y_test) = pickle.load(f)

word_to_id = keras.datasets.imdb.get_word_index()
word_to_id["<PAD>"] = 0
word_to_id["<START>"] = 1
word_to_id["<UNK>"] = 2

id_to_word = {value:key for key,value in word_to_id.items()}
x_test = [' '.join(id_to_word[index] for index in data) for data in x_test]


sid = SentimentIntensityAnalyzer()
y_pred = []
for sentence in x_test:
    #print(sentence)
    ss = sid.polarity_scores(sentence)['compound']
    y_pred.append(1 if ss > 0 else 0)

print("Accuracy: %.2f%%" % (100*accuracy_score(y_test, y_pred)))
