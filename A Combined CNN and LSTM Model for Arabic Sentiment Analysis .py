#Khalil Daibes 315688960
#Fidaa Khoury 206012155
#Milad Irhayil 208863001
#
#
#

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve,  roc_auc_score, classification_report
from gensim import models
import gensim
from gensim.models import KeyedVectors
import gensim.models.keyedvectors as word2vec #importing word to vectore 
from nltk.tokenize import RegexpTokenizer #importing toknizer  to clean the text 
import logging
from keras.models import Sequential #importing sequential as desierd model arch
from keras.layers.embeddings import Embedding #importing embedding 
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.layers import LSTM, Conv1D, Dense, Flatten, MaxPooling1D, Dropout


# *********************************************************************************
#                                 DEFINING USFUL THINGS 
# *********************************************************************************

logging.basicConfig(format='%(asctime)s : %(levelname) s : %(message)s', level=logging.INFO)

#Set random seed
np.random.seed(24)
unicode_errors='ignore'


# *********************************************************************************
#                               GET DATASET AND WORD2VEC MODEL
# *********************************************************************************



#read CSV file containing tweets and labels, using Pandas , to get a dataframe
tweetsData = pd.read_csv('Arabic Sentiment Analysis Dataset.csv', skiprows=[8835, 535881]) #skiping these two rows as they have some bad data
tweetsData.head()


unicode_errors='ignore'
w2vModel = gensim.models.Word2Vec.load('full_grams_sg_100_twitter.mdl')

#check the distribution of lebels
labels_count = labels.value_counts()
labels_count.plot(kind="bar")
print(labels.value_counts())


# *********************************************************************************
#                    TEXT CLEANING AND CONSTRUCTING VOCABULARY
# *********************************************************************************

#length of tweet to consider
maxlentweet = 30
#Dividing the dataset into features and lables
tweets = tweetsData['text']
labels = tweetsData['Sentiment']




#Looks like the distribution is even
#Lower and split the dialog
#and use regular expression to keep only letters we will use nltk Regular expression package
tkr = RegexpTokenizer('[ء-ي]+')

tweets_split = []

for i, line in enumerate(tweets):
    #print(line)
    tweet = str(line).lower().split()
    tweet = tkr.tokenize(str(tweet))

    tweets_split.append(tweet)



#Convert words to integers
#Only top num_words-1 most frequent words will be taken into account.
#Only words known by the tokenizer will be taken into account.
#Returns A list of sequences.
tokenizer = Tokenizer()
tokenizer.fit_on_texts(tweets_split)
#most frequent words will be taken into account.
# make a vocabulary,  here we give each word a index number  
X = tokenizer.texts_to_sequences(tweets_split)

#add padding
X = pad_sequences(X, maxlen=maxlentweet)

# *********************************************************************************
#                           CONSTRYCTING THE MODEL SEQUANCE
# *********************************************************************************






# *********************************************************************************
#create a embedding layer using Arabic to vector model (w2vModel is a pre triained word2vec)
embedding_layer = Embedding(input_dim= w2vModel.wv.syn0.shape[0], output_dim=w2vModel.wv.syn0.shape[1], weights=[w2vModel.wv.syn0], input_length=X.shape[1])


#create model combining LSTM with 1D Convonutional layer and MaxPool layer

lstm_out = 150

# sequential model is to ensure that we are working with layers oine after the other 
# in this sequential model we used first 
# embedding layer (hidden layer) ?
# cnn to extract the feature 
# we use a max-pooling with size 2 ?? 
model = Sequential()
#
model.add(embedding_layer)
#64 filter every one sized 5 , relu must be positive 

# *********************************************************************************
model.add(Conv1D(filters=64, kernel_size=5, activation='relu', padding='causal'))
# *********************************************************************************
model.add(MaxPooling1D(pool_size=2))
# *********************************************************************************
model.add(Dropout(0.5))
# *********************************************************************************
model.add(LSTM(units=lstm_out))
# *********************************************************************************
model.add(Dense(1, activation='sigmoid')) #function(sigmoid)
# *********************************************************************************
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])#adam ,,, binary_crossentropy
# *********************************************************************************
print(model.summary())



#split dataset
X_train, X_test, Y_train, Y_test = train_test_split(X, labels, test_size= 0.2, random_state = 24)

#fit model
batch_size = 32
# eboch = how many runs
model.fit(X_train, Y_train, epochs=5, verbose=1, batch_size=batch_size)



#analyze the results
score, acc = model.evaluate(X_test, Y_test, verbose=2, batch_size=batch_size)
y_pred = model.predict(X_test)




#ROC AUC curve
rocAuc = roc_auc_score(Y_test, y_pred)

falsePositiveRate, truePositiveRate, _ = roc_curve(Y_test, y_pred)

plt.figure()

plt.plot(falsePositiveRate, truePositiveRate, color='green',
         lw=3, label='ROC curve (area = %0.2f)' % rocAuc)
plt.plot([0, 1], [0, 1], color='navy', lw=3, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic of Sentiiment Analysis Model')
plt.legend(loc="lower right")
plt.show()







#Other accuracy metrices
y_pred = (y_pred > 0.5)

#confusion metrix
cm = confusion_matrix(Y_test, y_pred)
print(cm)

#F1 Score, Recall and Precision
print(classification_report(Y_test, y_pred, target_names=['Positive', 'Negative']))

model.save('sentiment_analysis_twitter_cnn_lstm.h5')



