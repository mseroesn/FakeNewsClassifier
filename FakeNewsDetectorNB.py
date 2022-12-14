#importing libraries
import string
import pandas
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

# Setting Dataset variable
dataSet = pandas.read_csv("DataSet.csv")

# Cleaning Data

dataSet.drop_duplicates(keep="first")
dataSet.dropna(axis=0, inplace=True)

# print(trainingDataSet.shape)

dataSet["content"] = dataSet["author"] + " " + dataSet["title"]

nltk.download('stopwords')

# Processing text


def extract_punctuations(text):
    text_without_punctuation = [character for character in text if character not in string.punctuation]
    text_without_punctuation = ''.join(text_without_punctuation)

    return text_without_punctuation


def extract_stopwords(text):
    text_without_stop_words = [word for word in extract_punctuations(text).split() if word.lower() not in stopwords.words("english")]

    return text_without_stop_words


dataSet = CountVectorizer(analyzer=extract_stopwords).fit_transform(dataSet["content"])

xtrain, xtest, ytrain, ytest = train_test_split(dataSet, dataSet['label'], test_size=0.20, random_state=0)


# setting model

classifier = MultinomialNB()
classifier.fit(xtrain, ytrain)

# Setting up the model's report

predicateTraining = classifier.predict(xtrain)
print(classification_report(ytrain, predicateTraining))

predicateTest = classifier.predict(xtest)
print(classification_report(ytest, predicateTest))
