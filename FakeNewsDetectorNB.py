#importing libraries

import string
import pandas
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
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


processed_text = CountVectorizer(analyzer=extract_stopwords).fit_transform(dataSet["content"])

xtrain, xtest, ytrain, ytest = train_test_split(processed_text, dataSet['label'], test_size=0.20, random_state=0)

# setting model Naive Bayes

classifier = MultinomialNB()
classifier.fit(xtrain, ytrain)

# Setting up the model's report

print("Multinomial Naive Bayes Classifier Classification Report")

predicateTraining = classifier.predict(xtrain)
print(classification_report(ytrain, predicateTraining))

predicateTest = classifier.predict(xtest)
print(classification_report(ytest, predicateTest))


# setting model Decision Tree Classifier

classifier_DT = DecisionTreeClassifier()
classifier_DT.fit(xtrain, ytrain)

# Setting up the model's report

print("Decision Tree Classifier Classification Report")

predicateTraining_DT = classifier_DT.predict(xtrain)
print(classification_report(ytrain, predicateTraining_DT))

predicateTest_DT = classifier_DT.predict(xtest)
print(classification_report(ytest, predicateTest_DT))

# Setting up Random Forest Model
classifier_RF = RandomForestClassifier()
classifier_RF.fit(xtrain, ytrain)

# Setting up the model's report

print("Random Forest Classifier Classification Report")

predicateTraining_RF = classifier_RF.predict(xtrain)
print(classification_report(ytrain, predicateTraining_RF))

predicateTest_RF = classifier_RF.predict(xtest)
print(classification_report(ytest, predicateTest_RF))
