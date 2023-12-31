#import pandas and numpy
import pandas as pd
import numpy as np

#importing libraries which help to clean data
import re
import nltk
import string

#importing stopwords
from nltk.corpus import stopwords
stopwords = set(stopwords.words("english"))

#importing stemmer
stemmer = nltk.SnowballStemmer("english")

#reading our file
dataset = pd.read_csv("labeled_data.csv")

#new column named 'labels'
dataset["labels"] = dataset["class"].map({0: "Hate Speech",
                                          1: "Offensive Language",
                                          2: "Neither Hate nor Offensive"})

#new data frame with only tweets and labels
data = dataset[["tweet", "labels"]]

#cleaning the data
def clean_data(text):
    text = str(text).lower()
    text = re.sub("https?://\S+|www\.S+", "", text)
    text = re.sub("\[.*?\]", "", text)
    text = re.sub("<.*?>+", "", text)
    text = re.sub("[%s]" %re.escape(string.punctuation), "", text)
    text = re.sub("\n", "", text)
    text = re.sub("\w*\d\w*", "", text)
    #remove stopwards
    text = [word for word in text.split(' ') if word not in stopwords]
    text = " ".join(text)
    #stemming
    text = [stemmer.stem(word) for word in text.split(" ")]
    text = " ".join(text)
    return text

data["tweet"] = data["tweet"].apply(clean_data)

x = np.array(data["tweet"])
y = np.array(data["labels"])

#importing libraries to help build our model
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

#vectorising the data
cv = CountVectorizer()
x = cv.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.33, random_state = 42)

#building the model
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
dt.fit(x_train, y_train)
y_pred = dt.predict(x_test)

#confusion matrix and accuracy
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

#plot to show confusion matrix and accuracy
"""import seaborn as sbn
import matplotlib.pyplot as plt

sbn.heatmap(cm, annot=True, fmt=".1f", cmap="YlGnBu")
plt.show()

#accuracy score
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, y_pred))
"""

sample = input("Enter the tweet:")
sample = clean_data(sample)
data1 = cv.transform([sample]).toarray()

print(dt.predict(data1))
