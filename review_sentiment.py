import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv(r'C:\Users\vaish\Desktop\files for python proj\amazon_reviews.csv')
df.info()
df.describe()

# Plot the count plot for the ratings
sns.countplot(df['rating'])
# Getting the length of the verified_reviews column
df['length'] = df['verified_reviews'].apply(len)
df['length'].hist(bins=100)

# Plot the countplot for feedback
sns.countplot(x = df['feedback']) # shows mostly positive feedback

# Obtain only the positive reviews
positive = df[df['feedback'] == 1]
# Obtain the negative reviews only
negative = df[df['feedback'] == 0]
# Convert to list format
sentences = positive['verified_reviews'].tolist()
# Join all reviews into one large string
sentences_as_one_string =" ".join(sentences)
sentences_as_one_string

from wordcloud import WordCloud
plt.figure(figsize=(20,20))
plt.imshow(WordCloud().generate(sentences_as_one_string))

import string
import nltk # Natural Language tool kit
nltk.download('stopwords')

# Download stopwords Package to execute this command
from nltk.corpus import stopwords
stopwords.words('english')

def message_cleaning(message):
    Test_punc_removed = [char for char in message if char not in string.punctuation]
    Test_punc_removed_join = ''.join(Test_punc_removed)
    Test_punc_removed_join_clean = [word for word in Test_punc_removed_join.split() if word.lower() not in stopwords.words('english')]
    return Test_punc_removed_join_clean

# Test the function
reviews_df_clean =  df['verified_reviews'].apply(message_cleaning)

from sklearn.feature_extraction.text import CountVectorizer
# Pipeline for vectorisation
vectorizer = CountVectorizer(analyzer = message_cleaning)
reviews_countvectorizer = vectorizer.fit_transform(df['verified_reviews'])
print(reviews_countvectorizer.toarray())
reviews = pd.DataFrame(reviews_countvectorizer.toarray())
X = reviews
y = df['feedback']

# Train and test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

from sklearn.naive_bayes import MultinomialNB
NB_classifier = MultinomialNB()
NB_classifier.fit(X_train, y_train)
from sklearn.metrics import classification_report, confusion_matrix

# Predicting the Test set results using NB classifier
y_predict_test = NB_classifier.predict(X_test)
cm = confusion_matrix(y_test, y_predict_test)
sns.heatmap(cm, annot=True)
print(classification_report(y_test, y_predict_test))

# Using lr classifier
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
cm = confusion_matrix(y_pred, y_test)
sns.heatmap(cm, annot = True)
print(classification_report(y_test, y_pred))

# Using gradient boost classifier
from sklearn.ensemble import GradientBoostingClassifier
model = GradientBoostingClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
cm = confusion_matrix(y_pred, y_test)
sns.heatmap(cm, annot = True)
print(classification_report(y_test, y_pred))

# For this data set the accuracy in classifier ranks as Naive bayes > Linear regression > Gradient boost










