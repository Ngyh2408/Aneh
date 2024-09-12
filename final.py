import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, HashingVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn import metrics
from matplotlib import pyplot as plt
import numpy as np
import itertools

# Function to plot the confusion matrix
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Streamlit UI
st.title('Fake News Detection System')

# Load dataset
df = pd.read_csv("fake_or_real_news.csv")
df = df.set_index("Unnamed: 0")

# Separate the labels and set up training and test datasets
y = df.label 
X_train, X_test, y_train, y_test = train_test_split(df['text'], y, test_size=0.33, random_state=53)

# Building the Count and Tfidf Vectors
count_vectorizer = CountVectorizer(stop_words='english')
count_train = count_vectorizer.fit_transform(X_train)
count_test = count_vectorizer.transform(X_test)

tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
tfidf_train = tfidf_vectorizer.fit_transform(X_train)
tfidf_test = tfidf_vectorizer.transform(X_test)

# Naive Bayes classifier for Multinomial model
clf = MultinomialNB()
clf.fit(tfidf_train, y_train)
pred = clf.predict(tfidf_test)
score = metrics.accuracy_score(y_test, pred)
st.write(f"Naive Bayes accuracy (TF-IDF): {score:.3f}")
cm = metrics.confusion_matrix(y_test, pred, labels=['FAKE', 'REAL'])
st.write("Confusion Matrix (TF-IDF):")
fig, ax = plt.subplots()
plot_confusion_matrix(cm, classes=['FAKE', 'REAL'])
st.pyplot(fig)

# Applying Passive Aggressive Classifier
linear_clf = PassiveAggressiveClassifier(max_iter=50)
linear_clf.fit(tfidf_train, y_train)
pred = linear_clf.predict(tfidf_test)
score = metrics.accuracy_score(y_test, pred)
st.write(f"Passive Aggressive accuracy (TF-IDF): {score:.3f}")
cm = metrics.confusion_matrix(y_test, pred, labels=['FAKE', 'REAL'])
st.write("Confusion Matrix (Passive Aggressive):")
fig, ax = plt.subplots()
plot_confusion_matrix(cm, classes=['FAKE', 'REAL'])
st.pyplot(fig)

# HashingVectorizer
hash_vectorizer = HashingVectorizer(stop_words='english', non_negative=True)
hash_train = hash_vectorizer.fit_transform(X_train)
hash_test = hash_vectorizer.transform(X_test)

# Naive Bayes classifier for Hashing Vectorizer
clf = MultinomialNB(alpha=.01)
clf.fit(hash_train, y_train)
pred = clf.predict(hash_test)
score = metrics.accuracy_score(y_test, pred)
st.write(f"Naive Bayes accuracy (Hashing Vectorizer): {score:.3f}")
cm = metrics.confusion_matrix(y_test, pred, labels=['FAKE', 'REAL'])
st.write("Confusion Matrix (Hashing Vectorizer):")
fig, ax = plt.subplots()
plot_confusion_matrix(cm, classes=['FAKE', 'REAL'])
st.pyplot(fig)

# Applying Passive Aggressive Classifier with Hashing Vectorizer
clf = PassiveAggressiveClassifier(max_iter=50)
clf.fit(hash_train, y_train)
pred = clf.predict(hash_test)
score = metrics.accuracy_score(y_test, pred)
st.write(f"Passive Aggressive accuracy (Hashing Vectorizer): {score:.3f}")
cm = metrics.confusion_matrix(y_test, pred, labels=['FAKE', 'REAL'])
st.write("Confusion Matrix (Hashing Vectorizer):")
fig, ax = plt.subplots()
plot_confusion_matrix(cm, classes=['FAKE', 'REAL'])
st.pyplot(fig)

# User input
user_input = st.text_area("Enter a sentence to check if it's Real or Fake news:", "")

if user_input:
    # You can add prediction logic here if desired
    # Example:
    input_tfidf = tfidf_vectorizer.transform([user_input])
    prediction = clf.predict(input_tfidf)
    if prediction == 'REAL':
        st.success("The news is likely Real!")
    else:
        st.error("The news is likely Fake!")
