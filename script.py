import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
import pickle
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

# reading the spam dataset
df = pd.read_csv("smsspamcollection.tsv",sep="\t")


# Here we have made the predictions based on just text 
# other attributes like length and punct may also be used
X = df["message"]

# the label consist of spam and ham classification
y = df["label"]

# the dataset is split into train and test dataset 
# it can be split into validation also
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Creating the pipeline and training the model
spam_clf2 = Pipeline([("tfdif",TfidfVectorizer()),('text_clf',LinearSVC())])

spam_clf2.fit(X_train,y_train)

# Getting the predictions on test dataset 
predictions = spam_clf2.predict(X_test)

print(confusion_matrix(y_test,predictions))

print(classification_report(y_test,predictions))

print(accuracy_score(y_test,predictions))

pickle.dump(spam_clf2,open("spam_classifier",'wb'))