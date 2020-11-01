import pickle

spam_clf = pickle.load(open("spam_classifier","rb"))

message = input("Enter the message ")

pd = spam_clf.predict([message])

if pd[0] == 'spam':
    print("The message provided is a spam\n")
else:
    print("The message is not a spam")