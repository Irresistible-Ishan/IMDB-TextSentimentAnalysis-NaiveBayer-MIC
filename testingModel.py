# this code is for testing the trained and stores model using pickle , to test ill use the metrics given in sklearn
# ill use accuracy score to check how accurate the model is

from sklearn.metrics import accuracy_score
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split

data = pd.read_csv("cleaned_data.csv")
reviews = data["review"]
sentiment = data["sentiment"]

with open("trainedModel.pkl", "rb") as f:
    vectorizer , model = pickle.load(f)

with open("vectorembedding.pkl", "rb") as f:
    vectorizer = pickle.load(f)

xtrain, xtest, ytrain, ytest =train_test_split(reviews, sentiment, test_size=0.2, random_state=42)
# taking same randomness seed 

xvectest = vectorizer.transform(xtest)
ypred = model.predict(xvectest)

accuracy = accuracy_score(ytest, ypred)  # we are just doing ratio correct / total predictions
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# RESULT : Model  Accuracy: 86.40 %
# hmmm haha there's time constraint but i would love to apply more ml models on this, this dataset was fun 
# thank you for  this problem 
# im out of time otherwise i would also  apply more testing  methods to analyse the model
# also i would  love to do hyperparameter tuning and cross validation to improve the model
# but this was  fun


