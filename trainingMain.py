# first ill split the data randomly in 80:20 ratio to later check for model accuracy as mentioned in the 
# thinking process file

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB  
#from sklearn.metrics import accuracy_score
  # using this i can test the accuracy for which i seprated the train variable
import pickle
import pandas as pd

data = pd.read_csv("cleaned_data.csv")
reviews = data["review"]
sentiment = data["sentiment"]

xtrain, xtest, ytrain, ytest = train_test_split(reviews , sentiment ,test_size=0.2 ,random_state=42)
# i saw in docs that randomstate is like the seed so that we can 
# know how we splitted the data randomly so we can 
# do it again when needed 


vectorizer = TfidfVectorizer()
Xvec = vectorizer.fit_transform(xtrain)
Xvectest = vectorizer.transform(xtest)
# we only vectorise x and not y is simply because x is the main input and y will be the output segment as 1 or 0


model = MultinomialNB()
model.fit(Xvec, ytrain)
print("model trained")

# now ill store the model in a pickle file and then reuse it in a different file to test it 
# so obviously i dont need to retrain it every time

with open("trainedModel.pkl", "wb") as f:
    pickle.dump((vectorizer, model), f)

with open("vectorembedding.pkl", "wb") as f:
    pickle.dump(vectorizer, f)


# i dont understand how so trained so fast even my logistic regression didnt train so fast 
# and the dataset is so large still 

# please go to accuracy testing file to see how accurate the model is now
