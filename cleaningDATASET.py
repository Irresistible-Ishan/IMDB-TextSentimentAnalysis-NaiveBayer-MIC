

# ill start by importing the dataset and clearning it 
# last time i used directly the csv file handling using import csv but this time 
# ill use pandas ig
import pandas as pd
import re

data = pd.read_csv("IMDB_Dataset.csv")
review = data["review"]
#sentiment = data["sentiment"]

for i, j in enumerate(review):
    review[i] = re.sub(r"\bbr\b", " ", re.sub(r"\s+", " ", re.sub(r"[^a-zA-Z\s]", " ", str(j))).strip().lower())

# here i made the pos and neg as 1 and 0 as i did in logistic regression project
# so it makes a 1 or 0 segrment in the whole data when its trained 
def cleaningsentiment(s):
    if s == "positive":
        return 1
    elif s == "negative":
        return 0
    else:
        return None

data["sentiment"] = data['sentiment'].map(cleaningsentiment)
data = data.dropna(subset=['sentiment'])

sentiment = data["sentiment"]
# save the clean dataset in the new file 
pd.DataFrame({"review":review , "sentiment":sentiment}).to_csv("cleaned_data.csv",index=False)

