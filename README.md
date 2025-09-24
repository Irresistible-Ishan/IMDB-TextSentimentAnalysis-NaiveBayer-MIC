# IMDB-TextSentimentAnalysis-NaiveBayer-MIC
This project was made under time constraint of 2 days for the recruitment round 2 , Microsoft Innovations Club AIML dept 2025 for On-Campus VIT CHENNAI . 

GOOGLE DRIVE LINK TO ACCESS ALL THE FILES IN A ZIP = 

## IMPORTANT : PLEASE USE GOOGLE DRIVE TO ACCESS THE DATASET AND ALL THE CODE FILES 
### as the dataset and the cleaneddataset csv were more than 25 mb so couldnt upload on the github thankyou. 
### just include dataset.csv and run the codes in the order down below and it would automatically form all the files 

### Language used : Python
### Libraries used : Sklearn , pandas , csv , pickle
### ML Method used : Naive Bayer

# RESULT : accuracy of 86.4%

I Recommend the order in which you must run the codes to come to conclusion and result 
###  cleaningDATASET.py -> trainingMain.py -> testingModel.py

Here is my thought process why , what , how on model picking and more 
# my approach 
first this is sentimental analysis 
text sentimental analysis is new to me rather than face sentimental analysis 

initital thought of me is to put all words in array and remove extras and punctuations and spaces and putting all unique set of words 
then remove the words that comes in both the positive cases adn negative cases 

but this problem is not that easy since in English sequence also matter to make sentence positive sentimental or negative sentimental 
so even if i do put all unique words in a list as positive or negative it would still get wrong result 

a very good example might be "not good" and "good" since not can be removed due to less uniqueness of the word and while good may signify that statement is good but its actually bad tho this issue can be solved using one thing that comes in my mind as per previous experience which is TF-IDR method Term frequency and Inverse Document Frequency which basically find the actual uniqueness , but more on this later but still keeping this in mind it might be useful in further methods 

so i must use a machine learning model which can analyse the sequence and pattern as well 

now the real problem is which one to pick and use 

as i have just use CNN convolutional neural network for my SIH project it was storing similar 128 dimensional features at nearby coordinates in that space to know the degree of similarity 
and as per my knowledge how LLMs work they also use the similar method of embedding similar words in a vector embedding to compare relativeness to each other 

here we can do it for 2 segments like positive or negative sentiment analysis , but idk how yet i still need to think and analyse further to see all options and the dataset 

it might just be a multi dimensional classification problem with use of vector embedding since the output is always either one or zero 1 or 0 so its not as complex as making an LLM thats a whole different case with whole different architecture which is transformer architecture but yes worth taking into context as this and that hold so much similarity to the approach like how i initially thought i should break it in array of all words same way like tokenisation but simpler in my case (much simpler)

i want to initially test it with something like classification model just like my previous project with logistic regression which is a classification model to classify 1 or 0 as the output but since like the last time i dont think text behaviour of English is so simple that we can solve it in a linear way or maybe we can taking TF-IDF and words of importance in axis of multiple dimensions and see the general trend of the areas that became either 0 or 1 according to classification then in a single sentence check for the ratio of how much % comes as 1 and as 0 then predict accordingly 
but i wanna take a non linear approach this time which should be better as above 


Ill be honest here i forgot to open the instructions.pdf lol so now my perspective has changed

I was gonna initially use distilBert or bert but after reading the instructions pdf i realised its for seniors and being a fresher i think i should just work with linear regression or classification based model since im super familiar with them , i could also use CNN even for text as i have worked with it before but not anymore 
and data is simple enough so ill be taking the same approach as i initially thought above 

ill be going with Naive Bayes since linear regression is way too simple for this , but im also thinking of reporting and testing both but first ill be applying Naive Bayes 

NOW , Im moving forward with cleaning the dataset so i can implement the dataset preparation , so im removing all punctuation and the br tags given and all the unnecessary things also removing all sorts of numbering as they can mean anything and so wont matter in the prediction , also made every letter lower() , and i saved the new clean file as a new dataset as csv 

now ill move forward with the main ML logic 

now as i tried to search how i can know its accuracy , like in previous logistic regression i trained i was using normal accuracy methods to test its accuracy but in this i got to know i must split the data in 2 parts randomly one for training and one to accuracy testing to see the model isn't overfitting rather its actually generalising and learning 

now to create an embedding like i used to in CNN vector 128 dimensional vector space i need to do the same here to know which words tends towards positivity and which negativity actually thats the task of the model but still we need to specify the special words here like if the is everywhere then model can confuse it with a unique or important word so ill use TF IDF here , even tho instructions linked count vectorizer i feel like i should be using TF IDF because it can help it show a better behaviour and its almost the same approach just different method , its better than just raw counting as i mentioned earlier like "the" "is" etc , TF IDF is more needed here since i didn't clean the dataset of the non important words like is and the.


