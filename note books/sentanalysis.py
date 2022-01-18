#Sentiment Analysis With Naive Bayes Classifier!

import pandas as pd
from sklearn.model_selection import train_test_split
import joblib
from sklearn.feature_extraction.text import CountVectorizer

data = pd.read_csv('google_play_store_apps_reviews_training.csv')

#preprocess the data
def preprocess_data(data):
    # Remove package name as it's not relevant
    data = data.drop('package_name', axis=1)
    
    # Convert text to lowercase
    data['review'] = data['review'].str.strip().str.lower()
    return data

data = preprocess_data(data)


#Splitting the data
x = data['review']
y = data['polarity']
x, x_test, y, y_test = train_test_split(x,y, stratify=y, test_size=0.25, random_state=42)

# Vectorize text reviews to numbers
vec = CountVectorizer(stop_words='english')
x = vec.fit_transform(x).toarray()
x_test = vec.transform(x_test).toarray()

#Model generation
from sklearn.naive_bayes import MultinomialNB

model = MultinomialNB()
model.fit(x, y)

print("Accuracy : ", model.score(x_test, y_test))

review1 = "Love this app simply awesome!"
review2 = "Navigation in this app is hard."
print("prediction for the review - ", review1, " is ", model.predict(vec.transform([review1])))
print("prediction for the review - ", review2, " is ", model.predict(vec.transform([review2])))

