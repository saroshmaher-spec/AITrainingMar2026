from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import pipeline
import joblib

vectorization = TfidfVectorizer()

Data = ["Win money now", "Limited Offer Click here", "Hello Friend how are you", "Congragulations you won a prize", "important update aboout your account","Let us meet tomorrow"]
label = [1,1,0,1,0,0]

X = vectorization.fit_transform(Data).toarray()
print(X.shape)

model = MLPClassifier(hidden_layer_sizes =(100,), max_iter=300)
model.fit(X,label)

joblib.dump(model, "model.pkl")

test_sample = ["Win a free phone now", "meeting scheduled tomorrow"]

X_test = vectorization.transform(test_sample).toarray()
prediction = model.predict(X_test)
print(prediction)
for email , pred in zip(test_sample, prediction):
    print(email, '->', "Spam" if pred == 1 else "Not Spam")

model1 = joblib.load("model.pkl")

print(model1.predict(["won a prize"]))



