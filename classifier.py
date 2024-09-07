import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# using pandas to open the csv file and specifying the delimiter
spotify_features = pd.read_csv("SpotifyFeatures.csv", delimiter=",")
# using the .shape function to return both the rows and columns number
samples_features = spotify_features.shape

# indexing to get the wanted genres along with their features
classical = spotify_features[spotify_features["genre"] == "Classical"]
pop = spotify_features[spotify_features["genre"] == "Pop"]

print(f"Spotify Features CSV \nTotal samples: {samples_features[0]} \nTotal features: {samples_features[1]}")

# Making a single DataFrame with classical and pop
pop_classical = pd.concat([classical, pop])
# Using the np.where function to add a new row with the column values being either 1 or 0 depending on what the genre is
pop_classical["classified"] = np.where(pop_classical["genre"] == "Classical", 1, 0)
# Taking out the rows and columns we require
pop_classical = pop_classical[["genre", "loudness", "liveness", "classified"]]
# Initalizing the feature matrix and classify vecotr using np.array
feature_matrix = np.array(pop_classical[["loudness", "liveness"]])
classify_vector = np.array(pop_classical["classified"])

# using sklearn to make the train and test sets
x_train, x_test, y_train, y_test = train_test_split(feature_matrix, classify_vector, test_size=0.2)

# Sigmoid function to transform the predictions to a value between 0 and 1
def sigmoid(x):
    return 1/(1+np.exp(-x))

# Initalizing self.error and self.epoch_list to use in the CEL method
class LogisticRegression():
    def __init__(self, y_set, lr=0.035, epochs=1000):
        self.lr = lr
        self.epochs = epochs
        self.y_set = y_set
        self.weights = None
        self.bias = None
        self.epoch_error = np.zeros(shape=(self.epochs, 2))

        # Figuring out the linear predictions then passing it into the sigmoid function
        # using the updated weights and bias values to improve the gradient
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for epoch in range(self.epochs):
            lin_pred = (X @ self.weights) + self.bias
            predictions = sigmoid(lin_pred)
            dw = (1/n_samples) * (X.T @ (predictions - y))
            db = (1/n_samples) * np.sum(predictions - y)
            self.weights = self.weights - self.lr * dw
            self.bias = self.bias - self.lr * db
            error = self.cross_entropy_loss(predictions, n_samples)
            self.epoch_error[epoch] = [epoch, error]    
    # after fitting it we use this method to actually predict the values
    def predict(self, X):
        lin_pred = (X @ self.weights) + self.bias
        y_pred = sigmoid(lin_pred)
        class_pred =  [0 if y <= 0.5 else 1 for y in y_pred]
        return class_pred
    
    # using the binary cross entropy loss function, since we have a Yes or No scenario
    def cross_entropy_loss(self, y_pred, n_samples):
        loss =  -(1/n_samples) * np.sum(self.y_set * np.log(y_pred) + (1 - self.y_set) * np.log(1 - y_pred))
        return loss
    
    
 # Decided to make a function for both train and test, seemed to structure for me
def clf_train():
    classifier = LogisticRegression(y_train)
    classifier.fit(x_train, y_train)
    y_train_pred = classifier.predict(x_train)
    epoch_error = classifier.epoch_error
    return y_train_pred, epoch_error

def clf_test():
    classifier = LogisticRegression(y_test)
    classifier.fit(x_test, y_test)
    y_test_pred = classifier.predict(x_test)
    epoch_error = classifier.epoch_error
    return y_test_pred, epoch_error

 # simple accuracy function
def accuracy(y_pred, y_true, state):
    acc = np.sum(y_true == y_pred) / len(y_true)
    acc = np.round(acc, decimals=4) * 100
    if state == "train":
        print(f"Accuracy on training set: {acc}%")
    else:
        print(f"Accuracy on test set: {acc}%")

 # plotting error as a function of epochs
def plot_err(epoch_error):
    epochs = epoch_error[:, 0]
    errors = epoch_error[:, 1]
    plt.plot(epochs, errors)
    plt.xlabel("Epochs")
    plt.ylabel("Error")
    plt.title("Logistic Regression")
    plt.show()



 # making the confusion matrix
def confusion_matrix(y_pred, y_test):
    TP = TN = FP = FN = 0
    for true, pred in zip(y_test, y_pred):
        if true == pred:
            if true == 1:
                TP += 1
            else:
                TN += 1
        else:
            if pred == 0:
                FN += 1
            else:
                FP += 1
    matrix = np.array([[TP, FP], [FN, TN]])
    return matrix
 # all code tests
if __name__== "__main__":
    clf_train_pred, clf_train_error_epoch = clf_train()
    clf_test_pred, clf_test_error_ = clf_test()

    ConfusionMatrix = confusion_matrix(clf_test_pred, y_test)

    train_acc = accuracy(clf_train_pred, y_train, "train")
    test_acc = accuracy(clf_test_pred, y_test, "test")

    print(ConfusionMatrix)
    plot_err(clf_train_error_epoch)
    conf_acc = (ConfusionMatrix[0][0] + ConfusionMatrix[1][1]) / (np.sum(ConfusionMatrix))

                








