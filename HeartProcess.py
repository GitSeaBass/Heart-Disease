import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, KFold, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score



def main():
    directory = './heart+disease/data'
    knn_accuracies = []
    sv_accuracies = []
    dt_accuracies = []
    nb_accuracies = []
    rf_accuracies = []
    lr_accuracies = []
    for filename in os.listdir(directory):
        print('*' * 60)
        filepath = os.path.join(directory, filename)
        if os.path.isfile(filepath):  # Check if it's a file
            print('Results for:\t\t\t' +os.path.basename(filepath))
            print('*' * 60)
            knn_accuracies.append(KNN(file_path=filepath))
            print('-' * 60)
            sv_accuracies.append(SupportVector(file_path=filepath))
            print('-' * 60)
            dt_accuracies.append(DecisionTree(file_path=filepath))
            print('-' * 60)
            nb_accuracies.append(NaiveBayes(file_path=filepath))
            print('-' * 60)
            rf_accuracies.append(RandomForest(file_path=filepath))
            print('-' * 60)
            lr_accuracies.append(LogRegression(file_path=filepath))
    print('*' * 60)
    # Change the return value of the accuracy in the methods to account for 
    # weight of the accuracy based on the number of samples in the dataset
    '''mean_accuracy = np.mean(knn_accuracies)
    print(f'KNN Average Accuracy:\t\t\t{mean_accuracy*100:.2f}%')
    mean_accuracy = np.mean(sv_accuracies)
    print(f'SVC Average Accuracy:\t\t\t{mean_accuracy*100:.2f}%')
    mean_accuracy = np.mean(dt_accuracies)
    print(f'Decision Tree Average Accuracy:\t\t{mean_accuracy*100:.2f}%')
    mean_accuracy = np.mean(nb_accuracies)
    print(f'Naive Bayes Average Accuracy:\t\t{mean_accuracy*100:.2f}%')
    mean_accuracy = np.mean(rf_accuracies)
    print(f'Random Forest Average Accuracy:\t\t{mean_accuracy*100:.2f}%')
    mean_accuracy = np.mean(lr_accuracies)
    print(f'Logistic Regression Average Accuracy:\t{mean_accuracy*100:.2f}%')'''
    

def KNN(file_path):
    classifier = KNNClassifier(n_neighbors=4)
    return classifier.classify_file(file_path)
    
def SupportVector(file_path):
    classifier = SupportVectorClassifier(kernel='rbf', C=1, gamma=.1)  # Default SVM with RBF kernel
    return classifier.classify_file(file_path)
    
def DecisionTree(file_path):
    classifier = DecisionTreeModel(max_depth=5)  # You can adjust the max_depth
    return classifier.classify_file(file_path)
    
def NaiveBayes(file_path):
    classifier = NaiveBayesClassifier()
    return classifier.classify_file(file_path)
    
def RandomForest(file_path):
    classifier = RandomForestModel(n_estimators=45, max_depth=5)  # Default settings
    return classifier.classify_file(file_path)
    
def LogRegression(file_path):
    classifier = LogisticRegressionModel()
    return classifier.classify_file(file_path)
    
class KNNClassifier:
    def __init__(self, n_neighbors=3):
        self.n_neighbors = n_neighbors
        self.classifier = KNeighborsClassifier(n_neighbors=self.n_neighbors)
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='mean')  # Default to mean imputation

    def load_and_prepare_data(self, file_path):
        # Load data
        data = pd.read_csv(file_path, header=None, na_values='?')
        # Impute missing values
        data = pd.DataFrame(self.imputer.fit_transform(data), columns=data.columns)
        # Separate features and target
        X = data.iloc[:, :-1]
        y = data.iloc[:, -1]
        
        kf = StratifiedKFold(n_splits=5)
        scores = cross_val_score(self.classifier, X, y, cv=kf, scoring='accuracy')
        print(f"\nKNN Mean accuracy = {scores.mean():.2f} (+/- {scores.std() * 2:.2f})")
        
        return X, y

    def train_classifier(self, X, y):
        # Split data into training and testing
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        # Scale features
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)
        # Train classifier
        self.classifier.fit(X_train, y_train)
        # Predict on test data
        y_pred = self.classifier.predict(X_test)
        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        # Assume y_pred and y_test are your predicted and actual labels respectively
        conf_matrix = confusion_matrix(y_test, y_pred)
        print("\nKNN Confusion Matrix:\n\n", conf_matrix)

        class_report = classification_report(y_test, y_pred, zero_division=0)
        print("\nKNN Classification Report:\n\n", class_report)
        
        return accuracy

    def predict(self, X):
        # Scale features
        X_scaled = self.scaler.transform(X)
        return self.classifier.predict(X_scaled)

    def classify_file(self, file_path):
        X, y = self.load_and_prepare_data(file_path)
        accuracy = self.train_classifier(X, y)
        print(f"KNN Accuracy:\t\t\t{accuracy * 100:.2f}%")
        return accuracy
    
class SupportVectorClassifier:
    def __init__(self, kernel='rbf', C=1.0, gamma=1.0):
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.classifier = SVC(kernel=self.kernel, C=self.C, gamma=self.gamma)
        self.scaler = StandardScaler()

    def load_and_prepare_data(self, file_path):
        # Load data assuming no headers and '?' as NA indicators
        data = pd.read_csv(file_path, header=None, na_values='?')
        # Impute missing values and separate features and target
        data = data.fillna(data.median())
        X = data.iloc[:, :-1]
        y = data.iloc[:, -1]
        
        kf = StratifiedKFold(n_splits=5)
        scores = cross_val_score(self.classifier, X, y, cv=kf, scoring='accuracy')
        print(f"\nSupport Vector Mean accuracy = {scores.mean():.2f} (+/- {scores.std() * 2:.2f})")
        
        return X, y

    def train_classifier(self, X, y):
        # Split data, scale features, train classifier
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)
        self.classifier.fit(X_train, y_train)
        y_pred = self.classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        conf_matrix = confusion_matrix(y_test, y_pred)
        print("\nSVC Confusion Matrix:\n\n", conf_matrix)

        class_report = classification_report(y_test, y_pred, zero_division=0)
        print("\nSVC Classification Report:\n\n", class_report)
        return accuracy

    def classify_file(self, file_path):
        X, y = self.load_and_prepare_data(file_path)
        accuracy = self.train_classifier(X, y)
        print(f"SVC Accuracy:\t\t\t{accuracy * 100:.2f}%")
        return accuracy
        
class DecisionTreeModel:
    def __init__(self, max_depth=None, random_state=42):
        self.max_depth = max_depth
        self.random_state = random_state
        self.classifier = DecisionTreeClassifier(max_depth=self.max_depth, random_state=self.random_state)
        self.scaler = StandardScaler()

    def load_and_prepare_data(self, file_path):
        # Load data assuming no headers and '?' as NA indicators
        data = pd.read_csv(file_path, header=None, na_values='?')
        # Impute missing values and separate features and target
        data.fillna(data.median(), inplace=True)
        X = data.iloc[:, :-1]
        y = data.iloc[:, -1]
        
        kf = StratifiedKFold(n_splits=5)
        scores = cross_val_score(self.classifier, X, y, cv=kf, scoring='accuracy')
        print(f"\nDecision Tree Mean accuracy = {scores.mean():.2f} (+/- {scores.std() * 2:.2f})")
        
        return X, y

    def train_classifier(self, X, y):
        # Split data, scale features, train classifier
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)
        self.classifier.fit(X_train, y_train)
        y_pred = self.classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Assume y_pred and y_test are your predicted and actual labels respectively
        conf_matrix = confusion_matrix(y_test, y_pred)
        print("\nDecision Tree Confusion Matrix:\n\n", conf_matrix)

        class_report = classification_report(y_test, y_pred, zero_division=0)
        print("\nDecision Tree Classification Report:\n\n", class_report)
        
        return accuracy

    def classify_file(self, file_path):
        X, y = self.load_and_prepare_data(file_path)
        accuracy = self.train_classifier(X, y)
        print(f"Decision Tree Accuracy:\t\t{accuracy * 100:.2f}%")
        return accuracy
        
class NaiveBayesClassifier:
    def __init__(self):
        self.classifier = GaussianNB()
        self.scaler = StandardScaler()

    def load_and_prepare_data(self, file_path):
        # Load data assuming no headers and '?' as NA indicators
        data = pd.read_csv(file_path, header=None, na_values='?')
        # Impute missing values and separate features and target
        data.fillna(data.median(), inplace=True)
        X = data.iloc[:, :-1]
        y = data.iloc[:, -1]
        
        kf = StratifiedKFold(n_splits=5)
        scores = cross_val_score(self.classifier, X, y, cv=kf, scoring='accuracy')
        print(f"\nNaive Bayes Mean accuracy = {scores.mean():.2f} (+/- {scores.std() * 2:.2f})")    
        
        return X, y

    def train_classifier(self, X, y):
        # Split data, scale features, train classifier
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)
        self.classifier.fit(X_train, y_train)
        y_pred = self.classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Assume y_pred and y_test are your predicted and actual labels respectively
        conf_matrix = confusion_matrix(y_test, y_pred)
        print("\nNaive Bayes Confusion Matrix:\n\n", conf_matrix)

        class_report = classification_report(y_test, y_pred, zero_division=0)
        print("\nNaive Bayes Classification Report:\n\n", class_report)
        
        return accuracy

    def classify_file(self, file_path):
        X, y = self.load_and_prepare_data(file_path)
        accuracy = self.train_classifier(X, y)
        print(f"Naive Bayes Accuracy:\t\t{accuracy * 100:.2f}%")
        return accuracy

class RandomForestModel:
    def __init__(self, n_estimators=100, max_depth=None, random_state=42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.classifier = RandomForestClassifier(n_estimators=self.n_estimators, max_depth=self.max_depth, random_state=self.random_state)

    def load_and_prepare_data(self, file_path):
        # Load data assuming no headers and '?' as NA indicators
        data = pd.read_csv(file_path, header=None, na_values='?')
        # Impute missing values and separate features and target
        data.fillna(data.median(), inplace=True)
        X = data.iloc[:, :-1]
        y = data.iloc[:, -1]
        
        kf = StratifiedKFold(n_splits=5)
        scores = cross_val_score(self.classifier, X, y, cv=kf, scoring='accuracy')
        print(f"\nRandom Forest Mean accuracy = {scores.mean():.2f} (+/- {scores.std() * 2:.2f})")
        
        return X, y

    def train_classifier(self, X, y):
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        # Train classifier
        self.classifier.fit(X_train, y_train)
        # Predict on test data
        y_pred = self.classifier.predict(X_test)
        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        
        # Assume y_pred and y_test are your predicted and actual labels respectively
        conf_matrix = confusion_matrix(y_test, y_pred)
        print("\nRandom Forest Confusion Matrix:\n\n", conf_matrix)

        class_report = classification_report(y_test, y_pred, zero_division=0)
        print("\nRandom Forest Classification Report:\n\n", class_report)
        return accuracy

    def classify_file(self, file_path):
        X, y = self.load_and_prepare_data(file_path)
        accuracy = self.train_classifier(X, y)
        print(f"Random Forest Accuracy:\t\t{accuracy * 100:.2f}%")
        return accuracy
        
class LogisticRegressionModel:
    def __init__(self, random_state=42, max_iter=100000):
        self.random_state = random_state
        self.max_iter = max_iter
        self.classifier = LogisticRegression(random_state=self.random_state, max_iter=self.max_iter)
        self.scaler = StandardScaler()

    def load_and_prepare_data(self, file_path):
        # Load data assuming no headers and '?' as NA indicators
        data = pd.read_csv(file_path, header=None, na_values='?')
        # Impute missing values and separate features and target
        data.fillna(data.median(), inplace=True)
        X = data.iloc[:, :-1]
        y = data.iloc[:, -1]
        
        return X, y

    def train_classifier(self, X, y):
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        # Scale features
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)
        # Train classifier
        self.classifier.fit(X_train, y_train)
        # Predict on test data
        y_pred = self.classifier.predict(X_test)
        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        
        # Assume y_pred and y_test are your predicted and actual labels respectively
        conf_matrix = confusion_matrix(y_test, y_pred)
        print("\nLogistic Regression Confusion Matrix:\n\n", conf_matrix)

        class_report = classification_report(y_test, y_pred, zero_division=0)
        print("\nLogistic Regression Classification Report:\n\n", class_report)
        
        return accuracy

    def classify_file(self, file_path):
        X, y = self.load_and_prepare_data(file_path)
        accuracy = self.train_classifier(X, y)
        print(f"Logistic Regression Accuracy:\t{accuracy * 100:.2f}%")
        return accuracy

if __name__ == '__main__':
    main()