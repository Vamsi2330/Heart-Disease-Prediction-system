# Importing Essential Libraries

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import seaborn as sns

# loading the csv heart_disease to pandas heart_diseaseframe
heart_disease = pd.read_csv(
    'C:\\Users\\91994\\Documents\\GitHub\\multiple-dir\\samples\\Hackthon\\Heart-Disease-Prediction-using-Machine-Learning\\heart.csv')  # noqa: E501


def heatmap_corr():
    sns.heatmap(heart_disease.corr(), annot=True, linewidths=2)
    plt.show()


def correlation_with_target():
    corr_matrix = heart_disease.corr()
    sns.scatterplot(corr_matrix['target'].sort_values(ascending=False))
    plt.show()


def age_analysis():
    plt.figure(figsize=(25, 12))
    sns.set_context('notebook', font_scale=1.5)
    sns.barplot(x=heart_disease.age.value_counts()[:10].index, y=heart_disease.age.value_counts()[:10].values)
    plt.show()


def age_range():
    minAge = min(heart_disease.age)
    maxAge = max(heart_disease.age)
    meanAge = heart_disease.age.mean()
    print(minAge)
    print(maxAge)
    print(meanAge)


def age_divide():
    Young = heart_disease[(heart_disease.age >= 29) & (heart_disease.age < 40)]
    Middle = heart_disease[(heart_disease.age >= 40) & (heart_disease.age < 55)]
    Elder = heart_disease[(heart_disease.age > 55)]

    plt.figure(figsize=(23, 10))
    sns.set_context('notebook', font_scale=1.5)
    sns.barplot(x=['young ages', 'middle ages', 'elderly ages'], y=[len(Young), len(Middle), len(Elder)],
                heart_disease=heart_disease)
    plt.show()


def heart_disease_pie_chart():
    Young = heart_disease[(heart_disease.age >= 29) & (heart_disease.age < 40)]
    Middle = heart_disease[(heart_disease.age >= 40) & (heart_disease.age < 55)]
    Elder = heart_disease[(heart_disease.age > 55)]
    colors = ['blue', 'green', 'yellow']
    explode = [0, 0, 0.1]
    plt.figure(figsize=(10, 10))
    sns.set_context('notebook', font_scale=1.2)
    plt.pie([len(Young), len(Middle), len(Elder)], labels=['young ages', 'middle ages', 'elderly ages'],
            explode=explode, colors=colors, autopct='%1.1f%%')
    plt.show()


def sex_slop():
    plt.figure(figsize=(18, 9))
    sns.set_context('notebook', font_scale=1.5)
    sns.countplot(x=heart_disease['sex'], hue=heart_disease["slope"], heart_disease=heart_disease)
    plt.show()


def cp_analysis():
    plt.figure(figsize=(18, 9))
    sns.set_context('notebook', font_scale=1.5)
    sns.countplot(x=heart_disease['cp'])
    plt.show()


def cp_analysis_vs_target():
    x = heart_disease['cp']
    sns.countplot(x, y=heart_disease['target'])
    plt.show()


def thal_analysis():
    plt.figure(figsize=(18, 9))
    sns.set_context('notebook', font_scale=1.5)
    sns.countplot(x=heart_disease['thal'])
    plt.show()


def target():
    plt.figure(figsize=(18, 9))
    sns.set_context('notebook', font_scale=1.5)
    sns.countplot(x=heart_disease['target'])
    plt.show()


def info():
    print(heart_disease.info())


def has_null():
    print(heart_disease.isnull().sum())


def description_of_heart_disease():
    print(heart_disease.describe())


# checking the distribution of Target Variable

def no_of_heart_disease():
    print(heart_disease['target'].value_counts())
    # 0 -->Healthy Heart
    # 1 -->Heart Disease


# Splitting the Features and Target

X = heart_disease.drop(columns='target', axis=1)
Y = heart_disease['target']
# splitting the heart_disease into training heart_disease and testing heart_disease
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)


# ---------------------------------------> Logistic Regressor <---------------------------------------------------------
class Logistic_regressor:
    # --------------> Model Evaluation <----------------
    def accuracy(self):
        # Accuracy Score
        model = LogisticRegression()
        # Training the LogisticRegression Model with Training heart_disease
        model.fit(X_train, Y_train)
        # accuracy on test heart_disease
        X_test_prediction = model.predict(X_test)
        test_heart_disease_accuracy = accuracy_score(X_test_prediction, Y_test)
        print("Accuracy on Test data is : ", test_heart_disease_accuracy)

        # accuracy on training heart_disease
        X_train_prediction = model.predict(X_train)
        training_heart_disease_accuracy = accuracy_score(X_train_prediction, Y_train)
        print('Accuracy on Train data using Logistic Regressions  : ', training_heart_disease_accuracy)

    # --------> Confusion Matrix <---------
    def Confusion_matrix(self):
        model = LogisticRegression()
        model.fit(X_train, Y_train)
        y_prediction_test = model.predict(X_test)
        print("Confusion Matrix on Test data using Logistic Regression is ",
              confusion_matrix(y_prediction_test, Y_test))
        y_prediction_train = model.predict(X_train)
        print("Confusion Matrix on Train data using Logistic Regression is ",
              confusion_matrix(y_prediction_train, Y_train))
        print("Correct prediction are ",
              confusion_matrix(y_prediction_train, Y_train)[0][0] + confusion_matrix(y_prediction_train, Y_train)[1][1])
        print("Incorrect predction are ",
              confusion_matrix(y_prediction_train, Y_train)[0][1] + confusion_matrix(y_prediction_train, Y_train)[1][0])

    # ---->Building the Predictive System <-----
    def Logistic_regression_predictive_system(self):
        model = LogisticRegression()
        model.fit(X_train, Y_train)
        # Change the input heart_disease to numpy array
        input_heart_disease_as_numpy = np.asarray([37, 1, 2, 130, 250, 0, 1, 187, 0, 3.5, 0, 0, 2])
        # Reshaping the numpy as we are predicting for only on instance
        input_heart_disease_reshaped = input_heart_disease_as_numpy.reshape(1, -1)
        prediction = model.predict(input_heart_disease_reshaped)

        if prediction == 0:
            print("The Person does not have Heart Disease ")
        else:
            print("The Person Has Heart Disease")


# ----------------------------------------> KNN Algorithm <-------------------------------------------------------------

class knn:
    # --------------> Model Evaluation <----------------
    def accuracy(self):
        Knn = KNeighborsClassifier(10)
        Knn.fit(X_train, Y_train)
        X_test_pared_knn = Knn.predict(X_test)
        print("Accuracy of test heart_disease in knn is : ", accuracy_score(X_test_pared_knn, Y_test))
        X_train_pared_knn = Knn.predict(X_train)
        print("Accuracy of train heart_disease in Knn is : ", accuracy_score(X_train_pared_knn, Y_train))

    # --------> Confusion Matrix <---------
    def Confusion_matrix(self):
        Knn = KNeighborsClassifier(10)
        Knn.fit(X_train, Y_train)
        X_test_pared_knn = Knn.predict(X_test)
        print("Confusion Matrix on Test Data using knn is : ", confusion_matrix(X_test_pared_knn, Y_test))
        X_train_pared_knn = Knn.predict(X_train)
        print("Confusion Matrix on Train Data using knn  is : ", confusion_matrix(X_train_pared_knn, Y_train))
        print("Correct prediction are ",
              confusion_matrix(X_train_pared_knn, Y_train)[0][0] + confusion_matrix(X_train_pared_knn, Y_train)[1][1])
        print("Incorrect predction are ",
              confusion_matrix(X_train_pared_knn, Y_train)[0][1] + confusion_matrix(X_train_pared_knn, Y_train)[1][0])

    # ---->Building the Predictive System <-------------
    def knn_predictive_system(self):
        Knn = KNeighborsClassifier(2)
        Knn.fit(X_train, Y_train)
        # Change the input heart_disease to numpy array
        input_heart_disease_as_numpy = np.asarray([37, 1, 2, 130, 250, 0, 1, 187, 0, 3.5, 0, 0, 2])
        # Reshaping the numpy as we are predicting for only on instance
        input_heart_disease_reshaped = input_heart_disease_as_numpy.reshape(1, -1)
        prediction = Knn.predict(input_heart_disease_reshaped)

        if prediction == 0:
            print("The Person does not have Heart Disease ")
        else:
            print("The Person Has Heart Disease")


# ----------------------------------------> Random Forest <----------------------------------------------------------
class random:
    # --------------> Model Evaluation <----------------
    def acuraccy(self):
        x = heart_disease.drop('target', axis=1)
        y = heart_disease['target']
        x_train, x_test, y_trian, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
        classifier = RandomForestClassifier(n_estimators=10, criterion='entropy')
        classifier.fit(x_train, y_trian)
        # Accuracy of Test Data
        y_predict = classifier.predict(x_test)
        print("Accuracy of Test data using Random Forest is ", accuracy_score(y_test, y_predict))
        # Accuracy of Train Data
        y_predict_train = classifier.predict(x_train)
        print("Accuracy of Train data using Random Forest is ", accuracy_score(y_trian, y_predict_train))

    # --------> Confusion Matrix <---------
    def Confusion_matrix(self):
        x = heart_disease.drop('target', axis=1)
        y = heart_disease['target']
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
        classifier = RandomForestClassifier(n_estimators=10, criterion='entropy')
        classifier.fit(x_train, y_train)
        # Confusion Matrix
        y_predict_train = classifier.predict(x_train)
        cm = confusion_matrix(y_train, y_predict_train)
        print("Confusion Matrix for train data in RandomForest Model is  ", cm)
        # Prediction Analysis
        print("Correct Prediction are ", cm[0][0] + cm[1][1])
        print("Incorrect Prediction are ", cm[0][1] + cm[1][0])

    # ------->Building the Predictive System <---------
    def predictive_system(self):
        x = heart_disease.drop('target', axis=1)
        y = heart_disease['target']
        x_train, x_test, y_trian, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

        classifier = RandomForestClassifier(n_estimators=10, criterion='entropy')
        classifier.fit(x_train, y_trian)
        input_data = [37, 1, 2, 130, 250, 0, 1, 187, 0, 3.5, 0, 0, 2]
        input_data_as_array = np.asarray(input_data)
        input_data_reshaped = input_data_as_array.reshape(1, -1)
        prediction = classifier.predict(input_data_reshaped)
        if prediction == 0:
            print("The persons heart is Healthy.")
        else:
            print("The person has heart disease.")


# Logistic Regression
#log = Logistic_regressor()
# log.accuracy()
#log.Confusion_matrix()
# log.Logistic_regression_predictive_system()


# KNN Classifier
#k = knn()
# k.knn_predictive_system(self)
# k.accuracy()
#k.Confusion_matrix()

# RandomForest
ra = random()
# ra.acuraccy()
ra.Confusion_matrix()
ra.predictive_system()
