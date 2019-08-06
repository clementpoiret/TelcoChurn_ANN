# Artificial Neural Network

## Importing the libraries
import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

## Importing the dataset
dataset = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

## Column's ID related to binomial & categorical variable; corrected by taking in consideration that onehotencoder is adding columns
BINOMIAL_COLS = [0, 2, 3, 5, 15]
CATEGORICAL_COLS = [6, 8, 10, 12, 14, 16, 18, 20, 22, 25]


def prep_pipeline(X, binomial, categorical):
    """Preprocessing pipeline's docstring.

    The following function is returning a corrected matrix, converting
    binomial and categorical variables to dummy variables.

    Args:
        X (matrix): Original dataset.
        binomial (array): indices of columns with only two unique 
            values (Yes/No; True/False; 0/1; etc).
        categorical (array): same as binomial but for columns with
            more than one category.

    Returns:
        X: The corrected matrix.

    Todo: 
        * Automatically correct indices.

    """

    for col in binomial:
        labelencoder_X = LabelEncoder()
        X[:, col] = labelencoder_X.fit_transform(X[:, col])

    for col in categorical:
        ct = ColumnTransformer(
            [("col", OneHotEncoder(categories="auto"), [col])],
            remainder="passthrough")
        X = ct.fit_transform(X)
        X = X[:, 1:]

    return X


# Correcting X through the pipeline; Removing empty strings by the mean of the column
X = prep_pipeline(X, BINOMIAL_COLS, CATEGORICAL_COLS)
X[:, -1] = pd.to_numeric(X[:, -1], errors='coerce')

imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
imputer = imputer.fit(X[:, -1].reshape(-1, 1))
X[:, -1] = imputer.transform(X[:, -1].reshape(-1, 1)).reshape(1, -1)[0]

# Saving corrected X
pd.DataFrame(X).to_csv("X.csv", index=False)

# Encoding Churn from Yes/No to 0/1
labelencoder_Y = LabelEncoder()
y = labelencoder_Y.fit_transform(y)

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# Now, let's make the ANN
def build_classifier(optimizer="rmsprop", nb_layers=1, dropout=False, rate=.1):
    """Classifier Builder's docstring.

    The following function is returning a classifier through Keras with a
    TensorFlow backend. The following classifier is usable for future 
    Hyperparameters tunning.

    Args:
        optimizer (str): Optimizer for backpropagation (such as "adam" or
            "rmsprop").
        nb_layers (int): The number of hidden layers.
        dropout (bool): Boolean to choose if hidden layers should be complemented
            with a dropout to randomly disable neurons and reduce their
            interdependence to counter overfitting.
        rate (float): Rate of the dropout, from 0 (no neuron disabled) to 1 (all 
            neurons disabled).

    Returns:
        classifier: A built Sequential classifier with given parameters.
    """

    classifier = Sequential()

    classifier.add(
        Dense(activation="relu",
              input_dim=30,
              units=15,
              kernel_initializer="random_uniform"))

    i = 1
    while i <= nb_layers:
        classifier.add(
            Dense(activation="relu",
                  units=15,
                  kernel_initializer="random_uniform"))
        if dropout:
            classifier.add(Dropout(rate=rate))
        i += 1

    classifier.add(
        Dense(activation="sigmoid",
              units=1,
              kernel_initializer="random_uniform"))

    classifier.compile(optimizer=optimizer,
                       loss="binary_crossentropy",
                       metrics=["accuracy"])

    return classifier


classifier = KerasClassifier(build_fn=build_classifier)

# Setting parameters to pass into a GridSearchCV for Hyperparameters computation
parameters = {
    "batch_size": [10, 32],
    "epochs": [100, 500],
    "optimizer": ["adam", "rmsprop"],
    "nb_layers": [1, 2],
    "dropout": [False, True]
}

grid_search = GridSearchCV(estimator=classifier,
                           param_grid=parameters,
                           scoring="accuracy",
                           cv=10,
                           n_jobs=-1)

grid_search = grid_search.fit(X_train, y_train)

# Getting best parameters from the GridSearchCV
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_

# Building and Fitting the ANN with bests parameters
classifier = build_classifier(optimizer=best_parameters.get("optimizer"),
                              nb_layers=best_parameters.get("nb_layers"),
                              dropout=best_parameters.get("dropout"))
classifier.fit(X_train,
               y_train,
               batch_size=best_parameters.get("batch_size"),
               epochs=best_parameters.get("epochs"))

# Saving the model for reuse
classifier.save("churn.h5")

# Making the predictions and evaluating the model
y_pred = classifier.predict(X_test)
y_pred = (y_pred > .5)

cm = confusion_matrix(y_test, y_pred)

accuracy = (cm[0, 0] + cm[1, 1]) / X_test.shape[0]
