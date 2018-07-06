import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC



def main():
    data = pd.read_csv(sys.argv[1])
    y = data['city']
    X = data.drop(['city', 'year'])  # excluding these 2 columns
    X_train, X_test, y_train, y_test = train_test_split(X, y)
