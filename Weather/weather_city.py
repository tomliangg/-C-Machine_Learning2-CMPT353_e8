import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler


def main():
    data = pd.read_csv(sys.argv[1])
    scaler = StandardScaler()
    
    y = data['city']
    X = data.drop(['city', 'year'], axis=1)  # excluding these 2 columns
    scaler.fit(X)
    data = scaler.transform(X) # normalize the data for X

    X_train, X_test, y_train, y_test = train_test_split(X, y)

    # Thru my epxeriments, SVC has decent scores
    svc_weather_model = SVC(kernel='linear', C=1)

    svc_weather_model.fit(X_train, y_train)
    print('SVC score: ', svc_weather_model.score(X_test, y_test))
    
    data_unlabelled = pd.read_csv(sys.argv[2])
    X_unlabelled = data_unlabelled.drop(['city', 'year'], axis=1)
    scaler.fit(X_unlabelled)
    data = scaler.transform(X_unlabelled) # normalize the data
    predictions = svc_weather_model.predict(X_unlabelled)
    pd.Series(predictions).to_csv(sys.argv[3], index=False)

if __name__ == '__main__':
    main()
