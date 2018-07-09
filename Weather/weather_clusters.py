import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


def get_pca(X):
    """
    Transform data to 2D points for plotting. Should return an array with shape (n, 2).
    """
    flatten_model = make_pipeline(
        # TODO
        MinMaxScaler(),
        PCA(2)
    )
    X2 = flatten_model.fit_transform(X)
    assert X2.shape == (X.shape[0], 2)
    return X2


def get_clusters(X):
    """
    Find clusters of the weather data.
    """
    model = make_pipeline(
        # TODO
        MinMaxScaler(),
        KMeans(n_clusters=10)
    )
    model.fit(X)
    return model.predict(X)


def main():
    data = pd.read_csv(sys.argv[1])

    y = data['city']
    X = data.drop(['city', 'year'], axis=1)  # excluding these 2 columns
    
    X2 = get_pca(X)
    clusters = get_clusters(X)
    plt.scatter(X2[:, 0], X2[:, 1], c=clusters, cmap='Set1', edgecolor='k', s=20)
    plt.savefig('clusters.png')

    df = pd.DataFrame({
        'cluster': clusters,
        'city': y,
    })
    counts = pd.crosstab(df['city'], df['cluster'])
    print(counts)

    """
    cluster          0   1   2   3   4   5   6   7   8   9
    city
    Anchorage        0   0   0   0   0   0   0   0   0  56
    Atlanta          0   0   0   0   0   0   0   0  47   0
    Atlantic City    0   0   0   0   0   0  40   0   5   0
    Calgary          0   1   0  51   0   0   0   0   0   0
    Chicago          0   1   0   0   0   0  51   0   0   0
    Denver           0   0   0   0   0   0   8   0   1   0
    Edmonton         0   0   0  49   0   1   0   0   0   1
    Gander           0   1   0   0  49   1   0   0   0   0
    Halifax          0  41   0   0   9   0   0   0   0   0
    London           0  41   0   0   0   0   1   0   0   0
    Los Angeles      0   0   0   0   0   0   0  39   0   0
    Miami           43   0   0   0   0   0   0   0   0   0
    Montreal         0  19   0   0   0   9   0   0   0   0
    New Orleans     42   0   0   0   0   0   0   0   3   0
    Ottawa           0  41   0   0   0  10   0   0   0   0
    Portland         0   0  27   0   0   0  10   0   1   0
    Québec           0   0   0   0   0  35   0   0   0   0
    Raleigh Durham   0   0   0   0   0   0   0   0  56   0
    Regina           0   0   0  42   0   0   0   0   0   0
    San Francisco    0   0   0   0   0   0   0  38   0   0
    Saskatoon        0   0   0  45   0   0   0   0   0   0
    Seattle          0   0  43   0   0   0   0   0   0   0
    Toronto          0  39   0   0   0   0  14   0   0   0
    Vancouver        0   0  53   0   0   0   0   0   0   0
    Victoria         0   0  52   0   0   0   0   0   0   0
    Winnipeg         0   0   0  43   0   0   0   0   0   0
    """

if __name__ == '__main__':
    main()