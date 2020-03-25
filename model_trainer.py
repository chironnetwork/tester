import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import pickle

def data_split(data, ratio):
    np.random.seed(42)
    shuffled = np.random.permutation(len(data))
    test_set_size = int(len(data)*ratio)
    test_indices = shuffled[:test_set_size]
    train_indices = shuffled[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]


if __name__ == '__main__':
    # Read the file
    df = pd.read_csv('data1.csv')
    train, test = data_split(df, 0.3)
    X_train = train[['fever','age','tiredness','cough','feverDays']].to_numpy()
    X_test  = test[['fever','age','tiredness','cough','feverDays']].to_numpy()

    Y_train = train[['infectionProb']].to_numpy().reshape(len(train),)
    Y_test = test[['infectionProb']].to_numpy().reshape(len(test),)

    clf = LogisticRegression()
    clf.fit(X_train, Y_train)

    file = open('model2.pkl','wb')


    # dumb informaiton to that file
    pickle.dump(clf,file)

    file.close()

