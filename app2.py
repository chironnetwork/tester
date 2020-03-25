import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import pickle
from flask import Flask , render_template ,request, jsonify
app = Flask(__name__)

'''
Train Model before starting Up the Server
'''
def data_split(data, ratio):
    np.random.seed(42)
    shuffled = np.random.permutation(len(data))
    test_set_size = int(len(data)*ratio)
    test_indices = shuffled[:test_set_size]
    train_indices = shuffled[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]

df = pd.read_csv('data1.csv')
train, test = data_split(df, 0.3)
X_train = train[['fever','age','tiredness','cough','feverDays']].to_numpy()
X_test  = test[['fever','age','tiredness','cough','feverDays']].to_numpy()

Y_train = train[['infectionProb']].to_numpy().reshape(len(train),)
Y_test = test[['infectionProb']].to_numpy().reshape(len(test),)

clf = LogisticRegression()
clf.fit(X_train, Y_train)

@app.route('/', methods = ["GET","POST"])
def check_result():

    inputFeatures = [
        int(request.args.get('fever')),
        int(request.args.get('age')),
        int(request.args.get('tiredness')),
        int(request.args.get('cough')),
        int(request.args.get('feverDays')),
    ]
    # print(inputFeatures)
    infProb = clf.predict_proba([inputFeatures])[0][1]
    return jsonify({'result':infProb*100})


if __name__ == '__main__':

    print("Server Up")
    app.run()
