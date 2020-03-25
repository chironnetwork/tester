from flask import Flask , render_template ,request, jsonify
app = Flask(__name__)

import pickle
file = open('model.pkl','rb')
clf = pickle.load(file)
file.close()

@app.route('/', methods = ["GET","POST"])
def hello_world():
    print(request.args)
    if request.method == "POST":
        myDict = request.form
        fever = int(myDict['fever'])
        age = int(myDict['age'])
        tiredness = int(myDict['tiredness'])
        cough = int(myDict['cough'])
        feverDays = int(myDict['feverDays'])

        inputFeatures = [fever,age,tiredness,cough,feverDays]
        # inputFeatures = [102,27,-1,1,12]
        infProb = clf.predict_proba([inputFeatures])[0][1]

        return jsonify({'result':infProb*100})
    else:
        return jsonify({'result':'NO'})
    # return render_template('index.html')


if __name__ == "__main__":
    print("Tester is Up")
    app.run(debug=True)


