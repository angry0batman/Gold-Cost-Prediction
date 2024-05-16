from flask import Flask,request,jsonify,render_template
import pickle
import pandas as pd

app=Flask(__name__)



with open("gold_price_model.pkl", 'rb') as file:
    model = pickle.load(file)

@app.route("/")
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    data=request.json
    df= pd.DataFrame(data, index=[0])
    prediction=model.predict(df)

    return jsonify({"prediction": prediction[0]})

if __name__ =='__main__':
    app.run(debug="True")