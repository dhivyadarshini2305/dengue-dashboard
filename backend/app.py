
from flask import Flask, request, jsonify
import joblib
import pandas as pd
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

model = joblib.load("dengue_model.pkl")
le = joblib.load("label_encoder.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json

    state = data["state"].title()
    year = int(data["year"])
    deaths = int(data["deaths"])

    state_encoded = le.transform([state])[0]

    input_df = pd.DataFrame({
        "year": [year],
        "state_encoded": [state_encoded],
        "deaths": [deaths]
    })

    prediction = model.predict(input_df)[0]

    return jsonify({"cases": int(prediction)})

if __name__ == "__main__":
    app.run(debug=True)
