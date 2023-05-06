from flask import Flask, request, render_template, jsonify
import flask_cors
import pandas as pd
import pickle

app = Flask(__name__)
flask_cors.CORS(app)

model = pickle.load(open('cat_model.pkl','rb'))

@app.route('/')
@flask_cors.cross_origin()
def home():
    return render_template('index.html')

@app.route('/predict')
@flask_cors.cross_origin()
def predict():
    specific_gravity = float(request.args.get('specific_gravity'))
    albumin = float(request.args.get('albumin'))
    blood_glucose_random = float(request.args.get('blood_glucose_random'))
    blood_urea = float(request.args.get('blood_urea'))
    serum_creatinine = float(request.args.get('serum_creatinine'))
    haemoglobin = float(request.args.get('haemoglobin'))
    packed_cell_volume = float(request.args.get('packed_cell_volume'))
    white_blood_cell_count = float(request.args.get('white_blood_cell_count'))
    hypertension = float(request.args.get('hypertension'))
    diabetes_mellitus = float(request.args.get('diabetes_mellitus'))

    data = pd.DataFrame(
        {
            'specific_gravity':[specific_gravity],
            'albumin':[albumin],
            'blood_glucose_random':[blood_glucose_random],
            'blood_urea':[blood_urea],
            'serum_creatinine':[serum_creatinine],
            'haemoglobin':[haemoglobin],
            'packed_cell_volume':[packed_cell_volume],
            'white_blood_cell_count':[white_blood_cell_count],
            'hypertension':[hypertension],
            'diabetes_mellitus':[diabetes_mellitus]
        }
    )
    res = model.predict_proba(data)[0]
    res = int(res[0]*100)
    response = jsonify({'result': res})
    print(res)
    return response

if __name__ == '__main__':
    app.run(host="127.0.0.1",port="5000",debug=True)