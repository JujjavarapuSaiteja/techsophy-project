from flask import Flask, render_template, request
import pandas as pd
import joblib
import os
app = Flask(__name__)
class RiskModel:
    def __init__(self):
        self.diabetes_model = joblib.load('models/diabetes_model.pkl')
        self.heart_model = joblib.load('models/heart_model.pkl')
        self.obesity_model = joblib.load('models/obesity_model.pkl')

    def diabetes_risk(self, age, bmi, glucose, bp):
        input_df = pd.DataFrame([[age, bmi, glucose, bp]], columns=['Age', 'BMI', 'Glucose', 'BloodPressure'])
        prob = self.diabetes_model.predict_proba(input_df)[0][1]
        return prob * 100

    def heart_disease_risk(self, age, bp, cholesterol, smoker):
        input_df = pd.DataFrame([[age, bp, cholesterol, smoker]], columns=['Age', 'BloodPressure', 'Cholesterol', 'Smoker'])
        prob = self.heart_model.predict_proba(input_df)[0][1]
        return prob * 100

    def obesity_risk(self, bmi, activity_level):
        activity_encoded = 0 if activity_level == 'high' else 1 if activity_level == 'moderate' else 2
        input_df = pd.DataFrame([[bmi, activity_encoded]], columns=['BMI', 'ActivityLevel'])
        prob = self.obesity_model.predict_proba(input_df)[0][1]
        return prob * 100

class RecommendationEngine:
    def recommend(self, risks):
        actions = []
        if risks['diabetes'] > 60:
            actions.append(("Reduce sugar intake", 1))
        if risks['heart'] > 60:
            actions.append(("Control blood pressure", 2))
        if risks['obesity'] > 50:
            actions.append(("Start physical activity", 1))
        if risks['diabetes'] > 60 and risks['obesity'] > 50:
            actions.append(("Weight management plan", 1))
        return sorted(actions, key=lambda x: x[1])
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/assess', methods=['POST'])
def assess():
    data = request.form
    age = int(data['age'])
    bmi = float(data['bmi'])
    glucose = float(data['glucose'])
    family_history = data.get('family_history') == 'on'
    bp = float(data['bp'])
    cholesterol = float(data['cholesterol'])
    smoker = 1 if data.get('smoker') == 'on' else 0
    activity_level = data['activity_level']

    model = RiskModel()
    risks = {
        'diabetes': model.diabetes_risk(age, bmi, glucose, bp),
        'heart': model.heart_disease_risk(age, bp, cholesterol, smoker),
        'obesity': model.obesity_risk(bmi, activity_level),
    }

    recommender = RecommendationEngine()
    recommendations = recommender.recommend(risks)

    return render_template('result.html', risks=risks, recommendations=recommendations)
if __name__ == '__main__':
    app.run(debug=True)