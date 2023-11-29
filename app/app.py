from flask import Flask, render_template, request, jsonify
import pandas as pd
from joblib import load
from predict_match_result import predict_result, predict_result_group_stage
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_countries')
def get_countries():
    countries_df = pd.read_csv(".\data\FIFA_country_list.csv")
    countries_df.columns = ["Country"]
    countries_df['Country'] = countries_df['Country'].str.rstrip(';')
    countries_list = countries_df['Country'].tolist()
    return jsonify(countries_list)


@app.route('/match_predictor', methods=['GET', 'POST'])
def match_predictor():
    winner = None
    proba = None

    if request.method == 'POST':
        team1 = request.form['team1']
        team2 = request.form['team2']
        neutral = request.form['neutral']

        winner, proba = predict_result_group_stage(team1, team2, neutral=neutral)

    return render_template('match_predictor.html', winner=winner, proba=proba)

@app.route('/tournament_predictor')
def tournament_predictor():
    # Implement your tournament predictor logic here
    winner = "Team X"  # Replace with the actual winner
    return render_template('tournament_predictor.html', winner=winner)

if __name__ == '__main__':
    app.run(debug=True)
