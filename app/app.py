from flask import Flask, render_template, request
import pandas as pd
from joblib import load
from predict_match_result import predict_result

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/match_predictor', methods=['GET', 'POST'])
def match_predictor():
    winner = None

    if request.method == 'POST':
        team1 = request.form['team1']
        team2 = request.form['team2']
        neutral = request.form['neutral']
        winner, proba = predict_result(team1, team2, neutral=neutral)

    return render_template('match_predictor.html', winner=winner, proba = proba)

@app.route('/tournament_predictor')
def tournament_predictor():
    # Implement your tournament predictor logic here
    winner = "Team X"  # Replace with the actual winner
    return render_template('tournament_predictor.html', winner=winner)


if __name__ == '__main__':
    app.run(debug=True)
