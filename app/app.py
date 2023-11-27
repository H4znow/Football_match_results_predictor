from flask import Flask, render_template, request
from joblib import load

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

        # Implement your match predictor logic here
        # Replace the following with your actual prediction logic
        winner = "Team A" if team1 == "Team A" else "Team B"

    return render_template('match_predictor.html', winner=winner)

@app.route('/tournament_predictor')
def tournament_predictor():
    # Implement your tournament predictor logic here
    winner = "Team X"  # Replace with the actual winner
    return render_template('tournament_predictor.html', winner=winner)

def predict_match(team1, team2):
    model = load("../assets/rf_clf_default_86_acc.joblib")
    team1 = team_last_info(team1)
    team2 = team_last_info(team2)
    

def team_last_info(team):
    df = load("../assets/rera.joblib")
    filtered_df = df[(df['home_country'] == team) | (df['away_country'] == team)]
    last_row_with_country = filtered_df.iloc[-1]
    # Determine if the country is a home team or an away team
    is_home_team = last_row_with_country['home_country'] == team
    # Drop columns containing the word 'away' if the country is a home team, otherwise drop columns containing 'home'
    columns_to_drop = [col for col in last_row_with_country.index if (is_home_team and 'away' in col) or (not is_home_team and 'home' in col)]
    last_row_without_columns = last_row_with_country.drop(columns=columns_to_drop)
    return last_row_without_columns


if __name__ == '__main__':
    app.run(debug=True)
