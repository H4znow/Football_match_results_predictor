from flask import Flask, render_template, request

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

if __name__ == '__main__':
    app.run(debug=True)
