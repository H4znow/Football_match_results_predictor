from flask import Flask, render_template, request, jsonify
import csv
import os
import pandas as pd
from joblib import load
from predict_match_result import predict_result, predict_result_group_stage
from predict_championship import championship
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

@app.route('/save_groups', methods=['POST'])
def save_groups():
    groups = {}
    csv_file_path = './app/data/groups.csv'
    progress_info = None

    for group in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']:
        teams = request.form.getlist('group{}[]'.format(group))
        groups[group] = teams

    # Save the groups to a CSV file
    save_groups_to_csv(groups, csv_file_path)

    # Start the tournament and get the progress information
    progress_info = championship(csv_file_path)

    # Pass the progress information to the template
    return render_template('tournament_predictor.html', saved_groups=groups, progress_info=progress_info)


def save_groups_to_csv(groups, csv_file_path):

    # Check if the directory exists, and create it if not
    directory = os.path.dirname(csv_file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

    with open(csv_file_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)

        # Write the header
        csv_writer.writerow(['Group A', 'Group B', 'Group C', 'Group D', 'Group E', 'Group F', 'Group G', 'Group H'])

        # Write the teams for each group
        for index in range(4):
            csv_writer.writerow([groups['A'][index], groups['B'][index], groups['C'][index], groups['D'][index],
                                groups['E'][index], groups['F'][index], groups['G'][index], groups['H'][index]])

@app.route('/tournament_predictor')  # Add this route
def tournament_predictor():
    # Placeholder for the 'tournament_predictor' route
    return render_template('tournament_predictor.html', saved_groups={}, progress_info=[])


if __name__ == '__main__':
    app.run(debug=True)
