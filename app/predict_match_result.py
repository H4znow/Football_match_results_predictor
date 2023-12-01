import sys

from joblib import load

import pandas as pd



# Extract features from both countries to feed the model for prediction
# We take the most recent features data available in the dataset 
def extract_features(home_team: str, away_team:str, neutral: bool):

    # Load dataset to get features
    df = load("./assets/rera.joblib")


    ### Extract home_team features ###

    # Get the most recent features
    df_home_team = df[(df["home_team"] == home_team) | (df["away_team"] == home_team)]
    features_home_team = df_home_team[df_home_team["date"] == df_home_team["date"].max()].drop(columns=["date", "winner", "neutral"])

    # Select the features corresponding to the team we're considering
    # The most recent data could be as home_team OR away_team, we don't know in advance
    # So we may have to rename columns if it does not correspond to the team status for the match considered
    if features_home_team["home_team"].iloc[0] == home_team:
        features_home_team = features_home_team.drop(features_home_team.filter(regex="away").columns, axis=1)
    else:
        features_home_team = features_home_team.drop(features_home_team.filter(regex="home").columns, axis=1)
        # Adjust columns names
        features_home_team.rename(columns={col: col.replace('away_', 'home_') for col in features_home_team.columns}, inplace=True)
        features_home_team.rename(columns={col: col.replace('_away', '_home') for col in features_home_team.columns}, inplace=True)

    # Drop duplicates: it's possible to have duplicates since we oversampled the dataset to have balanced classes
    features_home_team = features_home_team.drop_duplicates()




    ### Extract away_team features ###

    # Get the most recent features
    df_away_team = df[(df["home_team"] == away_team) | (df["away_team"] == away_team)]
    features_away_team = df_away_team[df_away_team["date"] == df_away_team["date"].max()].drop(columns=["date", "winner", "neutral"])

    # Select the features corresponding to the team we're considering
    if features_away_team["home_team"].iloc[0] == away_team:
        features_away_team = features_away_team.drop(features_away_team.filter(regex="away").columns, axis=1)
        # Adjust columns names
        features_away_team.rename(columns={col: col.replace('home_', 'away_') for col in features_home_team.columns}, inplace=True)
        features_away_team.rename(columns={col: col.replace('_home', '_away') for col in features_home_team.columns}, inplace=True)
    else:
        features_away_team = features_away_team.drop(features_away_team.filter(regex="home").columns, axis=1)
        

    # Drop duplicates: it's possible to have duplicates since we oversampled the dataset to have balanced classes
    features_away_team = features_away_team.drop_duplicates()



    ### Merge features to feed to the model ###

    final_feature = pd.merge(features_home_team, features_away_team, how="cross")
    # Set neutral to its value
    final_feature["neutral"] = neutral
    # Reorder columns to match the order of the training dataset
    final_feature = final_feature[df.drop(columns=["winner", "date"]).columns]


    return final_feature

# Predict the result of the match according to country names and neutral
# By default, the Random Forest model ("rf") is used
def predict_result(home_team: str, away_team: str, neutral: bool, model="rf"):
    winner = None
    probas_win = None
    # Load the model
    if model == "mlpc":
        model = load("./assets/mlpc_clf_gridsearch.joblib")
    else:
        model = load("./assets/rf_clf_gridsearch.joblib")


    # Get features to feed the model
    features = extract_features(home_team, away_team, neutral)


    # Predict probabilities to win for each country
    probas = model.predict_proba(features)


    # If probas are close to 0.50, we consider a draw
    if probas[0][0] >= 0.5 :
        winner = home_team
        probas_win = probas[0][0]
    else:
        winner = away_team
        probas_win = probas[0][1]

    return winner, probas_win



# Predict the result of the match according to country names and neutral in group stages' match (because this function determine if there is
# a draw)
# By default, the Random Forest model ("rf") is used
def predict_result_group_stage(home_team: str, away_team: str, neutral: bool, model="rf"):
    
    # Load the model
    if model == "mlpc":
        model = load("./assets/mlpc_clf_gridsearch.joblib")
    else:
        model = load("./assets/rf_clf_gridsearch.joblib")


    # Get features to feed the model
    features = extract_features(home_team, away_team, neutral)


    # Predict probabilities to win for each country
    probas = model.predict_proba(features)

    # If probas are close to 0.50, we consider a draw
    if (probas[0][0] <= 0.55) & (probas[0][0] >= 0.45):
        winner = "Draw"
        proba_winner = probas[0][0]
    elif probas[0][0] > 0.55:
        winner = home_team
        proba_winner = probas[0][0]
    else:
        winner = away_team
        proba_winner = probas[0][1]
    
    return winner, proba_winner


# Define a main function to use the script if need
def __main__():
    if len(sys.argv) != 5:
        print("Usage: python predict_match_result.py <home_team> <away_team> <neutral> <model_type>")
        print("<model_type> can be 'rf' for Random Forest or 'mlpc' for MLP Classifier")
        sys.exit(1)

    home_team = sys.argv[1]
    away_team = sys.argv[2]


    rera = load("./assets/rera.joblib")
    available_teams = set(rera["home_team"].unique()) | set(rera["away_team"].unique())
    available_teams = list(available_teams)

    # If country not available in list, we can't predict the result
    if home_team not in available_teams or away_team not in available_teams:
        print("There's at least one of the countries you entered that is not available.")
        sys.exit(1)

    if sys.argv[3] == "true":
        neutral =  True
    else:
        neutral = False

    if sys.argv[4] == "mlpc":
        model = "mlpc"
    else:
        model = "rf"

    result = predict_result(home_team, away_team, neutral, model)

    if result == "Draw":
        print("It's a draw!")
    else:
        print(f"The winner is: {result}")

    

if __name__ == "__main__":
    __main__()







