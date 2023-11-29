import itertools
import sys
import pandas as pd
import csv
import predict_match_result as pmr

# -------------------------------------------------------------------------------------------------------------------- #
# Function : Run a championship                                                                                        #
# -------------------------------------------------------------------------------------------------------------------- #
# Predict the final result of a championship of a given list of teams from a .CSV file. The championship is structured #
# in 2 phases: the group stage and the knockout stage.                                                                 #
# -------------------------------------------------------------------------------------------------------------------- #
def championship(teams: str, model="rf"):

    def import_teams(file_path):                                                   # Import group teams from a .CSV file
        with open(file_path, newline='') as csvfile:

            reader  = csv.reader(csvfile, delimiter=';')
            next(reader)                                                                           # Skip the header row
            columns = list(zip(*reader))                                                               # Rows to columns

        return [list(column) for column in columns]

    groups = import_teams(teams)                                                                          # Import teams
    groups = [{country.strip(): 0 for country in group} for group in groups]              # Create a dict for each group

    print(f"Groups: {groups}")

    # ---------------------------------------------------------------------------------------------------------------- #
    # The group stage                                                                                                  #
    # ---------------------------------------------------------------------------------------------------------------- #
    # For each group in the championship we predict the result of each match between teams of this group               #
    # ---------------------------------------------------------------------------------------------------------------- #

    print("\n---------------------------------------------------------------------------------------------------------")
    print("Results of the group stage:")
    print("---------------------------------------------------------------------------------------------------------\n")

    for group in groups:

        # Get all possible match combinations in the group
        combinations = pd.DataFrame(list(itertools.combinations(group.keys(), 2)),
                                    columns=["home_team", "away_team"])

        for home_team, away_team in combinations.values:                              # Predict the result of each match

            winner, probability_winner = pmr.predict_result_group_stage(home_team, away_team, False, model)

            if winner == "Draw":                                                            # Update points in the group
                group[home_team] += 1
                group[away_team] += 1
            else:
                group[winner]    += 3

            print(f"[{home_team} vs {away_team}]: {winner} wins with {probability_winner} probability")

    print("\n---------------------------------------------------------------------------------------------------------")
    print(f"Final group points: {groups}")
    print("---------------------------------------------------------------------------------------------------------\n")

    # ---------------------------------------------------------------------------------------------------------------- #
    # The knockout stage                                                                                               #
    # ---------------------------------------------------------------------------------------------------------------- #
    # For each group in the championship we predict the result of each match between teams of this group               #
    # ---------------------------------------------------------------------------------------------------------------- #

    # Get the 2 best teams of each group
    best_teams = [sorted(group.items(), key=lambda x: x[1], reverse=True)[:2] for group in groups]
    print(f"Best teams, knockout : {best_teams}")

    # New list order for the knockout stage
    knockout_teams = []
    for i in range(2):
        for group in best_teams:
            knockout_teams.append(group[i])

    print("\n---------------------------------------------------------------------------------------------------------")
    print(f"Knockout teams: {knockout_teams}")
    print("---------------------------------------------------------------------------------------------------------\n")

    # Until there only one team left, we predict the result of each match between 2 teams of the list in order
    while len(knockout_teams) > 1:

        print("\n-----------------------------------------------------------------------------------------------------")
        print("New knockout phase:")
        print("-----------------------------------------------------------------------------------------------------\n")

        knockout_teams_updated = []

        for i in range(0, len(knockout_teams), 2):

            home_team                  = knockout_teams[i][0]
            away_team                  = knockout_teams[i + 1][0]

            winner, probability_winner = pmr.predict_result(home_team, away_team, False, model)

            print(f"[{home_team} vs {away_team}]: {winner} wins with {probability_winner} probability")

            knockout_teams_updated.append([winner, probability_winner])

        knockout_teams = knockout_teams_updated

    print("\n---------------------------------------------------------------------------------------------------------")
    print(f"Winner of the championship: {knockout_teams[0][0]}")
    print("---------------------------------------------------------------------------------------------------------\n")

# -------------------------------------------------------------------------------------------------------------------- #
# Function : Check if chosen teams are valid                                                                           #
# -------------------------------------------------------------------------------------------------------------------- #

def check_teams(home_team: str, away_team: str):
    available_teams = pd.read_csv("../data/FIFA_country_list.csv",
                                  sep=";",
                                  names=["country"],
                                  index_col=False
                                  )["country"].unique()

    # If country not available in list, we can't predict the result

    if home_team not in available_teams or away_team not in available_teams:
        print("There's at least one of the countries you entered that is not available.")
        sys.exit(1)


# -------------------------------------------------------------------------------------------------------------------- #
# Function : main()                                                                                                    #
# -------------------------------------------------------------------------------------------------------------------- #
# Parameters:                                                                                                          #
#   - teams: path of the .CSV file containing the list of teams                                                        #
#   - model: type of model to use for prediction                                                                       #
# -------------------------------------------------------------------------------------------------------------------- #

def __main__():
    if len(sys.argv) != 3:

        print("Usage: python predict_championship.py <teams_csv_path> <model_type>")
        print("<model_type> can be 'rf' for Random Forest or 'mlpc' for MLP Classifier")
        sys.exit(1)

        # Example: python predict_championship.py ../data/championship.csv rf

    championship(sys.argv[1], sys.argv[2])

if __name__ == "__main__":
    __main__()







