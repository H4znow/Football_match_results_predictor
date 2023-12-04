import sys
import sklearn
import pandas                as     pd
import numpy                 as     np
import seaborn               as     sns
import matplotlib.pyplot     as     plt
from   sklearn.utils         import resample
from   joblib                import dump
from   sklearn.decomposition import PCA
from   sklearn.preprocessing import StandardScaler

# -------------------------------------------------------------------------------------------------------------------- #
# Data Preprocessing : Football match prediction project                                                               #
# -------------------------------------------------------------------------------------------------------------------- #
# Script version of preprocess_dataset.ipynb used in this project for preprocessing the dataset.                       #
# Université Côte d'Azur, 20 nov 2023.                                                                                 #
# -------------------------------------------------------------------------------------------------------------------- #
# Authors : HADDOU Amine, BOULLI Marouan, BAPTISTA Rafael.                                                             #
# -------------------------------------------------------------------------------------------------------------------- #

print("The preprocessing can take up to 15 min. Please be patient :)")

def preprocessing():

    # ---------------------------------------------------------------------------------------------------------------- #
    # Import Datasets                                                                                                  #
    # ---------------------------------------------------------------------------------------------------------------- #

    results = pd.read_csv("../data/results.csv")
    ranking = pd.read_csv("../data/fifa_ranking-2023-07-20.csv")

    results.head(5)                                                    # Display the first 5 rows of the results dataset
    results.info()                                                      # Display the information of the results dataset

    ranking.head(5)                                                    # Display the first 5 rows of the ranking dataset
    ranking.info()                                                      # Display the information of the ranking dataset

    # ---------------------------------------------------------------------------------------------------------------- #
    # 1. Replace old countries name by the most precise name                                                           #
    # ---------------------------------------------------------------------------------------------------------------- #
    # Given the fact that some country names have evolved over time, while others no longer exists, it's crucial to
    # alter the data by replacing these instances with accurately updated country names.                               #
    # ---------------------------------------------------------------------------------------------------------------- #
    # We have opted to use OpenAI's GPT-4, with a finely-tuned prompt, to evaluate the " false countries " in the      #
    # dataset and suggest appropriate replacements. After a series of trials and errors, the following prompt has been #
    # perfected to yield optimal results :                                                                             #
    # ---------------------------------------------------------------------------------------------------------------- #

    prompt = ("You will be presented with a list of countries. These could either still exist, have ceased to exist, "
              "or be listed in a different language. We refer to these as the false countries. Your task is to generate"
              "a JSON where the keys represent these false countries, and the corresponding values indicate the "
              "current, correct English name."
              )

    # ---------------------------------------------------------------------------------------------------------------- #
    # 1.1. Replacing country names in results dataset                                                                  #
    # ---------------------------------------------------------------------------------------------------------------- #
    # The goal of this approach is to maintain a dataset consisting solely of accurate English country names.          #
    # The JSON file converted to a dictionary returned by CHATGPT-4 :                                                  #
    # ---------------------------------------------------------------------------------------------------------------- #

    false_countries = {
        "Czechoslovakia": "Czech Republic", "Catalonia": "Spain","Basque Country": "Spain","Brittany": "France",
        "Yugoslavia": "Serbia","Galicia": "Spain","Andalusia": "Spain","Central Spain": "Spain","Silesia": "Poland",
        "Manchukuo": "China","German DR": "Germany","Kernow": "United Kingdom","Saarland": "Germany",
        "Vietnam Republic": "Vietnam","North Vietnam": "Vietnam","Zanzibar": "Tanzania","Eswatini": "Swaziland",
        "Western Australia": "Australia","Northern Cyprus": "Cyprus","Yemen DPR": "Yemen","Ynys Môn": "United Kingdom",
        "Shetland": "United Kingdom","Isle of Wight": "United Kingdom","Canary Islands": "Spain",
        "Frøya": "Norway","Hitra": "Norway","Corsica": "France","Saare County": "Estonia","Rhodes": "Greece",
        "Orkney": "United Kingdom","Sark": "United Kingdom","Alderney": "United Kingdom",
        "Falkland Islands": "United Kingdom","Republic of St. Pauli": "Germany","Găgăuzia": "Moldova",
        "Tibet": "China","Occitania": "France","Sápmi": "Norway","Menorca": "Spain","Provence": "France",
        "Arameans Suryoye": "Syria","Padania": "Italy","Iraqi Kurdistan": "Iraq","Gozo": "Malta",
        "Bonaire": "Netherlands","Western Sahara": "Morocco","Raetia": "Switzerland","Darfur": "Sudan",
        "Tamil Eelam": "Sri Lanka","Abkhazia": "Georgia","Madrid": "Spain","Ellan Vannin": "Isle of Man",
        "South Ossetia": "Georgia","County of Nice": "France","Székely Land": "Romania","Romani people": "India",
        "Felvidék": "Slovakia","Chagos Islands": "United Kingdom","United Koreans in Japan": "North Korea",
        "Western Armenia": "Turkey","Barawa": "Somalia","Kárpátalja": "Ukraine","Yorkshire": "United Kingdom",
        "Matabeleland": "Zimbabwe","Cascadia": "United States","Kabylia": "Algeria","Parishes of Jersey": "Jersey",
        "Chameria": "Albania","Yoruba Nation": "Nigeria","Biafra": "Nigeria","Mapuche": "Chile", "Aymara": "Bolivia",
        "Ticino": "Switzerland","Hmong": "China","Somaliland": "Somalia","Western Isles": "United Kingdom"
    }

    # ---------------------------------------------------------------------------------------------------------------- #
    # Calculate and display the number of unique countries before and after the replacement                            #
    # ---------------------------------------------------------------------------------------------------------------- #

    num_unique_countries = len(set(results["home_team"].unique()) | set(results["away_team"].unique()))
    print(f"Number of countries before the replacement : {num_unique_countries}")

    results.replace(false_countries, inplace=True)

    num_unique_countries = len(set(results["home_team"].unique()) | set(results["away_team"].unique()))
    print(f"Number of countries after the replacement : {num_unique_countries}")

    # ---------------------------------------------------------------------------------------------------------------- #
    # 1.2. Replacing country names in ranking dataset                                                                  #
    # ---------------------------------------------------------------------------------------------------------------- #
    # JSON converted to dictionary that CHATGPT-4 returned with the previous prompt.                                   #
    # ---------------------------------------------------------------------------------------------------------------- #

    false_countries = {
        "Korea DPR": "North Korea", "Cape Verde Islands": "Cape Verde","Brunei Darussalam": "Brunei",
        "USA": "United States","Côte d'Ivoire": "Ivory Coast","Yugoslavia": "Serbia","Czechoslovakia": "Czech Republic",
        "Zaire": "Democratic Republic of the Congo","IR Iran": "Iran","China PR": "China",
        "Trinidad and Tobago": "Trinidad","Korea Republic": "South Korea","FYR Macedonia": "North Macedonia",
        "Kyrgyz Republic": "Kyrgyzstan","Chinese Taipei": "Taiwan","Serbia and Montenegro": "Serbia",
        "Swaziland": "Eswatini","St. Vincent / Grenadines": "St. Vincent and the Grenadines",
        "Timor-Leste": "East Timor", "North Macedonia": "Macedonia","São Tomé e Príncipe": "Sao Tome and Principe",
        "Curaçao": "Curacao","Cabo Verde": "Cape Verde","Czechia": "Czech Republic","Türkiye": "Turkey",
        "St Vincent and the Grenadines": "St. Vincent and the Grenadines", "St Lucia": "St. Lucia",
        "The Gambia": "Gambia", "St Kitts and Nevis": "St. Kitts and Nevis","Hong Kong, China": "Hong Kong",
        "Aotearoa New Zealand": "New Zealand"
    }

    # ---------------------------------------------------------------------------------------------------------------- #
    # Calculate and display the number of unique countries before and after the replacement                            #
    # ---------------------------------------------------------------------------------------------------------------- #

    num_unique_countries = len(ranking["country_full"].unique())
    print(f"Number of countries before the replacement : {num_unique_countries}")

    ranking.replace(false_countries, inplace=True)

    num_unique_countries = len(ranking["country_full"].unique())
    print(f"Number of countries after the replacement : {num_unique_countries}")

    # ---------------------------------------------------------------------------------------------------------------- #
    # 1.3. Choosing a cutting year                                                                                     #
    # ---------------------------------------------------------------------------------------------------------------- #
    # After careful consideration we have decided to choose a “cut-off year” based on our project’s performance rather #
    # than changes in country names. We deem the data prior to 2000 as mostly irrelevant as it does not significantly  #
    # influence current game results because:                                                                          #
    #                                                                                                                  #
    #   - Matches from that era do not contribute to today’s game outcomes, mainly due to their age.                   #
    #   - There have been considerable changes in the game dynamics and player preparation over the years. As a result #
    #     comparisons between past and present matches are not highly meaningful.                                      #
    #   - The structure of competitions has undergone numerous transformations. Therefore, team performances from the  #
    #     past may not accurately reflect their current capabilities.                                                  #
    #                                                                                                                  #
    # Once we implement this initial cut-off, we will further fine-tune the “cut-off year” to refine our results.      #
    # ---------------------------------------------------------------------------------------------------------------- #

    # Convert "date columns" of datasets to date time type
    # Drop rows with date before 2000

    results['date']         = pd.to_datetime(results['date'], format='%Y-%m-%d')
    ranking['rank_date']    = pd.to_datetime(ranking['rank_date'], format='%Y-%m-%d')
    results                 = results[results['date'].dt.year >= 2000]
    ranking                 = ranking[ranking['rank_date'].dt.year >= 2000]

    # Ensure that the result is as expected
    print(results.head(2))
    print("*"*90)
    print(ranking.head(2))

    # ---------------------------------------------------------------------------------------------------------------- #
    # 2. Resempling Data                                                                                               #
    # ---------------------------------------------------------------------------------------------------------------- #

    # ---------------------------------------------------------------------------------------------------------------- #
    # 2.1. Resetting indexes                                                                                           #
    # ---------------------------------------------------------------------------------------------------------------- #
    # The dataset containing results will be indexed based solely on dates, while the dataset for ranking will also be #
    # indexed by the names of countries to assist with resampling.                                                     #
    # ---------------------------------------------------------------------------------------------------------------- #

    results.set_index(['date'], inplace=True)
    ranking.set_index(['rank_date', 'country_full'], inplace=True)

    # ---------------------------------------------------------------------------------------------------------------- #
    # 2.2. Grouping ranking datas by date and country's name.                                                          #
    # ---------------------------------------------------------------------------------------------------------------- #

    # Grouping by date and country name
    grouped_ranking = ranking.groupby(by=['rank_date', 'country_full']).ffill()

    # ---------------------------------------------------------------------------------------------------------------- #
    # 2.3. Upsampling                                                                                                  #
    # ---------------------------------------------------------------------------------------------------------------- #

    # Reset Ranking's indexs for the upsampling
    grouped_ranking.reset_index(inplace=True)
    grouped_ranking.set_index("rank_date", inplace=True)

    # An example to observe the impact of upsampling on our dataset
    count_france_lines = len(grouped_ranking.loc[grouped_ranking['country_full'] == "France"])

    # For each country, we create a sub-dataset. We upsample it by day, assign the latest non-null values to each date,
    # and add new rows to `sampled_ranking`, which represents the final dataset.
    list_of_countries   = grouped_ranking["country_full"].unique()
    sampled_ranking     = pd.DataFrame()

    for country in list_of_countries:

        df_country      = grouped_ranking[grouped_ranking['country_full'] == country].copy()
        df_country      = df_country.resample("D").last().ffill()
        sampled_ranking = pd.concat([sampled_ranking, df_country], axis=0)

    sampled_ranking     = sampled_ranking.sort_index()
    count_france_lines  = len(sampled_ranking.loc[sampled_ranking['country_full'] == "France"])

    print(f"Count lines where France appears after the upsamling : {count_france_lines}")

    # ---------------------------------------------------------------------------------------------------------------- #
    # 3. Merging datasets                                                                                              #
    # ---------------------------------------------------------------------------------------------------------------- #

    # Reset datasets' indexs for the merging
    sampled_ranking.reset_index(inplace=True)
    results.reset_index(inplace=True)

    # ---------------------------------------------------------------------------------------------------------------- #
    # 3.1. First merging on home team                                                                                  #
    # ---------------------------------------------------------------------------------------------------------------- #

    # Merging results with sampled_ranking on home team
    sampled_ranking.columns = [col + "_home" if col != "rank_date" and col != "country_full"
                                             else col for col in sampled_ranking.columns]

    rera                    = results.merge(sampled_ranking,
                                        left_on     =["date", "home_team"],
                                        right_on    =["rank_date", "country_full"],
                                        suffixes    =[None,'_home']).drop(["rank_date",
                                                                           "country_full",
                                                                           "country_abrv_home"],
                                        axis        =1)

    # ---------------------------------------------------------------------------------------------------------------- #
    # 3.2. Second merging on away team                                                                                 #
    # ---------------------------------------------------------------------------------------------------------------- #

    # Merging results with sampled_ranking on away team
    sampled_ranking.columns = [col.replace("_home", "_away") for col in sampled_ranking.columns]
    rera                    = rera.merge(sampled_ranking,
                                         left_on   =["date", "away_team"],
                                         right_on  =["rank_date", "country_full"],
                                         suffixes  =[None,'_away']).drop(["rank_date",
                                                                          "country_full",
                                                                          "country_abrv_away"],
                                         axis      =1)

    # ---------------------------------------------------------------------------------------------------------------- #
    # 3.3 Observe results from the merging                                                                             #
    # ---------------------------------------------------------------------------------------------------------------- #

    rera.sort_values(by='date', inplace=True)
    rera.head()
    rera.info()

    # ---------------------------------------------------------------------------------------------------------------- #
    # 4. Feature Engineering                                                                                           #
    # ---------------------------------------------------------------------------------------------------------------- #

    # ---------------------------------------------------------------------------------------------------------------- #
    # 4.1. New Features                                                                                                #
    # ---------------------------------------------------------------------------------------------------------------- #

    # ---------------------------------------------------------------------------------------------------------------- #
    # 4.1.1. A winner feature                                                                                          #
    # ---------------------------------------------------------------------------------------------------------------- #
    # winner will be an integer column, containing only three distinct values: {0, 1, 2}.                              #
    # 0: Indicates that the home_team has won.                                                                         #
    # 1: Indicates that the away_team has won.                                                                         #
    # 2: Represents a draw, signifying that both teams have an equal outcome.                                          #
    # -----------------------------------------------------------------------------------------------------------------#

    # Function to define a winner
    def define_winner(line) :

        if rera.loc[line.name, "home_score"] > rera.loc[line.name, "away_score"]:
             return 0

        elif rera.loc[line.name, "home_score"] < rera.loc[line.name, "away_score"]:
            return 1

        else :
            return 2

    # Create winner column and defining winners of each match
    rera['winner'] = rera.apply(lambda line : define_winner(line), axis=1)

    # ---------------------------------------------------------------------------------------------------------------- #
    # 4.1.1.1. Draw matches issue                                                                                      #
    # ---------------------------------------------------------------------------------------------------------------- #
    # In the case of a draw, we will update the winner column with the results of the shootouts data. If the shootout  #
    # column is NaN, we will drop the row. Since our model has to predict the outcome of a match, with a binary output,#
    # we cannot have draws in our dataset.                                                                             #
    # ---------------------------------------------------------------------------------------------------------------- #

    # ---------------------------------------------------------------------------------------------------------------- #
    # 4.1.1.2. Drop all friendly matches from the dataset that where draw                                              #
    # ---------------------------------------------------------------------------------------------------------------- #
    # First we need to drop all friendly draw machs from the dataset. Since we don’t have the shootouts results for    #
    # friendly matches.                                                                                                #
    # ---------------------------------------------------------------------------------------------------------------- #

    rera = rera.drop(rera[(rera['tournament'] == "Friendly") & (rera['winner'] == 2)].index)

    # ---------------------------------------------------------------------------------------------------------------- #
    # 4.1.1.3. Updating winner of draw matches with the shootouts results                                              #
    # ---------------------------------------------------------------------------------------------------------------- #
    # Now we can import the shootouts dataset and update the winner column by comparing the shootout dataset with the  #
    # date, home_team and away_team columns of the rera dataset.                                                       #
    # ---------------------------------------------------------------------------------------------------------------- #

    # Import shootouts dataset
    shootouts         = pd.read_csv("../data/shootouts.csv")
    shootouts['date'] = pd.to_datetime(shootouts['date'], format='%Y-%m-%d')

    # For each row in rera that has a draw, look for the corresponding match in shootouts,
    # And then update the winner column.

    for index, row in rera.iterrows():

        if row['winner'] == 2:

            shootout = shootouts.loc[(shootouts['date']         == row['date'])         &
                                     (shootouts['home_team']    == row['home_team'])    &
                                     (shootouts['away_team']    == row['away_team'])]

            if len(shootout) > 0:

                # Update the winner column as so :
                # If the home team won the shootout the winner is 0 else the winner is 1

                #print(row['date'], row['home_team'], row['away_team'], row['winner'], row['tournament'])

                if shootout['winner'].values[0] == row['home_team']:
                    rera.loc[index, 'winner'] = 0
                else:
                    rera.loc[index, 'winner'] = 1

    # ---------------------------------------------------------------------------------------------------------------- #
    # 4.1.1.4. Drop all left draw matches from the dataset                                                             #
    # ---------------------------------------------------------------------------------------------------------------- #

    rera = rera.drop(rera[rera['winner'] == 2].index)

    # ---------------------------------------------------------------------------------------------------------------- #
    # 4.1.2. Average goal feature                                                                                      #
    # ---------------------------------------------------------------------------------------------------------------- #
    # We are adding two new features, home_goals_avg and away_goals_avg. Two columns containing the average number of  #
    # goals scored by the home/away team in the last 7 matches.                                                        #
    # (Seven matches on average are sufficient to form an idea about the current form of a football team.)             #
    # ---------------------------------------------------------------------------------------------------------------- #

    # A Function to compute the average goal of a team on the last 7 matches
    # If there is less than 7 matches, we compute the average on the number of matches available
    # If there is no match, we return 0

    def compute_avg_country_goals(line, country):

        # Get the date of the current match and get the 7 previous matches
        date                = line.date
        previous_matches    = rera[(rera['home_team'] == country) | (rera['away_team'] == country)]
        previous_matches    = previous_matches[previous_matches["date"] < date].tail(7)

        # If there is no match, we return 0
        if len(previous_matches) == 0:
            return 0

        # If there is less than 7 matches, we compute the average on the number of matches available
        elif len(previous_matches) < 7:
            goals = previous_matches.apply(lambda row: row["home_score"] if (row["home_team"] == country) else
            row["away_score"], axis=1).sum()
            return goals / len(previous_matches)


        # If there is 7 matches, we compute the average on the 7 matches
        else:
            goals = previous_matches.apply(lambda row: row["home_score"] if (row["home_team"] == country) else
            row["away_score"], axis=1).sum()
            return goals / 7

    # Create home_win_avg and away_win_avg columns and defining the average win of each team

    rera['home_goals_avg'] = rera.apply(lambda line : compute_avg_country_goals(line, line['home_team']), axis=1)
    rera['away_goals_avg'] = rera.apply(lambda line : compute_avg_country_goals(line, line['away_team']), axis=1)

    # ---------------------------------------------------------------------------------------------------------------- #
    # 4.1.3. An average win feature                                                                                    #
    # ---------------------------------------------------------------------------------------------------------------- #
    # We are adding two new features, home_win_avg and away_win_avg. Two columns containing the average number of wins #
    # of the home/away team in the last 7 matches.                                                                     #
    # (Seven matches on average are sufficient to form an idea about the current form of a football team.)             #
    # ---------------------------------------------------------------------------------------------------------------- #

    # A Function to compute the average win of a team on the last 7 matches
    # If there is less than 7 matches, we compute the average on the number of matches available
    # If there is no match, we return 0

    def compute_avg_country_win(line, country):

        # Get the date of the current match and get the 7 previous matches
        date                = line.date
        previous_matches    = rera[(rera['home_team'] == country) | (rera['away_team'] == country)]
        previous_matches    = previous_matches[previous_matches["date"] < date].tail(7)

        # If there is no match, we return 0
        if len(previous_matches) == 0:
            return 0

        # If there is less than 7 matches, we compute the average on the number of matches available
        elif len(previous_matches) < 7:
            wins = previous_matches.apply(lambda row: 1 if ((row["home_team"] == country) &
                                                            (row["winner"] == 0)) | ((row["away_team"] == country) &
                                                            (row["winner"] == 1)) else 0, axis=1).sum()
            return wins / len(previous_matches)

        # If there is 7 matches, we compute the average on the 7 matches
        else:
            wins = previous_matches.apply(lambda row: 1 if ((row["home_team"] == country) &
                                                            (row["winner"] == 0)) | ((row["away_team"] == country) &
                                                            (row["winner"] == 1)) else 0, axis=1).sum()
            return wins / 7


    # Create home_win_avg and away_win_avg columns and defining the average win of each team

    rera['home_win_avg'] = rera.apply(lambda line : compute_avg_country_win(line, line['home_team']), axis=1)
    rera['away_win_avg'] = rera.apply(lambda line : compute_avg_country_win(line, line['away_team']), axis=1)

    # ---------------------------------------------------------------------------------------------------------------- #
    # 4.1.4. Number of victories in the last five matches between to teams                                             #
    # ---------------------------------------------------------------------------------------------------------------- #
    def compute_wins_direct_confrontation(row, country, opponent):
        date = row.date
        # All direct confrontaions between the 2 countries
        confrontations = rera[((rera["home_team"] == country) &
                               (rera["away_team"] == opponent)) | ((rera["home_team"] == opponent) &
                                                                   (rera["away_team"] == country))]
        # 5 last confrontations
        previous_matches = confrontations[confrontations["date"] < date].tail(5)

        if len(previous_matches) == 0:
            return 0

        # Get sum of wins for country against opponent
        nb_wins_country = previous_matches.apply(lambda row: 1 if ((row["home_team"] == country) &
                                                                   (row["winner"] == 0)) | ((row["away_team"] == country) &
                                                                   (row["winner"] == 1)) else 0, axis=1).sum()
        return nb_wins_country

    rera["home_last_wins"] = rera.apply(
        lambda row: compute_wins_direct_confrontation(row, row["home_team"], row["away_team"]), axis=1)


    rera["away_last_wins"] = rera.apply(
        lambda row: compute_wins_direct_confrontation(row, row["away_team"], row["home_team"]), axis=1)

    # ---------------------------------------------------------------------------------------------------------------- #
    # 4.2. Dropping useless features                                                                                   #
    # ---------------------------------------------------------------------------------------------------------------- #
    # Correlation heatmap and basic statistics on rera. Used to have an overview on the relationship between features. #
    # ---------------------------------------------------------------------------------------------------------------- #

    # Select only numeric columns
    numeric_columns = rera.select_dtypes(include=['float64', 'int64'])

    # Compute the correlation matrix
    correlation_matrix = numeric_columns.corr()

    # Create a heatmap with the correlation matrix
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")

    # Set plot title
    plt.title('Correlation Matrix')

    # Show the plot
    plt.show()

    numeric_columns.describe()

    # ---------------------------------------------------------------------------------------------------------------- #
    # 4.2.1. Dropping city feature                                                                                     #
    # ---------------------------------------------------------------------------------------------------------------- #
    # Having a city feature is not relevant to our analysis since we already have country. Therefore, we will drop it. #
    # ---------------------------------------------------------------------------------------------------------------- #

    rera.drop("city", axis=1, inplace=True)

    # ---------------------------------------------------------------------------------------------------------------- #
    # 4.2.2. Dropping home_score and away_score feature                                                                #
    # ---------------------------------------------------------------------------------------------------------------- #
    # We will drop home_score and away_score since we already have winner, home_goals_avg and away_goals_avg features. #
    # ---------------------------------------------------------------------------------------------------------------- #
    # The average goal feature is more relevant in the soccer context, since one team can score multiple times against #
    # a weaker team but then score nothing against a stronger opponent. It also gives a better overview on the team’s  #
    # performance over the years. Therefor, dropping the score columns will help us to avoid overfitting and           #
    # misleading the model.                                                                                            #
    # ---------------------------------------------------------------------------------------------------------------- #

    rera.drop(["home_score", "away_score"], axis=1, inplace=True)

    # ---------------------------------------------------------------------------------------------------------------- #
    # 4.2.3. Dropping country feature                                                                                  #
    # ---------------------------------------------------------------------------------------------------------------- #
    # Regarding the country feature, taking into account the fact that we can already identify the country where the   #
    # match takes place by looking at home_team or away_team depending on the value of neutral feature, country        #
    # gets redundant. Therefore, we can drop it.                                                                       #
    # ---------------------------------------------------------------------------------------------------------------- #

    rera.drop("country", axis=1, inplace=True)

    # ---------------------------------------------------------------------------------------------------------------- #
    # 4.2.4. Dropping previous_points_home and previous_points_away features                                           #
    # ---------------------------------------------------------------------------------------------------------------- #
    # As shown in the correlation heatmap, previous_points_home and previous_points_away exhibit high correlations     #
    # with  total_points_home and total_points_away, respectively. Across multiple matches, the total number of points #
    # remains relatively constant, possibly accounting for this correlation. We hypothesize that the total points      #
    # alone provide sufficient information, making the details from previous_points redundant, despite its minor       #
    # fluctuations.                                                                                                    #
    # ---------------------------------------------------------------------------------------------------------------- #

    rera.drop(["previous_points_home", "previous_points_away"], axis=1, inplace=True)

    # ---------------------------------------------------------------------------------------------------------------- #
    # 4.2.5. Dropping rank_change_home and rank_change_away features                                                   #
    # ---------------------------------------------------------------------------------------------------------------- #

    rera.drop(["rank_change_home", "rank_change_away"], axis = 1, inplace=True)

    # ---------------------------------------------------------------------------------------------------------------- #
    # 4.2.6. Dropping tournament feature                                                                               #
    # ---------------------------------------------------------------------------------------------------------------- #
    # There are too many values in this column, some of these values only appear once. Therefore, when we split the    #
    # dataset into training and test sets, some of these values may appear in one dataset but not the other. In such a #
    # scenario, the model won’t be able to make predictions on data containing a value that was not present during     #
    # training, especially if it appears only in the test set.                                                         #
    # ---------------------------------------------------------------------------------------------------------------- #

    rera.drop("tournament", axis = 1, inplace=True)

    # ---------------------------------------------------------------------------------------------------------------- #
    # 4.3. Tuning features                                                                                             #
    # ---------------------------------------------------------------------------------------------------------------- #
    # Theses features are not accurate enough to determine the performance of a team. The features *_win_avg are more  #
    # accurate and give a better perception of teams performance.                                                      #
    # ---------------------------------------------------------------------------------------------------------------- #

    # ---------------------------------------------------------------------------------------------------------------- #
    # 4.3.1. Standardization                                                                                           #
    # ---------------------------------------------------------------------------------------------------------------- #
    # We will perform standardization, encoding, and dimensionality reduction on `rera_standerdized_encoded` rather    #
    # than on the original DataFrame `rera`. This choice is deliberate as `rera_standerdized_encoded` is intended fo   #
    # hyperparameter analysis. The application of these steps to `rera` will be carried out in the model's notebook    #
    # through a pipeline, streamlining and automating the entire process. Leveraging a pipeline is advantageous for    #
    # swift preprocessing of new data, particularly when predicting the outcomes of new matches.                       #
    # ---------------------------------------------------------------------------------------------------------------- #

    # Make the standardisation of every numerical columns

    rera_standerdized_encoded = rera.copy()

    # Selecting only the numerical columns from the DataFrame
    numerical_columns = rera_standerdized_encoded.select_dtypes(include=['float64', 'int64'])
    scaler            = StandardScaler()
    rera_scaled       = scaler.fit_transform(numerical_columns)
    rera_scaled_df    = pd.DataFrame(rera_scaled, columns=numerical_columns.columns)

    rera_standerdized_encoded.update(rera_scaled_df)
    rera_standerdized_encoded.head()

    # ---------------------------------------------------------------------------------------------------------------- #
    # 4.3.2. Encoding                                                                                                  #
    # ---------------------------------------------------------------------------------------------------------------- #
    # For the PCA to be efficient, we need to drop all categorical features because it relies on the variation of      #
    # numerical data. Therefore, there is no need for encoding in our particular case.                                 #
    # ---------------------------------------------------------------------------------------------------------------- #

    columns_to_drop = ['home_team', 'away_team', 'confederation_home', 'confederation_away']
    rera_standerdized_encoded.drop(columns_to_drop, axis = 1, inplace=True)

    # ---------------------------------------------------------------------------------------------------------------- #
    # 4.3.3. UpSampling minority class                                                                                 #
    # ---------------------------------------------------------------------------------------------------------------- #
    # As observed, there is a variation in the distribution of values in the ‘winner’ column. They are not equally     #
    # represented, which could introduce bias to the model.                                                            #
    # ---------------------------------------------------------------------------------------------------------------- #

    # La classe minoritaire (Home team a gagne)
    minority = rera[rera["winner"] == 1]
    majority = rera[rera["winner"] == 0]

    minority_upsampled = resample(minority, replace=True, n_samples=len(majority))
    rera = pd.concat([majority, minority_upsampled], axis=0)

    # ---------------------------------------------------------------------------------------------------------------- #
    # 4.3.4. Dimensionality Reduction: Principal Component Analysis                                                    #
    # ---------------------------------------------------------------------------------------------------------------- #
    # We will apply dimensionality reduction through the Principal Component Analysis (PCA) algorithm. In this case we #
    # are utilizing it on our encoded dataframe rera_standerdized_encoded to determine the optimal value for the       #
    # hyperparameter n_component. This identified value will subsequently be employed in the model notebook’s pipeline #
    # facilitating the automated encoding and application of dimensionality reduction.                                 #
    # ---------------------------------------------------------------------------------------------------------------- #

    rera_standerdized_encoded = rera_standerdized_encoded.drop("date", axis = 1)

    pcN = PCA()
    pcN.fit(rera_standerdized_encoded)

    plt.figure(figsize=(10, 8))
    plt.title("Principal Component Analysis")
    plt.plot(np.cumsum(pcN.explained_variance_ratio_))
    plt.legend('Explained Variance')
    plt.xlabel('Number of Components')
    plt.ylabel('Explained Variance Ratio')
    plt.show()

    pc = PCA(n_components=5)
    pc.fit(rera_standerdized_encoded)
    pc.transform(rera_standerdized_encoded)

    # ---------------------------------------------------------------------------------------------------------------- #
    # 6. Exporting data set                                                                                            #
    # ---------------------------------------------------------------------------------------------------------------- #

    rera.head()

    dump(rera, "../assets/rera.joblib")

# -------------------------------------------------------------------------------------------------------------------- #
# Main()                                                                                                               #
# -------------------------------------------------------------------------------------------------------------------- #
def __main__():

    preprocessing()

if __name__ == "__main__":
    __main__()