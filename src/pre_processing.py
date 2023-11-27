# import library
import pandas            as pd
import numpy             as np
import seaborn           as sns
import matplotlib.pyplot as plt
import sklearn

print("The preprocessing can take up to 15mn. Please be patient :)")

results = pd.read_csv("../data/results.csv")
ranking = pd.read_csv("../data/fifa_ranking-2023-07-20.csv")

results.head(5)

results.info()

ranking.head(5)

ranking.info()

# the JSON file converted to a dictionary returned by CHATGPT-4

false_countries = {
"Czechoslovakia": "Czech Republic", "Catalonia": "Spain","Basque Country": "Spain","Brittany": "France","Yugoslavia": "Serbia","Galicia": "Spain","Andalusia": "Spain","Central Spain": "Spain","Silesia": "Poland","Manchukuo": "China","German DR": "Germany","Kernow": "United Kingdom","Saarland": "Germany", "Vietnam Republic": "Vietnam","North Vietnam": "Vietnam","Zanzibar": "Tanzania","Eswatini": "Swaziland","Western Australia": "Australia","Northern Cyprus": "Cyprus","Yemen DPR": "Yemen","Ynys Môn": "United Kingdom","Shetland": "United Kingdom","Isle of Wight": "United Kingdom","Canary Islands": "Spain",
"Frøya": "Norway","Hitra": "Norway","Corsica": "France","Saare County": "Estonia","Rhodes": "Greece","Orkney": "United Kingdom","Sark": "United Kingdom","Alderney": "United Kingdom","Western Isles": "United Kingdom","Falkland Islands": "United Kingdom","Republic of St. Pauli": "Germany","Găgăuzia": "Moldova", "Tibet": "China","Occitania": "France","Sápmi": "Norway","Menorca": "Spain","Provence": "France","Arameans Suryoye": "Syria","Padania": "Italy","Iraqi Kurdistan": "Iraq","Gozo": "Malta","Bonaire": "Netherlands","Western Sahara": "Morocco","Raetia": "Switzerland","Darfur": "Sudan","Tamil Eelam": "Sri Lanka", "Abkhazia": "Georgia","Madrid": "Spain","Ellan Vannin": "Isle of Man","South Ossetia": "Georgia","County of Nice": "France","Székely Land": "Romania","Romani people": "India","Felvidék": "Slovakia","Chagos Islands": "United Kingdom","United Koreans in Japan": "North Korea","Somaliland": "Somalia",
"Western Armenia": "Turkey","Barawa": "Somalia","Kárpátalja": "Ukraine","Yorkshire": "United Kingdom","Matabeleland": "Zimbabwe","Cascadia": "United States","Kabylia": "Algeria","Parishes of Jersey": "Jersey","Chameria": "Albania","Yoruba Nation": "Nigeria","Biafra": "Nigeria","Mapuche": "Chile", "Aymara": "Bolivia","Ticino": "Switzerland","Hmong": "China"
}

# Calculate and display the number of unique countries before and after the replacement

num_unique_countries = len(set(results["home_team"].unique()) | set(results["away_team"].unique()))

results.replace(false_countries, inplace=True)

num_unique_countries = len(set(results["home_team"].unique()) | set(results["away_team"].unique()))

# JSON converted to dictionary that CHATGPT-4 returned with the previous prompt.
false_countries = {
"Korea DPR": "North Korea", "Cape Verde Islands": "Cape Verde","Brunei Darussalam": "Brunei","USA": "United States","Côte d'Ivoire": "Ivory Coast","Yugoslavia": "Serbia","Czechoslovakia": "Czech Republic","Zaire": "Democratic Republic of the Congo","IR Iran": "Iran","China PR": "China", "Trinidad and Tobago": "Trinidad","Korea Republic": "South Korea","FYR Macedonia": "North Macedonia","Kyrgyz Republic": "Kyrgyzstan","Chinese Taipei": "Taiwan","Serbia and Montenegro": "Serbia","Swaziland": "Eswatini","St. Vincent / Grenadines": "St. Vincent and the Grenadines", "Timor-Leste": "East Timor","North Macedonia": "Macedonia","São Tomé e Príncipe": "Sao Tome and Principe","Curaçao": "Curacao","Cabo Verde": "Cape Verde","Czechia": "Czech Republic","Türkiye": "Turkey","St Vincent and the Grenadines": "St. Vincent and the Grenadines", "St Lucia": "St. Lucia","The Gambia": "Gambia","St Kitts and Nevis": "St. Kitts and Nevis","Hong Kong, China": "Hong Kong","Aotearoa New Zealand": "New Zealand"
}

# Calculate and display the number of unique countries before and after the replacement

num_unique_countries = len(ranking["country_full"].unique())

ranking.replace(false_countries, inplace=True)

num_unique_countries = len(ranking["country_full"].unique())

# Convert "date columns"  of datasets to date time type

results['date']         = pd.to_datetime(results['date'], format='%Y-%m-%d') 
ranking['rank_date']    = pd.to_datetime(ranking['rank_date'], format='%Y-%m-%d') 

results = results[results['date'].dt.year >= 2000]
ranking = ranking[ranking['rank_date'].dt.year >= 2000]

# Ensure that the result is as expected

results.set_index(['date'], inplace=True)
ranking.set_index(['rank_date', 'country_full'], inplace=True)



# Grouping by date and country name
grouped_ranking = ranking.groupby(by=['rank_date', 'country_full']).ffill()
grouped_ranking

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


# Reset datasets' indexs for the merging
sampled_ranking.reset_index(inplace=True)
results.reset_index(inplace=True)

# Merging results with sampled_ranking on home team
sampled_ranking.columns = [col + "_home" if col != "rank_date" and col != "country_full" 
                                         else col for col in sampled_ranking.columns]
rera                    = results.merge(sampled_ranking, 
                                        left_on     =["date", "home_team"], 
                                        right_on    =["rank_date", "country_full"], 
                                        suffixes    =[None,'_home']).drop(["rank_date","country_full", "country_abrv_home"],
                                        axis        =1)

# Merging results with sampled_ranking on away team
sampled_ranking.columns = [col.replace("_home", "_away") for col in sampled_ranking.columns]
rera                    = rera.merge(sampled_ranking, 
                                     left_on   =["date", "away_team"], 
                                     right_on  =["rank_date", "country_full"], 
                                     suffixes  =[None,'_away']).drop(["rank_date","country_full", "country_abrv_away"],
                                     axis      =1)

rera.sort_values(by='date', inplace=True)
rera.head()

rera.info()

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


rera = rera.drop(rera[(rera['tournament'] == "Friendly") & (rera['winner'] == 2)].index)


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
            #print('SHOOTOUT', shootout['date'].values[0], shootout['home_team'].values[0], shootout['away_team'].values[0], shootout['winner'].values[0])
            
            if shootout['winner'].values[0] == row['home_team']:
                rera.loc[index, 'winner'] = 0
            else:
                rera.loc[index, 'winner'] = 1


rera = rera.drop(rera[rera['winner'] == 2].index)


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
        goals = previous_matches.apply(lambda row: row["home_score"] if (row["home_team"] == country) else row["away_score"], axis=1).sum()
        return goals / len(previous_matches)
    
    
    # If there is 7 matches, we compute the average on the 7 matches
    else:
        goals = previous_matches.apply(lambda row: row["home_score"] if (row["home_team"] == country) else row["away_score"], axis=1).sum()
        return goals / 7
    
# Create home_win_avg and away_win_avg columns and defining the average win of each team

rera['home_goals_avg'] = rera.apply(lambda line : compute_avg_country_goals(line, line['home_team']), axis=1)
rera['away_goals_avg'] = rera.apply(lambda line : compute_avg_country_goals(line, line['away_team']), axis=1)

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
        wins = previous_matches.apply(lambda row: 1 if ((row["home_team"] == country) & (row["winner"] == 0)) | ((row["away_team"] == country) & (row["winner"] == 1)) else 0, axis=1).sum()
        return wins / len(previous_matches)
    
    # If there is 7 matches, we compute the average on the 7 matches
    else:
        wins = previous_matches.apply(lambda row: 1 if ((row["home_team"] == country) & (row["winner"] == 0)) | ((row["away_team"] == country) & (row["winner"] == 1)) else 0, axis=1).sum()
        return wins / 7
    

# Create home_win_avg and away_win_avg columns and defining the average win of each team

rera['home_win_avg'] = rera.apply(lambda line : compute_avg_country_win(line, line['home_team']), axis=1)
rera['away_win_avg'] = rera.apply(lambda line : compute_avg_country_win(line, line['away_team']), axis=1)

def compute_wins_direct_confrontation(row, country, opponent):
    date = row.date
    # All direct confrontaions between the 2 countries
    confrontations = rera[((rera["home_team"] == country) & (rera["away_team"] == opponent)) |  ((rera["home_team"] == opponent) & (rera["away_team"] == country))]
    # 5 last confrontations
    previous_matches = confrontations[confrontations["date"] < date].tail(5)

    if len(previous_matches) == 0:
        return 0

    # Get sum of wins for country against opponent
    nb_wins_country = previous_matches.apply(lambda row: 1 if ((row["home_team"] == country) & (row["winner"] == 0)) | ((row["away_team"] == country) & (row["winner"] == 1)) else 0, axis=1).sum()
    return nb_wins_country

rera["home_last_wins"] = rera.apply(lambda row: compute_wins_direct_confrontation(row, row["home_team"], row["away_team"]), axis=1)
rera["away_last_wins"] = rera.apply(lambda row: compute_wins_direct_confrontation(row, row["away_team"], row["home_team"]), axis=1)

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

rera.drop("city", axis=1, inplace=True)

rera.drop(["home_score", "away_score"], axis=1, inplace=True)

rera.drop("country", axis=1, inplace=True)

rera.drop(["previous_points_home", "previous_points_away"], axis=1, inplace=True)

rera.drop(["rank_change_home", "rank_change_away"], axis = 1, inplace=True)

rera.drop("tournament", axis = 1, inplace=True)

# Make the standardisation of every numerical columns
from sklearn.preprocessing import StandardScaler

rera_standerdized_encoded = rera.copy()

# Selecting only the numerical columns from the DataFrame
numerical_columns = rera_standerdized_encoded.select_dtypes(include=['float64', 'int64'])
scaler            = StandardScaler()
rera_scaled       = scaler.fit_transform(numerical_columns) 
rera_scaled_df    = pd.DataFrame(rera_scaled, columns=numerical_columns.columns) 

rera_standerdized_encoded.update(rera_scaled_df)
rera_standerdized_encoded.head()

columns_to_drop = ['home_team', 'away_team', 'confederation_home', 'confederation_away']
rera_standerdized_encoded.drop(columns_to_drop, axis = 1, inplace=True)



from sklearn.utils import resample 

# La classe minoritaire (Home team a gagne)
minority = rera[rera["winner"] == 1]
majority = rera[rera["winner"] == 0]

minority_upsampled = resample(minority, replace=True, n_samples=len(majority))
rera = pd.concat([majority, minority_upsampled], axis=0)


from sklearn.decomposition import PCA

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

rera.head()

from joblib import dump

dump(rera, "../assets/rera.joblib")