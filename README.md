# Football match results predictor

The purpose of this project is to train models that are able to predict football match results between 2 teams.

This repository contains:
- Models and preprocessed data exported to joblib format: [assets](./assets/) directory.
- Datasets used to train the models: [data](./data/) directory.
- All necesary notebooks and scripts to perform data preprocessing, models training and evaluation: [notebooks](./notebooks/) and [src](./src/) directories.
- Images confusion matrices, learning curves and other analysis plots: [img](./img/) directory.
- A Flask application integrating a trained model able to predict match results.  
  
We recommand to have a look at the notebooks to understand well what we did, as there are a lot of comments inside.  
  
## Prepare the environment
Necesary dependencies are listed in [Pipfile](./Pipfile) file.  
If you are sure to have all the dependencies installed in your computer or your own virtual environment, you can skip this section.  
  
First, install pipenv if not already installed:  
```
pip install pipenv
```
Then, create a virtual environment and install all the needed dependencies:
```
pipenv install
```
Finally, enter into the virtual environment:
```
pipenv shell
```
You are now ready to execute scripts.

## Models training

To train a model, execute the [train_clf](./src/train_clf.py) script.  
After executing the script, the model is exported and available in joblib format in the [assets](./assets/) directory.  
     
This script takes one argument, corresponding to the type of model you want to train:
- "rf" for Random Forest
- "mlpc" for Multi Layers Perceptron Classifier

Go to src directory:
```
cd src
```
Execute the script:
```
python3 train_clf.py rf
```

## Models evaluation

To evaluate a model, execute the [eval_clf](./src/eval_clf.py) script.  
After executing the script, the confusion matrix is saved in [img](./img/confusion_matrices/) directory, and the accuracy score is printed.
     
This script takes one argument, corresponding to the type of model you want to train:
- "rf" for Random Forest
- "mlpc" for Multi Layers Perceptron Classifier

Go to src directory:
```
cd src
```
Execute the script:
```
python3 eval_clf.py mlpc
```

## Match result prediction

To predict the result of a match, execute the [predict_match_result](./src/predict_match_result.py) script.  
  
This script takes 4 arguments: 
- The home team
- The away team
- Neutral: true or false
- The model type: rf or mlpc
  
Go to src directory:

```
cd src
```
Execute the script:
```
python3 predict_match_result.py Morocco Brazil true rf
```
The result is a draw if probabilities are between 0.45 and 0.55, else the winner is printed.

## Championship result prediction

To predict the result of a Championship, execute the [predict_championship](./src/predict_championship.py) script.  
  
This script takes two arguments: 
- The filepath of the file containing the list of team groups : ../data/championship.csv
- The model type : rf or mlpc

Go to src directory:

```
cd src
```
Execute the script:
```
python3 predict_championship.py ../data/championship.csv rf
```

## Data preprocessing script

Although it's not required, if you want to re-run all the tasks necessary for the preprocessing of the dataset, please execute the [pre_processing](./src/pre_processing.py) script.  
  
The script takes no arguments, and uses the datasets located in the [/data](./data) directory, [results.csv](./data/results.csv)  and [fifa_ranking-2023-07-20.csv](./data/fifa_ranking-2023-07-20.csv).

Go to src directory:

```
cd src
```
Execute the script:
```
python3 pre_processing.py
```

The preprocessing can take up to 15 minutes.

## Run the User Interface

The User Interface is a website locally hosted using the `Flask` library in Python.

The application and all the files necessary to run it correctly are located in [app](./app/). In this directory, you will find the following subdirectories:

- [assets](./app/assets/): Contains the models used in the UI.
- [data](./app/data/): Contains the data used in the file, specifically for the tournament.
- [static](./app/static/): Contains the CSS file.
- [templates](./app/templates/): Contains the UI templates, including the welcome page template, match predictor page template, and tournament predictor template.
- [app.py](./app/app.py): The main function of the UI.
- [predict_match_result.py](./app/predict_match_result.py) and [predict_championship.py](./app/predict_championship.py): Contain the necessary modules and functions to predict the winner of a match or tournament.

To run the UI, navigate to the app directory:

```bash
cd app
```
Then run the UI

```bash
python3 app.py
```

