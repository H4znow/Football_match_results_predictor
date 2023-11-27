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