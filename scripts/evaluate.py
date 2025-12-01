import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate
import joblib
import json
import yaml
import os

# Model evaluation
def evaluate_model():
    # Load hyperparameters
    with open('params.yaml', 'r') as fd:
        params = yaml.safe_load(fd)
    # Load data
    data = pd.read_csv('data/initial_data.csv')
    # Load model
    with open('models/fitted_model.pkl', 'rb') as fd:
        pipeline = joblib.load(fd)
        
    # Cross validation
    cv_strategy = StratifiedKFold(n_splits=params['n_splits'])
    cv_res = cross_validate(
        pipeline,
        data,
        data[params['target_col']],
        cv=cv_strategy,
        n_jobs=params['n_jobs'],
        scoring=params['metrics']
    )
    
    # Prepare for saving
    for key, value in cv_res.items():
        cv_res[key] = round(value.mean(), 3)
    
    # Save results
    os.makedirs('cv_results', exist_ok=True)
    with open('cv_results/cv_res.json', 'w') as fd:
        json.dump(cv_res, fd)
        
# Run main function
if __name__ == '__main__':
    evaluate_model()