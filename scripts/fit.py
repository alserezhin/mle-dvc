import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from category_encoders import CatBoostEncoder
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from catboost import CatBoostClassifier
import yaml
import os
import joblib

# Model fitting
def fit_model():
    # Load hyperparameters
    with open('params.yaml', 'r') as fd:
        params = yaml.safe_load(fd)
    # Load data from previous step
    data = pd.read_csv('data/initial_data.csv')
    # Feature preprocessing
    cat_features = data.select_dtypes(include='object')
    potential_binary_features = cat_features.nunique() == 2
    
    binary_cat_features = cat_features[potential_binary_features[potential_binary_features].index]
    other_cat_features = cat_features[potential_binary_features[~potential_binary_features].index]
    num_features = data.select_dtypes(include='float')
    # Create preprocessor
    preprocessor = ColumnTransformer(
        [
            ('binary', OneHotEncoder(drop=params['one_hot_drop']), binary_cat_features.columns.tolist()),
            ('cat', CatBoostEncoder(return_df=False), other_cat_features.columns.tolist()),
            ('num', StandardScaler(), num_features.columns.tolist())
        ],
        remainder='drop',
        verbose_feature_names_out=False
    )
    # Create model
    model = CatBoostClassifier(auto_class_weights=params['auto_class_weights'])
    # Create pipeline
    pipeline = Pipeline(
        [
            ('preprocessor', preprocessor),
            ('model', model)
        ]
    )
    # Fit
    pipeline.fit(data, data[params['target_col']])
    
    # Save model
    os.makedirs('models', exist_ok=True)
    with open('models/fitted_model.pkl', 'wb') as fd:
        joblib.dump(pipeline, fd)
        
# Run main function
if __name__ == '__main__':
    fit_model()