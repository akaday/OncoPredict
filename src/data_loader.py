import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

def load_data(filepath):
    data = pd.read_csv(filepath)
    return data

def clean_data(data):
    # Handle missing values
    imputer = SimpleImputer(strategy='mean')
    data = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)
    
    # Handle outliers (example: capping at 1st and 99th percentiles)
    for col in data.select_dtypes(include=['float64', 'int64']).columns:
        lower_bound = data[col].quantile(0.01)
        upper_bound = data[col].quantile(0.99)
        data[col] = data[col].clip(lower_bound, upper_bound)
    
    return data

def preprocess_data(data):
    # Separate features and target
    X = data.drop('target', axis=1)
    y = data['target']
    
    # Define preprocessing for numeric features
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())])
    
    # Define preprocessing for categorical features
    categorical_features = X.select_dtypes(include=['object']).columns
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))])
    
    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)])
    
    # Apply preprocessing
    X = preprocessor.fit_transform(X)
    
    return X, y
