

import pandas as pd
from sklearn.preprocessing import StandardScaler

#loading 'Give Me Some Credit' dataset training data
def load_data(path="data/GiveMeSomeCredit-training.csv"):
    df = pd.read_csv(path, index_col=0)

    # renaming the target 
    df = df.rename(columns={"SeriousDlqin2yrs": "Target"})
    return df


# deviding the input and the target , encoding the data , normalization
def preprocess(df):

    #handling the missing values 
    df['MonthlyIncome'] = df['MonthlyIncome'].fillna(df['MonthlyIncome'].median())
    df['NumberOfDependents'] = df['NumberOfDependents'].fillna(df['NumberOfDependents'].mode()[0])

    #deviding the target into input and predict value
    X = df.drop("Target", axis=1)
    y = df["Target"]

    # one-hot encoding
    X_encoded = pd.get_dummies(X, drop_first=True)

    # normalization 
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_encoded)

    return X_scaled, y,  X_encoded.columns.tolist()
