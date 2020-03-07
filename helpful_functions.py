import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler



def load_datasets(dataset_name):
    if dataset_name == 'bank_marketing':
        le = LabelEncoder()  # encoder
        # pd.set_option("display.max_columns", 999)
        df = pd.read_csv('bank_marketing_dataset.csv')
        df = df.sample(frac=1)
        y = df.loc[:, 'y']
        df = df.loc[:, df.columns != 'y']
        y = le.fit_transform(y)
        # print(df.describe(include = object)) #statistics of dataset (omit @param:include to include numerical variables)
        df.drop(columns=['default'], inplace=True)
        df = pd.get_dummies(df)  # one hot encoding categorical variables
        scaler = StandardScaler()
        df = scaler.fit_transform(df)
        df = np.array(df, dtype=np.float64)
        y = np.array(y)
        return df,y

    if dataset_name == 'cars':
        df = pd.read_csv("cars_dataset.csv")
        df = df.sample(frac=1)
        y = df.loc[:, 'car']
        df = df.loc[:, df.columns != 'car']
        le = LabelEncoder()  # encoder
        y = le.fit_transform(y)
        df = pd.get_dummies(df)  # one hot encoding categorical variables
        scaler = StandardScaler()
        df = scaler.fit_transform(df)
        df = np.array(df, dtype=np.float64)
        y = np.array(y)
        return df,y

    if dataset_name == 'vehicle_silhouette':
        df = pd.read_csv("vehicle_silhouette_dataset.csv")
        df = df.sample(frac=1)
        y = df.loc[:,'vehicle_class']
        df = df.loc[:,df.columns != 'vehicle_class']
        le = LabelEncoder()
        y = le.fit_transform(y)
        scaler = StandardScaler()
        df = scaler.fit_transform(df)
        df = np.array(df, dtype=np.float64)
        y = np.array(y)
        return df,y

    if dataset_name == 'diabetes':
        df = pd.read_csv("diabetes.csv")
        df = df.sample(frac=1)
        y = df.loc[:, 'Outcome']
        df = df.loc[:, df.columns != 'Outcome']
        scaler = StandardScaler()
        df = scaler.fit_transform(df)
        df = np.array(df, dtype=np.float64)
        y = np.array(y)
        return df,y
    else:
        print('No such dataset!!!')
        raise ValueError

def most_common_value(array):
    unique, counts = np.unique(array,return_counts=True)
    max_count = 0
    max_count_class = 0
    unique_counts = dict(zip(unique, counts))
    for cls in unique_counts.keys():
        if unique_counts[cls] >= max_count:
            max_count = unique_counts[cls]
            max_count_class = cls
    return max_count_class