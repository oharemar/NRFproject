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

    if dataset_name == 'valley':
        df = pd.read_csv('hill_valey.csv', delimiter=",")
        df = df.sample(frac=1)
        df.loc[df['Class'] == 1, 'Class'] = 0
        df.loc[df['Class'] == 2, 'Class'] = 1
        y = df.loc[:, 'Class']
        df = df.loc[:, df.columns != 'Class']
        scaler = StandardScaler()
        df = scaler.fit_transform(df)
        df = np.array(df, dtype=np.float64)
        y = np.array(y, dtype=np.int64)
        return df,y
    if dataset_name == 'valley_no_scale':
        df = pd.read_csv('hill_valey.csv', delimiter=",")
        df = df.sample(frac=1)
        df.loc[df['Class'] == 1, 'Class'] = 0
        df.loc[df['Class'] == 2, 'Class'] = 1
        y = df.loc[:, 'Class']
        df = df.loc[:, df.columns != 'Class']
        scaler = StandardScaler()
        #df = scaler.fit_transform(df)
        df = np.array(df, dtype=np.float64)
        y = np.array(y, dtype=np.int64)
        return df,y

    if dataset_name == 'OBSnetwork':
        df = pd.read_csv('OBSnetwork.arff', delimiter=",", header=None)
        df = df.sample(frac=1)
        y = df.loc[:, 21]
        le = LabelEncoder()  # encoder
        y = le.fit_transform(y)
        df = df.loc[:, df.columns != 21]
        df = pd.get_dummies(df)  # one hot encoding categorical variables
        scaler = StandardScaler()
        df = scaler.fit_transform(df)
        df = np.array(df, dtype=np.float64)
        y = np.array(y)
        return df,y

    if dataset_name == 'messidor':
        df = pd.read_csv('messidor.arff', delimiter=",", header=None)
        df = df.sample(frac=1)
        y = df.loc[:, 19]
        df = df.loc[:, df.columns != 19]
        scaler = StandardScaler()
        df = scaler.fit_transform(df)
        df = np.array(df, dtype=np.float64)
        y = np.array(y, dtype=np.int64)
        return  df,y

    if dataset_name == 'drive diagnosis':
        df = pd.read_csv('Sensorless_drive_diagnosis.txt', delimiter=" ", header=None)
        df = df.sample(frac=1)
        y = df.loc[:, 48]
        df = df.loc[:, df.columns != 48]
        scaler = StandardScaler()
        df = scaler.fit_transform(df)
        df = np.array(df, dtype=np.float64)
        y = np.array(y)
        return df,y

    if dataset_name == 'wine':
        from sklearn.datasets import load_wine
        dataset = load_wine()
        scaler = StandardScaler()
        df = pd.DataFrame(scaler.fit_transform(dataset.data))
        #df = pd.DataFrame(dataset.data) # musíme pak využít tuto možnost, jestli chceme porovnávat s jinou studíí, oni totiž nescalovali

        y = pd.DataFrame(dataset.target)
        dataset = pd.DataFrame(pd.concat([df, y], axis=1).sample(frac=1).values)
        y = np.array(dataset.loc[:, 13])
        df = np.array(dataset.loc[:, dataset.columns != 13], dtype=np.float64)
        return  df,y

    if dataset_name == 'wine_no_scale':
        from sklearn.datasets import load_wine
        dataset = load_wine()
        scaler = StandardScaler()
        #df = pd.DataFrame(scaler.fit_transform(dataset.data))
        df = pd.DataFrame(dataset.data) # musíme pak využít tuto možnost, jestli chceme porovnávat s jinou studíí, oni totiž nescalovali

        y = pd.DataFrame(dataset.target)
        dataset = pd.DataFrame(pd.concat([df, y], axis=1).sample(frac=1).values)
        y = np.array(dataset.loc[:, 13])
        df = np.array(dataset.loc[:, dataset.columns != 13], dtype=np.float64)
        return  df,y

    if dataset_name == 'USPS':
        import h5py
        with h5py.File('usps.h5', 'r') as hf:
            train = hf.get('train')
            X_tr = train.get('data')[:]
            y_tr = train.get('target')[:]
            test = hf.get('test')
            X_te = test.get('data')[:]
            y_te = test.get('target')[:]
        scaler = StandardScaler()
        df = np.concatenate((X_tr, X_te), axis=0)
        y = np.concatenate((y_tr, y_te), axis=0)
        df = scaler.fit_transform(df)
        df = np.array(df, dtype=np.float64)
        return df,y
    if dataset_name == 'USPS_no_scale':
        import h5py
        with h5py.File('usps.h5', 'r') as hf:
            train = hf.get('train')
            X_tr = train.get('data')[:]
            y_tr = train.get('target')[:]
            test = hf.get('test')
            X_te = test.get('data')[:]
            y_te = test.get('target')[:]
        scaler = StandardScaler()
        df = np.concatenate((X_tr, X_te), axis=0)
        y = np.concatenate((y_tr, y_te), axis=0)
        #df = scaler.fit_transform(df)
        df = np.array(df, dtype=np.float64)
        return df, y

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