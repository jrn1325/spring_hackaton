import sys

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn import linear_model, neighbors
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR



def read_csv(filename):
    return pd.read_csv(filename)

def choose_model(algo):
    '''
    Input: name of the algorithm
    Output: model
    Purpose: choose a regression algorithm
    '''
    if algo == "lr":
        model = linear_model.LinearRegression()
    elif algo == "knn":
        model = neighbors.KNeighborsRegressor(n_neighbors = 2)
        '''
        rmse_val = [] #to store rmse values for different k
        for k in range(1, 20):
            model = neighbors.KNeighborsRegressor(n_neighbors = k)
            model.fit(x_train, y_train) 
            pred = model.predict(x_test) 
            error = sqrt(mean_squared_error(y_test, pred)) #calculate rmse
            rmse_val.append(error) #store rmse values
            print('RMSE value for k= ' , K , 'is:', error)
        '''
    elif algo == "rf":
        model = RandomForestRegressor(n_estimators = 100, random_state = 0)
    elif algo == "svr":
        model = SVR(kernel = "rbf")
    return model


def perform_regression(df, model):
    X = df[['ECG_Rate', 'HRV_RMSSD', 'HRV_MeanNN', 'HRV_SDNN', 'HRV_SDSD',
            'EDA_Phasic_Mean', 'EDA_Phasic_Std', 'EDA_Tonic_Mean', 'EDA_Tonic_std',
            'SCR_Onsets', 'SCR_Magnitude', 'SCR_Amplitude_Mean',
            'SCR_RiseTime_Mean', 'SCR_RecoveryTime_Mean', 'Pupil_Mean', 'Pupil_Std']]
    y = df[['comfort', 'surprise', 'anxiety', 'calmness', 'boredom']]

    # Define scaler
    scaler = MinMaxScaler()
    # Split the dataset in train and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state=2)
    # Standardize data
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    print("RSME =", metrics.mean_squared_error(y_test, y_pred))
    print("MAE =", metrics.mean_absolute_error(y_test, y_pred))
    print("CCC =", np.corrcoef(y_test, y_pred)[0, 1] * (2.0 * np.std(y_test) * np.std(y_pred)) / (np.var(y_test) + np.var(y_pred) + (np.mean(y_test) - np.mean(y_pred))**2))
    # Calculate pearson correlation coefficient
    correlations = []
    for idx, column in enumerate(y_test):
        corr, _ = pearsonr(pd.DataFrame(y_test).iloc[:, idx], pd.DataFrame(y_pred).iloc[:, idx])
        correlations.append(corr)

    # calculate the mean of the correlation coefficients
    mean_corr = sum(correlations) / len(correlations)
    print("PCC =", mean_corr)


def main():
    #filename, algo = sys.argv[-2:]
    filename = "/home/jrn1325/spring_hackaton/bio_features.csv"
    algo = "lr"
    df = read_csv(filename)
    model = choose_model(algo)
    perform_regression(df, model)


if __name__ == "__main__":
    main()