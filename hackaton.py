import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn import linear_model, neighbors
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn import preprocessing
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR



def read_csv(filename):
    return pd.read_csv(filename)


def knn(X_train, X_test, y_train, y_test):
    rmse_val = {} #to store rmse values for different k
    for k in range(1, 20):
        model = neighbors.KNeighborsRegressor(n_neighbors = k)
        model.fit(X_train, y_train) 
        y_pred = model.predict(X_test) 
        error = metrics.mean_squared_error(y_test, y_pred) #calculate rmse
        rmse_val[k] = error #store rmse values
        #print('RMSE value for k= ' , K , 'is:', error)
    #plt.figure(figsize=(10,6))
    plt.plot(rmse_val.keys(), rmse_val.values(), color = "blue", linestyle = "dashed", marker ='o', markerfacecolor = "red",  markersize=10)
    plt.title("Error Rate vs. K Value")
    plt.xlabel('K')
    plt.ylabel("Error Rate")
    plt.show()
    plt.savefig('elbow.png')
    

def choose_model(algo):
    '''
    Input: name of the algorithm
    Output: model
    Purpose: choose a regression algorithm
    '''
    if algo == "lr":
        return linear_model.LinearRegression()
    elif algo == "knn":
        return neighbors.KNeighborsRegressor(n_neighbors = 6)
    elif algo == "rf":
        return RandomForestRegressor(n_estimators = 100, random_state = 0)
    elif algo == "svr":
        return SVR(kernel = "rbf")

def evaluate_performance(model, X_train_scaled, X_test_scaled, y_train, y_test):

    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    print("RSME =", np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
    print("MAE =", metrics.mean_absolute_error(y_test, y_pred))
    print("CCC =", np.corrcoef(y_test, y_pred)[0, 1] * (2.0 * np.std(y_test) * np.std(y_pred)) / (np.var(y_test) + np.var(y_pred) + (np.mean(y_test) - np.mean(y_pred))**2))
    # Calculate pearson correlation coefficient
    correlations = []
    for idx, _ in enumerate(y_test):
        corr, _ = pearsonr(pd.DataFrame(y_test).iloc[:, idx], pd.DataFrame(y_pred).iloc[:, idx])
        correlations.append(corr)
    # calculate the mean of the correlation coefficients
    mean_corr = sum(correlations) / len(correlations)
    print("PCC =", mean_corr)

def process_data(df):
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
    # Return data
    return X_train_scaled, X_test_scaled, y_train, y_test


    


def main():
    #filename, algo = sys.argv[-2:]
    filename = "/home/jrn1325/spring_hackaton/bio_features.csv"
    algo = "knn"
    df = read_csv(filename)
    X_train, X_test, y_train, y_test = process_data(df)
    #if algo == "knn":
    #    knn(X_train, X_test, y_train, y_test)
    model = choose_model(algo)
    evaluate_performance(model, X_train, X_test, y_train, y_test)


if __name__ == "__main__":
    main()