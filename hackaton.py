import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn import metrics
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split

def read_csv(filename):
    return pd.read_csv(filename)


def linear_regression(df):
    model = linear_model.LinearRegression()
    X = df[['ECG_Rate', 'HRV_RMSSD', 'HRV_MeanNN', 'HRV_SDNN', 'HRV_SDSD',
            'EDA_Phasic_Mean', 'EDA_Phasic_Std', 'EDA_Tonic_Mean', 'EDA_Tonic_std',
            'SCR_Onsets', 'SCR_Magnitude', 'SCR_Amplitude_Mean',
            'SCR_RiseTime_Mean', 'SCR_RecoveryTime_Mean', 'Pupil_Mean', 'Pupil_Std']]
    y = df[['comfort', 'surprise', 'anxiety', 'calmness', 'boredom']]

    # Split the dataset in train and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state=2)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("RSME =", metrics.mean_squared_error(y_test, y_pred))
    print()

def main():
    filename = "/home/jrn1325/spring_hackaton/bio_features.csv"
    df = read_csv(filename)
    linear_regression(df)


if __name__ == "__main__":
    main()