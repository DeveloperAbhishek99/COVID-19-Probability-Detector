import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import pickle

def data_split(data,ratio):      
    np.random.seed(42)          
    shuffled = np.random.permutation(len(data))
    test_set_size = int(len(data) * ratio)
    test_indices = shuffled[:test_set_size]
    train_indices = shuffled[test_set_size:]
    
    return data.iloc[train_indices], data.iloc[test_indices]

if __name__ =="__main__":

    # Read the data
    df = pd.read_csv('data.csv')
    train, test = data_split(df, 0.3)
    X_train = train[['fever','bodyPain','age','runnyNose','diffBreath']].values
    X_test = test[['fever','bodyPain','age','runnyNose','diffBreath']].values

    Y_train = train[['infectionProb']].values.reshape(1750,)
    Y_test = test[['infectionProb']].values.reshape(749,)

    clf = LogisticRegression()
    clf.fit(X_train, Y_train)

    # opens a file where you want to store the data
    file = open('model.pkl','wb')

    #dump information into that file
    pickle.dump(clf, file)
    file.close()



    

