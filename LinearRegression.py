import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from argparse import Namespace

def Init_nameSpace():
    # 参数
    args = Namespace(
        seed=1234,
        data_file="sample_data.csv",
        num_samples=100,
        train_size=0.75,
        test_size=0.25,
        num_epochs=100,
    )

    # 设置随机种子来保证结果可复现
    np.random.seed(args.seed)
    
    return args

def CreateDataSet(num_samples):
    data_X = pd.Series(range(num_samples))
    random_noise = np.random.uniform(-10,10,size=num_samples)
    data_X += random_noise
    
    data_Y = 3.65*data_X + 10 + random_noise # add some noise

    print(data_X,data_Y)
    print("X data type :")

    data_X[0:50].to_csv('studentscores.csv')
    dataset = pd.read_csv('studentscores.csv')
    X = dataset.iloc[ : , :1 ].values
    Y = dataset.iloc[ : , 1 ].values
    X = data_X.iloc[ : , 1 ].values
    print(X)
    pass
    return dataset

def ReadDataSet(parameter_list):
    dataset = pd.read_csv('studentscores.csv')
    X = dataset.iloc[ : ,   : 1 ].values
    Y = dataset.iloc[ : , 1 ].values
    pass

if __name__ == "__main__":
    args = Init_nameSpace()
    dataSet = CreateDataSet(args.num_samples)
    pass

if __name__ != "__main__":
    dataset = pd.read_csv('studentscores.csv')
    X = dataset.iloc[ : ,   : 1 ].values
    Y = dataset.iloc[ : , 1 ].values

    from sklearn.model_selection import train_test_split
    X_train, X_test, Y_train, Y_test = train_test_split( X, Y, test_size = 1/4, random_state = 0) 

    from sklearn.linear_model import LinearRegression
    regressor = LinearRegression()
    regressor = regressor.fit(X_train, Y_train)

    Y_pred = regressor.predict(X_test)

    plt.scatter(X_train , Y_train, color = 'red')
    plt.plot(X_train , regressor.predict(X_train), color ='blue')
    plt.show()

    plt.scatter(X_test , Y_test, color = 'red')
    plt.plot(X_test , regressor.predict(X_test), color ='blue')
    plt.show()
    pass