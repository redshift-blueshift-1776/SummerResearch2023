import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as seaborn
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.model_selection import GridSearchCV, RepeatedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn import svm, datasets
from sklearn.svm._libsvm import cross_validation
import h5py
import numpy as np
import warnings
warnings.filterwarnings('ignore')

fname = "gpscut3.csv"
# fname = "gps230101g.002.hdf5"  # input("file name:\n")
try:
    # myarray = np.fromfile(fname)
    fhand = open(fname)
    # fhand = h5py.File(fname, 'r+')
    """with h5py.File(fname, "r") as f:
        # Print all root level object names (aka keys)
        # these can be group or dataset names
        print("Keys: %s" % f.keys())
        # get first object name/key; may or may NOT be a group
        a_group_key = list(f.keys())[0]

        # get the object type for a_group_key: usually group or dataset
        print(type(f[a_group_key]))

        # If a_group_key is a group name,
        # this gets the object names in the group and returns as a list
        data = list(f[a_group_key])

        # If a_group_key is a dataset name,
        # this gets the dataset values and returns as a list
        data2 = list(f[a_group_key])
        # preferred methods to get dataset values:
        ds_obj = f[a_group_key]  # returns as a h5py dataset object
        # ds_arr = f[a_group_key][()]  # returns as a numpy array
        """
except:
    print('File cannot be opened:', fname)
    exit()
mode = 3  # int(input("1 if svr, 0 if svm\n")), 1 skips cv, 3 does it.
column_start = 3  # int(input("Column start\n"))
column_end = 5  # int(input("Column end\n")) + 1
categories = 5  # int(input("Categories column\n"))
C = 50  # float(input("C\n")), fix att 50 or 100
gamma = 0  # float(input("gamma=\n")), change C, make standard deviation for error.
focus = 3  # int(input("focus\n"))

if mode == 1 or mode == 2 or mode == 3:
    # dataset = myarray
    dataset = pd.read_csv(fname)
    print(dataset.shape)
    # dataset2 = np.split(dataset, [200, 400])[0]
    # print(dataset2.shape)
    X = dataset.iloc[:, column_start:column_end].values
    y = dataset.iloc[:, categories].values
    y2 = dataset.iloc[:, categories + 1].values
    X2 = dataset.iloc[:, focus].values
    X3 = dataset.iloc[:, focus:(focus + 2)].values

    sc_X = StandardScaler()
    sc_X2 = StandardScaler()
    sc_X3 = StandardScaler()
    sc_y = StandardScaler()
    sc_y2 = StandardScaler()
    X = sc_X.fit_transform(X)
    y = sc_y.fit_transform(y.reshape(-1, 1))
    y2 = sc_y2.fit_transform(y2.reshape(-1, 1))
    X2 = sc_X2.fit_transform(X2.reshape(-1, 1))
    X3 = sc_X3.fit_transform(X3)

    regressor = SVR(kernel='rbf', gamma=gamma, C=C)  # actual predictor
    regressor.fit(X, y)
    regressor2 = SVR(kernel='rbf')  # used for graph
    regressor2.fit(X2, y)
    regressor3 = SVR(kernel='rbf')  # used for graph
    regressor3.fit(X3, y)

    if mode != 3:
        """
        X_grid = np.arange(np.amin(X2), np.amax(X2), 0.01)  # this step required because data is feature scaled.
        X_grid = X_grid.reshape((len(X_grid), 1))

        plt.scatter(X2, y, color='red')
        plt.plot(X_grid, regressor2.predict(X_grid), color='blue')
        plt.title('SVR')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.show()
        """
    else:
        X_grid = np.arange(np.amin(X2), np.amax(X2), 0.01)  # this step required because data is feature scaled.
        X_grid = X_grid.reshape((len(X_grid), 1))
        li = []
        for j in range(-18, 19):
            for i in range(-36, 37):
                X_pred = sc_X.transform([[j * 5, i * 5]])
                y_scale_back = sc_y.inverse_transform([regressor.predict(X_pred)])
                li.append([j * 5, i * 5, y_scale_back[0][0]])
        # print(li)
        fout = open('output.txt', 'w')
        for x in li:
            fout.write(str(x[0]) + "," + str(x[1]) + "," + str(x[2]) + "\n")
        # plt.plot(X_grid, regressor2.predict(X_grid), color='blue')
        # for x in li:
            # plt.scatter(x[0], x[1], s=x[2], alpha=0.5)
        # plt.title('SVR')
        # plt.xlabel('X')
        # plt.ylabel('Y')
        # plt.show()
        # seaborn.heatmap(li)

    print("Instances: " + str(len(y)))
    abs_error_sum = 0
    percent_error_sum = 0
    spare = dataset.iloc[:, column_start:column_end].values
    for i in range(0, len(y)):
        X_pred = sc_X.transform([spare[i]])
        y_scale_back = sc_y.inverse_transform([regressor.predict(X_pred)])
        # print(y_scale_back[0][0])
        abs_error_sum += abs(sc_y.inverse_transform([y[i]]) - y_scale_back)
        percent_error_sum += abs((abs(sc_y.inverse_transform([y[i]]) - y_scale_back) + 0.0)
                                 / sc_y.inverse_transform([y[i]]))

    print("Absolute error sum: " + str(abs_error_sum))
    print("Average absolute error: " + str((abs_error_sum + 0.0) / len(y)))
    print("Average percent error: " + str(100 * (percent_error_sum + 0.0) / len(y)))

    if mode == 2 or mode == 3:
        # Tenfold cross-validation
        cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)

        # define the pipeline to evaluate
        model = SVR(kernel='sigmoid', C=50, gamma=50)

        # print stats
        cv = cross_validate(model, X, y, cv=10)
        print('SVR\n')
        print('cross-validation:' + '\n')
        print(str(cv) + '\n')
        print('cross validation test score:' + '\n')
        print(str(cv['test_score']) + '\n')
        print('average cross validation score:' + str(cv['test_score'].mean()) + '\n')

    while True:
        inputs = []
        for i in range(column_start, column_end):
            try:
                get = input("attribute\n")
                inputs.append(float(get))
            except:
                exit()
        X_pred = sc_X.transform([inputs])
        y_scale_back = sc_y.inverse_transform([regressor.predict(X_pred)])
        print(y_scale_back[0][0])
