import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.model_selection import GridSearchCV, RepeatedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn import svm, datasets
from sklearn.svm._libsvm import cross_validation

fname = input("file name:\n")
try:
    fhand = open(fname)
except:
    print('File cannot be opened:', fname)
    exit()
mode = int(input("1 if svr, 0 if svm\n"))
column_start = int(input("Column start\n"))
column_end = int(input("Column end\n")) + 1
categories = int(input("Categories column\n"))
C = float(input("C\n"))
gamma = float(input("gamma=\n"))
focus = int(input("focus\n"))

if mode == 1 or mode == 2:
    dataset = pd.read_csv(fname, header=None)
    X = dataset.iloc[:, column_start:column_end].values
    y = dataset.iloc[:, categories].values
    X2 = dataset.iloc[:, focus].values

    sc_X = StandardScaler()
    sc_X2 = StandardScaler()
    sc_y = StandardScaler()
    X = sc_X.fit_transform(X)
    y = sc_y.fit_transform(y.reshape(-1, 1))
    X2 = sc_X2.fit_transform(X2.reshape(-1, 1))

    if gamma < 0:
        regressor = SVR(kernel='rbf', C=C)  # actual predictor
        regressor2 = SVR(kernel='rbf', C=C)  # used for graph
    else:
        regressor = SVR(kernel='rbf', C=C, gamma=gamma)  # actual predictor
        regressor2 = SVR(kernel='rbf', C=C, gamma=gamma)  # used for graph
    regressor.fit(X, y)
    regressor2.fit(X2, y)

    X_grid = np.arange(np.amin(X2), np.amax(X2), 0.01)  # this step required because data is feature scaled.
    X_grid = X_grid.reshape((len(X_grid), 1))

    plt.scatter(X2, y, color='red')
    plt.plot(X_grid, regressor2.predict(X_grid), color='blue')
    plt.title('SVR')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()

    print("Instances: " + str(len(y)))
    abs_error_sum = 0
    percent_error_sum = 0
    spare = dataset.iloc[:, column_start:column_end].values
    for i in range(0, len(y)):
        X_pred = sc_X.transform([spare[i]])
        y_scale_back = sc_y.inverse_transform([regressor.predict(X_pred)])
        print(y_scale_back[0][0])
        abs_error_sum += abs(sc_y.inverse_transform([y[i]]) - y_scale_back)
        percent_error_sum += abs((abs(sc_y.inverse_transform([y[i]]) - y_scale_back) + 0.0)
                                 / sc_y.inverse_transform([y[i]]))

    print("Absolute error sum: " + str(abs_error_sum))
    print("Average absolute error: " + str((abs_error_sum + 0.0) / len(y)))
    print("Average percent error: " + str(100 * (percent_error_sum + 0.0) / len(y)))

    if mode == 2:
        # Tenfold cross-validation
        cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)

        # define the pipeline to evaluate
        model = SVR(kernel='rbf')

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
        print(y_scale_back)
        # result = regressor.predict([inputs])
        # result = sc_y.inverse_transform([result])
        # print(result[0])
else:
    dataset = pd.read_csv(fname, header=None)
    X = dataset.iloc[:, focus:(focus + 2)].values
    X2 = dataset.iloc[:, column_start:column_end].values
    target1 = dataset.iloc[:, categories].values
    map = dict()
    map2 = dict()
    target = []
    for i in target1:
        if i not in map:
            j = len(map)
            map[i] = j
            map2[j] = i
        target.append(map[i])
    y = np.array(target)
    # iris = datasets.load_iris()
    # X = iris.data[:, :2]
    # y = iris.target
    # print(y)
    # print(type(y))

    svc = svm.SVC(kernel='linear', C=C, gamma=gamma).fit(X, y)  # used for graph
    svc1 = svm.SVC(kernel='linear', C=C, gamma=gamma).fit(X2, y)  # actual predictor
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    h = (x_max - x_min) / 100
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    plt.subplot(1, 1, 1)
    Z = svc.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.xlim(xx.min(), xx.max())
    plt.title('SVC (predict)')
    plt.show()

    print("Testing accuracy")
    right1 = 0
    right = 0
    for i in range(0, len(target)):
        actual = target[i]
        # print(actual)
        predict1 = svc1.predict([X2[i]])[0]
        # print(predict1)
        predict = svc.predict([X[i]])[0]
        # print(predict)
        if predict1 == actual:
            right1 += 1
        else:
            print("Attributes: " + str(X2[i]))
            print("actual: " + map2.get(actual) + ". predict1: " + map2.get(predict1))
        if predict == actual:
            right += 1
        # else:
        # print("Attributes: " + str(X[i]))
        # print("actual: " + map2.get(actual) + ". predict: " + map2.get(predict))

    print(right1)
    # print(right)

    while True:
        inputs = []
        for i in range(column_start, column_end):
            try:
                get = input("attribute\n")
                inputs.append(float(get))
            except:
                exit()
        result = svc1.predict([inputs])
        # print(result[0])
        result2 = map2.get(result[0])
        print(result2)
