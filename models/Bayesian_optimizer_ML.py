import pickle
from functools import partial
from sklearn.model_selection import train_test_split

from evaluations import ROC_PR
import numpy as np

from models.Bayesian_optimizer import get_random_string


def get_model_SVM(kernel=0, degree=1, C=1, gamma=1):
    from sklearn.svm import SVC
    all_scores = 0
    C = 10 ** (int(C))
    gamma = 10 ** (int(gamma))
    degree = int(degree)
    kernel = int(kernel)

    for i in range(0, len(labels)):
        dfCurrentDrug = labels[i]
        X = df_train.values.tolist()
        y = dfCurrentDrug.values.tolist()
        for i2 in range(len(y) - 1, -1, -1):
            if y[i2][0] != 0.0 and y[i2][0] != 1.0:
                del y[i2]
                del X[i2]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42,
                                                            shuffle=True)

        if kernel == 0:
            svm_model_linear = SVC(kernel='linear', C=C).fit(X_train, y_train)
        elif kernel == 1:
            svm_model_linear = SVC(kernel='poly', C=C, degree=degree).fit(X_train, y_train)
        else:
            svm_model_linear = SVC(kernel='rbf', C=C, gamma=gamma).fit(X_train, y_train)

        try:
            score1 = ROC_PR.ROC_ML(svm_model_linear, X_test, y_test, "SVM", 0)
        except:
            score1 = svm_model_linear.score(X_test, y_test)
        print(i, flush=True)
        print(score1, flush=True)
        all_scores = all_scores + score1

    print(all_scores/len(labels), flush=True)
    return all_scores/len(labels)


def get_model_SVM_new(kernel=0, degree=1, C=1, gamma=1):
    from sklearn.svm import SVC
    all_scores = 0
    C = 10 ** (int(C))
    gamma = 10 ** (int(gamma))
    degree = int(degree)
    kernel = int(kernel)

    global X_train
    global X_test
    global X_val
    global y_train
    global y_test
    global y_val

    res_test = []
    res_val = []
    res_sr = []
    res_pr = []
    string_random = get_random_string(20)
    for i in range(0, len(y_train[0])):
        X_train2 = X_train.tolist()
        X_test2 = X_test.tolist()
        X_val2 = X_val.tolist()

        y_train2 = y_train[:, i]
        y_test2 = y_test[:, i]
        y_val2 = y_val[:, i]
        y_train2 = y_train2.tolist()
        y_test2 = y_test2.tolist()
        y_val2 = y_val2.tolist()

        for i2 in range(len(y_train2) - 1, -1, -1):
            if y_train2[i2] != 0.0 and y_train2[i2] != 1.0:
                del y_train2[i2]
                del X_train2[i2]

        for i2 in range(len(y_test2) - 1, -1, -1):
            if y_test2[i2] != 0.0 and y_test2[i2] != 1.0:
                del y_test2[i2]
                del X_test2[i2]

        for i2 in range(len(y_val2) - 1, -1, -1):
            if y_val2[i2] != 0.0 and y_val2[i2] != 1.0:
                del y_val2[i2]
                del X_val2[i2]
        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42,
        #                                                     shuffle=True)

        if kernel == 0:
            svm_model_linear = SVC(kernel='linear', C=C).fit(X_train2, y_train2)
        elif kernel == 1:
            svm_model_linear = SVC(kernel='poly', C=C, degree=degree).fit(X_train2, y_train2)
        else:
            svm_model_linear = SVC(kernel='rbf', C=C, gamma=gamma).fit(X_train2, y_train2)
        # try:
        #     score1 = ROC_PR.ROC_ML(svm_model_linear, X_test, y_test, "SVM", 0)
        # except:
        #     score1 = svm_model_linear.score(X_test, y_test)

        score_val, _, _ = ROC_PR.ROC_ML(svm_model_linear, X_val2, y_val2, "LR", 0)
        score_test, score_sr, score_pr = ROC_PR.ROC_ML(svm_model_linear, X_test2, y_test2, "LR", 0)
        print(i, flush=True)
        # print(score1, flush=True)
        res_test.append(score_test)
        res_val.append(score_val)
        res_sr.append(score_sr)
        res_pr.append(score_pr)
        all_scores = all_scores + score_val
        print('svm' + str(i) + string_random + '.sav')
        pickle.dump(svm_model_linear, open('svm' + str(i) + string_random + '.sav', 'wb'))


    global rf_val_score, rf_test_score
    res_val.append(all_scores / len(y_train[0]))
    rf_val_score.append(res_val)
    rf_test_score.append(res_test)
    rf_sr_score.append(res_sr)


    print("val score", res_val)
    print("test score", res_test)
    print("recall at 95 spec: ", res_sr)
    print("precision recall: ", res_pr)
    print(all_scores / len(y_train[0]), flush=True)
    print(string_random)
    return all_scores / len(y_train[0])


def get_model_LR(C=1, penalty=1, solver=1, l1_ratio=1, max_iter=2):
    from sklearn.linear_model import LogisticRegression
    all_scores = 0
    C = 10 ** (int(C))
    penalty = int(penalty)
    solver = int(solver)
    l1_ratio = l1_ratio / 10
    max_iter = 10 ** max_iter
    print(max_iter)
    for i in range(0, len(labels)):
        dfCurrentDrug = labels[i]
        X = df_train.values.tolist()
        y = dfCurrentDrug.values.tolist()
        for i2 in range(len(y) - 1, -1, -1):
            if y[i2][0] != 0.0 and y[i2][0] != 1.0:
                del y[i2]
                del X[i2]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42,
                                                            shuffle=True)
        if penalty == 0:
            lr_model_linear = LogisticRegression(C=C, penalty='l1', solver='liblinear', max_iter=max_iter).fit(X_train, y_train)
        elif penalty == 1:
            if solver == 0:
                lr_model_linear = LogisticRegression(C=C, penalty='l2', solver='newton-cg', max_iter=max_iter).fit(X_train, y_train)
            elif solver == 1:
                lr_model_linear = LogisticRegression(C=C, penalty='l2', solver='sag', max_iter=max_iter).fit(X_train, y_train)
            else:
                lr_model_linear = LogisticRegression(C=C, penalty='l2', solver='lbfgs', max_iter=max_iter).fit(X_train, y_train)
        elif penalty == 2:
            lr_model_linear = LogisticRegression(C=C, penalty='elasticnet', solver='saga', max_iter=max_iter, l1_ratio=l1_ratio).fit(X_train, y_train)
        else:
            lr_model_linear = LogisticRegression(C=C, penalty='none', max_iter=max_iter).fit(X_train, y_train)

        score1 = ROC_PR.ROC_ML(lr_model_linear, X_test, y_test, "LR", 0)
        # accuracy = svm_model_linear.score(X_test, y_test)
        print(i, flush=True)
        print(score1, flush=True)
        all_scores = all_scores + score1

    print(all_scores / len(labels), flush=True)
    return all_scores / len(labels)


df_train, labels = 0, 0


def get_model_LR_new(C=1, penalty=1, solver=1, l1_ratio=1, max_iter=2):
    from sklearn.linear_model import LogisticRegression
    all_scores = 0
    C = 10 ** (int(C))
    penalty = int(penalty)
    solver = int(solver)
    l1_ratio = l1_ratio / 10
    max_iter = 10 ** max_iter
    print(max_iter)

    global X_train
    global X_test
    global X_val
    global y_train
    global y_test
    global y_val

    res_test = []
    res_val = []
    res_sr = []
    res_pr = []
    string_random = get_random_string(20)
    for i in range(0, len(y_train[0])):
        X_train2 = X_train.tolist()
        X_test2 = X_test.tolist()
        X_val2 = X_val.tolist()

        y_train2 = y_train[:, i]
        y_test2 = y_test[:, i]
        y_val2 = y_val[:, i]
        y_train2 = y_train2.tolist()
        y_test2 = y_test2.tolist()
        y_val2 = y_val2.tolist()

        for i2 in range(len(y_train2) - 1, -1, -1):
            if y_train2[i2] != 0.0 and y_train2[i2] != 1.0:
                del y_train2[i2]
                del X_train2[i2]

        for i2 in range(len(y_test2) - 1, -1, -1):
            if y_test2[i2] != 0.0 and y_test2[i2] != 1.0:
                del y_test2[i2]
                del X_test2[i2]

        for i2 in range(len(y_val2) - 1, -1, -1):
            if y_val2[i2] != 0.0 and y_val2[i2] != 1.0:
                del y_val2[i2]
                del X_val2[i2]
        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42,
        #                                                     shuffle=True)

        if penalty == 0:
            lr_model_linear = LogisticRegression(C=C, penalty='l1', solver='liblinear', max_iter=max_iter).fit(X_train2,
                                                                                                               y_train2)
        elif penalty == 1:
            if solver == 0:
                lr_model_linear = LogisticRegression(C=C, penalty='l2', solver='newton-cg', max_iter=max_iter).fit(
                    X_train2, y_train2)
            elif solver == 1:
                lr_model_linear = LogisticRegression(C=C, penalty='l2', solver='sag', max_iter=max_iter).fit(X_train2,
                                                                                                             y_train2)
            else:
                lr_model_linear = LogisticRegression(C=C, penalty='l2', solver='lbfgs', max_iter=max_iter).fit(X_train2,
                                                                                                               y_train2)
        elif penalty == 2:
            lr_model_linear = LogisticRegression(C=C, penalty='elasticnet', solver='saga', max_iter=max_iter,
                                                 l1_ratio=l1_ratio).fit(X_train2, y_train2)
        else:
            lr_model_linear = LogisticRegression(C=C, penalty='none', max_iter=max_iter).fit(X_train2, y_train2)

        score_val, _, _ = ROC_PR.ROC_ML(lr_model_linear, X_val2, y_val2, "LR", 0)
        score_test, score_sr, score_pr = ROC_PR.ROC_ML(lr_model_linear, X_test2, y_test2, "LR", 0)
        print(i, flush=True)
        # print(score1, flush=True)
        res_test.append(score_test)
        res_val.append(score_val)
        res_sr.append(score_sr)
        res_pr.append(score_pr)
        all_scores = all_scores + score_val
        print('lr' + str(i) + string_random + '.sav')
        pickle.dump(lr_model_linear, open('lr' + str(i) + string_random + '.sav', 'wb'))


    global rf_val_score, rf_test_score
    res_val.append(all_scores / len(y_train[0]))
    rf_val_score.append(res_val)
    rf_test_score.append(res_test)
    rf_sr_score.append(res_sr)


    print("val score", res_val)
    print("test score", res_test)
    print("recall at 95 spec: ", res_sr)
    print("precision recall: ", res_pr)
    print(all_scores / len(y_train[0]), flush=True)
    print(string_random)
    return all_scores / len(y_train[0])


def get_model_RF(n_estimators=10, min_samples_split=2, max_depth=1, bootstrap=0):
    from sklearn.ensemble import RandomForestClassifier
    all_scores = 0
    n_estimators = 10 * int(n_estimators)
    min_samples_split = int(min_samples_split)
    if bootstrap < 0:
        bootstrap = False
    else:
        bootstrap = True
    if max_depth > 15:
        max_depth = None
    else:
        max_depth = 10 * int(max_depth)

    global X_train
    global X_test
    global X_val
    global y_train
    global y_test
    global y_val

    res_test = []
    res_val = []
    res_sr = []
    res_pr = []
    string_random = get_random_string(20)
    for i in range(0, len(y_train[0])):
        X_train2 = X_train.tolist()
        X_test2 = X_test.tolist()
        X_val2 = X_val.tolist()

        y_train2 = y_train[:, i]
        y_test2 = y_test[:, i]
        y_val2 = y_val[:, i]
        y_train2 = y_train2.tolist()
        y_test2 = y_test2.tolist()
        y_val2 = y_val2.tolist()

        for i2 in range(len(y_train2) - 1, -1, -1):
            if y_train2[i2] != 0.0 and y_train2[i2] != 1.0:
                del y_train2[i2]
                del X_train2[i2]

        for i2 in range(len(y_test2) - 1, -1, -1):
            if y_test2[i2] != 0.0 and y_test2[i2] != 1.0:
                del y_test2[i2]
                del X_test2[i2]

        for i2 in range(len(y_val2) - 1, -1, -1):
            if y_val2[i2] != 0.0 and y_val2[i2] != 1.0:
                del y_val2[i2]
                del X_val2[i2]
        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42,
        #                                                     shuffle=True)
        rf_model = RandomForestClassifier(n_estimators=n_estimators, min_samples_split=min_samples_split,
                                          bootstrap=bootstrap, max_depth=max_depth).fit(X_train2, y_train2)

        score_val, _, _ = ROC_PR.ROC_ML(rf_model, X_val2, y_val2, "RF", 0, rf=True)
        score_test, score_sr, score_pr = ROC_PR.ROC_ML(rf_model, X_test2, y_test2, "RF", 0, rf=True)
        print(i, flush=True)
        # print(score1, flush=True)
        res_test.append(score_test)
        res_val.append(score_val)
        res_sr.append(score_sr)
        res_pr.append(score_pr)
        all_scores = all_scores + score_val
        print('rf' + str(i) + string_random + '.sav')
        pickle.dump(rf_model, open('rf' + str(i) + string_random + '.sav', 'wb'))


    global rf_val_score, rf_test_score
    res_val.append(all_scores / len(y_train[0]))
    rf_val_score.append(res_val)
    rf_test_score.append(res_test)
    rf_sr_score.append(res_sr)


    print("val score", res_val)
    print("test score", res_test)
    print("recall at 95 spec: ", res_sr)
    print("precision recall: ", res_pr)
    print(all_scores / len(y_train[0]), flush=True)
    print(string_random)
    return all_scores / len(y_train[0])


def get_model_GBT(n_estimators=10, min_samples_split=2, max_depth=1, random_state=0):
    import xgboost.sklearn as xgb
    all_scores = 0
    n_estimators = 10 * int(n_estimators)
    min_samples_split = int(min_samples_split)
    if random_state < 0:
        random_state = None
    else:
        random_state = int(random_state)
    if max_depth > 15:
        max_depth = None
    else:
        max_depth = 10 * int(max_depth)

    global X_train
    global X_test
    global X_val
    global y_train
    global y_test
    global y_val

    res_test = []
    res_val = []
    res_sr = []
    res_pr = []
    string_random = get_random_string(20)

    for i in range(0, len(y_train[0])):
        X_train2 = X_train.tolist()
        X_test2 = X_test.tolist()
        X_val2 = X_val.tolist()

        y_train2 = y_train[:, i]
        y_test2 = y_test[:, i]
        y_val2 = y_val[:, i]
        y_train2 = y_train2.tolist()
        y_test2 = y_test2.tolist()
        y_val2 = y_val2.tolist()

        for i2 in range(len(y_train2) - 1, -1, -1):
            if y_train2[i2] != 0.0 and y_train2[i2] != 1.0:
                del y_train2[i2]
                del X_train2[i2]

        for i2 in range(len(y_test2) - 1, -1, -1):
            if y_test2[i2] != 0.0 and y_test2[i2] != 1.0:
                del y_test2[i2]
                del X_test2[i2]

        for i2 in range(len(y_val2) - 1, -1, -1):
            if y_val2[i2] != 0.0 and y_val2[i2] != 1.0:
                del y_val2[i2]
                del X_val2[i2]
        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42,
        #                                                     shuffle=True)

        param = {'n_estimators': n_estimators, 'min_samples_split': min_samples_split, 'random_state': random_state, 'max_depth': max_depth}
        print(n_estimators)
        print(min_samples_split)
        print(random_state)
        print(max_depth)
        try:
            gbt_model = xgb.XGBModel(n_estimators=n_estimators, min_samples_split=min_samples_split,
                                     random_state=random_state, max_depth=max_depth).fit(np.array(X_train2), np.array(y_train2))
            score_val, _, _ = ROC_PR.ROC_ML(gbt_model, np.array(X_val2), np.array(y_val2), "GBT", 0, xgb=True)
            score_test, score_sr, score_pr = ROC_PR.ROC_ML(gbt_model, np.array(X_test2), np.array(y_test2), "GBT", 0,
                                                           xgb=True)
            print('gbt' + str(i) + string_random + '.sav')
            pickle.dump(gbt_model, open('gbt' + str(i) + string_random + '.sav', 'wb'))
        except():
            print("errorrrrrr in GBT", flush=True)
            score_test, score_sr, score_pr,score_val = 0, 0, 0, 0

        print(i, flush=True)
        # print(score1, flush=True)
        res_test.append(score_test)
        res_val.append(score_val)
        res_sr.append(score_sr)
        res_pr.append(score_pr)
        all_scores = all_scores + score_val


    global rf_val_score, rf_test_score
    res_val.append(all_scores / len(y_train[0]))
    rf_val_score.append(res_val)
    rf_test_score.append(res_test)
    rf_sr_score.append(res_sr)


    print("val score", res_val)
    print("test score", res_test)
    print("recall at 95 spec: ", res_sr)
    print("precision recall: ", res_pr)
    print(all_scores / len(y_train[0]), flush=True)

    print(string_random)

    return all_scores / len(y_train[0])


def BO_SVM():
    # global df_train
    # df_train = X
    # global labels
    # labels = y
    # X_train2, X_test2, y_train2, y_test2 = train_test_split(X, y, test_size=0.1, random_state=42, shuffle=True)

    # global X_train
    # X_train = X_train2
    # global X_test
    # X_test = X_test2
    # global y_train
    # y_train = y_train2
    # global y_test
    # y_test = y_test2

    fit_with_partial = partial(get_model_SVM_new)

    fit_with_partial(kernel=0, degree=1, C=1, gamma=1)

    from bayes_opt import BayesianOptimization

    # Bounded region of parameter space
    pbounds = {'C': (-2, 2), "degree": (0.9, 10), "kernel": (0.9, 3.1), 'gamma': (-3, 3)}

    optimizer = BayesianOptimization(
        f=fit_with_partial,
        pbounds=pbounds,
        verbose=2,  # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
        random_state=1,
    )
    optimizer.maximize(init_points=15, n_iter=15, )

    for i, res in enumerate(optimizer.res):
        print("Iteration {}: \n\t{}".format(i, res), flush=True)

    print("resultttttttttttttt SVM" + str(i), flush=True)
    print(optimizer.max, flush=True)

def BO_LR():
    # global df_train
    # df_train = X
    # global labels
    # labels = y

    # X_train2, X_test2, y_train2, y_test2 = train_test_split(X, y, test_size=0.1, random_state=42, shuffle=True)
    #
    # global X_train
    # X_train = X_train2
    # global X_test
    # X_test = X_test2
    # global y_train
    # y_train = y_train2
    # global y_test
    # y_test = y_test2

    fit_with_partial = partial(get_model_LR_new)

    fit_with_partial(C=1, penalty=1, solver=1, l1_ratio=1, max_iter=2)

    from bayes_opt import BayesianOptimization

    # Bounded region of parameter space
    pbounds = {'C': (-2, 2), 'penalty': (0.9, 3.1), 'solver': (0.9, 2.1), 'l1_ratio': (0, 10), 'max_iter': (1.9, 3.0)}

    optimizer = BayesianOptimization(
        f=fit_with_partial,
        pbounds=pbounds,
        verbose=2,  # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
        random_state=1,
    )
    optimizer.maximize(init_points=15, n_iter=15, )

    for i, res in enumerate(optimizer.res):
        print("Iteration {}: \n\t{}".format(i, res), flush=True)

    print("resultttttttttttttt LR" + str(i), flush=True)
    print(optimizer.max, flush=True)


def BO_RF():
    fit_with_partial = partial(get_model_RF)

    fit_with_partial(n_estimators=10, min_samples_split=2, max_depth=1, bootstrap=0)

    from bayes_opt import BayesianOptimization

    # Bounded region of parameter space
    pbounds = {'n_estimators': (5, 15), 'min_samples_split': (2, 5), 'max_depth': (5, 20), 'bootstrap': (-1, 3)}

    optimizer = BayesianOptimization(
        f=fit_with_partial,
        pbounds=pbounds,
        verbose=2,  # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
        random_state=1,
    )
    optimizer.maximize(init_points=15, n_iter=15, )

    for i, res in enumerate(optimizer.res):
        print("Iteration {}: \n\t{}".format(i, res), flush=True)

    print("resultttttttttttttt RF" + str(i), flush=True)
    print(optimizer.max, flush=True)


def BO_GBT():
    fit_with_partial = partial(get_model_GBT)

    fit_with_partial(n_estimators=10, min_samples_split=2, max_depth=1, random_state=0)

    from bayes_opt import BayesianOptimization

    # Bounded region of parameter space
    pbounds = {'n_estimators': (1, 10), 'min_samples_split': (2, 5), 'max_depth': (5, 15), 'random_state': (-1, 2)}

    optimizer = BayesianOptimization(
        f=fit_with_partial,
        pbounds=pbounds,
        verbose=2,  # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
        random_state=1,
    )
    optimizer.maximize(init_points=15, n_iter=15, )

    for i, res in enumerate(optimizer.res):
        print("Iteration {}: \n\t{}".format(i, res), flush=True)

    print("resultttttttttttttt GBT" + str(i), flush=True)
    print(optimizer.max, flush=True)


X_train, X_test, X_val, y_train, y_test, y_val = 0, 0, 0, 0, 0, 0
rf_test_score = []
rf_val_score = []
rf_sr_score = []

def prepare_data(features, label):
    FrameSize = 200

    y = []
    for i in range(0, len(label)):
        label[i] = label[i].values.tolist()

    for j in range(0, len(label[0])):
        tmp = []
        for i in range(0, len(label)):
            if label[i][j][0] != 0.0 and label[i][j][0] != 1.0:
                tmp.extend([-1])
            else:
                tmp.extend(label[i][j])
        y.append(tmp)

    y = np.array(y)
    features = np.array(features)
    return features, y, FrameSize


def run_bayesian(df_train, labels):
    X, y, FrameSize = prepare_data(df_train, labels)

    global X_train
    global X_test
    global X_val
    global y_train
    global y_test
    global y_val

    #TODO
    print("GBT")
    for i in range(0, 10):
        print("fold: " + str(i))
        length = int(len(X) / 10)
        if i == 0:
            X_train2 = X[length:]
            X_test2 = X[0:length]
            y_train2 = y[length:]
            y_test2 = y[0:length]
        elif i != 9:
            X_train2 = np.append(X[0:length * i], X[length * (i + 1):], axis=0)
            X_test2 = X[length * i:length * (i + 1)]
            y_train2 = np.append(y[0:length * i], y[length * (i + 1):], axis=0)
            y_test2 = y[length * i:length * (i + 1)]
        else:
            X_train2 = X[0:length * i]
            X_test2 = X[length * i:]
            y_train2 = y[0:length * i]
            y_test2 = y[length * i:]
        X_train2, X_val2, y_train2, y_val2 = train_test_split(X_train2, y_train2, test_size=0.1, random_state=1, shuffle=False)
        X_train = X_train2
        X_test = X_test2
        X_val = X_val2
        y_train = y_train2
        y_test = y_test2
        y_val = y_val2
        # loaded_model = pickle.load(open('gbt0uapobneyqjzmpcasd.sav', 'rb'))
        loaded_model = pickle.load(open('gbt0vshfzjsqdrfvjecxwdyp.sav', 'rb'))
        result = loaded_model.predict(X_test)
        print(result)
        # BO_GBT()

    # print("LR")
    # for i in range(0, 10):
    #     print("fold: " + str(i))
    #     length = int(len(X) / 10)
    #     if i == 0:
    #         X_train2 = X[length:]
    #         X_test2 = X[0:length]
    #         y_train2 = y[length:]
    #         y_test2 = y[0:length]
    #     elif i != 9:
    #         X_train2 = np.append(X[0:length * i], X[length * (i + 1):], axis=0)
    #         X_test2 = X[length * i:length * (i + 1)]
    #         y_train2 = np.append(y[0:length * i], y[length * (i + 1):], axis=0)
    #         y_test2 = y[length * i:length * (i + 1)]
    #     else:
    #         X_train2 = X[0:length * i]
    #         X_test2 = X[length * i:]
    #         y_train2 = y[0:length * i]
    #         y_test2 = y[length * i:]
    #     X_train2, X_val2, y_train2, y_val2 = train_test_split(X_train2, y_train2, test_size=0.1, random_state=1,
    #                                                           shuffle=False)
    #     X_train = X_train2
    #     X_test = X_test2
    #     X_val = X_val2
    #     y_train = y_train2
    #     y_test = y_test2
    #     y_val = y_val2
    #     BO_LR()
    #
    # print("SVM")
    # for i in range(0, 10):
    #     print("fold: " + str(i))
    #     length = int(len(X) / 10)
    #     if i == 0:
    #         X_train2 = X[length:]
    #         X_test2 = X[0:length]
    #         y_train2 = y[length:]
    #         y_test2 = y[0:length]
    #     elif i != 9:
    #         X_train2 = np.append(X[0:length * i], X[length * (i + 1):], axis=0)
    #         X_test2 = X[length * i:length * (i + 1)]
    #         y_train2 = np.append(y[0:length * i], y[length * (i + 1):], axis=0)
    #         y_test2 = y[length * i:length * (i + 1)]
    #     else:
    #         X_train2 = X[0:length * i]
    #         X_test2 = X[length * i:]
    #         y_train2 = y[0:length * i]
    #         y_test2 = y[length * i:]
    #     X_train2, X_val2, y_train2, y_val2 = train_test_split(X_train2, y_train2, test_size=0.1, random_state=1,
    #                                                           shuffle=False)
    #     X_train = X_train2
    #     X_test = X_test2
    #     X_val = X_val2
    #     y_train = y_train2
    #     y_test = y_test2
    #     y_val = y_val2
        BO_SVM()
        # global rf_val_score, rf_test_score
        # print("rf_val_score")
        # print(rf_val_score)
        # print("rf_test_score")
        # print(rf_test_score)


if __name__ == '__main__':
    # | iter | target | C | degree | kernel |
    # | 1 | 0.8609 | -1.66 | 3.853 | 0.9001 |
    # C = -1.66
    # degree = 3.853
    # kernel = 0.9001
    # C = 10 ** (int(C))
    # degree = int(degree)
    # kernel = int(kernel)
    # if kernel == 0:
    #     svm_model_linear = SVC(kernel='linear', C=C).fit(X_train, y_train)
    # else:
    #     svm_model_linear = SVC(kernel='poly', C=C, degree=degree).fit(X_train, y_train)
    # resultttttttttttttt
    # LR29
    # {'target': 0.8636191441305866,
    #  'params': {'C': -7.953311423443483, 'l1_ratio': 4.140559878195683, 'max_iter': 3.427680347001039,
    #             'penalty': 1.811194392959186, 'solver': 0.9599441507353046}}

    # LR
    # {'target': 0.8601012372054825,
    #  'params': {'C': 7.023950121643671, 'l1_ratio': 2.8797628407599776, 'max_iter': 4.1, 'penalty': 0.9, 'solver': 0.9}}

    # LR
    # {'target': 0.8787893887726396,
    #  'params': {'C': -1.5906622846886966, 'l1_ratio': 4.140559878195683, 'max_iter': 1.9694400157727745,
    #             'penalty': 1.811194392959186, 'solver': 0.9599441507353046}}

    # RF
    # {'target': 0.9053981940680434,
    #  'params': {'bootstrap': -0.922544907560638, 'max_depth': 5.847050981700784, 'min_samples_split': 4.994497043033141,
    #             'n_estimators': 14.815287020046105}}

    # LR
    # {'target': 0.8787893887726396,
    #  'params': {'C': -1.5906622846886966, 'l1_ratio': 4.140559878195683, 'max_iter': 3.427680347001039,
    #             'penalty': 1.811194392959186, 'solver': 0.9599441507353046}}

    # max_depth =  13.007445686755366
    # min_samples_split =  4.904784727158193
    # n_estimators =  3.8208176034331855
    # random_state =  1.0769678470079422

    # max_depth =  14.888610889064948
    # min_samples_split =  4.244496963139518
    # n_estimators =  3.5239959285796467
    # random_state =  1.3678379853544653
    # #
    max_depth =  9.17304802367127
    min_samples_split =  3.676069485337255
    n_estimators =  2.263482447357104
    random_state =  -0.40569553274536363
    #
    max_depth =  6.467558908171131
    min_samples_split =  2.2770157843063936
    n_estimators =  2.676341902399038
    random_state =  0.03668218112914312
    #
    max_depth =  7.11628116000059
    min_samples_split =  2.7966399781166786
    n_estimators =  5.424158433523044
    random_state =  -0.8399123646487588


    max_depth =  9.17304802367127
    min_samples_split =  3.676069485337255
    n_estimators =  2.263482447357104
    random_state =  -0.40569553274536363
    #
    max_depth =  5.35974045182736
    min_samples_split =  2.1953193008319913
    n_estimators =  9.924797504376732
    random_state =  1.2160124089096085
    #
    max_depth =  6.467558908171131
    min_samples_split =  2.2770157843063936
    n_estimators =  2.676341902399038
    random_state =  0.03668218112914312
    #
    max_depth =  5.068061943154863
    min_samples_split =  4.97552730560031
    n_estimators =  1.1581061127115535
    random_state =  -0.29315445063695056
    #
    max_depth =  7.11628116000059
    min_samples_split =  2.7966399781166786
    n_estimators =  5.424158433523044
    random_state =  -0.8399123646487588

    n_estimators = 10 * int(n_estimators)
    min_samples_split = int(min_samples_split)
    if random_state < 0:
        random_state = None
    else:
        random_state = int(random_state)
    if max_depth > 15:
        max_depth = None
    else:
        max_depth = 10 * int(max_depth)


    print(max_depth)
    print(min_samples_split)
    print(n_estimators)
    print(random_state)
    # C = -1.5906622846886966
    # l1_ratio = 4.140559878195683
    # max_iter = 1.9694400157727745
    # penalty = 1.811194392959186
    # solver = 0.9599441507353046
    # max_iter = 10 ** max_iter
    # C = 10 ** (int(C))
    # penalty = int(penalty)
    # solver = int(solver)
    # l1_ratio = l1_ratio / 10
    # print(C)
    # print(l1_ratio)
    # print(penalty)
    # print(solver)
    # print(max_iter)
    #
    # print("___")
    #
    # bootstrap = -0.922544907560638
    # max_depth = 5.847050981700784
    # min_samples_split = 4.994497043033141
    # n_estimators = 14.815287020046105
    #
    # n_estimators = 10 * int(n_estimators)
    # min_samples_split = int(min_samples_split)
    # if bootstrap < 0:
    #     bootstrap = False
    # else:
    #     bootstrap = True
    # if max_depth > 15:
    #     max_depth = None
    # else:
    #     max_depth = 10 * int(max_depth)
    #
    # print(bootstrap)
    # print(max_depth)
    # print(min_samples_split)
    # print(n_estimators)