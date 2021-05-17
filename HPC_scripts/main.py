import pandas as pd
import numpy as np
import sklearn.model_selection as model_selection
import time

# pipe_line_gene.py
from loading_data import data_preprocess
from models import Bayesian_optimizer, model_gene_based, wide_n_deep, ML_methods, deepAMR_run
from feature_importance import base_approach, lime
from data_analyzer import source_analysis
from dataset_creator import gene_dataset_creator

def main():
    """Running the LRCN model on Imperial HPC"""

    df = pd.read_csv("Data/AllLabels.csv")
    f = pd.read_csv("Data/gene_data.csv")
    df_train, labels = data_preprocess.process(38, gene_dataset=True)

    limited=False
    portion=0.1

    X, y, FrameSize = model_gene_based.prepare_data(df_train, labels)
    X2 = X
    y2 = y
    j = 0
    print("drug: " + str(j))
    tmp = []
    tmp_x = []
    dele = []
    for k in range(0, len(y2)):
        if y2[k][j] == 1 or y2[k][j] == 0:
            tmp.append(y2[k][j])
            tmp_x.append(X2[k])
    y = tmp
    X = tmp_x
    i = 3
    print("fold: " + str(i))
    length = int(len(X) / 10)
    if i == 0:
        X_train = X[length:]
        X_test = X[0:length]
        y_train = y[length:]
        y_test = y[0:length]
    elif i != 9:
        X_train = np.append(X[0:length * i], X[length * (i + 1):], axis=0)
        X_test = X[length * i:length * (i + 1)]
        y_train = np.append(y[0:length * i], y[length * (i + 1):], axis=0)
        y_test = y[length * i:length * (i + 1)]
    else:
        X_train = X[0:length * i]
        X_test = X[length * i:]
        y_train = y[0:length * i]
        y_test = y[length * i:]

    X_train, X_val, y_train, y_val = model_selection.train_test_split(X_train, y_train, test_size=0.1, random_state=1, shuffle=False)
    Bayesian_optimizer.BO(X_train, X_test, X_val, y_train, y_test, y_val, limited, portion)

if __name__ == "__main__":
    tic = time.perf_counter()

    main()

    toc = time.perf_counter()
    print(f"Analysis took {toc - tic:0.4f} seconds")
