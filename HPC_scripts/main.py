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
    tic = time.perf_counter()

    df = pd.read_csv("Data/AllLabels.csv")
    f = pd.read_csv("Data/gene_data.csv")
    df_train, labels = data_preprocess.process(38, gene_dataset=True)

    toc = time.perf_counter()
    print(f"Analysis was run in {toc - tic:0.4f} seconds")

if __name__ == "__main__":
    main()