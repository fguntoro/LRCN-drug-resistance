from loading_data import data_preprocess
from models import model_gene_based, wide_n_deep, ML_methods, deepAMR_run
from feature_importance import base_approach, lime
from data_analyzer import source_analysis
from dataset_creator import gene_dataset_creator

epochs = 200


def train():
    # deepAMR_run.main()
    df_train, labels = data_preprocess.process(38, gene_dataset=True) # SNP or GENE
    # gene_dataset_creator.main()
    # df_train, labels = data_preprocess.process(38)
    # source_analysis.main(df_train)
    # df_train, labels = data_preprocess.process(38, shuffle_operon_group=True)
    # ML_methods.model_run(df_train, labels) # BO for ML and GBT
    # model_gene_based.run_model(df_train, labels, epochs)
    # wide_n_deep.run_bayesian(df_train, labels) # BO for wide n deep method
    # model_gene_based.run_bayesian(df_train, labels) # main diff in data folding nested CV # init_point 15, n_iter 15 # run_one_fold also run_k_fold and run_single_fold (section.3.5 help accuracy)
    # model_gene_based.run_bayesian_single(df_train, labels)
    # model_gene_based.run_all(df_train, labels, epochs)
    model_gene_based.run_model_kfold(df_train,labels,epochs)


    # base_approach.run_feature_importance(df_train, labels)
    # base_approach.find_feature_importance(file_name='feature_importance/score_results/lime_all.csv', k=200)
    # base_approach.find_feature_importance(file_name='feature_importance/score_results/feature_scores_lime_all_test.csv', k=200)
    # base_approach.find_feature_importance(file_name='feature_importance/score_results/feature_scores_lime_test_200.csv', k=200)
    # base_approach.find_feature_importance(file_name='feature_importance/score_results/feature_scores_lime_train.csv', k=200)
    #
    # print("______")
    # base_approach.find_feature_importance(file_name='feature_importance/score_results/lime_all.csv', k=100)
    # base_approach.find_feature_importance(file_name='feature_importance/score_results/feature_scores_lime_all_test.csv',
    #                                       k=100)
    # base_approach.find_feature_importance(file_name='feature_importance/score_results/feature_scores_lime_test_200.csv',
    #                                       k=100)
    # base_approach.find_feature_importance(file_name='feature_importance/score_results/feature_scores_lime_train.csv',
    #                                       k=100)
    # print("______")
    # base_approach.find_feature_importance(file_name='feature_importance/score_results/lime_all.csv', k=50)
    # base_approach.find_feature_importance(file_name='feature_importance/score_results/feature_scores_lime_all_test.csv',
    #                                       k=50)
    # base_approach.find_feature_importance(file_name='feature_importance/score_results/feature_scores_lime_test_200.csv',
    #                                       k=50)
    # base_approach.find_feature_importance(file_name='feature_importance/score_results/feature_scores_lime_train.csv',
    #                                       k=50)
    # print("______")
    # base_approach.find_feature_importance(file_name='feature_importance/score_results/lime_all.csv', k=20)
    # base_approach.find_feature_importance(file_name='feature_importance/score_results/feature_scores_lime_all_test.csv',
    #                                       k=20)
    # base_approach.find_feature_importance(file_name='feature_importance/score_results/feature_scores_lime_test_200.csv',
    #                                       k=20)
    # base_approach.find_feature_importance(file_name='feature_importance/score_results/feature_scores_lime_train.csv',
    #                                       k=20)
    # lime.main_function(df_train, labels)


def train_shuffle():
    df_train, labels = data_preprocess.process(38, shuffle_index=True, index_file=0)
    model_gene_based.run_model_kfold(df_train, labels, epochs, index=0)
    df_train, labels = data_preprocess.process(38, shuffle_index=True, index_file=1)
    model_gene_based.run_model_kfold(df_train, labels, epochs, index=1)
    df_train, labels = data_preprocess.process(38, shuffle_index=True, index_file=2)
    model_gene_based.run_model_kfold(df_train, labels, epochs, index=2)
    df_train, labels = data_preprocess.process(38, shuffle_index=True, index_file=3)
    model_gene_based.run_model_kfold(df_train, labels, epochs, index=3)


if __name__ == '__main__':
    train()
    # train_shuffle()
