"""
@author: Jimmy
"""
import pandas as pd
import os
from sklearn.metrics import pairwise_distances
os.chdir('C:/Users/Admin/Desktop/Project/RecommendSystem') #cd
pd.set_option('display.max_columns', None)
import DS_Model_function as func

'''
import matplotlib.pyplot as plt
import seaborn as sns
sns.distplot(Trainset.score,
             hist=True, 
             color = 'red').set(xlim=(0, 1))
'''

# Customer play record and exclude the recommend game(In SQL) because it is off line test
names = ['member_id', 'dates', 'game_id', 'game_name','game_category','bet_amount',
         'payoff','commissionable','RawDataType','gamecode', 'Expect_earning_ratio']
#train = pd.read_csv('TrainData_1101To1124.csv', header = 0, names = names) ##11/01~11/24 as training dataset

train = pd.read_csv('TrainData_1118To1124.csv',
                    header = 0,
                    names = names) ##11/01~11/24 as training dataset
test = pd.read_csv('TestData_1125To1130.csv', 
                   header = 0,
                   names = names) ##11/25~11/30 as test dataset


# 預處理
train_data = func.Pre_processing(train)
test_data = func.Pre_processing(test)

# filter & Exclude the just play one game people (Noise and it can't give our model any help)
train_data = func.Pre_processing_train(train_data)


# Define the hotgame list
HotGame_inTrain = func.Hot_Game(train_data, 
                                feature='commissionable',
                                n=12)

# get the train set, i.e., define the feature (or score namely) for recommending
Trainset = func.get_Trainset(train_data)
games = list(Trainset.game_id.unique())

# get sparse matrix
sparse_df = func.get_sparse(Trainset)

#cosine similarity
Neighbor_only_result, Neighbor_result = func.Recommendation_cosine(Trainset, sparse_df, games, N = 45)

#surprise svd
Latent_only_result, Latent_result_df = func.SVD_surprise(Trainset)
SVD_Neighbor, SVD_Neighbor_result = func.SVD_surprise_only(Trainset, 75)

# model of the bind result
Result_BindLatentandNeighbor = Neighbor_result.merge(Latent_result_df,
                                                     how = 'left',
                                                     on = ['member_id', 'game_id'])
Result_BindLatentandNeighbor = Result_BindLatentandNeighbor.sort_values(by = ['member_id', 'Pred_Return'],
                                                                        ascending=False).reset_index(drop=True)
Bind_result = Result_BindLatentandNeighbor.groupby('member_id').head(12)

#sparseSVD
#k = func.Dim_reduction(func.get_sparse_matrix(Trainset))=590
U, S, Vt = func.computeSVD(func.get_sparse_matrix(Trainset), 590)
similarity_matrix = pairwise_distances((S*Vt).T, (S*Vt).T, metric = 'cosine')
Neighbor_only_result_ = func.get_neighbor((S*Vt).T, 'cosine', Trainset, games, test_data, 50)

#bagging the model
Bagging_df = pd.concat([SVD_Neighbor, Neighbor_only_result_[['member_id', 'game_id']]], axis = 0)
Bagging_result = func.Bagging(Bagging_df, Trainset)


'''
Evaluate metrics
'''
#Hot Game
auc_notintrain = func.Get_AUC_notintrain(Neighbor_only_result, test_data, HotGame_inTrain)

#AUC
auc_Bagging = func.Get_AUC_intrain(Bagging_result, test_data)
auc_SVD_new = func.Get_AUC_intrain(SVD_Neighbor_result, test_data)
auc_cosinesimilarity = func.Get_AUC_intrain(Neighbor_only_result, test_data)
auc_SparseSVD = func.Get_AUC_intrain(Neighbor_only_result_, test_data)
auc_BindModel = func.Get_AUC_intrain(Bind_result, test_data)
auc_SVD = func.Get_AUC_intrain(Latent_only_result, test_data)

#NDCG
ndcg_Bagging = func.NDCG(Bagging_result, test_data)
ndcg_SVD_new = func.NDCG(SVD_Neighbor_result, test_data)
ndcg_cosinesimilarity = func.NDCG(Neighbor_only_result, test_data)
ndcg_SparseSVD = func.NDCG(Neighbor_only_result_, test_data)
ndcg_BindModel = func.NDCG(Bind_result, test_data)
ndcg_SVD = func.NDCG(Latent_only_result, test_data)

#Precision
Precision_Bagging = func.Get_Precision(Bagging_result, test_data)
Precision_SVD_new = func.Get_Precision(SVD_Neighbor_result, test_data)
Precision_cosinesimilarity = func.Get_Precision(Neighbor_only_result, test_data)
Precision_SparseSVD = func.Get_Precision(Neighbor_only_result_, test_data)
Precision_BindModel = func.Get_Precision(Bind_result, test_data)
Precision_SVD = func.Get_Precision(Latent_only_result, test_data)

#Recall
Recall_Bagging = func.Get_Recall(Bagging_result, test_data)
Recall_SVD_new = func.Get_Recall(SVD_Neighbor_result, test_data)
Recall_cosinesimilarity = func.Get_Recall(Neighbor_only_result, test_data)
Recall_SparseSVD = func.Get_Recall(Neighbor_only_result_, test_data)
Recall_BindModel = func.Get_Recall(Bind_result, test_data)
Recall_SVD = func.Get_Recall(Latent_only_result, test_data)

#Summary table
summary_dict = {'Bagging': [auc_Bagging, ndcg_Bagging, Precision_Bagging, Recall_Bagging],
                'SVD_new': [auc_SVD_new, ndcg_SVD_new, Precision_SVD_new, Recall_SVD_new],
                'Cosine Similarity': [auc_cosinesimilarity, ndcg_cosinesimilarity, Precision_cosinesimilarity, Recall_cosinesimilarity],
                'SparseSVD': [auc_SparseSVD, ndcg_SparseSVD, Precision_SparseSVD, Recall_SparseSVD],
                'DNN': [0.80, 0.76, 0.23, 0.611],
                'Bind Model': [auc_BindModel, ndcg_BindModel, Precision_BindModel, Recall_BindModel],
                'SVD': [auc_SVD, ndcg_SVD, Precision_SVD, Recall_SVD],
                'Sales Model': [0.32, 0.19, 0.039, 0.11]}
summary_df = pd.DataFrame(summary_dict, index = ['AUC', 'NDCG', 'Precision', 'Recall'])
print(summary_df)
summary_df.to_csv('recommender_result.csv')
