"""
@author: Jimmy
"""
import pandas as pd
import numpy as np
import os                   
import random
import scipy.sparse as sparse
from scipy.sparse.linalg import spsolve
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from surprise import Reader, Dataset, evaluate, Dataset, accuracy
from surprise import SVD, SVDpp, SlopeOne, NMF, NormalPredictor, KNNBaseline, KNNBasic
from surprise import KNNWithMeans, KNNWithZScore, BaselineOnly, CoClustering
from surprise.model_selection import cross_validate, train_test_split, GridSearchCV
from sklearn import metrics
os.chdir('C:/Users/ADMIN/Desktop/Project/RecommendSystem') #cd

## function zone
def Pre_processing(df): 
    result = df.copy()
    result['member_id'] = result['member_id'].astype('int64')
    result['dates'] = result['dates'].astype('int64')
    result['game_id'] = result['game_id'].astype('int64')
    result['bet_amount'] = result['bet_amount'].astype('float64')
    result['payoff'] = result['payoff'].astype('float64')
    result['commissionable'] = result['commissionable'].astype('float64')
    result['Expect_earning_ratio'] = result['Expect_earning_ratio'].astype('float64')
    result['gamecode'] = result['gamecode'].astype('category')
    # We will multiply the ratio from the gamehall in the future but the information we haven't received yet.
    # and assume every gamehall keep the same ==1
    result.insert(11, 'expect_earn', result['commissionable'] * result['Expect_earning_ratio'])
    #result.insert(12, 'expect_earn_per_day', result['expect_earn'] / result['dates'])
    result.insert(12, 'expect_earn_per_day', result['commissionable'] )#/ result['dates']
    result.insert(13, 'key', list(zip(result['RawDataType'], result['gamecode'])))
    return(result)
    
## filter & Exclude the just play one game people( Noise and it can't give our model any help)
def Pre_processing_train(x): 
    x = x[(x['dates'] > 0) & (x['bet_amount'] > 0) & (x['commissionable'] > 0)]
    group = x.groupby('member_id', as_index=False).agg({'game_name': ['count']})
    group.columns = ['member_id', 'count']
    group = group[group['count'] != 1]
    group = group.reset_index(drop=True)
    x = x[x.member_id.isin(group.member_id)]
    x = x.reset_index(drop = True)
    return(x)

#Define the hot game for the newuser or the user not in train data 
def Hot_Game(df, feature = 'commissionable', n = 12): 
    if feature == 'commissionable':
        FindHotGame = train_data.groupby('game_id', as_index=False).agg({'commissionable': ['sum']})        
    elif feature == 'member_id':
        FindHotGame = train_data.groupby('game_id', as_index=False).agg({'member_id': ['count']})
    else:
        print('Not Defined') 
    FindHotGame.columns = ['game_id', 'feature']
    FindHotGame = FindHotGame.sort_values(by = ['feature'], ascending = False).reset_index(drop = True)
    HotGame = list(FindHotGame.game_id[0:n])
    return(HotGame)

# what does the 30 come from?  95% percentile
def Get_neighbor_30(x): 
    #x[x>0.99] = 0.0
    return(gamelist[np.flip(np.argsort(x,axis=0))[0:30,]])
    
def Get_AUC_intrain(input_df): 
    #hot game must be a list and there were 12 container.
    df = test_data[['member_id']].drop_duplicates().merge(input_df[['member_id', 'game_id']], 
                                                          how = 'left',
                                                          on = ['member_id'])
    df = df[~df.game_id.isna()]
    df['game_id'] = df['game_id'].astype('int64')

    
    df = df.merge(test_data[['member_id', 'game_id', 'commissionable']], 
                  how = 'left', 
                  on = ['member_id', 'game_id'])
    df.commissionable = np.where(df.commissionable.isna(), 0, 1) #play as TP
    df.columns = ['member_id', 'game_id', 'TP']
    df.loc[:,'FP'] =  1 - df.TP
    
    aggregated = df[['member_id', 'TP']].groupby('member_id', as_index=False).sum()
    aggregated.loc[:,'FP_n'] =  12 - aggregated.TP
    aggregated.columns = ['member_id', 'TP_n', 'FP_n']
    grade0_memberid = aggregated[aggregated.TP_n == 0].member_id

    tmp = df[~df.member_id.isin(grade0_memberid)]
    tmp = tmp.merge(aggregated, how='left', on='member_id')
    tmp.loc[:,'TPR'] =  tmp.TP / tmp.TP_n
    tmp.loc[:,'FPR'] =  tmp.FP / tmp.FP_n
    auc_df = tmp[['member_id', 'TPR', 'FPR']].groupby(['member_id']).apply(lambda x:metrics.auc(np.cumsum(x.FPR), np.cumsum(x.TPR))) 
    auc_score = auc_df.sum()/len(np.unique(df.member_id))
    return(auc_score)

def Get_AUC_notintrain(HotGame): 
    #hot game must be a list and there were 12 container.
    raw = test_data[['member_id']].drop_duplicates().merge(Trainset[['member_id', 'game_id']], 
                                                          how = 'left',
                                                          on = ['member_id'])
    NotIntrain = raw[raw.game_id.isna()].member_id
    df = pd.DataFrame({'member_id': np.repeat(np.array(NotIntrain), 12, axis=0),
                                  'game_id': HotGame * len(NotIntrain)})
    df['game_id'] = df['game_id'].astype('int64')
    #df = pd.concat([df, NotIntraindata])
    
    df = df.merge(test_data[['member_id', 'game_id', 'commissionable']], 
                  how = 'left', 
                  on = ['member_id', 'game_id'])
    df.commissionable = np.where(df.commissionable.isna(), 0, 1) #play as TP
    df.columns = ['member_id', 'game_id', 'TP']
    df.loc[:,'FP'] =  1 - df.TP
    
    aggregated = df[['member_id', 'TP']].groupby('member_id', as_index=False).sum()
    aggregated.loc[:,'FP_n'] =  12 - aggregated.TP
    aggregated.columns = ['member_id', 'TP_n', 'FP_n']
    grade0_memberid = aggregated[aggregated.TP_n == 0].member_id

    tmp = df[~df.member_id.isin(grade0_memberid)]
    tmp = tmp.merge(aggregated, how='left', on='member_id')
    tmp.loc[:,'TPR'] =  tmp.TP / tmp.TP_n
    tmp.loc[:,'FPR'] =  tmp.FP / tmp.FP_n
    auc_df = tmp[['member_id', 'TPR', 'FPR']].groupby(['member_id']).apply(lambda x:metrics.auc(np.cumsum(x.FPR), np.cumsum(x.TPR))) 
    auc_score = auc_df.sum()/len(np.unique(df.member_id))
    return(auc_score)
   
## Customer play record and exclude the recommend game(In SQL) because it is off line test
names = ['member_id', 'dates', 'game_id', 'game_name','game_category','bet_amount', 
         'payoff','commissionable','RawDataType','gamecode', 'Expect_earning_ratio']
train = pd.read_csv('TrainData_1101To1124.csv', header = 0, names = names) ##11/01~11/24 as training dataset
test = pd.read_csv('TestData_1125To1130.csv', header = 0, names = names) ##11/25~11/30 as test dataset    
    
## Pre_processing(define the feature type)
train_data = Pre_processing(train)
test_data = Pre_processing(test)

## filter & Exclude the just play one game people( Noise and it can't give our model any help)
train_data = Pre_processing_train(train_data)
# =============================================================================
# test_memberid = np.unique(test_data.member_id)
# train_data_memberid = np.unique(train_data.member_id)
# test_notintrain = test_data[~test_data.member_id.isin(train_data_memberid)]
# test_intrain = test_data[test_data.member_id.isin(train_data_memberid)]
# print(len(np.unique(test_notintrain.member_id)), len(np.unique(test_intrain.member_id)), 
#       len(np.unique(test_data.member_id)))
# nb_choice = train_data.groupby('member_id', as_index = False).agg({'game_id': ['count']})
# nb_choice.columns = ['member_id', 'countgame']
# nb_choice = nb_choice.sort_values(by = ['countgame'], ascending = False).reset_index(drop = True)
# nb_choice.countgame.describe()
# np.percentile(nb_choice.countgame, [95])
# =============================================================================


#Define the hot game for the newuser or the user not in train data 
HotGame_inTrain = Hot_Game(train_data,
                           feature = 'commissionable',
                           n = 12)


Trainset  = train_data[['member_id','game_id','dates', 'expect_earn_per_day']].sort_values(by = ['member_id', 'game_id'], ascending=True).reset_index(drop=True)


#zmm = ( Trainset.loc[:,'expect_earn_per_day'] - np.median(Trainset.loc[:,'expect_earn_per_day']) ) / np.std(Trainset.loc[:,'expect_earn_per_day'])
min_expect_earn_per_day =  Trainset[['expect_earn_per_day']].min()
max_expect_earn_per_day = Trainset[['expect_earn_per_day']].max()
min_dates =  Trainset[['dates']].min()
max_dates = Trainset[['dates']].max()

zmm1 = np.float64((Trainset[['expect_earn_per_day']] - min_expect_earn_per_day) / (max_expect_earn_per_day - min_expect_earn_per_day))
zmm2 = np.float64((Trainset[['dates']] - min_dates) / (max_dates - min_dates))
zmm = 0.5 * zmm1 + 0.5 * zmm2
#zmm = ( Trainset.loc[:,'expect_earn_per_day'] - np.median(Trainset.loc[:,'expect_earn_per_day']) ) / np.std(Trainset.loc[:,'expect_earn_per_day'])
Trainset.loc[:, 'score'] = zmm


#model of the cosine similarity
members = list(Trainset.member_id.unique()) # Get our unique members
games = list(Trainset.game_id.unique()) # Get our unique games that were purchased
score_list = list(Trainset.score) # All of our score

# Get the associated row, column indices
cols = Trainset.member_id.astype('category', 
                                 categories = members).cat.codes
rows =  Trainset.game_id.astype('category', 
                                categories = games).cat.codes 
sparse_df = sparse.csr_matrix((score_list, (rows, cols)), 
                              shape=(len(games), len(members)))
print('num of members : {}\nnum of games : {}\nnum of score : {}'.format(len(members), 
      len(games), len(score_list)))
print("shape of record_sparse: " , sparse_df.shape)


cosine_sim = cosine_similarity(sparse_df, sparse_df) 
cosine_sim = pd.DataFrame(data = cosine_sim,
                          index = games,
                          columns = games)

## get the neighbor 30 game of every user 
gamelist    = np.array(cosine_sim.columns)
gamesplayed = Trainset.groupby(['member_id'])['game_id'].apply(list).reset_index(name='games')
gamesmax    = np.array(gamesplayed.games.map(lambda x: ((cosine_sim.loc[x,:].values).max(axis=0))))
filtered = list(map(Get_neighbor_30, gamesmax))

filtered_array = np.array(filtered)
filtered_array = filtered_array.reshape(filtered_array.shape[0]* filtered_array.shape[1],-1)
filtered_array = filtered_array.reshape(-1, )


Neighbor_result = pd.DataFrame({'member_id':np.repeat(np.array(np.unique(Trainset.member_id)), 
                                            30,
                                            axis=0),
                                'game_id': filtered_array})
Neighbor_only_result = Neighbor_result.groupby('member_id').head(12)
Neighbor_only_result = Neighbor_only_result.merge(Trainset[['member_id', 'game_id', 'score']], 
                                                  how = 'left', 
                                                  on = ['member_id', 'game_id'])
Neighbor_only_result.score = np.where(Neighbor_only_result.score.isna(), 0, Neighbor_only_result.score)
#sorted by the normalized expect return in training data
Neighbor_only_result = Neighbor_only_result.sort_values(by = ['member_id', 'score'], ascending = False)

# =============================================================================
# Use the cv to find the hyperParameter
# reader = Reader()
# Trainset_changetype = Dataset.load_from_df(Trainset[['member_id', 'game_id', 'score']], reader)
# Trainset[['member_id', 'game_id', 'score']].dtypes
# 
# param_grid = {'n_factors': [20], 
#               'n_epochs': [20], 
#               'lr_all': [0.0008,0.001]}
# gs = GridSearchCV(SVD, 
#                   param_grid, 
#                   measures = ['rmse'],
#                   cv = 3)
# gs.fit(Trainset_changetype)
# 
# # Best RMSE score
# print(gs.best_score['rmse'])
# 
# # combination of parameters that gave the best RMSE score
# print(gs.best_params['rmse'])
# =============================================================================

# model of the latent factor
reader = Reader()
Trainset_changetype = Dataset.load_from_df(Trainset[['member_id', 'game_id', 'score']], reader)
Trainset_changetype_result = Trainset_changetype.build_full_trainset()
svd = SVD(n_factors = 20,
          n_epochs = 20,
          lr_all = 0.0001,
          random_state = 1234)
svd.fit(Trainset_changetype_result)


members = list(Trainset.member_id.unique()) # Get our unique members
games = list(Trainset.game_id.unique()) # Get our unique games that were purchased
Latent_result = pd.DataFrame(data = np.dot(svd.pu, np.transpose(svd.qi)),
                             index = members,
                             columns = games)
Latent_result_array = np.array(Latent_result)
Latent_result_array = Latent_result_array.reshape(Latent_result.shape[0] * Latent_result.shape[1], -1)
Latent_result_array = Latent_result_array.reshape(-1, )

Latent_result_df = pd.DataFrame({'member_id':np.repeat(np.array(Latent_result.index.unique()), 
                                                       len(games), 
                                                       axis = 0), 
                                 'game_id':games * len(members),
                                 'Pred_Return':Latent_result_array})    
Latent_only = Latent_result_df.sort_values(by = ['member_id', 'Pred_Return'], ascending = False)
Latent_only_result = Latent_only.groupby('member_id').head(12)


# model of the bind result
Result_BindLatentandNeighbor = Neighbor_result.merge(Latent_result_df, 
                                                     how = 'left',
                                                     on = ['member_id', 'game_id'])   
Result_BindLatentandNeighbor = Result_BindLatentandNeighbor.sort_values(by = ['member_id', 'Pred_Return'], ascending=False).reset_index(drop=True)    
Bind_result = Result_BindLatentandNeighbor.groupby('member_id').head(12)


#Result
auc = Get_AUC_intrain(Bind_result[['member_id', 'game_id']])
auc2 = Get_AUC_intrain(Latent_only_result[['member_id', 'game_id']])
auc3 = Get_AUC_intrain(Neighbor_only_result[['member_id', 'game_id']])
auc_notintrain = Get_AUC_notintrain(HotGame_inTrain)
print(auc, auc2, auc3, auc_notintrain)