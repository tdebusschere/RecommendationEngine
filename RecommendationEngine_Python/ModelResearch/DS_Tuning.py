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
    result.insert(12, 'expect_earn_per_day', result['expect_earn'] / result['dates'])
    #result.insert(12, 'expect_earn_per_day', result['commissionable'] / result['dates'])
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
    
def Get_AUC(input_df, HotGame): 
    #hot game must be a list and there were 12 container.
    df = test_data[['member_id']].drop_duplicates().merge(input_df[['member_id', 'game_id']], 
                                                          how = 'left',
                                                          on = ['member_id'])
    NotIntrain = df[df.game_id.isna()].member_id
    NotIntraindata = pd.DataFrame({'member_id': np.repeat(np.array(NotIntrain), 12, axis=0),
                                    'game_id': HotGame * len(NotIntrain)})
    df = df[~df.game_id.isna()]
    df['game_id'] = df['game_id'].astype('int64')
    df = pd.concat([df, NotIntraindata])
    
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


Trainset  = train_data[['member_id','game_id','dates', 'expect_earn_per_day']].sort_values(by = ['member_id', 'game_id'], ascending=True).reset_index(drop=True)


#zmm = ( Trainset.loc[:,'expect_earn_per_day'] - np.median(Trainset.loc[:,'expect_earn_per_day']) ) / np.std(Trainset.loc[:,'expect_earn_per_day'])
min =  Trainset[['expect_earn_per_day']].min()
max = Trainset[['expect_earn_per_day']].max()
zmm = np.float64((Trainset[['expect_earn_per_day']] - min) / (max - min))
Trainset.loc[:, 'score'] = zmm



# =============================================================================
# Use the cv to find the hyperParameter
reader = Reader()
Trainset_changetype = Dataset.load_from_df(Trainset[['member_id', 'game_id', 'score']], reader)
# =============================================================================
# param_grid = {'n_factors': [16, 32 , 64, 128, 256], 
#                'n_epochs': [20, 50 , 80, 100, 130, 150 , 180, 200], 
#                'lr_all': [0.0001,0.0003,0.0005,0.00065,0.0008, 0.001, 0.002]}
# =============================================================================
param_grid = {'n_factors': [32,64,128], 
               'n_epochs': [20, 50 , 80, 100, 130, 150 , 180, 200], 
               'lr_all': [0.0001,0.0003,0.0005,0.00065,0.0008, 0.001, 0.002],
               'biased':[True, False],
               'reg_all':[0.005, 0.01, 0.015, 0.02],
               'random_state':[1234],
               'verbose':[True]}

gs = GridSearchCV(SVD,
                  param_grid, 
                  measures = ['rmse'],
                  cv = 5,
                  return_train_measures  = True)
gs.fit(Trainset_changetype)

# Best RMSE score, parameters
print(gs.best_score['rmse'])
print(gs.best_params['rmse'])
results_df = pd.DataFrame.from_dict(gs.cv_results)
results_df
results_df.iloc[np.array(results_df.mean_test_rmse).argmin(0), :].params
results_df.to_csv('surprise_result.csv', index = 0)