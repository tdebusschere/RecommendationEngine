import pandas as pd
import numpy as np
import scipy.sparse as sparse
from scipy.sparse import csr_matrix
from sparsesvd import sparsesvd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn import metrics
from surprise import SVD, Reader, Dataset

'''
Normalization Function
'''
def min_max_normalize(x):
    min = x.min()
    max = x.max()
    result = np.float64((x- min) / (max- min))
    return result

def StandardScaler(x):
    mean = x.mean()
    std = x.std()
    result = np.float64((x- mean)/ std)
    return result

def Mean_normalize(x):
    min = x.min()
    max = x.max()
    mean = x.mean()
    result = np.float64((x- mean)/ (max- min))
    return result  

'''
Pre_processing
'''
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
def Pre_processing_train(df):
    x = df.copy()
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
        FindHotGame = df.groupby('game_id', as_index=False).agg({'commissionable': ['sum']})
        FindHotGame.columns = ['game_id', 'feature']
    elif feature == 'member_id':
        FindHotGame = df.groupby('game_id', as_index=False).agg({'member_id': ['count']})
        FindHotGame.columns = ['game_id', 'feature']
    else:
        print('Not Defined')
        FindHotGame = pd.DataFrame(columns=['game_id', 'feature'])
    FindHotGame = FindHotGame.sort_values(by = ['feature'], ascending = False).reset_index(drop = True)
    HotGame = list(FindHotGame.game_id[0:n])
    return HotGame


def get_Trainset(df):
    Trainset = df[['member_id', 'game_id', 'dates', 'expect_earn_per_day','payoff']].copy()
    Trainset = Trainset.sort_values(by=['member_id', 'game_id'], ascending=True).reset_index(drop=True)
        
    zmm1 = min_max_normalize(Trainset[['expect_earn_per_day']])#min_max_normalize(Trainset[['expect_earn_per_day']])
    zmm2 = min_max_normalize(Trainset[['dates']])#min_max_normalize(Trainset[['dates']])
    zmm = 0.5 * zmm1 + 0.5 * zmm2
    Trainset.loc[:, 'score'] = zmm
    
    return Trainset

'''
Model -Cosine Similarity
'''
def get_sparse(Trainset):
    #model of the cosine similarity
    members = list(Trainset.member_id.unique())  # Get our unique members
    games = list(Trainset.game_id.unique())  # Get our unique games that were purchased
    score_list = list(Trainset.score) # All of our score
    # Get the associated row, column indices
    cols = Trainset.member_id.astype('category',
                                     categories = members).cat.codes
    rows = Trainset.game_id.astype('category',
                                    categories = games).cat.codes
    sparse_df = sparse.csr_matrix((score_list, (rows, cols)),
                                  shape=(len(games), len(members)))
    '''
    print('num of members : {}\nnum of games : {}\nnum of score : {}'.format(len(members),
          len(games), len(score_list)))
    print("shape of record_sparse: ", sparse_df.shape)
    '''
    return sparse_df


# what does the 30 come from?  95% percentile
def Recommendation_cosine(Trainset, sparse_df, games, N = 30):
    cosine_sim = cosine_similarity(sparse_df, sparse_df)
    cosine_sim = pd.DataFrame(data = cosine_sim, index=games, columns=games)
    ## get the neighbor 30 game of every user
    gamelist    = np.array(cosine_sim.columns)
    gamesplayed = Trainset.groupby(['member_id'])['game_id'].apply(list).reset_index(name='games')
    gamesmax    = np.array(gamesplayed.games.map(lambda x: ((cosine_sim.loc[x,:].values).max(axis=0))))
    def Get_neighbor_30(x):
        # x[x>0.99] = 0.0
        return (gamelist[np.flip(np.argsort(x, axis=0))[0:N, ]])
    filtered = list(map(Get_neighbor_30, gamesmax))
    filtered_array = np.array(filtered)
    filtered_array = filtered_array.reshape(filtered_array.shape[0] * filtered_array.shape[1], -1)
    filtered_array = filtered_array.reshape(-1,)
    Neighbor_result = pd.DataFrame({'member_id': np.repeat(np.array(np.unique(Trainset.member_id)), N, axis=0),
                                    'game_id': filtered_array})

    Neighbor_only_result = Neighbor_result.merge(Trainset[['member_id', 'game_id', 'score']], how = 'left', on = ['member_id', 'game_id'])
    Neighbor_only_result.score = np.where(Neighbor_only_result.score.isna(), 0, Neighbor_only_result.score)
    #sorted by the normalized expect return in training data
    Neighbor_only_result = Neighbor_only_result.sort_values(by = ['member_id', 'score'], ascending = False)
    Neighbor_only_result = Neighbor_only_result.groupby('member_id').head(12)
    
    return Neighbor_only_result, Neighbor_result


'''
Model - SVD_new
'''
def SVD_surprise_only(Trainset, N = 30):
    reader = Reader()
    Trainset_changetype = Dataset.load_from_df(Trainset[['member_id', 'game_id', 'score']], reader)
    Trainset_changetype_result = Trainset_changetype.build_full_trainset()
    svd = SVD(n_factors = 20,
              n_epochs = 20,
              lr_all = 0.01,#0.0001,
              random_state = 1234)
    svd.fit(Trainset_changetype_result)

    games = list(Trainset.game_id.unique()) # Get our unique games that were purchased

    #model SVD_New
    data = np.transpose(np.dot(svd.pu, np.transpose(svd.qi)))
    x = cosine_similarity(data, data)
    cosine_sim_x = pd.DataFrame(data = x, 
                                index=games,
                                columns=games)
    gamesplayed = Trainset.groupby(['member_id'])['game_id'].apply(list).reset_index(name='games')
    gamesmax = np.array(gamesplayed.games.map(lambda x: ((cosine_sim_x.loc[x,:].values).max(axis=0))))
    gamelist = np.array(cosine_sim_x.columns)
    
    def Get_neighbor_30(x):
            # x[x>0.99] = 0.0
            return (gamelist[np.flip(np.argsort(x, axis=0))[0:N, ]])
    filtered = list(map(Get_neighbor_30, gamesmax))
    filtered_array = np.array(filtered)
    filtered_array = filtered_array.reshape(filtered_array.shape[0] * filtered_array.shape[1], -1)
    filtered_array = filtered_array.reshape(-1,)
    SVD_Neighbor = pd.DataFrame({'member_id': np.repeat(np.array(np.unique(Trainset.member_id)), N, axis=0), 'game_id': filtered_array})
    #SVD_Neighbor_result = SVD_Neighbor.groupby('member_id').head(12)
    SVD_Neighbor_result = SVD_Neighbor.merge(Trainset[['member_id', 'game_id', 'score']], how = 'left', on = ['member_id', 'game_id'])
    SVD_Neighbor_result.score = np.where(SVD_Neighbor_result.score.isna(), 0, SVD_Neighbor_result.score)
    SVD_Neighbor_result = SVD_Neighbor_result.sort_values(by = ['member_id', 'score'], ascending = False)
    SVD_Neighbor_result = SVD_Neighbor_result.groupby('member_id').head(12)
    
    return SVD_Neighbor, SVD_Neighbor_result


'''
Model - SVD, Bind Model
'''
def SVD_surprise(Trainset):
    reader = Reader()
    Trainset_changetype = Dataset.load_from_df(Trainset[['member_id', 'game_id', 'score']], reader)
    Trainset_changetype_result = Trainset_changetype.build_full_trainset()
    svd = SVD(n_factors = 20,
              n_epochs = 20,
              lr_all = 0.01,#0.0001,
              random_state = 1234)
    svd.fit(Trainset_changetype_result)


    members = list(Trainset.member_id.unique()) # Get our unique members
    games = list(Trainset.game_id.unique()) # Get our unique games that were purchased
    
    #model SVD
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
    
    return Latent_only_result, Latent_result_df

'''
Model -SparseSVD
'''
def get_sparse_matrix(Trainset):
    members = list(Trainset.member_id.unique())  # Get our unique members
    games = list(Trainset.game_id.unique())  # Get our unique games that were purchased
    score_list = list(Trainset.score)  # All of our dates
    #print('num of members : {}\nnum of games : {}\nnum of score : {}'.format(len(members),
    #                                                                         len(games), len(score_list)))

    # Get the associated row,column indices
    rows = Trainset.member_id.astype('category', categories=members).cat.codes
    cols = Trainset.game_id.astype('category', categories=games).cat.codes
    sparse_df = sparse.csc_matrix((score_list, (rows, cols)), shape=(len(members), len(games)))
    #print("shape of record_sparse: ", sparse_df.shape)
    return sparse_df


def computeSVD(urm, K):
	U, s, Vt = sparsesvd(urm, K)

	dim = (len(s), len(s))
	S = np.zeros(dim, dtype=np.float32)
	for i in range(0, len(s)):
		S[i,i] = s[i]

	U = csr_matrix(np.transpose(U), dtype=np.float32)
	S = csr_matrix(S, dtype=np.float32)
	Vt = csr_matrix(Vt, dtype=np.float32)

	return U, S, Vt

def get_threshold(array, percentage):
    denom = sum(elem ** 2 for elem in array)
    k = 0
    sums = 0
    for elem in array:
        sums += elem ** 2
        k += 1
        if sums >= percentage * denom:
            return k

def Dim_reduction(sparse_df):
    #Original SVD
    U_, S_, Vt_ = computeSVD(sparse_df, min(sparse_df.shape[0], sparse_df.shape[1])-1)
    k = get_threshold(S_.diagonal(), 0.9)
    return k


#V = (S*Vt).T
def get_neighbor(V, distance, Trainset, games, test_data, N = 30):
    from sklearn.metrics import pairwise_distances
    similarity_matrix = pairwise_distances(V, V, metric = distance)
    similarity_matrix = pd.DataFrame(data = similarity_matrix, index = games, columns = games)
    gamelist = np.array(Trainset.game_id.unique())
    gamesplayed = Trainset.groupby(['member_id'])['game_id'].apply(list).reset_index(name='games')
    gamesmin    = np.array(gamesplayed.games.map(lambda x: ((similarity_matrix.loc[x,:].values).min(axis=0))))

    def Get_neighbor_30(x):
        # x[x>0.99] = 0.0
        return (gamelist[np.flip(np.argsort(-x, axis=0))[0:N, ]])

    filtered = list(map(Get_neighbor_30, gamesmin))

    filtered_array = np.array(filtered)
    filtered_array = filtered_array.reshape(filtered_array.shape[0]* filtered_array.shape[1],-1)
    filtered_array = filtered_array.reshape(-1, ) #將30*n的數組轉成(-1,)維的陣列

    Neighbor_result = pd.DataFrame({'member_id':np.repeat(np.array(np.unique(Trainset.member_id)),
                                                N,
                                                axis=0),
                                    'game_id': filtered_array}) #給定每個玩家在每款遊戲的名單
    Neighbor_result['recommend'] = 1
    Neighbor_only_result = Neighbor_result.merge(Trainset[['member_id', 'game_id', 'score']],
                                           how = 'left',
                                           on = ['member_id', 'game_id']) #把每個人的SCORE收集起來
    Neighbor_only_result.score = np.where(Neighbor_only_result.score.isna(), 0, Neighbor_only_result.score)
    #sorted by the normalized expect return in training data
    Neighbor_only_result = Neighbor_only_result.sort_values(by = ['member_id', 'score'], ascending = False)
    Neighbor_only_result = Neighbor_only_result.groupby('member_id').head(12)
    Neighbor_only_result.loc[:, 'Order'] = list(range(1,13)) * len(np.unique(Neighbor_only_result.member_id))
    
    Neighbor_only_result = Neighbor_only_result.drop(['recommend'], axis=1)
    
    return Neighbor_only_result

'''
Model - Bagging
'''
def Bagging(Bagging_df, Trainset):
    #value_counts() set assending =False as default.
    Bagging_df_groupby = Bagging_df.groupby(['member_id']).apply(lambda x: pd.DataFrame(x.game_id.value_counts().index).iloc[0:12, ])
    Bagging_result_12 = pd.DataFrame(Bagging_df_groupby)
    Bagging_result_12.columns = ['game_id']
    Bagging = pd.DataFrame({'member_id':np.repeat(np.array(np.unique(Trainset.member_id)), 12, axis=0),
    'game_id': Bagging_result_12.game_id}).reset_index(drop=True)
    
    Bagging_result = Bagging.merge(Trainset[['member_id', 'game_id', 'score']], how = 'left', on = ['member_id', 'game_id'])
    Bagging_result.score = np.where(Bagging_result.score.isna(), 0, Bagging_result.score)
    
    Bagging_result = Bagging_result.sort_values(by = ['member_id', 'score'], ascending = False)    
    
    return Bagging_result


'''
Evaluation metric
'''
def NDCG(input_df, test_df):
    def get_Testset_score(df):
        temp = df.copy()
        temp = temp.sort_values(by=['member_id', 'game_id'], ascending=True).reset_index(drop=True)
    
        zmm1 = min_max_normalize(temp[['expect_earn_per_day']])
        zmm2 = min_max_normalize(temp[['dates']])
        zmm = 0.5 * zmm1 + 0.5 * zmm2
        temp.loc[:, 'score'] = zmm
        return temp

    def DCG(df):
        data = df.copy()
        data['dcg_rank'] = list(range(2, 14)) * int(data.shape[0] / 12)
        from math import log
        data['dcg'] = data.apply(lambda row: (2 ** row.score - 1) / log(row.dcg_rank, 2), axis=1)
        dcg = data.groupby('member_id', as_index=False)['dcg'].sum()
        return dcg
    
    def iDCG(df):
        data = df.copy()
        data = data.sort_values(by=['member_id', 'score'], ascending=False)
        data['dcg_rank'] = list(range(2, 14)) * int(data.shape[0] / 12)
        from math import log
        data['idcg'] = data.apply(lambda row: (2 ** row.score - 1) / log(row.dcg_rank, 2), axis=1)
        idcg = data.groupby('member_id', as_index=False)['idcg'].sum()
        return idcg
    
    df = input_df.copy()
    test = get_Testset_score(test_df)
    member_list = np.unique(test.member_id)
    df = df[df.member_id.isin(member_list)]
    df = df[['member_id', 'game_id']].merge(test[['member_id', 'game_id', 'score']],
                                            how = 'left',
                                            on = ['member_id', 'game_id']).fillna(0)
    dcg = DCG(df)
    idcg = iDCG(df)
    ndcg = dcg.merge(idcg, on='member_id', how='left')
    ndcg['NDCG'] = ndcg['dcg']/ndcg['idcg']
    ndcg.NDCG = np.where(ndcg.NDCG.isna(), 0, ndcg.NDCG)
    
    NDCG_value = np.mean(ndcg.NDCG)
    
    return NDCG_value

def Get_AUC_intrain(input_df, test_df):
    #hot game must be a list and there were 12 container.
    test_data = test_df.copy()
    df = input_df.copy()
    member_list = np.unique(test_data.member_id)
    df = df[df.member_id.isin(member_list)]
    df = df[['member_id', 'game_id']].merge(test_data[['member_id', 'game_id', 'commissionable']],
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
    
    return auc_score

def Get_AUC_notintrain(input_df, test_df, HotGame):
    #hot game must be a list and there were 12 container.
    test_data = test_df.copy()
    raw = test_data[['member_id']].drop_duplicates().merge(input_df[['member_id', 'game_id']],
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
    
    return (auc_score)


def Get_Precision(df, test_df):
    data = df.copy()
    data = data.merge(test_df[['member_id', 'game_id', 'commissionable']],
                      how = 'left',
                      on = ['member_id', 'game_id'])
    data.commissionable = np.where(data.commissionable.isna(), 0, 1)
    
    Precision = sum(data.commissionable)/len(data.commissionable)

    return Precision

def Get_Recall(df, test_df):
    data = df.copy()
    test = test_df.copy()
    test_intrain = test[test.member_id.isin(data.member_id)]
    data.loc[:, 'Recommend'] = 1
    test_intrain = test_intrain.merge(data[['member_id', 'game_id', 'Recommend']],
                                      how = 'left',
                                      on = ['member_id', 'game_id'])
    test_intrain.Recommend = np.where(test_intrain.Recommend.isna(), 0, 1)
    group = test_intrain[['member_id', 'game_id', 'Recommend']].groupby(['member_id']).apply(lambda x:(sum(x.Recommend))/(len(x.member_id)))
       
    Recall = sum(group)/len(np.unique(test_intrain.member_id))#sum(test_intrain.Recommend)/len(test_intrain.Recommend)
    
    return Recall

