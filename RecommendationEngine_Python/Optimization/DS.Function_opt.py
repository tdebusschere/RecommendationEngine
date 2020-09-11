import pyodbc
import pandas as pd
import numpy as np
import time
import logging
import sys
import scipy.sparse as sparse
#from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from sklearn import metrics
from surprise import SVD, Reader, Dataset
import asyncio

'''
Function zone
'''

'''
Connect Function
'''
class DB:
    def __init__(self, driver, server, uid, pwd):
        self.driver = driver
        self.server = server
        self.uid = uid
        self.pwd = pwd
    
    def __getConnect(self):
        try:
            self.conn = pyodbc.connect(driver=self.driver,
                                       server=self.server,
                                       uid=self.uid,
                                       pwd=self.pwd, ApplicationIntent='READONLY')
            cur = self.conn.cursor()
        except Exception as ex:
            logging.error('SQL Server connecting error, reason is: {}'.format(str(ex)))
            sys.exit()
        return cur
    
    def ExecQuery(self, sql):
        cur = self.__getConnect()
        try:
            cur.execute(sql)
            rows = cur.fetchall()
            colList = []
            for colInfo in cur.description:
                colList.append(colInfo[0]) 
            resultList = []
            for row in rows:
                resultList.append(list(row))
            df = pd.DataFrame(resultList, columns=colList)
        except pyodbc.Error as ex:
            logging.error('SQL Server.Error: {}'.format(str(ex)))
            sys.exit()
        cur.close()
        self.conn.close()
        return df
    
    def Executemany(self, sql, obj):
        cur = self.__getConnect()
        try:
            cur.executemany(sql, obj)
            self.conn.commit()
        except pyodbc.Error as ex:
            logging.error('SQL Server.Error: {}'.format(str(ex)))
        cur.close()
        self.conn.close()
    
    def ExecNoQuery(self, sql):
        cur = self.__getConnect()
        try:
            cur.execute(sql)
            self.conn.commit()
        except pyodbc.Error as ex:
            logging.error('SQL Server.Error: {}'.format(str(ex)))
        cur.close()
        self.conn.close()

'''
Normalization Function
'''
def min_max_normalize(x):
    min = x.min()
    max = x.max()
    result = np.float64((x- min) / (max- min))
    return result

def get_data(JG, start, end):
    declare = "DECLARE @Start_Date DATETIME, @End_Date DATETIME SET @Start_Date = '{}' SET @End_Date = '{}'".format(start, end)
    sqlquery = " SELECT [siteid], [gameaccount], [gametypesourceid], Sum([commissionable]) [Commissionable], \
                 Count(1) [Dates] FROM \
                 (SELECT CONVERT(DATE, [dateplayed]) DAY, [gameaccount], [siteid], [gametypesourceid], \
                 Sum([commissionable]) [Commissionable] \
                 FROM [DataScientist].[dbo].[ds_recommendersystemdailyquery] (nolock) \
                 WHERE [dateplayed] >= @Start_Date AND \
                 [dateplayed] <= @End_Date \
                 GROUP BY CONVERT(DATE, [dateplayed]), [gameaccount], [siteid], [gametypesourceid])Y \
                 GROUP BY [siteid], [gameaccount], [gametypesourceid]"
    FullString = declare + sqlquery
    data = JG.ExecQuery(FullString)
    return data

def Pre_processing(df):
    result = df.copy()
    #Change columns name
    result.columns = ['SiteID', 'Member', 'Game', 'Commissionable', 'Dates']
    
    result['SiteID'] = result['SiteID'].astype('object')
    result['Member'] = result['Member'].astype('object')
    result['Game'] = result['Game'].astype('object')
    result['Commissionable'] = result['Commissionable'].astype('float64')
    result['Dates'] = result['Dates'].astype('int64')
    
    return result


def Pre_processing_train(df):
    result = df.copy()
    
    # Exclude the following freak conditions
    Condition1 = result['Commissionable'] > 0
    Condition2 = result['Dates'] > 0   
    result = result[Condition1 | Condition2]

    ## Exclude the just play one game people( Noise and it can't give our model any help)    
    group = result.groupby('Member', as_index=False).agg({'Game': ['count']})
    group.columns = ['Member', 'count']
    group = group[group['count'] != 1]
    group = group.reset_index(drop=True)
    result = result[result.Member.isin(group.Member)]
    result = result.reset_index(drop=True)
    
    return result

#Define the hot game for the newuser or the user not in train data
def Hot_Game(df, feature = 'Commissionable', n = 15):
    if feature == 'Commissionable':
        FindHotGame = df.groupby('Game', as_index=False).agg({'Commissionable': ['sum']})
        FindHotGame.columns = ['Game', 'feature']
    '''
    elif feature == 'member_id':
        FindHotGame = df.groupby('game_id', as_index=False).agg({'member_id': ['count']})
        FindHotGame.columns = ['game_id', 'feature']
    else:
        print('Not Defined')
        FindHotGame = pd.DataFrame(columns=['game_id', 'feature'])
    '''
    FindHotGame = FindHotGame.sort_values(by = ['feature'], ascending = False).reset_index(drop = True)
    HotGame = list(FindHotGame.Game[0:n])
    return HotGame


def get_Trainset(df):
    Trainset = df[['Member_encoding', 'Game_encoding', 'Dates', 'Commissionable']].copy()
    Trainset = Trainset.sort_values(by=['Member_encoding', 'Game_encoding'], ascending=True).reset_index(drop=True)
        
    zmm1 = min_max_normalize(Trainset[['Commissionable']])
    zmm2 = min_max_normalize(Trainset[['Dates']])
    zmm = 0.5 * zmm1 + 0.5 * zmm2
    Trainset.loc[:, 'score'] = zmm
    
    return Trainset

'''
Model - Cosine Similarity
'''
def get_sparse(Trainset):
    #model of the cosine similarity
    members = list(Trainset.Member_encoding.unique())  # Get our unique members
    games = list(Trainset.Game_encoding.unique())  # Get our unique games that were purchased
    score_list = list(Trainset.score) # All of our score
    # Get the associated row, column indices
    cols = Trainset.Member_encoding.astype('category',
                                     categories = members).cat.codes
    rows = Trainset.Game_encoding.astype('category',
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
    gamesplayed = Trainset.groupby(['Member_encoding'])['Game_encoding'].apply(list).reset_index(name='games')
    gamesmax    = np.array(gamesplayed.games.map(lambda x: ((cosine_sim.loc[x,:].values).max(axis=0))))
    def Get_neighbor_30(x):
        # x[x>0.99] = 0.0
        return (gamelist[np.flip(np.argsort(x, axis=0))[0:N, ]])
    filtered = list(map(Get_neighbor_30, gamesmax))
    filtered_array = np.array(filtered)
    filtered_array = filtered_array.reshape(filtered_array.shape[0] * filtered_array.shape[1], -1)
    filtered_array = filtered_array.reshape(-1,)
    Neighbor_result = pd.DataFrame({'Member_encoding': np.repeat(np.array(np.unique(Trainset.Member_encoding)), N, axis=0),
                                    'Game_encoding': filtered_array})

    Neighbor_only_result = Neighbor_result.merge(Trainset[['Member_encoding', 'Game_encoding', 'score']],
                                                 how = 'left',
                                                 on = ['Member_encoding', 'Game_encoding'])
    Neighbor_only_result.score = np.where(Neighbor_only_result.score.isna(), 0, Neighbor_only_result.score)
    #sorted by the normalized expect return in training data
    Neighbor_only_result = Neighbor_only_result.sort_values(by = ['Member_encoding', 'score'], ascending = False)
    Neighbor_only_result = Neighbor_only_result.groupby('Member_encoding').head(12)
    
    return Neighbor_only_result, Neighbor_result

'''
Model - SVD_new
'''
def SVD_surprise_only(Trainset, N = 30):
    reader = Reader()
    Trainset_changetype = Dataset.load_from_df(Trainset[['Member_encoding', 'Game_encoding', 'score']], reader)
    Trainset_changetype_result = Trainset_changetype.build_full_trainset()
    svd = SVD(n_factors = 20,
              n_epochs = 20,
              lr_all = 0.01,#0.0001,
              random_state = 1234)
    svd.fit(Trainset_changetype_result)

    games = list(Trainset.Game_encoding.unique()) # Get our unique games that were purchased

    #model SVD_New
    data = np.transpose(np.dot(svd.pu, np.transpose(svd.qi)))
    x = cosine_similarity(data, data)
    cosine_sim_x = pd.DataFrame(data = x, 
                                index=games,
                                columns=games)
    gamesplayed = Trainset.groupby(['Member_encoding'])['Game_encoding'].apply(list).reset_index(name='games')
    gamesmax = np.array(gamesplayed.games.map(lambda x: ((cosine_sim_x.loc[x,:].values).max(axis=0))))
    gamelist = np.array(cosine_sim_x.columns)
    
    def Get_neighbor_30(x):
            # x[x>0.99] = 0.0
            return (gamelist[np.flip(np.argsort(x, axis=0))[0:N, ]])
    filtered = list(map(Get_neighbor_30, gamesmax))
    filtered_array = np.array(filtered)
    filtered_array = filtered_array.reshape(filtered_array.shape[0] * filtered_array.shape[1], -1)
    filtered_array = filtered_array.reshape(-1,)
    SVD_Neighbor = pd.DataFrame({'Member_encoding': np.repeat(np.array(np.unique(Trainset.Member_encoding)), N, axis=0), 
                                 'Game_encoding': filtered_array})
    #SVD_Neighbor_result = SVD_Neighbor.groupby('member_id').head(12)
    SVD_Neighbor_result = SVD_Neighbor.merge(Trainset[['Member_encoding', 'Game_encoding', 'score']],
                                             how = 'left',
                                             on = ['Member_encoding', 'Game_encoding'])
    SVD_Neighbor_result.score = np.where(SVD_Neighbor_result.score.isna(), 0, SVD_Neighbor_result.score)
    SVD_Neighbor_result = SVD_Neighbor_result.sort_values(by = ['Member_encoding', 'score'], ascending = False)
    SVD_Neighbor_result = SVD_Neighbor_result.groupby('Member_encoding').head(12)
    
    return SVD_Neighbor, SVD_Neighbor_result

def SVD_surprise_only_tom(Trainset): #, N = 30):
    reader = Reader()
    Trainset_changetype = Dataset.load_from_df(Trainset[['Member_encoding', 'Game_encoding', 'score']], reader)

    Trainset_changetype_result = Trainset_changetype.build_full_trainset()
    svd = SVD(n_factors = 20,
              n_epochs = 20,
              lr_all = 0.01,#0.0001,
              random_state = 1234)
    svd.fit(Trainset_changetype_result)

    games = list(Trainset.Game_encoding.unique()) # Get our unique games that were purchased

    s = time.time()
    chunk = 300

    tmp2 = np.zeros([len(games), len(games)]) 
    norms = dict()
    cut0 = np.zeros((np.shape(svd.pu)[0],chunk))
    cut1 = np.zeros((np.shape(svd.pu)[0],chunk))

    # changed type
    #s = time.time()          
    
    for k in range(0, np.shape(tmp2)[0]// chunk +1):
        minxindex = k * chunk
        maxxindex = ((k+1) * chunk)
        
        if k == np.shape(tmp2)[0]// chunk :
            maxxindex = np.shape(tmp2)[1] + 1
            cut0 = np.zeros((np.shape(svd.pu)[0],(maxxindex-minxindex - 1)))

        np.dot(svd.pu, np.transpose(svd.qi[minxindex:maxxindex,:]),out=cut0)
        norms[str(minxindex)] = np.linalg.norm(cut0, axis = 0)


        for l in range(0,k+1):
            #s = time.time()          
            minyindex = l * chunk             
            maxyindex = ((l+1) * chunk) #- 1
            if l == np.shape(tmp2)[0]// chunk:
                maxyindex = np.shape(tmp2)[1] + 1
            if (minxindex == minyindex) & (maxxindex == maxyindex):
                cut1 = np.copy(cut0)
            else:              
                np.dot(svd.pu, np.transpose(svd.qi[minyindex:maxyindex,:]), out= cut1)
                
            if( str(minyindex) not in norms):
                norms[str(minyindex)] = np.linalg.norm(cut1,axis=0)
            #tmp3[minxindex:maxxindex,minyindex:maxyindex] = cosine_similarity(np.transpose(cut0), np.transpose(cut1))
            
            tmp2[minxindex:maxxindex,minyindex:maxyindex] = np.dot(np.transpose(cut0), cut1) / \
                np.outer(norms[str(minxindex)] , norms[str(minyindex)])
            tmp2[minyindex:maxyindex,minxindex:maxxindex] = np.transpose(tmp2[minxindex:maxxindex,minyindex:maxyindex] )
            
            e = time.time()
            print(str((k, l)) + '_end')
            print(e- s)    
            
    e1 = time.time()
    
    #model SVD_New
    cosine_sim_x = pd.DataFrame(data = tmp2, 
                                index = games,
                                columns = games)
    return(cosine_sim_x)


'''
Evaluation metric
'''
def NDCG(input_df, test_df):
    def get_Testset_score(df):
        temp = df.copy()
        temp = temp.sort_values(by=['Member', 'Game'], ascending=True).reset_index(drop=True)
    
        zmm1 = min_max_normalize(temp[['Commissionable']])
        zmm2 = min_max_normalize(temp[['Dates']])
        zmm = 0.5 * zmm1 + 0.5 * zmm2
        temp.loc[:, 'score'] = zmm
        return temp

    def DCG(df):
        data = df.copy()
        data['dcg_rank'] = list(range(2, 14)) * int(data.shape[0] / 12)
        #from math import log
        #data['dcg'] = data.apply(lambda row: (2 ** row.score - 1) / log(row.dcg_rank, 2), axis=1)
        data.loc[:, 'dcg'] = (2 ** data['score'] - 1)/ np.log2(data['dcg_rank'])
        
        dcg = data.groupby('Member', as_index=False)['dcg'].sum()
        return dcg
    
    def iDCG(df):
        data = df.copy()
        data = data.sort_values(by=['Member', 'score'], ascending=False)
        data['dcg_rank'] = list(range(2, 14)) * int(data.shape[0] / 12)
        #from math import log
        #data['idcg'] = data.apply(lambda row: (2 ** row.score - 1) / log(row.dcg_rank, 2), axis=1)
        data.loc[:, 'idcg'] = (2 ** data['score'] - 1)/ np.log2(data['dcg_rank'])
        idcg = data.groupby('Member', as_index=False)['idcg'].sum()
        return idcg
    
    df = input_df.copy()
    test = get_Testset_score(test_df)
    member_list = np.unique(test.Member)
    df = df[df.Member.isin(member_list)]
    df = df[['Member', 'Game']].merge(test[['Member', 'Game', 'score']],
                                            how = 'left',
                                            on = ['Member', 'Game']).fillna(0)
    dcg = DCG(df)
    idcg = iDCG(df)
    ndcg = dcg.merge(idcg, on='Member', how='left')
    ndcg['NDCG'] = ndcg['dcg']/ndcg['idcg']
    ndcg.NDCG = np.where(ndcg.NDCG.isna(), 0, ndcg.NDCG)
    
    NDCG_value = np.mean(ndcg.NDCG)
    
    return NDCG_value

def Get_AUC_intrain(input_df, test_df):
    #hot game must be a list and there were 12 container.
    test_data = test_df.copy()
    df = input_df.copy()
    member_list = np.unique(test_data.Member)
    df = df[df.Member.isin(member_list)]
    df = df[['Member', 'Game']].merge(test_data[['Member', 'Game', 'Commissionable']],
                                            how = 'left',
                                            on = ['Member', 'Game'])
    df.Commissionable = np.where(df.Commissionable.isna(), 0, 1) #play as TP
    df.columns = ['Member', 'Game', 'TP']
    df.loc[:,'FP'] =  1 - df.TP

    aggregated = df[['Member', 'TP']].groupby('Member', as_index=False).sum()
    aggregated.loc[:,'FP_n'] =  12 - aggregated.TP
    aggregated.columns = ['Member', 'TP_n', 'FP_n']
    grade0_memberid = aggregated[aggregated.TP_n == 0].Member
    grade1_memberid = aggregated[aggregated.TP_n == 12].Member
    
   
    tmp = df[(~df.Member.isin(grade0_memberid)) & (~df.Member.isin(grade1_memberid))]
    #tmp = df[~df.member_id.isin(grade1_memberid)]
    tmp = tmp.merge(aggregated, how='left', on='Member')
    tmp.loc[:,'TPR'] =  tmp.TP / tmp.TP_n
    tmp.loc[:,'FPR'] =  tmp.FP / tmp.FP_n
    auc_df = tmp[['Member', 'TPR', 'FPR']].groupby(['Member']).apply(lambda x:metrics.auc(np.cumsum(x.FPR), np.cumsum(x.TPR)))
    
    auc_score = (auc_df.sum()+ 1*len(grade1_memberid))/len(np.unique(df.Member))
    
    return auc_score