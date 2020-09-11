import pyodbc
import pandas as pd
import numpy as np
import logging
import sys
import scipy.sparse as sparse
#from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from sklearn import metrics
from surprise import SVD, Reader, Dataset

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

def get_data(JG, start, end, system):
    declare = "DECLARE @Start_Date DateTime, @End_Date DateTime, @sys nvarchar(50) set @Start_Date = '{}' set @End_Date = '{}' set @sys = '{}' ".format(start, end, system)
    sqlquery = " SELECT [SystemCode], [MemberId], [RawDataType], [Code], \
                        COUNT(DAY) dates, SUM([CommissionableSum]) [CommissionableSum], SUM([WagersCount]) [WagersCount] \
                 FROM ( \
                         SELECT [SystemCode], [MemberId], CONVERT(date, DATE) DAY, [RawDataType], [Code], \
                                SUM([CommissionableSum]) [CommissionableSum], SUM([WagersCount]) [WagersCount] \
                         FROM [DataPool].[dbo].[DS_VW_BetRecordQueryGameSum] \
                         where DATE >= @Start_Date and DATE <= @End_Date  \
                         GROUP BY [SystemCode], [MemberId], CONVERT(date, DATE), [RawDataType], [Code] ) day \
                 GROUP BY [SystemCode], [MemberId], [RawDataType], [Code]"
    FullString = declare + sqlquery
    data = JG.ExecQuery(FullString)
    return data

def Pre_processing(df):
    result = df.copy()
    
    result['SystemCode'] = result['SystemCode'].astype('object')
    result['MemberId'] = result['MemberId'].astype('object')
    result['RawDataType'] = result['RawDataType'].astype('int64')
    result['Code'] = result['Code'].astype('object')
    result['dates'] = result['dates'].astype('int64')
    result['CommissionableSum'] = result['CommissionableSum'].astype('float64')
    result['WagersCount'] = result['WagersCount'].astype('float64')

    result.insert(7, 'Member', list(zip(result['SystemCode'], result['MemberId'])) )
    result.insert(8, 'Game', list(zip(result['RawDataType'], result['Code'])))
    return(result)


def Pre_processing_train(df):
    x = df.copy()
    x = x[(x['dates'] > 0) & (x['CommissionableSum'] > 0)]
    group = x.groupby('Member', as_index=False).agg({'Game': ['count']})

    group.columns = ['Member', 'count']
    group = group[group['count'] != 1]
    group = group.reset_index(drop=True)
    x = x[x.Member.isin(group.Member)]
    x = x.reset_index(drop = True)
    return(x)

#Define the hot game for the newuser or the user not in train data
def Hot_Game(df, feature = 'CommissionableSum', n = 15):
    if feature == 'CommissionableSum':
        FindHotGame = df.groupby('Game', as_index=False).agg({'CommissionableSum': ['sum']})
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
    Trainset = df[['Member_encoding', 'Game_encoding', 'dates', 'CommissionableSum']].copy()
    Trainset = Trainset.sort_values(by=['Member_encoding', 'Game_encoding'], ascending=True).reset_index(drop=True)
        
    zmm1 = min_max_normalize(Trainset[['CommissionableSum']])
    zmm2 = min_max_normalize(Trainset[['dates']])
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
    
    x = np.zeros([len(games), len(games)])
    
    
    for k in range(0, round(np.shape(x)[0]/200)+1):
        for l in range(0, round(np.shape(x)[0]/200)+1):
            minxindex = k*200 
            minyindex = l*200
            maxxindex = ((k+1) * 200) #- 1
            maxyindex = ((l+1) * 200) #- 1
            if k == round(np.shape(x)[0]/200):
                maxxindex = np.shape(x)[1] + 1
            if l == round(np.shape(x)[0]/200):
                maxyindex = np.shape(x)[1] + 1
            cut0  =  np.dot(svd.pu, np.transpose(svd.qi[minxindex:maxxindex,:]))
            cut1  =  np.dot(svd.pu, np.transpose(svd.qi[minyindex:maxyindex,:]))
            x[minxindex:maxxindex,minyindex:maxyindex] = cosine_similarity(np.transpose(cut0), np.transpose(cut1))

    #model SVD_New
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

