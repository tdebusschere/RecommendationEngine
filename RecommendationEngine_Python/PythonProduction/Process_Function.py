import pandas as pd
import numpy as np
import time
import logging
import scipy.sparse as sparse
#from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from sklearn import metrics
from surprise import SVD, Reader, Dataset
import asyncio
import nest_asyncio
nest_asyncio.apply()
logging.basicConfig(filename= 'run.log', level = logging.ERROR) #loggin設定
pd.set_option('display.max_columns', None)

'''
Function Zone
'''
#Normalization Function
def min_max_normalize(x):
    min = x.min()
    max = x.max()
    result = np.float64((x- min) / (max- min))
    return result

def get_data(IP, DailyQueryTable, start, end):
    declare = "DECLARE @Start_Date DATETIME, @End_Date DATETIME SET @Start_Date = '{}' SET @End_Date = '{}'".format(start, end)
    sqlquery = " SELECT [siteid], [gameaccount], [gametypesourceid], Sum([commissionable]) [Commissionable], Count(1) [Dates] \
                 FROM \
                 (SELECT CONVERT(DATE, [dateplayed]) DAY, [gameaccount],\
                 [siteid], [gametypesourceid], Sum([commissionable]) [Commissionable] \
                 FROM {table} (nolock) \
                 WHERE [dateplayed] >= @Start_Date AND [dateplayed] <= @End_Date \
                 GROUP BY CONVERT(DATE, [dateplayed]), [gameaccount], [siteid], [gametypesourceid])Y \
                 GROUP BY [siteid], [gameaccount], [gametypesourceid]\
                 order by [gametypesourceid],[gameaccount],[siteid]".format(table=DailyQueryTable)
    FullString = declare + sqlquery
    data = IP.ExecQuery(FullString)
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
    result['Membercode'] = list(zip(result['SiteID'], result['Member']))

    
    return result

#Define the hot game for the newuser or the user not in train data
def Hot_Game(df, feature='Commissionable', n=15):
    if feature == 'Commissionable':
        FindHotGame = df.groupby('Game', as_index=False).agg({'Commissionable': ['sum']})
        FindHotGame.columns = ['Game', 'feature']

    elif feature == 'Member':
        FindHotGame = df.groupby('Game', as_index=False).agg({'Member': ['count']})
        FindHotGame.columns = ['game_id', 'feature']
    '''
    else:
        print('Not Defined')
        FindHotGame = pd.DataFrame(columns=['game_id', 'feature'])
    '''
    FindHotGame = FindHotGame.sort_values(by = ['feature'], ascending = False).reset_index(drop = True)
    HotGame = list(FindHotGame.Game[0:n])
    return HotGame

def Encoding_RS(users, games):
    
    userid2idx = {o:i for i,o in enumerate(users)}
    gameid2idx = {o:i for i,o in enumerate(games)}
    userid2Rraw = {i:o for i,o in enumerate(users)}
    gameid2iraw = {i:o for i,o in enumerate(games)}

    return (userid2idx, userid2Rraw, gameid2idx, gameid2iraw)


def Encoding_TrainData(train_data, userid2idx, gameid2idx):
    
    tmp = train_data.copy()
    tmp.loc[:, 'Member_encoding'] = tmp['Membercode'].apply(lambda x: userid2idx[x])
    tmp.loc[:, 'Game_encoding'] = tmp['Game'].apply(lambda x: gameid2idx[x])

    return tmp


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
Model
'''
def Calculate_Similarity(Trainset): #, N = 30):
    reader = Reader()
    Trainset_changetype = Dataset.load_from_df(Trainset[['Member_encoding', 'Game_encoding', 'score']], reader)

    Trainset_changetype_result = Trainset_changetype.build_full_trainset()
    svd = SVD(n_factors = 20,
              n_epochs = 20,
              lr_all = 0.01,#0.0001,
              random_state = 1234)
    svd.fit(Trainset_changetype_result)

    games = list(Trainset.Game_encoding.unique()) # Get our unique games that were purchased
    

    #s = time.time()
    chunk = 300

    tmp2 = np.zeros([len(games), len(games)]) 
    norms = dict()
    cut0 = np.zeros((np.shape(svd.pu)[0],chunk))
    cut1 = np.zeros((np.shape(svd.pu)[0],chunk))

    
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
            
            #e = time.time()
            print(str((k, l)) + '_end')
            #print(e- s)    
                
    #model SVD_New
    cosine_sim_x = pd.DataFrame(data = tmp2, 
                                index = games,
                                columns = games)
    
    return cosine_sim_x


def Cosine_Similarity(Trainset):
    
    similarity = Calculate_Similarity(Trainset)

    return similarity


def Exclude_Game(BalanceCenter_190, category_exclude, games, gameid2idx):
    #exclude by user
    #will be added in the future
    
    
    #exclude the game that can be connect directly: category ('視訊', '體育', '彩票')
    sqlquery = "SELECT a.[GameTypeSourceId], \
               a.[Type], \
               b.[Category] \
               FROM   [BalanceOutcome].[dbo].[lookuptable] a \
               LEFT JOIN (SELECT [RawDataType] type, \
                         [Category] \
                         FROM   [DataPool].[dbo].[dbo.vw_gamelookup] \
                         GROUP  BY [RawDataType], \
                            [Category]) b \
                          ON a.type = b.type"
    data_category = BalanceCenter_190.ExecQuery( sqlquery )
    data_category.columns = ['GameTypeSourceId', 'Type', 'Category']
    connect_indirectly = data_category[(data_category.Category.isin(category_exclude)) & \
                                       (~data_category.Category.isna()) &\
                                       (~data_category.GameTypeSourceId.isna())]
    
    connect_indirectly['GameTypeSourceId'] = connect_indirectly['GameTypeSourceId'].astype('int64')
    connect_indirectly = connect_indirectly[connect_indirectly.GameTypeSourceId.isin(games)].reset_index(drop=True)
    connect_indirectly.loc[:, 'Game_encoding'] = connect_indirectly['GameTypeSourceId'].apply(lambda x: gameid2idx[x])                                 
    #connect_indirectly = list(np.int64(connect_indirectly))  
    exclude_game_list_raw = list(connect_indirectly.GameTypeSourceId)
    exclude_list = list(connect_indirectly.Game_encoding)
    
    return (exclude_game_list_raw, exclude_list)


def Summarized_cosine_sim_df(cosine_sim, gameid2iraw, current, connect_indirectly, n = 100):
    
    df = cosine_sim.copy()
    df.loc[:, 'Game'] = df.index
    cosine_sim2 = pd.melt(df, 
                          id_vars=['Game'],
                          var_name='CorrespondGame', 
                          value_name='CS')
    cosine_sim2['CorrespondGame'] = cosine_sim2['CorrespondGame'].astype('int64')

    cosine_sim2 = cosine_sim2[cosine_sim2.Game != cosine_sim2.CorrespondGame]
    cosine_sim2 = cosine_sim2.reset_index(drop=True)

    cosine_sim3 = cosine_sim2.copy()    
    cosine_sim3.loc[:, 'Game_raw'] = cosine_sim3['Game'].apply(lambda x: gameid2iraw[x])
    cosine_sim3.loc[:, 'CorrespondGame_raw'] = cosine_sim3['CorrespondGame'].apply(lambda x: gameid2iraw[x])
    cosine_sim3 = cosine_sim3[~cosine_sim3.CorrespondGame_raw.isin(connect_indirectly)].reset_index(drop=True)  

    cosine_sim3 = cosine_sim3.sort_values(by = ['Game_raw', 'CS'],
                                          ascending = [True, False])
    cosine_sim3 = cosine_sim3.reset_index(drop=True)    
    cosine_sim3 = cosine_sim3.groupby('Game_raw').head(n).reset_index(drop=True)

    cosine_sim_final = cosine_sim3[['Game_raw', 'CorrespondGame_raw', 'CS']]
    
    cosine_sim_final.loc[:, 'UpdateTime'] = current
    cosine_sim_final.columns = ['Game', 'CorrespondGame', 'CosineSimilarity', 'UpdateTime']
    
    return cosine_sim_final

'''
def SVD_surprise_only_tom(Trainset, N = 30, top = 12):
    similarity = Calculate_Similarity(Trainset)
    xmm = asyncio.get_event_loop()
    data = xmm.run_until_complete( find_top_K(Trainset, similarity, N, step=1000))
    return (similarity, data)
    
def Get_svd_neighbor(Trainset, similarity, N = 30, top = 12):
    
    xmm = asyncio.get_event_loop()
    data = xmm.run_until_complete( find_top_K(Trainset, similarity, N, step=1000) )
    
    return (data)

def Original_Similarity(Trainset):
    reader = Reader()
    Trainset_changetype = Dataset.load_from_df(Trainset[['Member_encoding', 'Game_encoding', 'score']], reader)
    Trainset_changetype_result = Trainset_changetype.build_full_trainset()
    svd = SVD(n_factors = 20,
              n_epochs = 20,
              lr_all = 0.01,#0.0001,
              random_state = 1234)
    svd.fit(Trainset_changetype_result)

    games = list(Trainset.Game_encoding.unique()) # Get our unique games that were purchased
    data = np.transpose(np.dot(svd.pu, np.transpose(svd.qi)))
    x = cosine_similarity(data, data)
    cosine_sim = pd.DataFrame(data = x, 
                                index=games,
                                columns=games)

    return(cosine_sim)



async def find_top_K(Trainset, cosine_sim,  K = 30,  step = 10000):
    start = time.time()
    loop = asyncio.get_event_loop()
    keys, values = Trainset.loc[:,['Member_encoding','Game_encoding']].values.T 
    ukeys, index = np.unique(keys,True)
    arrays       = np.split(values, index[1:])
    gamesplayed          = pd.DataFrame({'a':ukeys,
                                         'b':[list(a) for a in arrays]})
    gamesplayed.columns = ['Member_encoding','games']
    print("--- %s seconds ---" % (time.time() - start))
    
    
    #gamesplayed = Trainset.groupby(['Member_encoding'])['Game_encoding'].apply(lambda x: list(x)).reset_index(name='games')
    cosine2 = cosine_sim.to_numpy()
    coskeys = np.array(cosine_sim.index)
    coskey_col = np.array(cosine_sim.columns)
    
    print("--- %s seconds ---" % (time.time() - start))
    lup = dict()
    for key in range(np.shape(cosine2)[0]):
        lup[coskeys[key]]= cosine2[key,:]

    totalsize = np.shape(gamesplayed)[0]

    batches = totalsize // step
    tasks = []
        
    for k in range(batches):
        l = k + 1
        batch = gamesplayed.iloc[ k*step:l*step,: ]
        tasks.append(loop.create_task(process(batch, lup, k*step, coskeys, coskey_col,step, K)))
    laststep  = batches *step

    try:
        lastbatch = gamesplayed.iloc[ laststep : (totalsize),:]
        tasks.append(loop.create_task(process(lastbatch,lup,laststep,coskeys, coskey_col,step,K)))
    except:
        pass
    res = await asyncio.gather(*tasks)
    filtered_array = np.vstack(res)
    filtered_array = filtered_array[0:(totalsize),:]
    filtered_array = filtered_array[filtered_array[:,0].argsort()]
    user           = filtered_array[:,0]
    filtered_array = filtered_array[:,1:(K+1)]
    filtered_array = filtered_array.reshape( filtered_array.shape[0] * filtered_array.shape[1], -1)

    SVD_Neighbor = pd.DataFrame({'Member_encoding': np.repeat(user, K), 
                                 'Game_encoding': filtered_array[:,0]})
        
    SVD_Neighbor['Member_encoding'] = SVD_Neighbor['Member_encoding'].astype('int64')
    SVD_Neighbor['Game_encoding'] = SVD_Neighbor['Game_encoding'].astype('int64')
    
    return (SVD_Neighbor) 


    
async def process(batch, lup, startindex, coskeys, coskey_col, batchsize, K):
    lookup  = batch.games.to_numpy()
    indx = 0
    goal = np.zeros((batchsize,K+1))
    for key in lookup: 
        #print(key)
        res = [ lup[val] for val in key ]
        #added by jy on 11/18 --start
        position_boolean = np.isin(coskey_col, key)
        position = np.where(position_boolean == True)
        for j in position:
            for b in range(len(res)):
                res[b][j] = 0
        #added by jy on 11/18 --end
        
        results = np.max(res, axis=0).argsort()[::-1][0:K]
        
        goal[indx,1:(K+1)] = coskey_col[results]
        goal[indx,0] = startindex + indx
        indx = indx + 1
    return (goal)



def summarized_data_tosql(SVD_Neighbor, userid2Rraw, current):
    SVD_Neighbor_result = SVD_Neighbor.copy()

    final_df = pd.DataFrame(np.array(SVD_Neighbor_result.Game_encoding).reshape(SVD_Neighbor_result.Member_encoding.nunique(), 12))
    final_df.columns = ['Game1', 'Game2', 'Game3', 'Game4', 'Game5', 'Game6', 'Game7', 'Game8', 'Game9', 'Game10', 'Game11', 'Game12']
    final_df.loc[:, 'Member_encoding'] = np.unique(SVD_Neighbor_result.Member_encoding)
    final_df.loc[:, 'Member'] = final_df['Member_encoding'].apply(lambda x: userid2Rraw[x])
    final_df = final_df.reset_index(drop=True)

    y = pd.DataFrame(list(final_df.Member))
    y.columns = ['SiteId', 'GameAccount']
    final_df.loc[:, 'SiteId'] = y.SiteId
    final_df.loc[:, 'GameAccount'] = y.GameAccount
    
    final_df.loc[:, 'Updatetime'] = current
    final_df_tosql = final_df[['SiteId', 'GameAccount', 'Game1', 'Game2', 'Game3',\
                               'Game4', 'Game5', 'Game6', 'Game7', 'Game8', 'Game9', 'Game10', \
                               'Game11', 'Game12', 'Updatetime']]
    final_df_tosql = final_df_tosql.reset_index(drop=True)
    
    return final_df_tosql
'''


'''
Evaluation metric
'''
def NDCG(input_df, test_df):
    def get_Testset_score(df):
        temp = df.copy()
        temp = temp.sort_values(by=['MemberCode', 'Game'], ascending=True).reset_index(drop=True)
    
        zmm1 = min_max_normalize(temp[['Commissionable']])
        zmm2 = min_max_normalize(temp[['Dates']])
        zmm = 0.5 * zmm1 + 0.5 * zmm2
        temp.loc[:, 'score'] = zmm
        return temp

    def DCG(df):
        data = df.copy()
        data['dcg_rank'] = list(range(2, 14)) * int(data.shape[0] / 12)
        data.loc[:, 'dcg'] = (2 ** data['score'] - 1)/ np.log2(data['dcg_rank'])
        
        dcg = data.groupby('MemberCode', as_index=False)['dcg'].sum()
        return dcg
    
    def iDCG(df):
        data = df.copy()
        data = data.sort_values(by=['MemberCode', 'score'], ascending=False)
        data['dcg_rank'] = list(range(2, 14)) * int(data.shape[0] / 12)

        data.loc[:, 'idcg'] = (2 ** data['score'] - 1)/ np.log2(data['dcg_rank'])
        idcg = data.groupby('MemberCode', as_index=False)['idcg'].sum()
        return idcg
    
    df = input_df.copy()
    test = get_Testset_score(test_df)
    member_list = np.unique(test.MemberCode)
    df = df[df.MemberCode.isin(member_list)]
    df = df[['MemberCode', 'Game']].merge(test[['MemberCode', 'Game', 'score']],
                                          how = 'left',
                                          on = ['MemberCode', 'Game']).fillna(0)
    dcg = DCG(df)
    idcg = iDCG(df)
    ndcg = dcg.merge(idcg, on='MemberCode', how='left')
    ndcg['NDCG'] = ndcg['dcg']/ndcg['idcg']
    ndcg.NDCG = np.where(ndcg.NDCG.isna(), 0, ndcg.NDCG)
    
    NDCG_value = np.mean(ndcg.NDCG)
    
    return NDCG_value

def Get_AUC_intrain(input_df, test_df):
    #hot game must be a list and there were 12 container.
    test_data = test_df.copy()
    df = input_df.copy()
    member_list = np.unique(test_data.MemberCode)
    df = df[df.MemberCode.isin(member_list)]
    df = df[['MemberCode', 'Game']].merge(test_data[['MemberCode', 'Game', 'Commissionable']],
                                          how = 'left',
                                          on = ['MemberCode', 'Game'])
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