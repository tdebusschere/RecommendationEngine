# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 09:39:27 2019

@author: DS.Tom
"""
path = "C:/Users/DS.Tom/.spyder-py3/db/"

import numpy as np
import os
import pandas as pd
import time
os.chdir(path)
import DS_function as func
import asyncio
import nest_asyncio
import keyring
nest_asyncio.apply()



async def main():

    #import matplotlib.pyplot as plt
    start = time.time() 

    '''
    Variable zone
    '''
    JG = func.DB('SQL Server Native Client 11.0', 'JG\MSSQLSERVER2016', 'DS.Jimmy', keyring.get_password('JG','DS.Jimmy'))
    train_start = '2019-10-10 00:00:00.000'  
    train_end = '2019-10-30 23:59:59.999'
    test_start = '2019-10-21 00:00:00.000'  
    test_end = '2019-10-21 23:59:59.999'

    #Get train & test with sql & ODBC
    train = func.get_data(JG, train_start, train_end)

    test = func.get_data(JG, test_start, test_end)

    print("--- %s seconds ---" % (time.time() - start))
    '''
    Exclude_rawdatatype = [35, 67, 68, 136]# Mako said that there aren't unique gameaccount in these gamehall(rawdatatype)
    train = train[~train.RawDataType.isin(Exclude_rawdatatype)]
    '''

    sql_end = time.time() 
    python_start = sql_end


    '''
    PreProcessing
    '''
    train_data = func.Pre_processing(train)
    test_data = func.Pre_processing(test)
    del(train)
    del(test)
    print("--- %s seconds ---" % (time.time() - start))
    '''
    train_data = train_data[train_data.SiteID == 154].reset_index()
    test_data = test_data[test_data.SiteID == 154].reset_index()
    '''



# filter & Exclude the just play one game people (Noise and it can't give our model any help)
    train_data = func.Pre_processing_train(train_data)

# Define the hotgame list
    HotGame_inTrain = func.Hot_Game(train_data, 
                                feature='Commissionable',
                                n=15)
    print("--- %s seconds ---" % (time.time() - start))

    users = train_data.Member.unique()
    games = train_data.Game.unique()
    userid2idx = {o:i for i,o in enumerate(users)}
    gameid2idx = {o:i for i,o in enumerate(games)}
    userid2Rraw = {i:o for i,o in enumerate(users)}
    gameid2iraw = {i:o for i,o in enumerate(games)}

    print("--- %s seconds ---" % (time.time() - start))

    train_data.loc[:, 'Member_encoding'] = train_data['Member'].apply(lambda x: userid2idx[x])
    train_data.loc[:, 'Game_encoding'] = train_data['Game'].apply(lambda x: gameid2idx[x])
    Trainset = func.get_Trainset(train_data)

    cosine_sim = func.SVD_surprise_only_tom(Trainset) # pd.read_csv("C:/Users/DS.Tom/Desktop/cosine_sim.csv", index_col = 0)
    #print(cosine_sim)
    print( "Cosine sim:" + str(np.shape(cosine_sim)))
    res = await find_top_K(Trainset,cosine_sim,30,50000)
    filtered_array = np.vstack(res)

    filtered_array = filtered_array[filtered_array[:,0].argsort()]
    users          = filtered_array[:,0]
    filtered_array = filtered_array[:,1:31]
    filtered_array = filtered_array.reshape( filtered_array.shape[0] * filtered_array.shape[1], -1)
    
    SVD_Neighbor = pd.DataFrame({'Member_encoding': np.repeat(users, 30), 
                                 'Game_encoding': filtered_array[:,0]})
    
    #SVD_Neighbor_result = SVD_Neighbor.groupby('member_id').head(12)
    SVD_Neighbor_result = SVD_Neighbor.merge(Trainset[['Member_encoding', 'Game_encoding', 'score']],
                                             how = 'left',
                                             on = ['Member_encoding', 'Game_encoding'])
    SVD_Neighbor_result.score = np.where(SVD_Neighbor_result.score.isna(), 0, SVD_Neighbor_result.score)
    SVD_Neighbor_result = SVD_Neighbor_result.sort_values(by = ['Member_encoding', 'score'], ascending = False)
    SVD_Neighbor_result = SVD_Neighbor_result.groupby('Member_encoding').head(12)
    print(len(users))
    print(SVD_Neighbor_result)
    print("--- %s seconds ---" % (time.time() - start))


    #print(np.shape(res[0]))
    #print(res[0])

async def find_top_K(Trainset, cosine_sim,  K = 30,  step = 10000):
    start = time.time()
    loop = asyncio.get_event_loop()
    keys,values  = Trainset.loc[:,['Member_encoding','Game_encoding']].values.T 
    ukeys, index = np.unique(keys,True)
    arrays       = np.split(values, index[1:])
    gamesplayed          = pd.DataFrame({'a':ukeys,'b':[list(a) for a in arrays]})
    gamesplayed.columns = ['Member_encoding','games']
    print("--- %s seconds ---" % (time.time() - start))
    
    #gamesplayed = Trainset.groupby(['Member_encoding'])['Game_encoding'].apply(lambda x: list(x)).reset_index(name='games')
    cosine2 = cosine_sim.to_numpy()
    #print("--- %s seconds ---" % (time.time() - start))
    
    lup = dict()
    for key in range(np.shape(cosine2)[0]):
        lup[key]= cosine2[key,:]
    goal = np.zeros((np.shape(gamesplayed)[0],K))

    totalsize = np.shape(gamesplayed)[0]

    batches = totalsize // step
    tasks = []
        
    for k in range(batches):
        l = k + 1
        batch = gamesplayed.iloc[ k*step:l*step,: ]
        tasks.append(loop.create_task(process(batch,lup,k*step, step,K)))
    laststep  = batches *step

    try:
        #pass
        lastbatch = gamesplayed.iloc[ laststep : (totalsize-1),:]
        tasks.append(loop.create_task(process(lastbatch,lup,laststep,step,K)))
    except:
        pass
    res = await asyncio.gather(*tasks)
    print("--- %s seconds ---" % (time.time() - start))
    return(res) 
    
async def process(batch, lup, startindex, batchsize,K):
    lookup  = batch.games.to_numpy()
    indx = 0
    goal = np.zeros((batchsize,31))
    for key in lookup:
        res = [ lup[val] for val in key ]
        results = np.max(res,axis=0).argsort()[::-1][0:K]
        goal[indx,1:31] = results
        goal[indx,0] = startindex + indx
        indx = indx + 1
    return(goal)

asyncio.run(main())


