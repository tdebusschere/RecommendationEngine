# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 09:39:27 2019

@author: DS.Tom
"""
import numpy as np
import os
import pandas as pd
import time
import DS_function as func
import asyncio


start = time.time() 

'''
Variable zone
'''
JG = func.DB('SQL Server Native Client 11.0', 'JG\MSSQLSERVER2016', 'DS.Tom', keyring::get_password('JG','DS_Tom'))
train_start = '2019-10-15 00:00:00.000'  
train_end = '2019-10-20 23:59:59.999'
test_start = '2019-10-21 00:00:00.000'  
test_end = '2019-10-21 23:59:59.999'

#Get train & test with sql & ODBC
train = func.get_data(JG, train_start, train_end)
test = func.get_data(JG, test_start, test_end)
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

users = train_data.Member.unique()
games = train_data.Game.unique()
userid2idx = {o:i for i,o in enumerate(users)}
gameid2idx = {o:i for i,o in enumerate(games)}
userid2Rraw = {i:o for i,o in enumerate(users)}
gameid2iraw = {i:o for i,o in enumerate(games)}

train_data.loc[:, 'Member_encoding'] = train_data['Member'].apply(lambda x: userid2idx[x])
train_data.loc[:, 'Game_encoding'] = train_data['Game'].apply(lambda x: gameid2idx[x])
Trainset = func.get_Trainset(train_data)

cosine_sim = pd.read_csv("C:/Users/DS.Tom/Desktop/cosine_sim.csv", index_col = 0)

async def main():
    print("--- %s seconds ---" % (time.time() - start))

    gamesplayed = Trainset.groupby(['Member_encoding'])['Game_encoding'].apply(lambda x: list(x)).reset_index(name='games')
    cosine2 = cosine_sim.to_numpy()
    print("--- %s seconds ---" % (time.time() - start))

    
    lup = dict()
    for key in range(np.shape(cosine2)[0]):
        lup[key]= cosine2[key,:]
    goal = np.zeros((np.shape(gamesplayed)[0],30))
    print( np.shape(goal))

    totalsize = np.shape(gamesplayed)[0]
    step = 100000
    
    batches = int(np.floor(totalsize/step))
    tasks = []
    K     = 30
    for k in range(batches):
        l = k + 1
        batch = gamesplayed.iloc[ k*step:l*step,: ]
        tasks.append(asyncio.create_task(process(batch,lup,k*step, step,K)))
    laststep  = batches *step
    print(str(laststep) + ' ' + str(totalsize) + ' ' + str(totalsize))
    try:
        lastbatch = gamesplayed.iloc[ laststep : totalsize,:]
        tasks.append(asyncio.create_task(process(lastbatch,lup,laststep,step,K)))
    except:
        pass
    res = await asyncio.gather(*tasks)
    print("--- %s seconds ---" % (time.time() - start))
     
    
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


