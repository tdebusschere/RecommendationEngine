import os

import pandas as pd
import time
import datetime
import Connect_SQL as Connect_SQL
import Parameter as parameter
import Process_Function as func
import Judge_dailyquery_status as step1
#import asyncio
import nest_asyncio
nest_asyncio.apply()
pd.set_option('max_columns', None)

'''
Variable zone
'''
JG, BalanceCenter_190 = Connect_SQL.JG(), Connect_SQL.BalanceCenter_190()
train_start = (datetime.datetime.now()+ datetime.timedelta(days = -parameter.CS_Delta)).strftime("%Y-%m-%d 01:00:00.000")  
train_end = datetime.datetime.now().strftime("%Y-%m-%d %H:00:00.000")
current = datetime.datetime.now().strftime("%Y-%m-%d %H:00:00.000") 


'''
Processing
'''
#Judge_dailyquery_status
start = time.time()
step1.Judge_dailyquery_status(parameter.StatsTable_bytype,
                              train_start,
                              train_end, 
                              JG, 
                              sleep_sec=30, 
                              last_sec=30*60)

# status table of CosineSimilarity
Running = pd.DataFrame({'Status_CS':'Running',
                        'UpDateTime':current}, index=[0])
BalanceCenter_190.Executemany("insert into {table}(\
                              [Status_CS], [UpDateTime]) values (?,?)".format(table=parameter.MedianTable_CSStatusTable), Running)
    

#Get data
train = func.get_data(BalanceCenter_190,
                      parameter.DailyQueryTable_190,
                      train_start,
                      train_end)

#PreProcessing
train_data = func.Pre_processing(train)
del(train)


# filter & Exclude the just play one game people (Noise and it can't give our model any help)
train_data = func.Pre_processing_train(train_data)

#change to encoding
users, games = train_data.Membercode.unique(), train_data.Game.unique()
userid2idx, userid2Rraw, gameid2idx, gameid2iraw = func.Encoding_RS(users, games)

train_data = func.Encoding_TrainData(train_data,
                                     userid2idx,
                                     gameid2idx)

Trainset = func.get_Trainset(train_data)
del(train_data)
print("---GetData & PreProcessing cost %s seconds ---" %(time.time() - start))


#get cosine_sim dataframe
cosine_sim = func.Cosine_Similarity(Trainset)

# Exclude the gamelist, connect_indirectly so far.
exclude_game_list_raw, exclude_game_list = func.Exclude_Game(BalanceCenter_190,
                                                             parameter.category_exclude,
                                                             games,
                                                             gameid2idx)

cosine_sim.drop(exclude_game_list,
                axis=1,
                inplace=True) 

cosine_sim_final = func.Summarized_cosine_sim_df(cosine_sim,
                                                 gameid2iraw,
                                                 current,
                                                 exclude_game_list,
                                                 n=12)

#Insert result 
BalanceCenter_190.Executemany("insert into {table}(\
                              [Game], [CorrespondGame], [CosineSimilarity], [UpdateTime]) \
                              values (?,?,?,?)".format(table=parameter.MedianTable_CSTable),
                              cosine_sim_final)

exe_time = ( time.time()- start )
BalanceCenter_190.ExecNoQuery("UPDATE {table}\
                              SET Exe_Time_sec = {exe_time}, Status_CS = 'Success' \
                              where UpDateTime = '{uptime}' and Status_CS = 'Running'".format(table=parameter.MedianTable_CSStatusTable,
                                                                                              exe_time=exe_time, 
                                                                                              uptime=current))
print("----Total cost %s seconds ---" % (exe_time))