import pandas as pd
import time
import datetime
import Connect_SQL as Connect_SQL
import Process_Function as func
import Parameter as parameter
import Judge_dailyquery_status as step1
#import asyncio
pd.set_option('max_columns', None)


'''
Variable zone
'''
start = time.time() 
JG = Connect_SQL.JG();  BalanceCenter_190 = Connect_SQL.BalanceCenter_190()
train_start = (datetime.datetime.now()+datetime.timedelta(days = -parameter.Get_Finallist_Delta)).strftime("%Y-%m-%d 13:00:00.000")  
train_end = datetime.datetime.now().strftime("%Y-%m-%d %H:00:00.000")
current = datetime.datetime.now().strftime("%Y-%m-%d %H:00:00.000") 

step1.Judge_dailyquery_status(parameter.StatsTable_bytype,
                              train_start,
                              train_end, 
                              JG, 
                              sleep_sec=30, 
                              last_sec=30*60)
        
# status table of CosineSimilarity
Running = pd.DataFrame({'Status_Result':'Running',
                        'Status_Default':'Running',
                        'UpDateTime':current}, index=[0])

BalanceCenter_190.Executemany("insert into {table}(\
                               [Status_Result],[Status_Default], [UpDateTime]) values (?,?,?)".format(table=parameter.ResultTable_Status), Running)
 

#Get train & test with sql & ODBC
train = func.get_data(BalanceCenter_190,
                      parameter.DailyQueryTable_190,
                      train_start,
                      train_end)
train_data = func.Pre_processing(train)
del(train)



# filter & Exclude the just play one game people (Noise and it can't give our model any help)
train_data = func.Pre_processing_train(train_data)
games = train_data.Game.unique()
gameid2idx = {o:i for i,o in enumerate(games)}


# Exclude the gamelist, connect_indirectly so far.
exclude_game_list_raw, exclude_game_list = func.Exclude_Game(BalanceCenter_190,
                                                             parameter.category_exclude,
                                                             games,
                                                             gameid2idx)


# Define the hotgame list
train_data_exclude = train_data[~train_data.Game.isin(exclude_game_list_raw)].reset_index(drop=True)
HotGame_inTrain = func.Hot_Game(train_data_exclude, 
                                feature='Commissionable',
                                n = 12)
HotGame_df = pd.DataFrame({'Rank':range(1, 13),
                           'Game':HotGame_inTrain,
                           'UpDateTime':current})

BalanceCenter_190.Executemany("insert into {table}(\
                              [Rank], [Game], [UpDateTime]) \
                              values (?,?,?)".format(table=parameter.ResultTable_Default), HotGame_df)
BalanceCenter_190.ExecNoQuery("UPDATE {table}\
                              SET  Status_Default = 'Success' \
                              where UpDateTime = '{uptime}' and Status_Default = 'Running'".format(table=parameter.ResultTable_Status,
                                                                                                   uptime=current))

exe_time = (time.time() -start)


'''
variable zone
'''

cs = BalanceCenter_190.ExecQuery("SELECT max(updatetime) updatetime FROM DataScientist.dbo.DS_RecommenderSystem_CSStatus")
csupdate_time = cs.updatetime.iloc[0].strftime("%Y-%m-%d %H:00:00.000")


hotgame_ = BalanceCenter_190.ExecQuery(" select max(updatetime) updatetime from {table}".format(table=parameter.ResultTable_Default))
hotgame_time = hotgame_.updatetime.iloc[0].strftime("%Y-%m-%d %H:00:00.000")


Execution_time = train_end
KG = Connect_SQL.BalanceCenter_190()
ids = KG.ExecQuery(" exec [Resultpool].[dbo].[RecommendationGeneratePersonalList] \
                         @execution_time='{execution_time}'".format(execution_time=Execution_time))

for k in ids.firstletters: #@execution_time datetime, @selectinfo int, @hotgame_time datetime
    ids = KG.ExecNoQuery(" exec [Resultpool].[dbo].[RecommendationGenerateResultPortion] \
                               @execution_time='{execution_time}',\
                               @selectinfo={letter},\
                               @hotgame_time='{hotgame_time}',\
                               @csupdate_time = '{csupdate_time}'".format(
                               letter = int(k), 
                               hotgame_time = hotgame_time, 
                               execution_time = Execution_time,
                               csupdate_time = csupdate_time))
    print(k)
Exe_Time = (time.time() - start)


BalanceCenter_190.ExecNoQuery("UPDATE {table} \
                              SET  Status_Result = 'Success', [Exe_Time_sec] = {Exe_Time} \
                              where UpDateTime = '{uptime}' and Status_Result = 'Running'".format(table=parameter.ResultTable_Status,
                                                                                                  Exe_Time=Exe_Time,
                                                                                                  uptime=current))