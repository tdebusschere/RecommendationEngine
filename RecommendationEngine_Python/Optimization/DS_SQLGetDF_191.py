# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 17:23:28 2019
@author: DS.Tom
"""

import os
import numpy as np
import pandas as pd
import datetime
import time
import DS_SQLGetDF_function as func

Server_list = ['10.80.16.191', '10.80.16.192', '10.80.16.193', '10.80.16.194']
Server = Server_list[0]
IP = func.DB('SQL Server Native Client 11.0', Server, 'DS.Reader', '8sGb@N3m')

#main parameter
now = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:00:00.000") #UTC+0
current = datetime.datetime.now().strftime("%Y-%m-%d %H:00:00.000") 
SQLQuery_df = func.select_from_sql(now, current, Server)
JG = func.DB('SQL Server Native Client 11.0', 'JG\MSSQLSERVER2016', 'DS.Jimmy', keyring::get_password('DS.Jimmy'))

#print(SQLQuery_df)


#solution
err = []
time_list = []
start_for = time.time()
data = pd.DataFrame([],columns=['Dateplayed','GameAccount','SiteId','GameTypeSourceId','Commissionable','WagersCount'])
for i in range(SQLQuery_df.shape[0]):
    #print(i)
    s = time.time()
       
    sqlquery = SQLQuery_df['Sqlquery'][i]

 
    try:
        df = IP.ExecQuery(sqlquery)
        
        if not df.empty:
           #print(df)
           #print(data)
           data =pd.concat([data,df])
           #pass # JG.Executemany("insert into DataScientist.dbo.DS_RecommenderSystemDailyQuery([DatePlayed], [GameAccount], [SiteId], [GameTypeSourceId], [Commissionable], [WagersCount]) values (?,?,?,?,?,?)", df.apply(lambda x: tuple(x.values), axis=1).tolist())
    except:
        err.append(SQLQuery_df.GameTypeSourceId[i])
        continue
    finally:
        e = time.time()
        time_list.append(e-s)
end_for = time.time()
print(end_for- start_for)
    
