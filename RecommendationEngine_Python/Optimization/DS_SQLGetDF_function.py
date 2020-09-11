# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 08:51:28 2019

@author: DS.Tom
"""

import pyodbc
import pandas as pd
import logging
import time
import datetime
from dateutil.relativedelta import relativedelta
'''
from sklearn.utils import resample
from scipy.spatial.distance import cdist
from functools import reduce
import smtplib
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
'''


'''
Function Zone
'''

class DB:
    def __init__(self, driver, server, uid, pwd):
        self.driver = driver
        self.server = server
        self.uid = uid
        self.pwd = pwd
        self.conn = None
        self.cur = self.__getConnect()
    
    def __getConnect(self):
        try:
            self.conn = pyodbc.connect(driver=self.driver,
                                       server=self.server,
                                       uid=self.uid,
                                       pwd=self.pwd, ApplicationIntent='READONLY')
            cur = self.conn.cursor()
        except Exception as ex:
            logging.error('SQL Server connecting error, reason is: {}'.format(str(ex)))
        return cur
    
    def ExecQuery(self, sql):
        #cur = self.cur #__getConnect()
        try:
            df = pd.read_sql_query(sql,self.conn)
        except pyodbc.Error as ex:
            logging.error('SQL Server.Error: {}'.format(str(ex)))
        #cur.close()
        #self.conn.close()       
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


def to_sql_set(gametypesourceid):
    string = 'select {} as gametypesourceid'.format(gametypesourceid[0])
    S = gametypesourceid[1:]
    for elem in S:
        string += ' union select {}'.format(elem)
    return string

#讀取對照表
def select_from_sql(now, current_day, serverip):
    def get_target():
        IP191 = DB('SQL Server Native Client 11.0', '10.80.16.191', 'DS.Reader', keyring.get_password('191','DS.Tom'))
        chart = IP191.ExecQuery("select * from [BalanceOutcome].[dbo].[LookUpTable]")
        Exclude_rawdatatype = [35, 67, 68, 136]# Mako said that there aren't unique gameaccount in these gamehall(rawdatatype)

        target = chart[~chart['GameTypeSourceId'].isin(Exclude_rawdatatype)].copy()
        target = target[target['ServerIP'] == serverip].reset_index()

        target['DBName_sql'] = target.apply(lambda row: 
            str(row['DBName'])
            + now[2:4]
            + now[5:7] if row['MonthDB'] else row['DBName'], axis=1)
        previous_month = (datetime.datetime.strptime(now, "%Y-%m-%d %H:00:00.000") 
        - relativedelta(months=1)).strftime("%Y-%m-%d %H:00:00.000")
        target['DBName_sql_previous_month'] = target.apply(lambda row: 
            str(row['DBName'])
            + previous_month[2:4]
            + previous_month[5:7] if row['MonthDB'] else row['DBName'], axis=1)
        return target
        
    def SQL_timezone_sqlstring(row):
        DBName_sql = row['DBName_sql']
        RawDataType = row['Type']
        return "SELECT [Timezone] FROM {DB}.[dbo].[VW_RawDataInfo] \
        where type = {type}".format(DB = DBName_sql, type = RawDataType)
    
    def SQL_timezone_value(row):
        Server = row['ServerIP']
        IP = DB('SQL Server Native Client 11.0', Server, 'DS.Reader', '8sGb@N3m')
        sqlstring = row['timezone_sqlstring']
        df = IP.ExecQuery(sqlstring)
        #result = df.Timezone
        return df['Timezone'].iloc[0]

    def SQL_EndTime(row, nowutc):
        Delta = row['timezone_value']
        End_time = (datetime.datetime.strptime(nowutc, "%Y-%m-%d %H:00:00.000") + datetime.timedelta(hours = Delta)).strftime("%Y-%m-%d %H:00:00.000")
        return End_time
    
    def SQL_StartTime(row, delta):
        End_time = row['End_Table_time']
        Start_time = (datetime.datetime.strptime(End_time, "%Y-%m-%d %H:00:00.000") + datetime.timedelta(hours = delta)).strftime("%Y-%m-%d %H:00:00.000")
        return Start_time

    def SQL_data(row, current_day):
        #variable zone
        game = row['GameTypeSourceId']
        current_db = row['DBName_sql']
        previous_db = row['DBName_sql_previous_month']
        table_name = 'VW_'+row['TableName']
        endtime = row['End_Table_time']
        starttime = row['Start_Table_time']
        
        Start_time = datetime.datetime.strptime(row['Start_Table_time'], "%Y-%m-%d %H:00:00.000")
        End_time = datetime.datetime.strptime(row['End_Table_time'], "%Y-%m-%d %H:00:00.000")
        Condition1 = (Start_time.month != End_time.month)
        Condition2 = row['MonthDB']
        if Condition1 & Condition2:
            sqlquery = "SELECT '{date}' [Dateplayed], [GameAccount], [SiteId], {game} [GameTypeSourceId],\
            Sum([Commissionable]) [Commissionable], Sum([WagersCount]) [WagersCount] \
            FROM (SELECT [GameAccount], [SiteId], Isnull(Sum([betamount]), 0) [Commissionable], Count(1) [WagersCount]\
            FROM {current_db}.[dbo].{table_name} (nolock) \
            WHERE [wagerstime] >= '{starttime}' AND [wagerstime] <= '{endtime}' AND [gametypesourceid] = {game} \
            GROUP BY [gameaccount], [siteid] HAVING Sum([betamount]) > 0\
            UNION ALL SELECT [GameAccount], [SiteId], Isnull(Sum([betamount]), 0) \
            [Commissionable], Count(1) [WagersCount] FROM {previous_db}.[dbo].{table_name} (nolock) \
            WHERE [wagerstime] >= '{starttime}' AND [wagerstime] <= '{endtime}' AND [gametypesourceid] = {game} \
            GROUP BY [gameaccount], [siteid] HAVING Sum([betamount]) > 0 )X GROUP BY [gameaccount], [siteid]".\
                  format(date=current_day, game=game, current_db= current_db, table_name= table_name, \
                         endtime = endtime, starttime = starttime, previous_db= previous_db)
        else:
            sqlquery = "SELECT '{date}' [Dateplayed], [GameAccount], [SiteId], [GameTypeSourceId],\
            Isnull(Sum([betamount]), 0) [Commissionable], Count(1) [WagersCount] \
            FROM {current_db}.dbo.{table_name} (nolock)\
            WHERE [wagerstime] >= '{starttime}' AND [wagerstime] <= '{endtime}' and [gametypesourceid] = {game} \
            GROUP BY [gameaccount], [siteid], [gametypesourceid] \
            HAVING Sum([betamount]) > 0".format(date=current_day, game=game, current_db= current_db,\
            table_name= table_name, endtime = endtime, starttime = starttime)

        return sqlquery 
    target = get_target()
    target_GroupByType = target[['ServerIP', 'Type', 'DBName_sql']].drop_duplicates()
    target_GroupByType = target_GroupByType.reset_index(drop = True)
    #target_GroupByType['timezone_sqlstring'], target_GroupByType['timezone_value'] = zip(*target_GroupByType.apply(lambda row: SQL_timezone2(row), axis=1))
    
    target_GroupByType.loc[:, 'timezone_sqlstring'] = target_GroupByType.apply(lambda row: SQL_timezone_sqlstring(row), axis=1)
    target_GroupByType.loc[:, 'timezone_value'] = target_GroupByType.apply(lambda row: SQL_timezone_value(row), axis=1)
    
    target = target.merge(target_GroupByType[['Type', 'timezone_value']],
                          how = 'left',
                          on = 'Type')
    target.loc[:, 'End_Table_time'] = target.apply(lambda row: SQL_EndTime(row, nowutc = now), axis=1)
    target.loc[:, 'Start_Table_time'] = target.apply(lambda row: SQL_StartTime(row, delta = -1), axis=1)
    target.loc[:, 'Sqlquery'] = target.apply(lambda row: SQL_data(row, current_day), axis=1)
    
    return target
