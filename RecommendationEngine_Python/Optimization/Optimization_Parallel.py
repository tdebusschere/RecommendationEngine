
import asyncio
import aioodbc
from concurrent.futures import ThreadPoolExecutor

import time
import os
import datetime
import DS_SQLGetDF_function as func
import numpy as np
import pandas as pd

async def connect_db(loop):
#dsn = 'Driver=SQL Server Native Client 11.0;Server=10.80.16.191;User=DS.Reader;Password=8sGb@N3m'
    conn = await aioodbc.create_pool(dsn = 'DRIVER={SQL Server Native Client 11.0};SERVER=10.80.16.191;\
                                     UID=DS.Reader;PWD=8sGb@N3m;ApplicationIntent=READONLY;',
                                     executor=ThreadPoolExecutor(max_workers=4),
                                     loop = loop)
    return(conn)
    
async def queryDAT( pool, sql):
    async with pool.acquire() as conn:
            cursor =  await conn.execute(sql)
            rows   =  await cursor.fetchall()
            colList = []
            for colInfo in cursor.description:
                colList.append(colInfo[0]) 
            resultList = []
            for row in rows:
                resultList.append(list(row))
            data = pd.DataFrame(resultList, columns=colList)
    #    except:
    #        data = None
    return(data)

async def main_connect(loop):
    pool = await connect_db(loop)
    started_at = time.monotonic()
    Server_list = ['10.80.16.191', '10.80.16.192', '10.80.16.193', '10.80.16.194']
    Server = Server_list[0]

    #main parameter
    now = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:00:00.000") #UTC+0
    current = datetime.datetime.now().strftime("%Y-%m-%d %H:00:00.000") 
    SQLQuery_df = func.select_from_sql(now, current, Server)
    #print(SQLQuery_df)
    
    tasklist = np.unique(SQLQuery_df.Sqlquery).tolist()
    print(tasklist)
    print(len(tasklist))
    tasks = [ queryDAT( pool ,x ) for x in tasklist ]
    results  = await asyncio.gather(*tasks)
    zm = pd.concat(results)
    print(zm)
    print(len(results))
    print(time.monotonic() - started_at)


loop = asyncio.get_event_loop()  
loop.run_until_complete(main_connect(loop))



    