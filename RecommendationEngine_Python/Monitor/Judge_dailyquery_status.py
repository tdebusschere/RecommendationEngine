import os
path = "C://Users//DS.Jimmy//Desktop//Project//RecommenderSystem//Run_Model_對帳//opt//V2"
os.chdir(path)
import sys
import time


def Judge_dailyquery_status(StatsTable_bytype, train_start, train_end, JG, sleep_sec=30, last_sec=30*60):
    last = int(last_sec/sleep_sec)

    for k in range(0, last):
        print(k)
        sqlquery = "SELECT * FROM {table} \
                    where [UpDateTime] <= '{train_end}' and [UpDateTime] >= '{train_start}'\
                    and [Status] not in ('Success', 'Empty')".format(table=StatsTable_bytype,
                                                                     train_start=train_start,
                                                                     train_end=train_end)
        df = JG.ExecQuery(sqlquery)
        if df.empty:
            break
        elif((df.shape[0] == 0) & (k == 59)):
            sys.exit()
        else:
            time.sleep(30)#stop 30sec, last to 30 min   