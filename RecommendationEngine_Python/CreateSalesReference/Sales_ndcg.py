import pandas as pd
import itertools
import numpy as np
import multiprocessing as mp
import time

start = time.time()

pd.set_option('display.max_columns', None)
df = pd.read_csv('C:/Users/ADMIN/PycharmProjects/Recommender system/sales_model.csv')
test = pd.read_csv('C:/Users/ADMIN/PycharmProjects/Recommender system/test.csv')  ##11/25~11/30 as test dataset
test = test.sort_values(by=['memberid', 'Commissionable'], ascending=False)
test = test.groupby('memberid', as_index=False).head(12)
test['rel'] = test.groupby('memberid')['Commissionable'].rank(ascending=False)

memberlist = test.memberid.unique().tolist()
memberlist = list(itertools.chain.from_iterable(itertools.repeat(x, 12) for x in memberlist))


def NDCG_calculation(i):
    print('第{}批'.format(i+1))
    gamelist = df.iloc[i].tolist() * (len(memberlist) // 12)
    DF = pd.DataFrame(np.column_stack([memberlist, gamelist]), columns=['memberid', 'gametypeid'])
    DF = DF.merge(test[['memberid', 'gametypeid', 'rel']], how='left', on=['memberid', 'gametypeid'])
    DF.rel = np.where(DF.rel.isna(), 0, DF.rel)  # play as TP
    DF.columns = ['member_id', 'game_id', 'rel']
    DF['i+1'] = list(range(2,14)) * (DF.shape[0] // 12)
    from math import log
    DF['dcg'] = DF.apply(lambda row: (2 ** row['rel'] - 1) / log(row['i+1'], 2), axis=1)
    DCG = DF.groupby('member_id', as_index=False)['dcg'].sum()
    DF_ = DF.sort_values(by=['member_id', 'rel'], ascending=False)
    DF_['i+1'] = list(range(2, 14)) * (DF_.shape[0] // 12)
    DF_['idcg'] = DF_.apply(lambda row: (2 ** row['rel'] - 1) / log(row['i+1'], 2), axis=1)
    iDCG = DF_.groupby('member_id', as_index=False)['idcg'].sum()
    NDCG = DCG.merge(iDCG, on='member_id', how='left')
    NDCG['ndcg'] = NDCG['dcg'] / NDCG['idcg']
    NDCG.ndcg = np.where(NDCG.ndcg.isna(), 0, NDCG.ndcg)
    return NDCG.ndcg.mean()


def multicore():
    pool = mp.Pool()
    result = pool.map(NDCG_calculation, range(500))
    print(result)
    print(sum(result) / len(result))

if __name__=='__main__':
    multicore()
    print(time.time() - start)