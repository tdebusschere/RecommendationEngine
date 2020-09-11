CS_Delta = 7
Get_Finallist_Delta = 7
#sql table in balance center 190
LookUptable_statustable = "[BalanceOutcome].[dbo].[DS_LookUptableStatus_STATUS]"
Balance190_lookuptablename = "[BalanceOutcome].[dbo].[LookUpTable]"
lookuptable_col = ['LinkServerName', 'GameTypeSourceId', 'DBName', 'MonthDB', 'TableName', 'Type']

DailyQueryTable_190 = "[BalanceCenterSummarize ].[dbo].[DS_BalanceCenterDailyQuery]"
ResultTable_Default = "[ResultPool].[dbo].[DS_RecommenderSystem_DefaultGame]"
ResultTable_Status = "[ResultPool].[dbo].[DS_RecommenderSystem_ResultStatus]"
MedianTable_CSTable = "[DataScientist].[dbo].[DS_RecommenderSystem_CosineSimilarity]"
MedianTable_CSStatusTable = "[DataScientist].[dbo].[DS_RecommenderSystem_CSStatus]"

#sql table in JG
StatsTable_bytype = "[DataScientist].[dbo].[DS_BalanceCenterDailyQueryStatus_byType]"
StatsTable_bytype_col = ['Server', 'Type', 'Status', 'Exe_Time_sec', 'UpDateTime']


CosineSimilarity_Statustable = "[DataScientist].[dbo].[DS_RecommenderSystem_CSStatus]"
DailyQueryTable = "[DataScientist].[dbo].[DS_BalanceCenterDailyQuery]"




# Exclude the gamelist, connect_indirectly so far.

category_exclude = ['彩票', '視訊', '體育']
