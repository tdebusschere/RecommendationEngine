
context("test_get_data")
setwd("..")
source("GlobalConfig.R")



switcher = odbcDriverConnect(connection = "driver={SQL Server};server=JG\\MSSQLSERVER2016;uid=DS.Jimmy;pwd=4wvb%ECX;")== -1
if (switcher) skip("No usable Database Connection")


###conformity of the database
test_that(
  'database_conformity',
  {
    expect_s3_class(new("getMSSQL_counts_data",TRUE)@conn,'RODBC')
    expect_s3_class(new("getMSSQL_counts_data",FALSE)@conn,'RODBC')
    target_db = data.frame(name = 'DataScientist')
    expect_equal( dim(sqlQuery(new('getMSSQL_counts_data',TRUE)@conn,
                               "select name from sys.databases where name = 'DataScientist'"))[1],1)
    #expect_equal( sqlQuery(new('getMSSQL_counts_data',FALSE)@conn,
    #                       "select name from sys.databases where name = 'DataScientist'"),target)
    
  }
)

context("table_conformity")
test_that(
  'table_conformity',
  {
    queryobj = new('getMSSQL_counts_data',TRUE)@conn
    sqlQuery(queryobj,"use DataScientist;")
    target_table = data.frame(name = c('IntelligentGame','GameType','BetRecordMonthlySystemCodeGame'))
    target_table$name = sort(target_table$name)
    expect_equal(sqlQuery(queryobj,"select name from sys.tables where name IN ('IntelligentGame','GameType','BetRecordMonthlySystemCodeGame') order by name"),target_table)
  }
)

test_that(
  'IntelligentGame_Conformity',
  {
    queryobj = new('getMSSQL_counts_data',TRUE)@conn
    sqlQuery(queryobj,"use DataScientist;")
    expect_equal(dim(sqlQuery(queryobj,"select column_name from information_schema.columns where
                              table_name ='IntelligentGame' and column_name in ('Date','RawDataType',
                              'GameCode','Top1Game','Top2Game','Top3Game','Top4Game','Top5Game',
                              'Top6Game','Top7Game','Top8Game','Top9Game','Top10Game')"))[1], 13)
    expect_equal(dim(sqlQuery(queryobj,"select count(*), date, rawdatatype, gamecode  from 
                              datascientist.dbo.intelligentgame group by date,rawdatatype,gamecode having
                              count(*) > 1"))[1],0)
    for (i in c(1:10))
    {
      expect_equal(dim(sqlQuery(queryobj,paste("select Top",i,"Game from datascientist.dbo.intelligentgame where Top",i,"Game not like '%,%'",
                                               collapse='',sep='')))[1],0)
    }
  }
    )

test_that(
  'GameType_Conformity',
  {
    queryobj = new('getMSSQL_counts_data',TRUE)@conn
    sqlQuery(queryobj,"use DataScientist;")
    expect_equal(dim(sqlQuery(queryobj,"select column_name from information_schema.columns where
                              table_name = 'GameType' AND column_name in ('Id','RawDataType','GameCode')"))[1],3)
    expect_is(sqlQuery(queryobj,"select top 100 id from datascientist.dbo.gametype")[1,1],"integer")
    expect_gt(dim(sqlQuery(queryobj,"select id from datascientist.dbo.gametype"))[1],100)
    
  }
)

test_that(
  'BetRecordMonthlySystemCodeGame_Conformity',
  {
    queryobj = new('getMSSQL_counts_data',TRUE)@conn
    sqlQuery(queryobj,"use DataScientist;")
    expect_equal(dim(sqlQuery(queryobj,"select column_name from information_schema.columns where
                              table_name = 'BetRecordMonthlySystemCodeGame' and column_name in ('Id','First30Days','SystemCode',
                              'MemberId','CategoryId','RawDataType','GameCode','Counts')"))[1],8)
    
  }
    )

test_that(
  'Test if there are games not in DataScientist.dbo.gametype',
  {
    queryobj = new('getMSSQL_counts_data',TRUE)@conn
    sqlQuery(queryobj,"use DataScientist;")
    expect_equal(dim(sqlQuery(queryobj,"select * from ( select distinct Categoryid, rawdatatype,gamecode from 
                              BetRecordMonthlySystemCodeGame ) x left join gametype y on x.gamecode = y.gamecode and x.rawdatatype = y.rawdatatype
                              where y.id is NULL"))[1],0)
    
  }
    )

test_that(
  'Test if date of last_update > 5  days ago and on the same day',
  {
    queryobj = new('getMSSQL_counts_data',TRUE)@conn
    sqlQuery(queryobj,"use DataScientist;")
    expect_gt(5,abs(as.numeric(as.Date(sqlQuery(queryobj,"select max(First30Days) from BetRecordMonthlySystemCodeGame")[[1]],'%Y-%m-%d') - Sys.Date())))
    expect_gt(as.numeric(Sys.Date() - as.Date(sqlQuery(queryobj,"select max(First30Days) from BetRecordMonthlySystemCodeGame")[[1]],'%Y-%m-%d')),0)
    expect_gt(5,abs(as.numeric(as.Date(sqlQuery(queryobj,"select max(date) from IntelligentGame")[[1]],'%Y-%m-%d') - Sys.Date())))
    expect_gt(as.numeric(Sys.Date() - as.Date(sqlQuery(queryobj,"select max(date) from IntelligentGame")[[1]],'%Y-%m-%d')),0)
    ##will indicate that it didn't run after update though
    expect_equal( abs(as.numeric(as.Date(sqlQuery(queryobj,"select max(date) from IntelligentGame")[[1]],'%Y-%m-%d'))),
                  abs(as.numeric(as.Date(sqlQuery(queryobj,"select max(First30Days) from BetRecordMonthlySystemCodeGame")[[1]],'%Y-%m-%d') )))
  }
)

context("test_get_data_performance")
test_that( 
  'Test actual data processing of this module:data',
  {
    object = new('getMSSQL_counts_data',TRUE)
    object = setTimeLastUpdatedDB(object)
    expect_s3_class(as.Date(getTimeLastUpdatedDB(object),'%Y-%m-%d'),'Date')
    expect_gt(0,as.numeric(as.Date(getTimeLastUpdatedDB(object),'%Y-%m-%d') - Sys.Date()))
  }
)

test_that(
  "Test actual data processing of this module: recommendations",
  {
    object = new('getMSSQL_counts_data',TRUE)
    object = setTimeLastUpdatedDB(object)
    Recs   = getRecommendationSet(object)
    expect_gt(dim(Recs)[1],50)
    expect_is(Recs$MemberId, 'integer')
    expect_is(Recs$key,'integer')
    Control= getControlSet(object)
    expect_equal(sum(Control$key %in% unique(Recs$key)),length(Control$key))
  }
)

test_that(
  "Test actual data processing of this module: control",
  {
    object = new('getMSSQL_counts_data',TRUE)
    object = setTimeLastUpdatedDB(object)
    Control= getControlSet(object)
    expect_gt(dim(Control)[1],50)
    expect_is(Control$key,'integer')
  }
)

test_that(
  "Test actual data processing of this module: Hot",
  {
    object = new('getMSSQL_counts_data',TRUE)
    object = setTimeLastUpdatedDB(object)
    Hot= getHotSet(object)
    expect_equal(dim(Hot)[1],10)
    expect_is(Hot$id,'integer')
  }
)
