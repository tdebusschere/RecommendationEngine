context("test_Log_To_Database")
setwd('..')
source('GlobalConfig.R')
switcher = odbcDriverConnect(connection = "driver={SQL Server};server=JG\\MSSQLSERVER2016;uid=DS.Jimmy;pwd=4wvb%ECX;")== -1
if (switcher) stop("No usable Database Connection")

prod = new('Log_To_Database', 'recommendation',TRUE)
test = new('Log_To_Database', 'recommendation',FALSE)


test_that(
  'initialization',
  {
    skip_if(switcher)
    expect_s4_class(prod,'Log_To_Database')
    expect_error(new('Log_To_Database','recommendation','production'))
    expect_s4_class(test,'Log_To_Database')
  }
)

test_that(
  'initializedb',
  {
    skip_if(switcher)
    expect_equal(sqlQuery(prod@dbhandle,paste0("select starttime from [Datascientist].[dbo].[RunningRecord]  where starttime = '",
    as.character(prod@starttime),"'"))[1],data.frame('starttime'=as.character(prod@starttime)))
    expect_equal(sqlQuery(test@dbhandle,paste0("select starttime from [Datascientist].[dbo].[RunningRecord]  where starttime = '",
     as.character(test@starttime),"'"))[1],data.frame('starttime'=as.character(test@starttime)))
    expect_equal(sqlQuery(test@dbhandle,paste0("select Environment from [Datascientist].[dbo].[RunningRecord]  where starttime = '",
    as.character(test@starttime),"'"))[1],data.frame('Environment'='staging'))
    expect_equal(sqlQuery(prod@dbhandle,paste0("select Environment from [Datascientist].[dbo].[RunningRecord]  where starttime = '",
                 as.character(prod@starttime),"'"))[1],data.frame('Environment'='production'))
    expect_equal(sqlQuery(prod@dbhandle,paste0("select Status from [Datascientist].[dbo].[RunningRecord]  where starttime = '",
                                               as.character(prod@starttime),"'"))[1],data.frame('Status'='initialized'))
    expect_equal(sqlQuery(test@dbhandle,paste0("select Status from [Datascientist].[dbo].[RunningRecord]  where starttime = '",
                                               as.character(test@starttime),"'"))[1],data.frame('Status'='initialized'))
  }
)

test_that(
  'logtodb',
  {
    skip_if(switcher)
    log_to_database(test,'Working...')
    expect_equal(sqlQuery(test@dbhandle,paste0("select Status from [Datascientist].[dbo].[RunningRecord]  
                                               where starttime = '",
                          as.character(test@starttime),"'"))[1],data.frame('Status'='Working...'))
    log_to_database(test,'Completed')
    expect_equal(sqlQuery(test@dbhandle,paste0("select Status from [Datascientist].[dbo].[RunningRecord]  
                                               where starttime = '",
                                               as.character(test@starttime),"'"))[1],data.frame('Status'='Completed'))
    expect(as.POSIXlt(sqlQuery(test@dbhandle,paste0("select endtime from [Datascientist].[dbo].[RunningRecord]  
                                               where starttime = '",
                                               as.character(test@starttime),"'"))[1]$endtime) > test@starttime)
    
    
  }
)


test_that(
  'set_timestamp',
  {
    skip_if(switcher)
    test = set_timestamp(test,'2018-11-04')
    expect_equal(sqlQuery(test@dbhandle,paste0("select Date from [Datascientist].[dbo].[RunningRecord]  
                                               where starttime = '",
                                               as.character(test@starttime),"'"))[1],data.frame('Date'='2018-11-04'))
    
    expect_equal(test@lastupdate,'2018-11-04')
  }
)

test_that(
  'getEnvironment',
  {
    skip_if(switcher)
    expect_equal('staging',getEnvironment(test))
    expect_equal('production',getEnvironment(prod))
  }
)

sqlQuery(prod@dbhandle,paste0("delete from [Datascientist].[dbo].[RunningRecord]  where starttime = '", 
                              as.character(prod@starttime),"'"))

sqlQuery(test@dbhandle,paste0("delete from [Datascientist].[dbo].[RunningRecord]  where starttime = '", 
                              as.character(test@starttime),"'"))