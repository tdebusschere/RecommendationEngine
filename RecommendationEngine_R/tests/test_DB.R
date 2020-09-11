context("test_db_connection")
setwd('..')
source('GlobalConfig.R')

switcher = odbcDriverConnect(connection = "driver={SQL Server};server=JG\\MSSQLSERVER2016;uid=DS.Jimmy;pwd=4wvb%ECX;")== -1
if (switcher) skip("No usable Database Connection")

##basic properties
test_that(
  'test_db_connection',
  {
    skip_if(switcher)
    expect_s4_class(new("connect_to_testMSSQL"),'connect_to_MSSQL')
    expect_s4_class(new("connect_to_productionMSSQL"),'connect_to_MSSQL')
    expect_s3_class(new("connect_to_testMSSQL")@dbhandle,'RODBC')
    ##we need to be able to select data
    expect_gt(dim(sqlQuery(new("connect_to_productionMSSQL")@dbhandle,'select name from sys.databases'))[1],1)
  }
)


###is the object usable
test_that(
  'getHandle',
  {
    skip_if(switcher)
    expect_gt(dim(sqlQuery(new("connect_to_productionMSSQL")@dbhandle,'select name from sys.databases'))[1],1)
    expect_s3_class(getHandle(new('connect_to_productionMSSQL')),'RODBC')
    expect_s3_class(getHandle(new('connect_to_testMSSQL')),'RODBC')
  }
)