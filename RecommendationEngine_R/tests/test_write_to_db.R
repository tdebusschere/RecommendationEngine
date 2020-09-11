context("Write_To_Data_Base")
require(RODBC)
setwd("..")

source('Write_To_Db_Refactored.R')
source("postprocessing.R")

test_that(
  'initialization',
  {
    expect_s4_class(new('write_to_db',production = TRUE,setdate = '2012-05-05'), 'write_to_db')
    expect_s4_class(new('write_to_db',production = FALSE,setdate = '2018-05-05'), 'write_to_db')
    expect_error(new('write_to_db'))
    expect_error(new('write_to_db', production = TRUE))
    expect_error(new('write_do_db', production = FALSE,setdate = 'appel'))
    expect_error(new('write_do_db', production = FALSE,setdate = '0'))
    expect_error(new('write_do_db', production = FALSE,setdate = '258-011-14'))
  }
)

test_that(
  'cleanup_past',
  {
    outputmachine = new('write_to_db',production = FALSE,setdate = '2012-05-05')
    expect_match(cleanup_past(outputmachine,5),'2012-04-30')
    expect_match(cleanup_past(outputmachine,6),'2012-04-29')
    expect_match(cleanup_past(outputmachine,4),'2012-05-01')
    expect_equal(cleanup_past(outputmachine,-10),0)
    expect_match(cleanup_past(outputmachine,4), 'delete from ')
    expect_match(cleanup_past(outputmachine,4), 'where Date <=')
  }
)

test_that(
  'cleanup_today',
  {
    outputmachine = new('write_to_db',production = FALSE,setdate = '2012-05-05')
    expect_match(cleanup_today(outputmachine),'2012-05-05')
    expect_match(cleanup_today(outputmachine),'delete from')
  }
)


zm = 
  rbind(
    c(5,  0,    0,    0,    3,    0,    4,    2,    0,     0),
    c(3,  3,    0,    4,    0,    5,    4,    0,    0,     2),
    c(1,  0,    4,    4,    3,    0,    0,    5,    0,     0),
    c(0,  0,    0,    0,    0,    0,    0,    4,    1,     0),
    c(0,  0,    0,    0,    0,    0,    0,    0,    1,     5),
    c(0,  0,    4,    0,    0,    0,    0,    0,    0,     0)
  )

demodata = cbind(1,'2012-05-05',c(1:6),zm)
colnames(demodata) = c('SN','Date','Game','Top1Game','Top2Game','Top3Game','Top4Game', 'Top5Game', 'Top6Game',
                       'Top7Game','Top8Game','Top9Game','Top10Game')



test_that(
  'insert_into_table',
  {
    outputmachine = new('write_to_db',production = FALSE,setdate = '2012-05-05')
    amm = insert_into_table(outputmachine,demodata)
    expect_match(amm[5],'2012-05-05')
    expect_match(amm[3],'2012-05-05')
    expect_match(amm[2],",0,")
    expect_length(amm,6)
    expect_match(amm[4],"insert into")
    expect_equal(amm[1],"insert into [DataScientist].[dbo].[IntelligentGame] values ('2012-05-05',0,'1','5','0','0','0','3','0','4','2','0','0')")
  }
)

test_that(
  'update_Tables',
  {
    outputmachine = new('write_to_db',production = FALSE,setdate = '2012-05-05')
    amm = updateTables(outputmachine)
    expect_match(amm,'2012-05-05')
    for (n in c(1:9))
    {
    expect_match(amm,paste0('Top',as.character(n),'Game = gt.key2'))
    expect_match(amm,paste0("',',ig.Top",as.character(n),"Game"))  
    }
    expect_match(amm,"update ig set ig.Top1Game = gt.key2 from")
  }
)

test_that(
  'update_keys',
  {
    outputmachine = new('write_to_db',production = FALSE,setdate = '2012-05-05')
    output = updateKeys(outputmachine)
    expect_equal(output,"update ig set ig.rawdatatype = gt.category, ig.gamecode = gt.[id] from DataScientist.dbo.IntelligentGame ig join ##tmp5848 gt on ig.GameCode = gt.[key] where date='2012-05-05'")
  }
)

test_that(
  'createTempDB',
  {
    outputmachine = new('write_to_db',production = FALSE,setdate = '2012-05-05')
    output = createTempDB(outputmachine)
    expect_match(output,"select a.GameCode as id,a.RawDataType as category,b.id as")
    expect_match(output," group by a.RawDataType,a.GameCode,b.id order by a.RawDataType")
  }
)

test_that(
  'write_to_table',
  {
    outputmachine = new('write_to_db',production = FALSE,setdate = '2012-05-05')
    expect_equal(write_to_table(outputmachine,demodata),0)
  }
)

switcher = odbcDriverConnect(connection = "driver={SQL Server};server=JG\\MSSQLSERVER2016;uid=DS.Jimmy;pwd=4wvb%ECX;")== -1
if (switcher) skip("No usable Database Connection")

