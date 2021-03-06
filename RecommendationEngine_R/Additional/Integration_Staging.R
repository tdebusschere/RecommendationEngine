library(RODBC)
library(testthat)
library(optparse)
##integration test
source('Code/Connect_To_SQL.R')

option_list = list(
  make_option(c("-d", "--directory"), type="character", default="", 
              help="directory", metavar="character"),
  make_option(c("-c", "--commit"), type='character',default="",
              help="commit", metavar="character"),
  make_option(c("-b", "--branch"), type='character',default="staging",
              help="staging", metavar="character"))
opt_parser = OptionParser(option_list=option_list);
opt = parse_args(opt_parser);


fail= function(mess)
{
  a = (paste0("\"C:\\Program Files\\R\\R-3.5.1\\bin\\Rscript.exe\" .\\Additional\\Write_Mail.R --directory ",
                opt$directory," --branch ", opt$branch, " --commit ", opt$commit, " --message ", mess))
  system(a)
}

#1: Test database versus production database
tryCatch({
  conn = getHandle(new('connect_to_productionMSSQL'))
  date = sqlQuery(conn,'select max(date) from datascientist.dbo.intelligentgame')[[1]]
}, error = function(e){
  fail('fail:DatabaseConnection')
  })

tryCatch({
proddate = sqlQuery(conn,paste0("select * from datascientist.dbo.intelligentgame where date ='",date,"' order by GameCode"))
testdate = sqlQuery(conn,paste0("select * from datascientist.dbo.intelligentgame_test where date ='",date,"' order by GameCode"))
}, error = function(e){
  fail('fail:Drawing data')
  })



dims = sum(dim(proddate) == dim(testdate))
dataset= rep(0,12)
if (dims == 2)
{
  dataset[1] = sum(proddate[,'RawDataType'] == testdate[,'RawDataType']) / dim(testdate)[1]  
  dataset[2] = sum(proddate[,'GameCode'] == testdate[,'GameCode']) / dim(testdate)[1]  
  for (k in c(1:10))
  {
   dataset[k+2] = sum(proddate[,paste0('Top',k,'Game')] == testdate[,paste0('Top',k,'Game')]) / dim(testdate)[1]
  }
} else {
  fail("fail:dimdifferences")
}


#2a: comma's, check if updated
commas = rep(0,10)
for (k in c(1:10))
{
  commas[k] = sum( grep(',',testdate[,paste0('Top',k,'Game')]) > 0)/ dim(testdate)[1]
}
if (sum(commas) <10)
{
  fail("fail:incompatible")
}

#2b: are there any NA's
tryCatch(
  {
rawdatatypes = sqlQuery(conn,paste0("select count(*) from datascientist.dbo.intelligentgame where date ='",
                                date,"' and Rawdatatype=0"))
}, error =function(e){ rawdatatypes = 0})

km = rep(0,10)
gm = rep(0,10)
for (k in c(1:10))
{
  km[k] = unlist(sqlQuery(conn,paste0("select count(*) from datascientist.dbo.intelligentgame where date ='",
                                    date,"' and Top",k,"Game='NA' "))[1])
  gm[k] = unlist(sqlQuery(conn,paste0("select count(*) from datascientist.dbo.intelligentgame where date ='",
                               date,"' and Top",k,"Game is NULL "))[1])
}
if (( rawdatatypes + sum(km) + sum(gm)) > 0)
{
  fail("fail:updates_top_insert")
}

#3: look at summarylog
tryCatch(
  { 
    print(date)
    query = sqlQuery(conn,paste0("select top 1 * from [Datascientist].[dbo].[RunningRecord] where Date = '",date,"' 
                     and Environment = 'staging' order by starttime desc"))

completed = grep('Completed',query$Status)
if (completed != 1)
{
  fail("fail:incomplete")
}

#3.1: compatability, amount of files
games = as.numeric(unlist(strsplit(as.character(query$Status[1]),':'))[2])
if (games != dim(proddate)[1])
{
  fail('fail:gamecount')
}
  },
error = function(msg){fail("No_Log") }
)

if (prod(dataset) < 0.2)
{
  fail('fail:different_results')
}

#4: diversity
#4.1: same category
#todo
#4.2: same datalabel
#todo
#4.3: hot
#todo

