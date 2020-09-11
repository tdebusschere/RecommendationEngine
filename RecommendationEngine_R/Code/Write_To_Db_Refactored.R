setClass('write_to_db',representation(conn ='RODBC',domain='logical', setdate = 'character'))

setMethod('initialize', 'write_to_db', function(.Object, production, setdate)
{
  if (production == TRUE) {dbhandle = getHandle(new('connect_to_productionMSSQL'))}
  else { dbhandle = getHandle(new('connect_to_productionMSSQL')) }
  .Object@conn    = dbhandle
  .Object@domain  = production
  .Object@setdate = as.character(as.Date(setdate,'%Y-%m-%d'))
  return(.Object)
})

setGeneric("cleanup_past",function(Object,days){return(integer)})
setMethod("cleanup_past",'write_to_db',function(Object,days)
{
  if( days < 0) { return(0)}
  query = paste0("delete from [DataScientist].[dbo].[IntelligentGame] where Date <= CONVERT(char(10),'",
                 as.character(as.Date(Object@setdate,'%Y-%m-%d')-days),"',126)")
  if (Object@domain == TRUE) {
    sqlQuery(Object@conn, query)
    return(1)
  } else {
    query = gsub('IntelligentGame','IntelligentGame_test',query)
    sqlQuery(Object@conn, query)
    return(query)
  }
})

setGeneric("cleanup_today",function(Object){return()})
setMethod("cleanup_today",'write_to_db',function(Object)
{
  query = paste0("delete from [DataScientist].[dbo].[IntelligentGame] where Date = '",as.character(Object@setdate),"'")
  if (Object@domain == TRUE)
  {
    sqlQuery(Object@conn, query)
  } else {
    query = gsub('IntelligentGame','IntelligentGame_test',query)
    sqlQuery(Object@conn, query)
    return(query)
  }
})

setGeneric('createTempDB',function(Object){return(integer)})
setMethod('createTempDB','write_to_db',function(Object){
  if (dim(sqlQuery(Object@conn,"select * from tempdb..sysobjects where name='##tmp5848'"))[1]==1)
  {
    sqlQuery(Object@conn,"drop table ##tmp5848")
  }
  tmp = Object@setdate
  query = c("select a.GameCode as id,a.RawDataType as category,b.id as [key],
            convert(nvarchar(20),a.RawDataType) + ',' + convert(nvarchar(20),a.GameCode) as [key2]
            into ##tmp5848 from [DataScientist].[dbo].[BetRecordMonthlySystemCodeGame] as a left join [DataScientist].[dbo].[GameType] as b
            on a.RawDataType=b.RawDataType and a.GameCode=b.GameCode
            group by a.RawDataType,a.GameCode,b.id order by a.RawDataType")
  if( Object@domain == TRUE)
  {
    res = sqlQuery(Object@conn,query)
    return(1)
  } else {
    return(query)
  }
  
})



setGeneric('updateKeys',function(Object){return(integer)})
setMethod('updateKeys','write_to_db',function(Object)
{

  #CONVERT(char(10), GetDate(),126)
  query2 =paste(c("update ig set ig.rawdatatype = gt.category, ig.gamecode = gt.[id]
                  from DataScientist.dbo.IntelligentGame ig join ##tmp5848 gt on ig.GameCode = gt.[key] where 
                  rawdatatype = 0 and date='",as.character(Object@setdate),"'"),sep='',collapse='')
  if(Object@domain == TRUE)
  {
    res = sqlQuery(Object@conn,query2)
  } else {
    query2 = gsub('IntelligentGame','IntelligentGame_test',query2)
    res = sqlQuery(Object@conn,query2)
    return(query2)
  }
  return(1)  
})


setGeneric('insert_into_table',function(Object,gmm){return(integer)})
setMethod('insert_into_table','write_to_db',function(Object,gmm)
{
  sqlString = "insert into [DataScientist].[dbo].[IntelligentGame] values "
  body = paste(paste("('",gmm[,2],"',0,'",gmm[,3],"','",
                     apply( gmm[,c(4:13)],1,paste,sep="",collapse="','"),"')",sep=""),sep="",collapse=",")
  FullString = paste0(sqlString,body)
  if (Object@domain == TRUE)
  {
    sqlQuery(Object@conn, FullString)
    return(1)
  } else {
    FullString = gsub('IntelligentGame','IntelligentGame_test',FullString)
    sqlQuery(Object@conn, FullString)
    return(FullString)
  }
})

setGeneric('write_to_table', function(Object,Dataset){ return(integer)})
setMethod('write_to_table','write_to_db',function(Object,Dataset)
{
  cleanup_today(Object)
  cleanup_past(Object,10)
  times=ceiling(nrow(Dataset)/1000)
  for (i in 1:times) {
    #1000 records at a time
    if(i!=times){gmm = Dataset[c((i*1000-999):(i*1000)),]}
    else{gmm=Dataset[c((i*1000-999):nrow(Dataset)),]}
    tryCatch({insert_into_table(Object,gmm)}, error= function(e){
      loginfo(paste0("Error in inserting the data:",e), logger='highlevel.module')
    })
  }
  createTempDB(Object)
  updateKeys(Object)
  updateTables(Object)
  return(1)
})

setGeneric('createTempDB',function(Object){return(integer)})
setMethod('createTempDB','write_to_db',function(Object)
{
  if (dim(sqlQuery(Object@conn,"select * from tempdb..sysobjects where name='##tmp5848'"))[1]==1)
  {
    sqlQuery(Object@conn,"drop table ##tmp5848")
  }
  tmp = Object@setdate
  query = c("select a.GameCode as id,a.RawDataType as category,b.id as [key],
            convert(nvarchar(20),a.RawDataType) + ',' + convert(nvarchar(20),a.GameCode) as [key2]
            into ##tmp5848 from [DataScientist].[dbo].[BetRecordMonthlySystemCodeGame] as a left 	join [DataScientist].[dbo].[GameType] as b
            on a.RawDataType=b.RawDataType and a.GameCode=b.GameCode
            group by a.RawDataType,a.GameCode,b.id order by a.RawDataType")
  if (Object@domain == TRUE)
  {
    res = sqlQuery(Object@conn,query)
  } else {
    query = gsub('IntelligentGame','IntelligentGame_test',query)
    sqlQuery(Object@conn, query)
    return(query)
  } 
})



setGeneric('updateTables', function(Object,Dataset){ return(integer)})
setMethod('updateTables','write_to_db',function(Object)
{
  updatequery = paste("update ig set ig.Top",c(1:10),"Game = gt.key2 from DataScientist.dbo.IntelligentGame ig
                      join ##tmp5848 gt on ig.Top",c(1:10),"Game = gt.[key] where 
                      ig.Date='", Object@setdate,"' 
                      and charindex(',',ig.Top",c(1:10),"Game) = 0;",sep='',collapse='')
  sqlQuery(Object@conn,updatequery)
  if (Object@domain == TRUE)
  {
    res = sqlQuery(Object@conn,updatequery)
    return(1)  
  } else {
    updatequery = gsub('IntelligentGame','IntelligentGame_test',updatequery)
    sqlQuery(Object@conn, updatequery)
    return(updatequery)
  }
}          
)             


