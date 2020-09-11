setClass('Log_To_Database', representation(dbhandle= 'RODBC', process='character',environment='logical', 
                                           starttime='POSIXlt', lastupdate ='Date'))

setMethod('initialize','Log_To_Database',function(.Object, process, environment)
{
  if (environment == TRUE) {dbhandle = getHandle(new('connect_to_productionMSSQL'))}
  else { dbhandle = getHandle(new('connect_to_testMSSQL')) }
  .Object@dbhandle = dbhandle
  .Object@environment = environment
  .Object@process = process
  .Object@starttime = as.POSIXlt(Sys.time())
  .Object@lastupdate
  insert_to_database(.Object)
  return(.Object)  
}
)

setGeneric('insert_to_database',function(object){return(string)})
setMethod('insert_to_database',signature ='Log_To_Database', function(object)
{
  string = paste0("insert into [Datascientist].[dbo].[RunningRecord] select '", 
                  as.character(as.Date(as.POSIXlt(object@starttime))),"','",
                  as.character(object@starttime),"','','Initialized','",
                  getEnvironment(object),"'")
  object@lastupdate = as.Date(object@starttime)
  sqlQuery(object@dbhandle,string)
})

setGeneric('log_to_database',function(object,update){return(string)})
setMethod('log_to_database',signature ='Log_To_Database', function(object, update)
{
  if (length(grep('Completed',update)) > 0)
  { finishtime = as.POSIXlt(Sys.time())}
  else {finishtime=''}
  string = paste0("update [Datascientist].[dbo].[RunningRecord] set Endtime='",as.character(finishtime),
                  "',Status='",update,"' where Starttime='", as.character(object@starttime),"' and Date = '",
                  as.character(object@lastupdate),"'")
  sqlQuery(object@dbhandle,string)
  return(1)
})
  
setGeneric('set_timestamp',function(object,timestamp){return(string)})
setMethod('set_timestamp',signature='Log_To_Database', function(object,timestamp)
{
  object@lastupdate = as.Date(timestamp)
  string = paste0("Update [Datascientist].[dbo].[RunningRecord] set Date ='", as.character(object@lastupdate),
            "' where starttime ='", as.character(object@starttime),"'")
  sqlQuery(object@dbhandle,string)
  return(object)
})

setGeneric('getEnvironment', function(object){return(string)})
setMethod('getEnvironment', signature='Log_To_Database', function(object)
  {
  if (object@environment == TRUE){return('production')}
  else { return('staging')}
})
