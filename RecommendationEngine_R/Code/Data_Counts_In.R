
setClass('getMSSQL_counts_data',representation( conn='RODBC', domain = 'logical', mock = 'logical', timestamp = "character"))
setGeneric('getRecommendationSet', function(object,...){ return(data.frame)})
setGeneric('getControlSet', function(object,...){return(data.frame)})
setGeneric("getHotSet",function(object,...){return(data.frame)})
setGeneric("getTimeLastUpdatedDB",function(object,...){return(character)})
setGeneric("setTimeLastUpdatedDB",function(object,...){return(object)})



setMethod('initialize', 'getMSSQL_counts_data', function(.Object, production, mock=FALSE )
{
  if (mock == TRUE) 
  {
    a =0
    attr(a,'class') = 'RODBC'
    .Object@domain = production
    .Object@mock   = TRUE
    .Object@conn   = a
    .Object@timestamp = '2018-11-03'
    return(.Object)
  }
  if (production == TRUE) {dbhandle = getHandle(new('connect_to_productionMSSQL'))}
  else { dbhandle = getHandle(new('connect_to_testMSSQL')) }
  .Object@conn = dbhandle
  .Object@domain = production
  return(.Object)
})

setGeneric('mock_getRecommendationSet',function(object,days){return(data.frame())})
setGeneric('mock_getControlSet',function(object,days){return(data.frame())})
setGeneric('mock_getHotSet',function(object,days){return(data.frame())})
setGeneric('getExcludedCategories', function(object){return(data.frame())})


setMethod('getRecommendationSet',signature ='getMSSQL_counts_data', function(object, days = 30)
{
  if (class(days) != 'numeric'){ return(data.frame())}
  if (days %%1 != 0){ return(data.frame())}
  
  if (object@domain != FALSE){
    recommend=sqlQuery(object@conn, 
                       paste0("select convert(int,rank() over( order by( convert(nvarchar(30),a.MemberId) + a.Systemcode))) MemberId,a.CategoryId,a.GameCode as GameTypeId , b.id as [key], a.Counts as count
                               from [DataScientist].[dbo].[BetRecordMonthlySystemCodeGame] as a left join [DataScientist].[dbo].[GameType] as b
                               on a.RawDataType=b.RawDataType and a.GameCode=b.GameCode where a.First30Days='", as.character(object@timestamp),
                               "' AND  b.id IS NOT NULL order by a.MemberId"))
  } else {
    recommend=sqlQuery(object@conn, paste0("select convert(int,rank() over( order by( convert(nvarchar(30),a.MemberId) + a.Systemcode))) MemberId,a.CategoryId,a.GameCode as GameTypeId ,
    b.id as [key], a.Counts as count,a.RawDataType
    from [DataScientist].[dbo].[BetRecordMonthlySystemCodeGame] as a left join [DataScientist].[dbo].[GameType] as b
    on a.RawDataType=b.RawDataType and a.GameCode=b.GameCode where a.First30Days='", as.character(object@timestamp),
    "' AND  b.id IS NOT NULL order by a.MemberId"))
  }
  recommend[,'MemberId'] = as.integer(recommend[,'MemberId'])
  print( class(recommend[,'MemberId']) )
  return(recommend[,c('MemberId','key','count')])
  })

setMethod('mock_getRecommendationSet',signature = 'getMSSQL_counts_data',function(object,days=30)
{
  recommendation = data.frame(
    rbind(
      c(1,1,5),
      c(1,3,3),
      c(1,5,4),
      c(1,6,2),
      c(1,9,4),
      c(2,1,3),
      c(2,2,4),
      c(2,4,5),
      c(2,5,4),
      c(2,8,2),
      c(2,10,3),
      c(3,1,1),
      c(3,2,4),
      c(3,3,3),
      c(3,6,5),
      c(3,11,4),
      c(4,7,1),
      c(4,6,4),
      c(5,7,1),
      c(5,8,5),
      c(6,9,2),
      c(6,11,4),
      c(7,10,2)))
  colnames(recommendation) = c('MemberId','key','count')
  recommendation$key = as.factor(recommendation$key)
  return(recommendation)
})			


setMethod('getControlSet', signature ='getMSSQL_counts_data',
          function(object, days = 30)
          {
            if (class(days) != 'numeric'){ return(data.frame())}
            if (days %%1 != 0){ return(data.frame())}
            if (object@domain == FALSE){
              control=sqlQuery(object@conn, paste0("select distinct a.GameCode as id,a.CategoryId as category,b.id as [key],a.RawDataType 
                                                   from [DataScientist].[dbo].[BetRecordMonthlySystemCodeGame] as a left join [DataScientist].[dbo].[GameType] as b
                                                   on a.RawDataType=b.RawDataType and a.GameCode=b.GameCode where First30Days= '", as.character(object@timestamp),
                                                   "' and b.id is not NULL  order by a.CategoryId"))
            } else {
              control=sqlQuery(object@conn, paste0("select distinct a.GameCode as id,a.CategoryId as category,b.id as [key],a.RawDataType 
                                                   from [DataScientist].[dbo].[BetRecordMonthlySystemCodeGame] as a left join [DataScientist].[dbo].[GameType] as b
                                                   on a.RawDataType=b.RawDataType and a.GameCode=b.GameCode where First30Days= '", as.character(object@timestamp),
                                                   "' and b.id is not NULL  order by a.CategoryId"))
            }                            
            return(control[,c('id','category','key','RawDataType')])
          }
              )

setMethod('mock_getControlSet', signature = 'getMSSQL_counts_data',function(object,days=30)
{
  return(data.frame(cbind(
    id = c('alm','alk','all','alp','alq','qls'),
    category = c('a','ab','a','ab','a','ab'),
    key = c(1,2,3,4,5,6,7,8,9,10,11,15),
    RawDataType= c('1','2','1','3','3','4','5','4','3','9','8','5')
  )))
})		


setMethod('getHotSet', signature='getMSSQL_counts_data',
          function(object,days=30)
          {
            if(class(days) != 'numeric'){return(data.frame())}
            if(days %%1 != 0){return(data.frame())}
            if (object@domain != FALSE)
            {
              hot = sqlQuery(object@conn, paste0("select a.[CategoryId],a.GameCode ,b.id, a.[counts] from (select top 10 CategoryId,[RawDataType],[GameCode],COUNT(*) as [counts]
                                                 from [DataScientist].[dbo].[BetRecordMonthlySystemCodeGame] where First30Days = '",as.character(object@timestamp),"' 
                                                 and RawDataType not in (1,2,3,4,16,17,22,23,24,28,29,36,42,79,80)
                                                 group by CategoryId,[RawDataType],[GameCode]
                                                 order by [counts] desc) as  a left join [DataScientist].[dbo].[GameType] as b
                                                 on a.RawDataType=b.RawDataType and a.GameCode=b.GameCode and b.id is not NULL order by a.[counts] desc") )         
            } else {
              hot = sqlQuery(object@conn, paste0("select a.[CategoryId],a.GameCode ,b.id, a.[counts] from (select top 10 CategoryId,[RawDataType],[GameCode],COUNT(*) as [counts]
                                                 from [DataScientist].[dbo].[BetRecordMonthlySystemCodeGame] where First30Days = '",as.character(object@timestamp),"'
                                                 and and RawDataType not in (1,2,3,4,16,17,22,23,24,28,29,36,42,79,80)
                                                 group by CategoryId,[RawDataType],[GameCode]
                                                 order by [counts] desc) as  a left join [DataScientist].[dbo].[GameType] as b
                                                 on a.RawDataType=b.RawDataType and a.GameCode=b.GameCode and b.id is not NULL order by a.[counts] desc") )     
            }
            return(hot[,c('id','counts')])
          })
		  
setMethod('mock_getHotSet',signature='getMSSQL_counts_data',function(object,days=30)
{
    return( data.frame(cbind( id= c(6,4,3,2,5,8,9,10,11,1),counts=c(11,9,8,7,6,5,4,3,2,1))))                       
})	

setMethod("setTimeLastUpdatedDB",signature='getMSSQL_counts_data',function(object)
{
	data = sqlQuery(object@conn,"SELECT max(First30Days) as date FROM [DataScientist].[dbo].[BetRecordMonthlySystemCodeGame]")
	object@timestamp = as.character(data[[1]])
	return(object)
})

setMethod("getTimeLastUpdatedDB",signature='getMSSQL_counts_data',function(object)
{
    time = object@timestamp
	return(time)
})

setMethod("getExcludedCategories", signature='getMSSQL_counts_data', function(object)
{
  return(data.frame(
    gamesupplier = c(1,1,1,1,1,1,6,6,6,6,9,9,9,11,11),
    rawdatatype  = c(1,2,3,4,42,79,16,17,36,80,22,23,24,28,29),
    urltext      = c('BB','BB','BB','BB','BB','BB','AG','AG','AG','AG','MG','MG','MG','PT','PT'),
    discountname = c('BBINbbsport', 'BBINvideo','BBINprobability','BBINlottery','BBINFish30','BBINFish38',
                     'AgBr','AgEbr','AgHsr','AgYoPlay','Mg2Real','Mg2Slot','Mg2Html5','Pt2Real','Pt2Slot')
  ))
}
)