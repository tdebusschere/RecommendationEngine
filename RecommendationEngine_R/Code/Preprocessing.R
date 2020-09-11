###Ideally Use A singleton Pattern For DataAccess



setClass('preprocessing', representation(Recommendationset = "data.frame", control = "data.frame",
                                         Hot = "data.frame",timestamp="character", Exclusions = 'data.frame',
                                         Members = 'vector', Games = 'vector', Sparse = 'dgCMatrix'))
setClass('preprocessing_counts', contains = 'preprocessing' )

setMethod('initialize','preprocessing',function(.Object){
  .Object@Recommendationset = data.frame(MemberId = integer(),
                                         key = factor(),
                                         count = numeric(),
                                         stringsAsFactors = TRUE)
  .Object@Hot = data.frame(id = integer(), counts=integer())
  .Object@control = data.frame(id = integer(), category = factor(), key = factor(),RawDataType=factor(), stringsAsFactors = FALSE)
  .Object@Exclusions = data.frame(gamesupplier = integer(), rawdatatype = integer(),urltext = factor(), discountname=factor(),
                                  stringsAsFactors = FALSE)
  .Object@Members = vector()
  .Object@Games   = vector()
  return(.Object)
})


setGeneric('validateData', function(object,recommendation,control,Hot){ return(object)})
setGeneric('validateRecommendation',function(object,recommendation) {return(object)})
setGeneric('validateControl',function(object,recommendation,control){return(object)})
setGeneric('validateHot',function(object,recommendation,Hot){return(object)})


setMethod('validateRecommendation','preprocessing_counts',function(object,recommendation){
   if (!(sum(colnames(recommendation) == colnames(object@Recommendationset)
           ) == length(colnames(recommendation))))
   { 
      loginfo("Headers recommendation incompatible", logger='debugger.module')
   }
  #for (i in c(1:length(recommendation)))
  #{
  #  if (class(object@Recommendationset[,i]) != class(recommendation[,i]))
  #  {
  #    loginfo("classes recommendation incompatible", logger='debugger.module')
  #  }
  #}
  if (length(unique(recommendation$key)) <= 1 ) 
  {
      loginfo("Recommendationset has at most one game, what's the point", logger='debugger.module')
  }
  object@Recommendationset = recommendation
  return(object)
})


setMethod('validateControl','preprocessing_counts',function(object,recommendation,control)
{
   if (!(sum(colnames(control) == colnames(object@control)) == length(colnames(control))))
   {
	  loginfo("Headers Control incompatible", logger='debugger.module')
   }
   #for (i in c(1:length(colnames(control))))
   #{
   #   if (class(object@control[,i]) != class(control[,i]))
   #   {
	 #   loginfo("classes control incompatible", logger='debugger.module')
   #   }
   #}
   if(sum(recommendation$key %in% control$key)==length(unique(recommendation$key)))
   {
     loginfo("Keys in Recommendation that are not in control", logger='debugger.module')
   }
   if(sum(control$key %in% recommendation$key)==length(unique(recommendation$key)))
   {
     loginfo("Keys in Control that are not in Recommendation", logger='debugger.module')
   }
   if(length(unique(control$category)) <=1)
   {
     loginfo("Only one key in control", logger='debugger.module')
   }
  object@control = control
  return(object)
})



setMethod('validateHot','preprocessing_counts',function(object,recommendation,Hot)
{
   if (!(sum(colnames(Hot) == colnames(object@Hot)
   ) == length(colnames(Hot))))
   {
  	loginfo("Headers Hot incompatible",logger='debugger.module')
   }
   #for (i in c(1:length(Hot)))
   #{
   # if (class(object@Hot[,i]) != class(Hot[,i]))
   # {
   #    loginfo("object Hot incompatible",logger='debugger.module')
	 # }
   #}
   if (length(Hot) > length(unique(recommendation$key)))
   {
     loginfo("Cannot have more Hot than movies in Recommendation", logger='debugger.module') 
   } else if (sum(!Hot$id %in% recommendation$key) > 1)
   {
     loginfo("Recommendations in Hot that are not Recommendations", logger='debugger.module')
   }
   object@Hot = Hot
   return(object)
})


setMethod('validateData', 'preprocessing_counts', function(object,recommendation,control,Hot){
  object = validateRecommendation(object,recommendation)
  object = validateControl(object,recommendation,control)
  object = validateHot(object,recommendation,Hot)
  return(object)
})



setGeneric('getvalidatedDataSets', function(object,...){ return(object)})
setMethod('getvalidatedDataSets','preprocessing_counts', function(object, production){
  tryCatch({
	processor = new('getMSSQL_counts_data', production)
  }, error = function(e){
    logerror("Couldn't initialize a 'getMSSQL_counts_data' object",logger='highlevel.module')
	  return("object")
  })
  tryCatch({      processor = setTimeLastUpdatedDB(processor)}, error = 
             function(e) {logerror("Error in setting timestamp", logger='highlevel.module'); stop()})
  tryCatch({ timestamp      = getTimeLastUpdatedDB(processor)}, 
           error = function(e) {logerror("error in getting timestamp",logger='highlevel.module'); stop();})
  tryCatch({ recommendation = getRecommendationSet(processor) }, 
           error = function(e) { print("error in getting recommendation data",logger='highlevel.module'); stop();})
  tryCatch({ control        = getControlSet(processor)}, 
           error=function(e) {print("error in getting control data",logger='highlevel.module'); stop();})
  tryCatch({ hot            = getHotSet(processor)}, 
           error = function(e) {print("error in getting hot data", logger='highlevel.module'); stop();})
  tryCatch({ exclusions     = getExcludedCategories(processor)}, 
           error = function(e) {print("error in getting hot data", logger='highlevel.module'); stop();})
  
  object@timestamp = timestamp
  #object@control = control
  #object@Recommendationset = recommendation
  #object@Hot = hot
  loginfo(paste("Object controls dim: ",toString(dim(control)),sep=""),logger='debugger.module')
  loginfo(paste("Object Recommendations dim: ",toString(dim(recommendation)),sep=""),logger='debugger.module')
  loginfo(paste("Object Hot dim: ",toString(dim(hot)),sep=""),logger='debugger.module')
  loginfo(paste("currently required memory:"),pryr::mem_used() / 10^6,logger='debugger.module')
  object = validateData(object,recommendation,control, hot)
  object@Exclusions = exclusions
  return(object)
})

setGeneric('preprocess', function(object,...){ return(object)})
setMethod('preprocess','preprocessing_counts', function(object)
{
  loginfo('preprocessing:_______________', logger='debugger.module')
  recommend = object@Recommendationset
  recommend1=recommend %>% group_by(MemberId) %>% dplyr::summarise(game=length(key))
  recommend1=recommend1[recommend1$game>1,]     
  recommend4 =data.frame(left_join(recommend1,recommend,by='MemberId'))
  recommend4$key = as.character(recommend4$key)
  control = object@control
  loginfo(paste0('Reformatting Complete; Currently required memory',pryr::mem_used() / 10^6), logger='debugger.module')

  recommend4$cat <- as.numeric(as.factor(recommend4$MemberId))
  recommend4$gtd <- as.numeric(as.factor(recommend4$key))
  object@Members = sort(unique(recommend4$MemberId))
  object@Recommendationset = recommend4
  object@Games   = sort(unique(recommend4$key))
  control$key=as.character(control$key)
  
  sparse_version = sparseMatrix(i=recommend4$cat,j=recommend4$gtd,x=as.numeric(recommend4$count))
  loginfo(paste0('Sparse Matrix Constructed; Currently required memory',pryr::mem_used() / 10^6), logger='debugger.module')
  loginfo(paste0("Dimension of the sparse matrix =", dim(sparse_version)), logger='debugger.module')
  loginfo(paste0("Number of games =", length(object@Games)), logger='debugger.module')
  loginfo(paste0("Number of games in control =", length(object@control$key)), logger='debugger.module')
  loginfo(paste0("Amount of members =", length(object@Members)), logger='debugger.module')
  object@Recommendationset = recommend4
  object@Sparse = sparse_version
  object@control = control[control$key %in% object@Games,]
  object@control = object@control[order(object@control$key),]
  object@control$idx = rownames(object@control)
  object@control[,'Exclusions'] = 0
  object@control[ object@control[,'RawDataType'] %in% object@Exclusions[,'rawdatatype'],'Exclusions'] = 1
  return(object)
})


setGeneric('getRecommendation', function(object,...){ return(data.frame)})
setMethod('getRecommendation','preprocessing_counts', function(object)
{
  return(object@Recommendationset)
})

setGeneric('getSparse', function(object,...){ return(matrix)})
setMethod('getSparse','preprocessing_counts', function(object)
{
  return(object@Sparse) 
})

setGeneric('getControl', function(object,...){ return(data.frame)})
setMethod('getControl','preprocessing_counts', function(object)
{
  return(object@control)
})

setGeneric('getGames', function(object,...){return(data.frame)})
setMethod('getGames','preprocessing_counts',function(object)
{
  return(object@Games)
})

setGeneric('getMembers', function(object,...){return(data.frame)})
setMethod('getMembers','preprocessing_counts',function(object)
{
  return(object@Members)
})

setGeneric('getHot', function(object,...){return(data.frame)})
setMethod('getHot', 'preprocessing_counts',function(object)
{
  return(object@Hot)
})

setGeneric("getTimeLastUpdated", function(object,...){return(character)})
setMethod("getTimeLastUpdated", 'preprocessing_counts',function(object)
{
 return(object@timestamp)
})

setGeneric("getExcluded", function(object){return(data.frame)})
setMethod("getExcluded", 'preprocessing_counts', function(object)
{
 return(object@Exclusions)  
}
)

setGeneric('mock_getvalidatedDataSets',function(object,production){return(object)})
setMethod('mock_getvalidatedDataSets','preprocessing_counts', function(object,production)
{
  coll = new('getMSSQL_counts_data',TRUE,TRUE)
  object@timestamp = getTimeLastUpdatedDB(coll)
  object@Recommendationset = mock_getRecommendationSet(coll)
  object@control = mock_getControlSet(coll)
  object@Hot = mock_getHotSet(coll)
  object@Exclusions = getExcludedCategories(coll)
  return(object)
})

setGeneric('mock_preprocessed',function(object){return(list)})
setMethod('mock_preprocessed','preprocessing_counts',function(object)
{
	object = mock_getvalidatedDataSets(object)
	object = preprocess(object)
	return(list(Games = object@Games, Sparse = object@Sparse, Recommendationset = object@Recommendationset, 
				control = object@control, Hot = object@Hot, timestamp = object@timestamp, Excludedlist = object@Exclusions))
}
)



