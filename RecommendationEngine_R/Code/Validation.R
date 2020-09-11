
assign_to_sets = function(Members,Games,Datasets,parameters)
{
  if( parameters == FALSE) {return(list(Recommendationset = Datasets,validationset = data.frame(),testset=data.frame()))}
  else{
    members = sample(Members,(parameters$test + parameters$validation)*length(Members ) / 100 )
    basicmembers = Members [!Members %in% members]
    testmembers = sample(members,(parameters$test)/(parameters$test + parameters$validation) * length(members))
    validationmembers = sample(members,parameters$validation/(parameters$test + parameters$validation) * length(members))
    basicset = Datasets[Datasets$MemberId %in% basicmembers,]
    testset  = Datasets[Datasets$MemberId %in% testmembers,]
    validationset = Datasets[Datasets$MemberId %in% validationmembers,]
    return(list(Recommendationset = basicset, Validationset = validationset, Testset = testset))
  }
}

toSparse = function(matrix, games)
{
  return(sparseMatrix(i=as.integer(factor(matrix$MemberId)),j=as.integer(matrix$gtd),x=matrix$count))
}

setClass('Validator', representation(dataset = "dgCMatrix", target = "data.frame"))
setMethod('initialize','Validator',function(.Object,dataset,target){
  .Object@dataset = dataset
  .Object@target  = target
  return(.Object)
})


##Proposed Games
setGeneric('get_games_width', function(object, post_processing_settings){ return(object)})
setMethod('get_games_width', 'Validator',function(object,post_processing_settings)
{
  games_to_select = post_processing_settings['games_to_select'][[1]]
  Selection = c(1:games_to_select)
  for (k in c(1:games_to_select))
  {
    Selection[k] = length(sort(unique(unlist(object@target[,c(4:(k+3))])))) / dim(object@target)[1]
  }
  print(Selection)
  plot(Selection)
  text(Selection,labels=c(1:games_to_select))
  return(Selection)
}
)

###Sensitivity / Precision:
setGeneric('SensitivityRec', function(object,post_processing_settings){return(object)})
setMethod('SensitivityRec','Validator', function(object,post_processing_settings)
{
  games_to_select = post_processing_settings['games_to_select'][[1]]
  co_occurrences = as.matrix(crossprod(object@dataset))
  diag(co_occurrences) <- max(co_occurrences) +4
  ranks =  t(apply(co_occurrences,2, order, decreasing=TRUE))
  Games = object@target[,'Game']
  first_zero = apply(co_occurrences,2, function(x){ return( min(which(x == 0)))})
  multimod = function(x,y){return(min(1 + games_to_select,which(x %in% y)))}
  targets = object@target[,c(4:13)]
  res = mapply( multimod, split(ranks,row(ranks)),first_zero)
  precision = rep(0,games_to_select)
  recall    = rep(0,games_to_select)
  for (k in c(1:games_to_select))
  {
    for (cm in c(1:dim(targets)[1]))
    {
      precision[k] = precision[k] +   map_to_precision_game(targets[cm,c(1:(games_to_select ))],Games[ranks[cm,c(2:(k+1))]])
      recall[k]    = recall[k] + map_to_recall_game(targets[cm,c(1:games_to_select )],Games[ranks[cm,c(2:(k+1))]])
    }
    precision[k] = 1.0*precision[k] / length(unlist(targets[,c(1:(games_to_select))]))
    recall[k] = 1.0* recall[k] / sum(sapply(res - 1 ,function(s,k){return(min(c(s,k)))},k))
  }
  return(list(precision=precision, recall = recall))
})

find_targets = function(x,y){
  returnvector = rep(0,games_to_select);
  if (y> 1)
  {   
    returnvector[c(1:(y-1))] = x[c(1:(y-1))]; 
    returnvector[c(1:(y-1))] = games[returnvector]
  }
  return(returnvector)
}


map_to_precision_game = function(intargets, predictions)
{
  return( sum(as.character(unlist(predictions)) %in% intargets) )
}

map_to_recall_game = function(targets, predictions)
{
  return(sum( targets %in% as.character(unlist(predictions ))))
}

