setClass('post_processing',representation(cutoff ='numeric', games_to_select = 'integer', 
                                          hot = 'data.frame', lastupdated='character',
                                          Excluded = 'data.frame'))

setMethod('initialize', 'post_processing', function(.Object, configuration, hot, lastupdated)
{
  if (!is.numeric(configuration['games_to_select']) || (!is.numeric(confiration['cutoff'])))
  {
    .Object@cutoff = 0.1
    .Object@games_to_select = as.integer(10)
    .Object@hot = hot
    .Object@lastupdated = lastupdated
    return(.Object)
  }
  if(round(configuration['games_to_select']) == configuration['games_to_select'])
  {
    .Object@games_to_select = as.integer(configuration['games_to_select'])
  }
  .Object@cutoff = confiration['cutoff']
  return(.Object)
})

setGeneric('postProcess',function(.Object,dataset,...){return(data.frame)})
setMethod('postProcess','post_processing', function(.Object, dataset, games)
{
  vects = t(apply(dataset, 2, order, decreasing=T))
  result5  = matrix(0,nrow = dim(vects)[1], ncol = 11)
  for (k in c(1:dim(vects)[1]))
  {
    tryCatch({
              result5[k,] = filter_category(vects[k,],dataset[k,],games[k,'key'],games,
                                            .Object@games_to_select,.Object@cutoff,.Object@hot)
              },error = function(msg) {
                logerror(paste0('fall-back to hot in row:',k), logger='debugger.module')
                logerror(paste0(mgs),logger='debugger.module')
                result5[k,] = Object@hot[c(1:.Object@games_to_select),'id']
              }
              )
  }  
  result_final=data.frame(result5,stringsAsFactors = FALSE)[,c(1:(.Object@games_to_select+1))]
  result_final2=data.frame(rep(1:nrow(result_final)),.Object@lastupdated,result_final[,1:(.Object@games_to_select+1)])
  cnames = paste('Top',c(1:.Object@games_to_select),'Game',sep='')
  colnames(result_final2)=c("SN","Date","Game", cnames)
  return(result_final2)
})

filter_category = function(ordered_row, distance_row, active_game, updated_control,games_to_select, cutoff, hot)
{
  results = rep(0,games_to_select)
  ##Exclude Categories
  ##1 fill with other categories, not in excluded
  results = filterCategory(ordered_row, active_game, updated_control,games_to_select,results)
  ##2 Cleanup
  results = cleanupStep(results, distance_row, cutoff, active_game, games_to_select, updated_control)
  if ( length(results) > games_to_select) { return(results) }
  
  ##Exclude RawDataType, Include Categories
  results = filterCategory(ordered_row, active_game, updated_control,games_to_select,results,'2')  
  ##2 Cleanup
  results = cleanupStep(results, distance_row, cutoff, active_game, games_to_select, updated_control)
  if ( length(results) > games_to_select) { return(results) }
  
  ##Include RawDataType, Include Categories
  results = filterCategory(ordered_row, active_game, updated_control,games_to_select,results,'3')
  ##2 Cleanup
  results = cleanupStep(results, distance_row, cutoff, active_game, games_to_select, updated_control)
  if ( length(results) > games_to_select) { return(results) }
  
  ###Fill the rest with hot
  filled_portion =   filled_portion = max(c(0,which(distance_row[results] > cutoff)))
  results = updated_control[results,'key']
  results[c((filled_portion + 1):games_to_select)] = hot[!hot[,'id'] %in% results,'id'][c(1:(games_to_select - filled_portion))]
  return(as.character(c(active_game,results)))
}


#cleanup
cleanupStep = function(results, distance_row, cutoff, active_game, games_to_select, updated_control)
{
  filled_portion = max(c(0,which(distance_row[results] > cutoff)))
  #RawDataType
  if(filled_portion == games_to_select) {
    results = updated_control[results,'key']  
    return(as.character(c(active_game,results)))
  }
  else {
    results[c((filled_portion+1):games_to_select)] = 0
    return(results)
  }
} 

#filterCategory
filterCategory = function(ordered_row, active_game, updated_control, games_to_select, results, type_filter='1')
{
  filled_portion = sum(results != 0)
  offset = 0
  active_category = updated_control[updated_control[,c('key')] == active_game,'category']
  active_datatype = updated_control[updated_control[,c('key')] == active_game,'RawDataType']
  if (type_filter == '1') {
    filterCategory = (updated_control[,'category'] != active_category & updated_control[,'Exclusions'] == 0)
  } else if (type_filter == '2') {
    filterCategory = (updated_control[,'category'] == active_category & updated_control[,'Exclusions'] == 0 & updated_control[,'RawDataType'] != active_datatype)
  } else {
    filterCategory = (updated_control[,'category'] == active_category & updated_control[,'Exclusions'] == 0 & updated_control[,'RawDataType'] == active_datatype)
    offset = 1
  }
  if(sum(filterCategory) == 0){return(results)}
  orderedIndicesFiltered = ordered_row %in% which(filterCategory)
  maxindex = min(c(games_to_select-filled_portion,length(ordered_row[orderedIndicesFiltered]) - offset))
  if (maxindex > 0)
  {
   results[c((filled_portion+1):(filled_portion+maxindex))] = ordered_row[orderedIndicesFiltered] [c((1+offset):(maxindex+offset))]
  }
  return( results )
}


