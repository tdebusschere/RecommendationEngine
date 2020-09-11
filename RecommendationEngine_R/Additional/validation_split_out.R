

sets = assign_to_sets(dmm)

prepare_sparse_matrix = function(valset,recommend4)
{
  valset = valset[ valset$key %in% games,]
  valset$cat <- as.numeric(as.factor(valset$MemberId))
  gamelookup = unique(recommend4[,c('key','gtd')])
  gamelookup = gamelookup[order(gamelookup$gtd),]
  valset =valset %>% inner_join(gamelookup,by='key')
  return(valset)  
}

###Proposed Games
valset = prepare_sparse_matrix(valset,recommend4)
tset   = prepare_sparse_matrix(tset,recommend4)

sparse_valset = sparseMatrix(i=valset$cat,j=valset$gtd,x=valset$count, 
                             dims = c(max(valset$cat) , max(recommend4$gtd)))
sparse_tset   = sparseMatrix(i=as.integer(tset$cat),j=as.integer(tset$gtd),x=tset$count,
                             dims = c(max(tset$cat) , max(recommend4$gtd)))

Selection = c(1:games_to_select)
for (k in c(1:games_to_select))
{
  Selection[k] = length(sort(unique(unlist(result_final[,c(2:(k+1))])))) / length(games)
}
plot(Selection)
text(Selection,labels=c(1:games_to_select))



###Sensitivity / Precision:
co_occurrences = as.matrix(crossprod(sparse_valset))
ranks =  t(apply(co_occurrences,2, order, decreasing=TRUE))
first_zero = apply(co_occurrences,2, function(x){ return( min(which(x == 0)))})
multimod = function(x,y){return(min(1 + games_to_select,which(x %in% y)))}
res = mapply( multimod, split(ranks,row(ranks)),first_zero)
find_targets = function(x,y){
  returnvector = rep(0,games_to_select);
  if (y> 1){returnvector[c(1:(y-1))] = x[c(1:(y-1))]; returnvector[c(1:(y-1))] = games[returnvector]}
  return(returnvector)
}
targets = t(mapply(find_targets, split(ranks,row(ranks)), res))


map_to_precision_game = function(intargets, predictions)
{
  return( sum(as.character(unlist(predictions)) %in% intargets) )
}

map_to_recall_game = function(targets, predictions)
{
  return(sum( targets %in% as.character(unlist(predictions ))))
}

precision = c(1:games_to_select)
recall    = c(1:games_to_select)
for (k in c(2:games_to_select))
{
  precision[k] = sum(mapply(map_to_precision_game, 
                            split(targets[,c(1:(games_to_select ))],row(targets)), 
                            split(result_final[,c(2:(k+1))], row(result_final))) / 
                       length(unlist(result_final[,c(2:k+1)])))
  recall[k] = sum(mapply(map_to_recall_game, 
                         split(targets[,c(1:(games_to_select ))],row(targets)), 
                         split(result_final[,c(2:(k+1))], row(result_final)))) / 
    sum(targets[,c(1:(games_to_select))] > 0)
}

