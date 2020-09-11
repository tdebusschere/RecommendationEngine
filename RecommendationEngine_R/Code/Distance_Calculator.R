
setClass('Distancecalculator')
setClass('CosineDistancecalculator',contains = c('Distancecalculator'))

setMethod('initialize', 'Distancecalculator', function(.Object)
{
  return(.Object)                     
})



setGeneric('calculate_distance', function(object,input,labels,...){return(data.frame)})
setMethod('calculate_distance','Distancecalculator',  function(object,input,labels){
  return(calculate_distance(new('CosineDistancecalculator'),input,labels))
}
)
setMethod('calculate_distance','CosineDistancecalculator',function(object,input,labels)
{
  self_inner = as.matrix(crossprod(input, input))
  cd_marginal_cumulative = colSums(input * input)
  cd_norms = (sqrt(cd_marginal_cumulative) %*% t(sqrt(cd_marginal_cumulative)))
  cos_distancematrix = as.matrix(self_inner / cd_norms)
  row.names(cos_distancematrix)= labels
  colnames(cos_distancematrix) = labels
  diag(cos_distancematrix)= 10
  return(cos_distancematrix)  
}
)



