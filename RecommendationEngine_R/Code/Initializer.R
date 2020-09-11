setClass('Recommendation_Configuration', representation( validation = 'logical', valpct = 'numeric', 
                                                         testpct = 'numeric', games_to_select = 'numeric',
                                                         cutoff = 'numeric', production = 'logical'))

setMethod('initialize', 'Recommendation_Configuration', function(.Object, validation = FALSE, valpct= 10, 
                                                                 testpct = 5, games_to_select = 10,
                                                                 cutoff = 0.1, production = TRUE)
{
  validObject(.Object)
  if(valpct > 100){ print("Validation over 100")}
  if(valpct < 0) {print("Validation can't be negative")}
  if(testpct > 100) {print("Testpct over 100")}
  if(testpct < 0) {print("Testpct can't be negative")}
  if(games_to_select < 0) {stop("cannot be negative")}
  
  if(cutoff < 0) {stop("cannot be negative")}
  .Object@validation = validation
  .Object@valpct     = valpct
  .Object@testpct    = testpct
  .Object@games_to_select = floor(games_to_select)
  .Object@cutoff = cutoff
  .Object@production = production
  validObject(.Object)
  return(.Object)
})

setGeneric('testValidation', function(object){return(boolean)})
setMethod('testValidation','Recommendation_Configuration', function(object){return(object@validation)})

setGeneric('getParameters', function(object){return(boolean)})
setMethod('getParameters','Recommendation_Configuration', 
          function(object){
            if (testValidation(object))
            {     return(list(validation = object@valpct, test = object@testpct))}})


setGeneric('getPostProcessingSettings', function(object){return(list)})
setMethod('getPostProcessingSettings', 'Recommendation_Configuration', function(object)
{
  return(list('games_to_select' = object@games_to_select, 'cutoff' = object@cutoff))
})
