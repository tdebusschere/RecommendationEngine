rm(list = ls())

library(logging)
require(pryr)

basicConfig()
addHandler(writeToFile, logger="highlevel", file=".//logging//Time_Tracing")
addHandler(writeToFile, logger='debugger', file='.//logging//Debug_Tracing')

loginfo("Starting to process:",logger='highlevel.module')
tryLoadFile = function(str){
  loginfo(paste0("loading ",str),logger='debugger.module')
  tryCatch({source(str)},
           error=function(msg){
             logerror(paste0("Couldn't load the ",str, toString(msg),sep=''), logger='debugger.module')
           },
           warning=function(msg){source(str);logwarn(toString(msg),logger='debugger.module')})
}

tryLoadFile('Initializer.R')
tryLoadFile('Preprocessing.R')
tryLoadFile('Distance_Calculator.R')
tryLoadFile('postprocessing.R')
tryLoadFile('Write_To_Db_Refactored.R')

config = new('Recommendation_Configuration')
loginfo(toString(names(unlist(attributes(config)))), logger='debugger.module')
loginfo(toString(unlist(attributes(config))), logger='debugger.module')


dmm = new('preprocessing_counts')
loginfo("Starting to process:",logger='highlevel.module')
dmm = getvalidatedDataSets(dmm, config@production)
dmm = preprocess(dmm)
loginfo("Data Successfully removed from the database:",logger='highlevel.module')
loginfo(paste("Current required memory:",pryr::mem_used() / 10^6,sep=''),logger='highlevel.module')
Games = getGames(dmm)
Members = getMembers(dmm)
Datasets = getRecommendation(dmm)
parameters = dist


###to implement
if(testValidation(config)){  
  dist = getParameters(config)
  sets = assign_to_sets(Games = Games,Members = Members, Datasets = Datasets, parameters = dist)
  Dataset    = toSparse(sets$Recommendationset)
} else {
  dist = FALSE
  Dataset = getSparse(dmm)
}
loginfo("Data Successfully assigned to sets:",logger='highlevel.module')
loginfo(paste("Current required memory:",pryr::mem_used() / 10^6,sep=''),logger='highlevel.module')


last_updated = getTimeLastUpdated(dmm)

Controlset = getControl(dmm)
Games      = getGames(dmm)
Hot        = getHot(dmm)

Distance = new('CosineDistancecalculator')
distance_matrix = calculate_distance(Distance,Dataset,Games)
loginfo("Distance matrix calculated:",logger='highlevel.module')
loginfo(paste("Current required memory:",pryr::mem_used() / 10^6,sep=''),logger='highlevel.module')

#how to deal with games that are removed from the dataset by sampling the dataset
post_processing_settings = getPostProcessingSettings(config)
outputmatrix = new('post_processing',post_processing_settings,Hot, last_updated)
recommendations = postProcess(outputmatrix, distance_matrix, Controlset)

###to implement
if(testValidation(config)){  
  new('Validator', dataset= sets$Validationset, target = recommendations)
  validation = new('Validator', dataset= sets$Validationset, target = recommendations)
  get_games_width(validation, post_processing_settings)
} 



write_out = new('write_to_db', config@production, last_updated)
write_to_table(write_out,recommendations)
loginfo("Process Completed:",logger='highlevel.module')
