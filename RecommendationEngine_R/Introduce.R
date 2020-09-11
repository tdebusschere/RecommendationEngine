rm(list = ls())

source('GlobalConfig.R')
logger_database = new('Log_To_Database','Recommendation_Engine',opt$production)

loginfo("Starting to process:",logger='highlevel.module')

config = new('Recommendation_Configuration', cutoff = 0.1, production = opt$production)
loginfo(toString(names(unlist(attributes(config)))), logger='debugger.module')
loginfo(toString(unlist(attributes(config))), logger='debugger.module')

dmm = new('preprocessing_counts')
loginfo("Starting to process:",logger='highlevel.module')
dmm = getvalidatedDataSets(dmm,TRUE)
logger_database = set_timestamp(logger_database,getTimeLastUpdated(dmm))
log_to_database(logger_database,paste0("Processing: Queried"))
dmm = preprocess(dmm)
loginfo("Data Successfully removed from the database:",logger='highlevel.module')
loginfo(paste("Current required memory:",pryr::mem_used() / 10^6,sep=''),logger='highlevel.module')
Games = getGames(dmm)
Excludedlist = getExcluded(dmm)
log_to_database(logger_database,paste0("PreProcessed:",length(Games)))

Members = getMembers(dmm)
Datasets = getRecommendation(dmm)

last_updated = getTimeLastUpdated(dmm)
Controlset = getControl(dmm)
Hot        = getHot(dmm)

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

rm(dmm)
rm(Datasets)
loginfo("Deleted dmm",logger='debugger.module')
loginfo(paste("Current required memory:",pryr::mem_used() / 10^6,sep=''),logger='debugger.module')
log_to_database(logger_database,paste0("Assigned_Sets:",length(Games)))

Distance = new('CosineDistancecalculator')
distance_matrix = calculate_distance(Distance,Dataset,Games)
loginfo("Distance matrix calculated:",logger='highlevel.module')
loginfo(paste("Current required memory:",pryr::mem_used() / 10^6,sep=''),logger='highlevel.module')
loginfo(paste0("Dimension of distance matrix:",toString(dim(distance_matrix))), logger='debugger.module')
log_to_database(logger_database,paste0("Distances:",dim(distance_matrix[1])))


#how to deal with games that are removed from the dataset by sampling the dataset
post_processing_settings = getPostProcessingSettings(config)
outputmatrix = new('post_processing',post_processing_settings,Hot, last_updated)
recommendations = postProcess(outputmatrix, distance_matrix, Controlset)
log_to_database(logger_database,paste0("post_processed:",dim(outputmatrix)[1]))


rm(distance_matrix)
rm(Dataset)
loginfo("Postprocessing is done:",logger='debugger.module')
loginfo(paste("Current required memory:",pryr::mem_used() / 10^6,sep=''),logger='debugger.module')


###to implement
if(testValidation(config)){  

  validation = new('Validator', dataset= toSparse(sets$Validationset), target = recommendations)
  coverage = get_games_width(validation, post_processing_settings)
  outputstats = SensitivityRec(validation,  post_processing_settings)
  loginfo(paste0("Validation,coverage:",paste0(coverage,collapse=',')), logger='highlevel.module')
  loginfo(paste0("Validation,precision:",paste0(outputstats$precision,collapse=',')), logger='highlevel.module')
  loginfo(paste0("Validation,recall:",paste0(outputstats$recall,collapse=',')), logger='highlevel.module')
} 


write_out = new('write_to_db', config@production, last_updated)
write_to_table(write_out,recommendations)
loginfo("Process Completed:",logger='highlevel.module')
log_to_database(logger_database,paste0("Completed:",dim(recommendations)[1]))

