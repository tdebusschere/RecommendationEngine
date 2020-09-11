library(testthat)
rm(list=ls())
if (.Platform$OS.type == 'Windows')
{
##test_log_todatabase
test_file(path=".//tests//test_Log_To_Database.R", 
          reporter = JunitReporter$new(file=".//Reports//test_Log_To_Database.xml"))

##test_initialization
test_file(path=".//tests//test_initialization.R", reporter = JunitReporter$new(file=".//Reports//test_initialization.xml"))

##test_connection
test_file(path=".//tests//test_DB.R", reporter = JunitReporter$new(file=".//Reports//test_db.xml"))

##getMSSQL_counts_data
test_file(path=".//tests//test_get_data.R", reporter = JunitReporter$new(file = ".//Reports//test_get_data.xml"))

##Test_preprocessing
test_file(path=".//tests//test_preprocessing.R", 
          reporter = JunitReporter$new(file=".//Reports//test_preprocessing.xml"))

##Test_calculate_Distance
test_file(path=".//tests//test_Distance_Cosine.R", 
          reporter = JunitReporter$new(file=".//Reports//test_distance_cosine.xml"))
		  
##Test_Post_Processing
test_file(path=".//tests//test_post_processing.R",
          reporter = JunitReporter$new(file=".//Reports//test_post_processing.xml"))

#test_database
test_file(path=".//tests//test_write_to_db.R",
          reporter = JunitReporter$new(file="./Reports/test_write_to_db.xml"))

} else {

test_file(path="./tests/test_Log_To_Database.R", 
            reporter = JunitReporter$new(file="./Reports/test_Log_To_Database.xml"))
  
    
test_file(path="./tests/test_initialization.R", reporter = JunitReporter$new(file="./Reports/test_initialization.xml"))
  
##test_connection
test_file(path="./tests/test_DB.R", reporter = JunitReporter$new(file="./Reports/test_db.xml"))
  
##getMSSQL_counts_data
test_file(path="./tests/test_get_data.R", reporter = JunitReporter$new(file = "./Reports/test_get_data.xml"))

##preprocessing
test_file(path="./tests/test_preprocessing.R", 
          reporter = JunitReporter$new(file="./Reports/test_preprocessing.xml"))

##Test_calculate_Distance
test_file(path="./tests/test_Distance_Cosine.R", 
          reporter = JunitReporter$new(file="./Reports/test_distance_cosine.xml"))

test_file(path="./tests/test_write_to_db.R",
          reporter = JunitReporter$new(file="./Reports/test_write_to_db.xml"))

}