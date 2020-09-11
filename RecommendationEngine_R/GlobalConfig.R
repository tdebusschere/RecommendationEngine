library("optparse")
library('logging')
library('RODBC')
library('coop')
require('dplyr')
require('Matrix')
require('pryr')
require('testthat')
require('methods') 

basicConfig()
addHandler(writeToFile, logger="highlevel", file=".//logging//Time_Tracing")
addHandler(writeToFile, logger='debugger', file='.//logging//Debug_Tracing')

tryLoadFile = function(str){
  loginfo(paste0("loading ",str),logger='debugger.module')
  tryCatch({source(str)},
           error=function(msg){
             logerror(paste0("Couldn't load the ",str, toString(msg),sep=''), logger='debugger.module')
           },
           warning=function(msg){source(str);logwarn(toString(msg),logger='debugger.module')})
}

tryLoadFile('Code/Log_To_Database.R')

option_list = list(
  make_option(c("-p", "--production"), type="character", default="TRUE", 
              help="production environment", metavar="character")
);
opt_parser = OptionParser(option_list=option_list);
opt = parse_args(opt_parser);

if (is.null(opt$production)){
  print_help(opt_parser)
  stop("At least one argument must be supplied (input file).n", call.=FALSE)
} else if (tolower(opt$production) == 'false') {
  opt$production = FALSE
} else {
  opt$production = TRUE
}

tryLoadFile('Code/Initializer.R')
tryLoadFile('Code/Connect_To_SQL.R')
tryLoadFile("Code/Data_Counts_In.R")
tryLoadFile('Code/Preprocessing.R')
tryLoadFile('Code/Distance_Calculator.R')
tryLoadFile('Code/postprocessing.R')
tryLoadFile('Code/Write_To_Db_Refactored.R')

