context("PostProcessing")
setwd("..")

source("GlobalConfig.R")


hot = data.frame(cbind(id = c('4607','6416','5517','1047','316','2948','162','4645','3057','4785'),
                       counts = c(200905,161377,141317,139639,128142,100860,94932,87617,86158,77712)),stringsAsFactors = FALSE)

games = c(1,2,3,4,5,6,7,8,9,10,11)

distance_matrix = data.frame(cbind(
  c(10.0000000,0.5070926,0.1195229,0.4780914,0.7171372,0.5070926,0.9561829,0.3779645,0.0000000,0.1883294,0.7559289),
  c(0.5070926 ,10.0000000,0.000000,0.7071068,0.0000000,1.0000000,0.7071068,0.0000000,0.0000000,0.3713907,0.0000000),
  c(0.1195229 ,0.0000000 ,10.00000,0.5000000,0.5000000,0.0000000,0.0000000,0.5270463,0.0000000,0.0000000,0.3162278),
  c(0.4780914 ,0.7071068,0.5000000,10.000000,0.5000000,0.7071068,0.5000000,0.5270463,0.0000000,0.2626129,0.0000000),
  c(0.7171372 ,0.0000000,0.5000000,0.5000000,10.000000,0.0000000,0.5000000,0.7378648,0.0000000,0.0000000,0.6324555),
  c(0.5070926 ,1.0000000,0.0000000,0.7071068,0.0000000,10.000000,0.7071068,0.0000000,0.0000000,0.3713907,0.0000000),
  c(0.9561829 ,0.7071068,0.0000000,0.5000000,0.5000000,0.7071068,10.000000,0.2108185,0.0000000,0.2626129,0.6324555),
  c(0.3779645 ,0.0000000,0.5270463,0.5270463,0.7378648,0.0000000,0.2108185,10.000000,0.4216370,0.0000000,0.2666667),
  c(0.0000000 ,0.0000000,0.0000000,0.0000000,0.0000000,0.0000000,0.0000000,0.4216370,10.000000,0.6565322,0.0000000),
  c(0.1883294 ,0.3713907,0.0000000,0.2626129,0.0000000,0.3713907,0.2626129,0.0000000,0.6565322,10.000000,0.0000000),
  c(0.7559289 ,0.0000000,0.3162278,0.0000000,0.6324555,0.0000000,0.6324555,0.2666667,0.0000000,0.0000000,10.000000)))

colnames(distance_matrix)  = games
row.names(distance_matrix) = games
vects = t(apply(distance_matrix, 2, order, decreasing=T))




control = data.frame(cbind(
  id = c('alm','alk','all','alp','alq','qls'),
  category = c('a','ab','a','ab','a','ab'),
  RawDataType = c('4','8','2','4','1','2','3','4','2','1','3','4'),
  key = c(1,2,3,4,5,6,7,8,9,10,11,15),
  Exclusions = c(0,0,0,0,0,0,0,0,1,0,0,0)
),stringsAsFactors = FALSE)

test_that(
  'initialization',
  {
    expect_s4_class(new('post_processing', configuration =list(cutoff = 0.1, games_to_select = 10),hot=hot,
                        lastupdated='2018-11-03'),'post_processing')
  }
)

test_that(
  'filtercategory',
  {
    post_processor = new('post_processing', configuration =list(cutoff = 0.1, games_to_select = 10), hot = hot,
                         lastupdated='2018-11-03')
    
    for (k in c(1:dim(distance_matrix)[1]))
    {
      expect_equal(vects[k,], order(as.numeric(distance_matrix[k,]),decreasing = TRUE))
    }
    
    
    output = c(2,7,8,3,8,7,2,5,10,7,8)
    for (k in c(1:dim(distance_matrix)[1]))
    {
      expect_equal(as.character(output[k]),filter_category(ordered_row = vects[k,],distance_row = distance_matrix[k,], 
                                              active_game = games[k], updated_control = control, 
                                              games_to_select = 1,  cutoff = 0.1, hot = hot)[2])
      expect_equal(as.character(games[k]), filter_category(ordered_row = vects[k,],distance_row = distance_matrix[k,], 
                                                           active_game = games[k], updated_control = control, 
                                                            games_to_select = 1,  cutoff = 0.1, hot = hot)[1])
    }
    
    k = 5
    expect_equal(c("5","8","4","1","11","3"),filter_category(ordered_row = vects[k,],distance_row = distance_matrix[k,], 
                                       active_game = games[k], updated_control = control, 
                                       games_to_select = 5,  cutoff = 0.1, hot = hot))
    expect_equal(c("5","4607","6416","5517","1047","316"),filter_category(ordered_row = vects[k,],
                                                                          distance_row = distance_matrix[k,], 
                                                             active_game = games[k], updated_control = control, 
                                                             games_to_select = 5,  cutoff = 0.8, hot = hot))
    expect_equal(c('5','8','4','1','11','3','7','4607','6416','5517','1047'),
                   filter_category(ordered_row = vects[k,],distance_row = distance_matrix[k,], 
                    active_game = games[k], updated_control = control, 
                    games_to_select = 10,  cutoff = 0.3, hot = hot))
    
  }
)

test_that(
  'postProcess',
  {
    ppc = new('post_processing', configuration =list(cutoff = 0.1, games_to_select = 1),hot=hot,
                        lastupdated='2018-11-03')
    outp = postProcess(ppc,dataset = distance_matrix,games = control)
    expect_equal(outp[1,1],1)
    expect_equal(outp[1,'Date'],as.factor("2018-11-03"))
    expect_equal(outp[1,'Game'],'1')
    cols = c('SN','Date','Game','Top1Game','Top2Game','Top3Game', 'Top4Game','Top5Game', 'Top6Game', 'Top7Game','Top8Game',
             'Top9Game','Top10Game')
    expect_equal(cols, colnames(outp))
  }
)




