library(testthat)
context("Preprocessing_Tests")

setwd("..")
source('GlobalConfig.R')



##basic properties
test_that(
  'initialization',
  {
    expect_s4_class(new("preprocessing"),'preprocessing')
    expect_s4_class(new("preprocessing_counts"),'preprocessing_counts')
    expect_s3_class(new("preprocessing_counts")@Recommendationset, 'data.frame')
    expect_s3_class(new("preprocessing_counts")@control, 'data.frame')
    expect_equal(new("preprocessing_counts")@Members, vector())
    expect_equal(new("preprocessing_counts")@Games, vector())
    #expect_equal(new("preprocessing_counts")@Hot, data.frame())
  }
)



###is the object usable
test_that(
  'getvalidatedDataSets',
  {
    ##mock resulted dataset
    coll = new('getMSSQL_counts_data',TRUE,TRUE)
    obj = new('preprocessing_counts')
	obj@timestamp = getTimeLastUpdatedDB(coll)
    
    
    obj@Recommendationset = mock_getRecommendationSet(coll)
    obj@control = mock_getControlSet(coll)
    obj@Hot = mock_getHotSet(coll)
    obj = validateData(obj,obj@Recommendationset,obj@control,obj@Hot)
	obj@Recommendationset$key = as.numeric(obj@Recommendationset$key)
	validateData(obj,rbind(obj@Recommendationset, c(1,50,3)), obj@control, obj@Hot)
  }
)


test_that(
	'getvalidatedRecommendation',
	{
	    coll = new('getMSSQL_counts_data',TRUE,TRUE)
		obj = new('preprocessing_counts')
		obj@timestamp = getTimeLastUpdatedDB(coll)    
		zm = mock_getRecommendationSet(coll)
		validateRecommendation(obj,zm)
		expect_equal(zm,getRecommendation(validateRecommendation(obj,zm)))
		
		zm$key = 'a0'
		#expect_message(validateRecommendation(obj,zm),'classes recommendation incompatible')
		#stub add further tests
	}
)
	
test_that(
	'getvalidatedControl',
	{
	    coll = new('getMSSQL_counts_data',TRUE,TRUE)
		obj = new('preprocessing_counts')
		obj@timestamp = getTimeLastUpdatedDB(coll)    
		zm = mock_getRecommendationSet(coll)
		ctr = mock_getControlSet(coll)
		validateControl(obj,zm,ctr)
		expect_equal(ctr,getControl(validateControl(obj,zm,ctr)))
		
		zm$key = 'a0'
		#expect_message(validateRecommendation(obj,zm),'classes recommendation incompatible')
		#stub add further tests
	}
)	
	
zm = 
rbind(
c(5,  0,    0,    0,    3,    0,    4,    2,    0,     0,     4),
c(3,  3,    0,    4,    0,    5,    4,    0,    0,     2,     0),
c(1,  0,    4,    4,    3,    0,    0,    5,    0,     0,     0),
c(0,  0,    0,    0,    0,    0,    0,    4,    1,     0,     0),
c(0,  0,    0,    0,    0,    0,    0,    0,    1,     5,     0),
c(0,  0,    4,    0,    0,    0,    0,    0,    0,     0,     2)
)

test_that(
  'preprocessing',
  {
    test = mock_getvalidatedDataSets(new('preprocessing_counts'))
    preprocess_result = preprocess(test)
    expect_s4_class(preprocess_result@Sparse,'dgCMatrix')
    expect_equal(preprocess_result@Members,c(1,2,3,4,5,6))
    expect_equal(preprocess_result@Games,c('1','10','11','2','3','4','5','6','7','8','9'))
    expect_equal(sum(as.matrix(preprocess_result@Sparse) - zm), 0)
  }
)


test_that(
  'getRecommendation',
  {
    coll = new('getMSSQL_counts_data',TRUE,TRUE)
    test = mock_getvalidatedDataSets(new('preprocessing_counts'))
    preprocess_result = preprocess(test)
    expect_s3_class(getRecommendation(preprocess_result), 'data.frame')
    expect_equal(getRecommendation(preprocess_result),mock_getRecommendationSet(coll) )
    expect_equal(sum(as.matrix(preprocess_result@Sparse) - zm), 0)
  }
)

test_that(
  'getSparse',
  {
    coll = new('getMSSQL_counts_data',TRUE,TRUE)
    test = mock_getvalidatedDataSets(new('preprocessing_counts'))
    preprocess_result = preprocess(test)
    expect_s4_class(getSparse(preprocess_result),'dgCMatrix')
    expect_equal(sum(as.matrix(preprocess_result@Sparse) - zm), 0)
  }
)

test_that(
  'getGames',
  {
    coll = new('getMSSQL_counts_data',TRUE,TRUE)
    test = mock_getvalidatedDataSets(new('preprocessing_counts'))
    preprocess_result = preprocess(test)
    expect_is(getGames(preprocess_result),'character')
    expect_equal(getGames(preprocess_result),c('1','10','11','2','3','4','5','6','7','8','9') )
  }
)

test_that(
  'getmembers',
  {
    coll = new('getMSSQL_counts_data',TRUE,TRUE)
    test = mock_getvalidatedDataSets(new('preprocessing_counts'))
    preprocess_result = preprocess(test)
    expect_is(getMembers(preprocess_result),'numeric')
    expect_equal(getMembers(preprocess_result),c(1,2,3,4,5,6))
  }
)

test_that(
  'getHot',
  {
    coll = new('getMSSQL_counts_data',TRUE,TRUE)
    test = mock_getvalidatedDataSets(new('preprocessing_counts'))
    preprocess_result = preprocess(test)
    expect_is(getHot(preprocess_result),'data.frame')
  }
)

test_that(
    'getControl',
    {
      coll = new('getMSSQL_counts_data',TRUE,TRUE)
      test = mock_getvalidatedDataSets(new('preprocessing_counts'))
      preprocess_result = preprocess(test)
      expect_is(getControl(preprocess_result),'data.frame')
    }
)

test_that(
	'getTimeLastUpdated',
	{
		coll = new('getMSSQL_counts_data',TRUE,TRUE)
		test = getTimeLastUpdatedDB(coll)
		expect_equal(test,'2018-11-03')
		
	}
)
