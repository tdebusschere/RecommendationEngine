library(testthat)
context("Test Initalizer")
setwd("..")
source("GlobalConfig.R")

##initalization
test_that(
  'initialization',
  {
    expect_s4_class(new('Recommendation_Configuration'),'Recommendation_Configuration')
    expect_s4_class(new('Recommendation_Configuration',valpct= 101),'Recommendation_Configuration')
    expect_s4_class(new('Recommendation_Configuration',valpct= 10),'Recommendation_Configuration')
    expect_s4_class(new('Recommendation_Configuration',valpct= -10),'Recommendation_Configuration')
    expect_s4_class(new('Recommendation_Configuration',testpct= 10),'Recommendation_Configuration') 
    expect_s4_class(new('Recommendation_Configuration',testpct= 101),'Recommendation_Configuration')
    expect_s4_class(new('Recommendation_Configuration',testpct= -10),'Recommendation_Configuration')
    expect_s4_class(new('Recommendation_Configuration',validation= FALSE),'Recommendation_Configuration')
    expect_error(   new('Recommendation_Configuration',validation= 1))
    expect_error(   new('Recommendation_Configuration',production= 1))
    expect_s4_class(new('Recommendation_Configuration',production= TRUE),'Recommendation_Configuration')
    expect_error(   new('Recommendation_Configuration',games_to_select= -11))
    expect_error(   new('Recommendation_Configuration',cutoff= -11))
    expect_error(   new('Recommendation_Configuration',banana= -11))
    
  }
)

##test_validation

test_that(
  'testValidation',
  {
    expect_equal(testValidation(new('Recommendation_Configuration')),FALSE)
    expect_equal(testValidation(new('Recommendation_Configuration', validation = FALSE)),FALSE)
    expect_equal(testValidation(new('Recommendation_Configuration', validation = TRUE)),TRUE)
  }
)

test_that(
  'getParameters',
  {
    tv = new('Recommendation_Configuration')
    expect_equal(getParameters(tv)$validation , NULL)
    expect_equal(getParameters(tv)$test , NULL)
    tv = new('Recommendation_Configuration', validation=TRUE)
    expect_equal(getParameters(tv)$validation , 10)
    expect_equal(getParameters(tv)$test , 5)
    tv = new('Recommendation_Configuration', validation=TRUE, valpct = 15, testpct = 8)
    expect_equal(getParameters(tv)$validation , 15)
    expect_equal(getParameters(tv)$test , 8)
  }
)


test_that(
  'getPostProcessingSettings',
  {
    expect_equal(getPostProcessingSettings(new('Recommendation_Configuration'))$cutoff,0.1)
    expect_equal(getPostProcessingSettings(new('Recommendation_Configuration'))$games_to_select,10)
    expect_equal(getPostProcessingSettings(new('Recommendation_Configuration',cutoff= 0.5))$cutoff,0.5)
    expect_equal(getPostProcessingSettings(new('Recommendation_Configuration',games_to_select= 5))$games_to_select,5)
    expect_equal(getPostProcessingSettings(new('Recommendation_Configuration',games_to_select= 5.9))$games_to_select,5)
    expect_equal(getPostProcessingSettings(new('Recommendation_Configuration',games_to_select= 5.1))$games_to_select,5)
    expect_equal(getPostProcessingSettings(new('Recommendation_Configuration',games_to_select= 4.99999999999999999))$games_to_select,5)
  }
)
