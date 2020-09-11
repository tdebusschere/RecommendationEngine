context("Distance Calculator")

setwd("..")
source('GlobalConfig.R')


tester = mock_preprocessed(new('preprocessing_counts'))

test_that(
  'initialization',
  {
    expect_s4_class(new('CosineDistancecalculator'), 'Distancecalculator')
    expect_s4_class(new('CosineDistancecalculator'), 'CosineDistancecalculator')
  }
)

test_that(
  'cosine_similarity',
  {
    tmm = tester$Sparse
    reference = coop::cosine(tmm)
    
    totest = calculate_distance(new('CosineDistancecalculator'),tmm,tester$Games)
    expect_equal(colnames(totest), tester$Games)
    expect_equal(rownames(totest), tester$Games)
    expect_lt(diag(totest)[1]-10,0.0001)
    diag(totest) = 0
    for (k in c(1:dim(totest)[1]))
    {
      expect_lt(sum(totest[k,]- reference[k,]),0.0001)
    }
    
  }
)