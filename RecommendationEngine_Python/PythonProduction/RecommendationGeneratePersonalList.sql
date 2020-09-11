/****** Object:  StoredProcedure [dbo].[RecommendationGeneratePersonalList]    Script Date: 2019/12/12 下午 04:44:08 ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
DROP PROCEDURE IF EXISTS RecommendationGeneratePersonalList
GO

CREATE PROCEDURE [dbo].[RecommendationGeneratePersonalList] 
	@execution_time DATETIME
AS
BEGIN


SET NOCOUNT ON
DECLARE @COUNT INT = 0, 
		@cnt INT = 1
	    @start_date DATETIME =  (SELECT dateadd(hh, -24*(7),@execution_time))

SELECT @COUNT = COUNT(1) FROM [datascientist].dbo.DS_RecommenderSystem_TemporaryUserData WHERE last_executed != @execution_time
SELECT @cnt   = COUNT(1) FROM [datascientist].dbo.DS_RecommenderSystem_TemporaryUserData WHERE last_executed = @execution_time

IF  ( @COUNT != 0 or @cnt = 0)
BEGIN
	TRUNCATE TABLE [datascientist].dbo.DS_RecommenderSystem_TemporaryUserData;

	INSERT INTO [datascientist].dbo.DS_RecommenderSystem_TemporaryUserData
	SELECT [gameacCOUNT], 
       	   [siteid], 
           [gametypesourceid] ,
	       @execution_time
	FROM   [BalanceCenterSummarize].[dbo].[ds_balancecenterdailyquery] (nolock)
	WHERE  dateplayed >= @start_date AND
           dateplayed <= @execution_time
	GROUP  BY [gameacCOUNT], 
              [siteid], 
              [gametypesourceid] 
END

SELECT ABS((CAST(HASHBYTES('SHA2_256',gameacCOUNT) AS BIGINT) ) %20  ) as FirstLetters
	   FROM [datascientist].dbo.DS_RecommenderSystem_TemporaryUserData
	   GROUP BY ABS((CAST(HASHBYTES('SHA2_256',gameacCOUNT) AS BIGINT) ) %20  ) 
	   ORDER BY abs((cast(hashbytes('SHA2_256',gameacCOUNT) AS BIGINT) ) %20  ) DESC

END


