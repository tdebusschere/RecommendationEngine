
drop procedure  if exists RecommendationGenerateResultPortion
GO


USE [resultpool]
GO
/****** Object:  StoredProcedure [dbo].[RecommendationGeneratePersonalList]    Script Date: 2019/12/19 上午 11:21:10 ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO

CREATE procedure [dbo].[RecommendationGenerateResultPortion] 
	@execution_time datetime, @selectinfo int, @hotgame_time datetime, @csupdate_time datetime
AS
BEGIN

	IF OBJECT_ID('tempdb.dbo.##jy2') IS NOT NULL DROP TABLE ##jy2;
	IF OBJECT_ID('tempdb.dbo.##result') IS NOT NULL DROP TABLE ##result;
   

	SELECT a.gameaccount, 
		   a.siteid, 
		   b.correspondgame, 
		   Max(b.cosinesimilarity) cosinesimilarity 
	INTO   ##jy2 
	FROM   [datascientist].dbo.DS_RecommenderSystem_TemporaryUserData a 
    JOIN   [datascientist].dbo.DS_RecommenderSystem_CosineSimilarity b 
	ON a.gametypesourceid = b.game and b.updatetime = @csupdate_time
	where abs((cast(hashbytes('SHA2_256',a.gameaccount) as bigint) ) %20  ) = @selectinfo
	GROUP  BY a.gameaccount, 
              a.siteid, 
              b.correspondgame 

	SELECT  jy.gameaccount, 
			jy.siteid, 
			jy.correspondgame , 
			row_number() over( partition by jy.gameaccount,jy.siteid order by cosinesimilarity DESC) RNS
	INTO   ##result 
	FROM   ##jy2 jy 
	WHERE not exists (select 1 FROM [datascientist].dbo.DS_RecommenderSystem_TemporaryUserData t 
	where t.gameaccount = jy.gameaccount and t.gametypesourceid = jy.CorrespondGame and t.siteid = jy.siteid)

	insert into ##result
	select x.gameaccount,x.siteid, rs.game, rs.[rank]
	FROM
	(
	select max(rns) RNS, gameaccount, siteid 
	from ##result 
	group by gameaccount, siteid
	) x JOIN
	[Resultpool].dbo.DS_RecommenderSystem_DefaultGame rs
	on rs.[rank] - x.RNS > 0
	where rs.UpDateTime = @hotgame_time


	insert into [ResultPool].[dbo].[DS_RecommenderSystem_ResultFinal]
	select siteid,
		   gameaccount,
		   Max( case when RNS = 3 then correspondgame else NULL END) [Game1],
		   Max( case when RNS = 1 then correspondgame else NULL END) [Game2],
		   Max( case when RNS = 2 then correspondgame else NULL END) [Game3],
		   Max( case when RNS = 4 then correspondgame else NULL END) [Game4],
		   Max( case when RNS = 7 then correspondgame else NULL END) [Game5],
		   Max( case when RNS = 5 then correspondgame else NULL END) [Game6],
		   Max( case when RNS = 6 then correspondgame else NULL END) [Game7],
		   Max( case when RNS = 8 then correspondgame else NULL END) [Game8],
		   Max( case when RNS = 11 then correspondgame else NULL END) [Game9],
		   Max( case when RNS = 9 then correspondgame else NULL END) [Game10],
		   Max( case when RNS = 10 then correspondgame else NULL END) [Game11],
		   Max( case when RNS = 12 then correspondgame else NULL END) [Game12],
		   @execution_time--convert(date,getdate())
	from ##result where RNS <= 12
	group by gameaccount, siteid
END
