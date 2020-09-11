  /*190 DS_RecommenderSystem_CosineSimilarity */
  create table [DataScientist].dbo.[DS_RecommenderSystem_CosineSimilarity]
  ( 
	rns int IDENTITY(1,1), 
	game int,
	correspondgame int,
	cosinesimilarity float,
	updatetime datetime
  )

 create table DS_RecommenderSystem_TemporaryUserData ( indx int not NULL IDENTITY(1,1) PRIMARY KEY,
					 gameaccount nvarchar(20),
					 siteid int,
					 gametypesourceid int,
					 last_executed datetime 
                    )

