
setClass('connect_to_MSSQL', representation(dbhandle= 'RODBC'))
setClass('connect_to_testMSSQL',contains='connect_to_MSSQL',representation(dbhandle='RODBC'))
setClass('connect_to_productionMSSQL',contains='connect_to_MSSQL',representation(dbhandle='RODBC'))


setMethod('initialize','connect_to_testMSSQL',function(.Object, DB='JG\\MSSQLSERVER2016', User = 'DS.Tom' )
{
  Password <- keyring::keyget('JG',username=User)	
  dbhandle <- odbcDriverConnect(paste('driver={SQL Server};server=',DB,';uid=',User,';pwd=',Password,';',sep=''))
  .Object@dbhandle = dbhandle
  return(.Object)  
}
)

setMethod('initialize','connect_to_productionMSSQL', function(.Object, DB='JG\\MSSQLSERVER2016', User = 'DS.Tom')
{
  Password <- keyring::keyget('JG',username=User)	
  dbhandle <- odbcDriverConnect(paste('driver={SQL Server};server=',DB,';uid=',User,';pwd=',Password,';',sep=''))
  .Object@dbhandle = dbhandle
  return(.Object)
}
)


setGeneric('getHandle', function(.Object){return('RODBC')})
setMethod('getHandle','connect_to_MSSQL',function(.Object){
  return(.Object@dbhandle)
})