title R test run
cd C:\test\stagingfolder\RecommendationEngine

git checkout staging
git reset --hard AzureCloud/staging
git pull AzureCloud staging
 
DEL Reports\* /F /Q
call "C:\Program Files\R\R-3.5.1\bin\Rscript.exe" TestConfiguration.R
DEL logging\* /F /Q
call "C:\Program Files\R\R-3.5.1\bin\Rscript.exe" Introduce.R --production false

set Mth=%Date:~5,2%
set Yr=%Date:~0,4%
set Day=%Date:~8,2%
set concatt=%Yr%%Mth%%Day%
SET hr=%time:~0,2%
SET min=%time:~3,2%
SET sec=%time:~6,2%
set timen=%hr%%min%%sec%
FOR /F %%g IN ('git rev-parse HEAD') do (SET COMMIT=%%g)
set foldername=C:\Users\DS.tom\documents\Testruns\%concatt%_%timen%
SET foldername2=%foldername: =%
md "%foldername2%"
xcopy Reports "%foldername2%" /F /Q
xcopy logging "%foldername2%" /F /Q
SET a="C:\Program Files\R\R-3.5.1\bin\Rscript.exe" Additional/Integration_Staging.R --directory %foldername2% --branch staging  --commit %COMMIT%
DEL Reports\* /F /Q
DEL logging\* /F /Q
call %a%
