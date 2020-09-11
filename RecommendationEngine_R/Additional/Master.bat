title R test run
cd C:\Test\DS_recommend

DEL logging\* /F /Q
call "C:\Program Files\R\R-3.5.1\bin\Rscript.exe" Introduce.R 

set Mth=%Date:~5,2%
set Yr=%Date:~0,4%
set Day=%Date:~8,2%
set concatt=%Yr%%Mth%%Day%
SET hr=%time:~0,2%
SET min=%time:~3,2%
SET sec=%time:~6,2%
set timen=%hr%%min%%sec%
FOR /F %%g IN ('git rev-parse HEAD') do (SET COMMIT=%%g)
set foldername=C:\Test\DS_recommend_log\%concatt%_%timen%
SET foldername2=%foldername: =%
md "%foldername2%"
xcopy Reports "%foldername2%" /F /Q
xcopy logging "%foldername2%" /F /Q
SET a="C:\Program Files\R\R-3.5.1\bin\Rscript.exe" Additional/Integration.R --directory %foldername2% --branch staging  --commit %COMMIT%
DEL Reports\* /F /Q
DEL logging\* /F /Q
call %a%