WITH diff AS (
  SELECT 
    [gameaccount], 
    [siteid], 
    [dateplayed], 
    Lag ([dateplayed], 1, 0) OVER (
      partition BY [gameaccount], 
      [siteid] 
      ORDER BY 
        [dateplayed]
    ) AS PRE, 
    Row_number() OVER(
      partition BY [gameaccount], 
      [siteid] 
      ORDER BY 
        [dateplayed]
    ) AS Row 
  FROM 
    [DataScientist].[dbo].[ds_recommendersystemdailyquery] 
  GROUP BY 
    [gameaccount], 
    [siteid], 
    [dateplayed]
) 
select 
  [siteid], 
  avg(HourDiff) HourDiff, 
  count(1) cnt 
from 
  (
    SELECT 
      [gameaccount], 
      [siteid], 
      [dateplayed], 
      pre, 
      Datediff(hour, pre, [dateplayed]) AS HourDiff 
    FROM 
      diff 
    WHERE 
      row >= 2
  ) x 
group by 
  [siteid] 
order by 
  avg(HourDiff) desc

