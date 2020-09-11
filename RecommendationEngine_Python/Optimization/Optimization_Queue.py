import asyncio
import time


import os
#os.chdir('C://Users//DS.Jimmy//Desktop//Project//RecommenderSystem')
import datetime
import DS_SQLGetDF_function as func


started_at = time.monotonic()

Server_list = ['10.80.16.191', '10.80.16.192', '10.80.16.193', '10.80.16.194']
Server = Server_list[3]

#main parameter
now = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:00:00.000") #UTC+0
current = datetime.datetime.now().strftime("%Y-%m-%d %H:00:00.000") 
SQLQuery_df = func.select_from_sql(now, current, Server)
JG = func.DB('SQL Server Native Client 11.0', 'JG\MSSQLSERVER2016', 'DS.Jimmy', '4wvb%ECX')
IP = func.DB('SQL Server Native Client 11.0', Server, 'DS.Reader', '8sGb@N3m')




async def worker(name, queue):
    IP = func.DB('SQL Server Native Client 11.0', Server, 'DS.Reader', '8sGb@N3m')
    while True:
        # Get a "work item" out of the queue.
        sleep_for = await queue.get()
        print(queue.qsize())
        # Sleep for the "sleep_for" seconds.
        try:
            df = IP.ExecQuery(sleep_for)
        except:
            pass
        #print(df)
        # Notify the queue that the "work item" has been processed.
        queue.task_done()



async def main():
    # Create a queue that we will use to store our "workload".
    queue = asyncio.Queue()

    # Generate random timings and put them into the queue.
    total_sleep_time = 0
    for index in SQLQuery_df.Sqlquery:
        queue.put_nowait(index)

    # Create three worker tasks to process the queue concurrently.
    tasks = []
    for i in range(3):
        task = asyncio.create_task(worker(f'worker-{i}', queue))
        tasks.append(task)

    # Wait until the queue is fully processed.
    started_at = time.monotonic()
    await queue.join()
    total_slept_for = time.monotonic() - started_at

    # Cancel our worker tasks.
    for task in tasks:
        task.cancel()
    # Wait until all worker tasks are cancelled.
    await asyncio.gather(*tasks, return_exceptions=True)

    print('====')
    print(f'3 workers slept in parallel for {total_slept_for:.2f} seconds')
    print(f'total expected sleep time: {total_sleep_time:.2f} seconds')


asyncio.run(main())