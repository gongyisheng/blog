+++
date = '2023-02-10T12:00:00-07:00'
draft = false
title = 'MySQL connection deadlock'
+++
### Observation

#### CronJob is taking more than 1h to complete

My colleagues told me that one of the cron job stuck in the middle after a random day. They received the warning: CronJob is taking more than 1h to complete. And the pod kept stucking there after a day, which is abnormal. However, another cron job which almost uses the same code works well. No database failure was reported during the period of time.

### Analysis

1. Which line of code stucks the cronjob  

    My colleague told me that she successfully found out that following code stuck the cronjob with pdb, but can’t tell why:

    ```
    async def fetch(db_pool, sql, params=None):
        async with db_pool.acquire() as conn:
            **async with conn.cursor(aiomysql.DictCursor) as cursor:**
                await cursor.execute(sql, params)
                data = await cursor.fetchall()
                return data
    ```

    This line of code was called to run a sql query from database. It tried to get a connection and used the cursor to execute the sql query. The query looks good, unlikely to cause errors:

    ```
    SELECT * FROM user WHERE user_id = XXX
    ```

2. Code change before/after the day  

    No obvious code change was found. But thay day was a deployment day and all of the changes we made this month was deployed at that day. My colleague told me that she found this line of code was added this month by another colleague in the infra team to fix a mysql connection concurrency error. Is that the code change cause the stuck?

    ```
    async with db_pool.acquire() as conn
    ```

3. Smoke test of sql query  

    To check that it’s not the code change cause the stuck, I wrote a one-line smoke test script and ran it on my laptop to execute the same sql qeury. The result was returned as expected. Although there was code change, it was not the reason to cause the cronjob to be stuck.

4. Add log and reproduce the issue  

    The issue can be reproduced, which is good. But I don’t know whether it was stuck at getting connection or cursor initialization. I was not quite familiar with pdb. Thus I decided to add some logs to see what really happened before being stuck. What we had already know was that everything was working because no error was thrown out, the job was simply stuck there.

    ```
    async def fetch(db_pool, sql, params=None):
        print("start fetch, sql=[%s]" % sql)
        async with db_pool.acquire() as conn:
            print("acquire connection")
            async with conn.cursor(aiomysql.DictCursor) as cursor:
                print("init cursor")
                await cursor.execute(sql, params)
                print("execute sql")
                data = await cursor.fetchall()
                print("fetch data")
                return data
    ```

    Result:

    ```
    INFO: Start processing XXX
    INFO: Making new db conn pool [host: ***, mid: ***, port: ***, maxsize: 1]
    ###Query 1### INFO: start fetch, sql=[select * from `connection` where `id` = XXX]
    INFO: acquire connection
    INFO: init cursor
    INFO: execute sql
    INFO: fetch data
    INFO: Another start processing XXX
    ###Query 2### INFO: start fetch, sql=[SELECT * FROM `user` WHERE `user_id`=XXXX]
    ```

    Here I noticed that the code was stuck at getting connection from connection pool. And I also noticed that the connection pool size was only 1. It’s a high-confidence evidence that there’s something like deadlock happens. As what we had known, the query can be executed as expected in smoke test, and no error was thrown out by the stuck cronjob, which means the connection is good. The only reason that fits the phenomenon is the connection deadlock. I need to search for root cause in the code carefully.

### Root Cause

1. Connection pool size too small caused deadlock  
    Following code requires at least 2 connections to work well. It caused deadlock when the connection size was only 1:

    ```
    itemiter = Global.model.fetch_iter(sql1)
    async for item in itemiter:
        Global.model.fetch(sql2)
    ```

    The first query is executed by an iterator, which means that the connection will not be returned until for loop completes. The second query is executed inside the for loop. In this case it requests for one more connection from the pool to execute the query. If the connection pool size is only 1, the code will run into deadlock.

2. Before/After deployment day  

    Before the deployment day there was no async with db_pool.acquire() as conn step. A connection can be used concurrently, which is a bug. However, it made the deadlock code work well. After the deployment, the bug was fixed and the code can no longer use one connection concurrently, which caused the deadlock.

3. Why the other cronjob used the same code worked well?  

    After reading the code, I found out that the other cron job created a duplicated db connection local variable inside the main function! It’s a bad practice but surprisingly makes the cronjob escape from deadlock. However, since we know that it was caused by a small-sized connection pool. We can remove the duplicated db connection local variable now. It’s bad practice because db connection local variable may cause db connection to be created and destroyed for many times, which hurts the performance.

### Learned

1. A small-sized connection pool (such as 1, 2) may cause deadlock. Count the minimum number of connection your code need to use concurrently and set it as the right connection pool size.  

2. Create db connection as singleton global variable, not local variable.
