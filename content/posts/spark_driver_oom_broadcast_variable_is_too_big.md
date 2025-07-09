---
date: '2023-03-15T12:00:00-07:00'
draft: false
title: 'Spark driver OOM: Broadcast variable is too big'
tags: ["data engineering", "incident", "software engineering", "spark", "oom"]
categories: ["spark"]
---
### Observation

1. Spark cluster master node OOM  

    Master node shutdown due to OOM. Error log:
    ```
    INFO Data stored in hdfs:///XXXX
    INFO XXXXX updated
    INFO Data has XXXXX records
    INFO Data stored in hdfs:///XXXX
    INFO XXXXX updated
    #
    # java.lang.OutOfMemoryError: Java heap space
    # -XX:OnOutOfMemoryError="kill -9 %p"
    #   Executing /bin/sh -c "kill -9 *****"...
    ```

2. Spark cluster worker node shutdown  

    Worker nodes worked well until driver shutdown. Error log:
    ```
    ERROR YarnCoarseGrainedExecutorBackend: Executor self-exiting due to : Driver ip-***-***-***-***.ec2.internal:***** disassociated! Shutting down.
    INFO MemoryStore: MemoryStore cleared
    INFO BlockManager: BlockManager stopped
    ERROR CoarseGrainedExecutorBackend: RECEIVED SIGNAL TERM
    ```

### Analysis

1. Look up driver log  

    This is a driver OOM problem. We need to figure out what caused the driver OOM.

    ```
    INFO UnifiedMemoryManager: Will not store broadcast_1159 as the required space (6290408084 bytes) exceeds our memory limit (4923378892 bytes)
    WARN MemoryStore: Not enough space to cache broadcast_1159 in memory! (computed 3.9 GiB so far)
    INFO MemoryStore: Memory use = 2.4 GiB (blocks) + 1024.0 KiB (scratch space shared across 1 tasks(s)) = 2.4 GiB. Storage limit = 4.6 GiB.
    WARN BlockManager: Persisting block broadcast_1159 to disk instead.
    ```

    The size of broadcast_1159 rdd is around 6G, however the free driver memory is only 4.58G, causing the error.

2. Look up the code  

    There’s a broadcast variable whose size is related to dafaframe row count during overwriting the out-of-date data:

    ```
    marker = update_df.select(update_df.<id>.alias('purge_key'))
    joined = base.join(sfunc.broadcast(marker), base.<id> == marker.purge_key, 'left_outer')
    purged = joined.where(joined.purge_key.isNull()).drop('purge_key')
    ```

    It’s kind of merge upsert process: Look up the id from dataframe. If id exists, update the row of data with new one. If id doesn’t exist, insert a new row of data  

### Learned

1. Be careful with the size of broadcast variable, especially when their size is growing or fluctuating. Make sure that you have monitoring for the size and it’ll not exceed the driver momory limit.

2. Merge upsert can be better supported by delta lake. Try delta lake if necessary. Demo code:

```
base.alias('base') \
    .merge(updates.alias('updates'), 'base.<key> ==updates.<key>') \
    .whenMatchedUpdateAll() \
    .whenNotMatchedInsertAll() \
    .execute()
```
