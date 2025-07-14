---
date: "2023-08-08"
draft: false
title: "Spark executor OOM: not enough memory to build hash map"
tags: ["data engineering", "incident", "software engineering", "spark", "oom"]
categories: ["spark"]
---

### Background

One of the pipelines I maintained failed last week. The first try failed but the second try was successful. The pipeline’s structure is quite simple, it contains 3 steps, like most of the ETL pipelines:
1. Load data from database
2. Boardcast join a 2GiB dataset
3. Write to delta table

Error message is `There is not enough memory to build the hash map`. It happened in step 2.

### Environment Details

```
Databricks Runtime: 11.3 LTS, Scala 2.12, Spark 3.3.0
Cluster: (AWS EC2 instances)
Driver: r5.xlarge · (32GiB memory, 4 vcores)
Workers: i3en.xlarge · 4 workers (32GiB memory, 4 vcores)
Spark config (selected):
spark.driver.memory 24g
spark.driver.maxResultSize 12g
spark.executor.cores 2
spark.executor.memory 12g
spark.executor.memoryOverhead 2048
```

### Observation

Detailed error log and traceback: 

```
Caused by: org.apache.spark.SparkException: Job aborted due to stage failure: Task 312 in stage 189.0 failed 4 times, most recent failure: Lost task 312.3 in stage 189.0 (TID 2483) (10.241.80.62 executor 11): org.apache.spark.sql.execution.OutOfMemorySparkException: There is not enough memory to build the hash map

Caused by: org.apache.spark.sql.execution.OutOfMemorySparkException: There is not enough memory to build the hash map
```

Okay, the first and most important question is:  
**Where did OOM happened? Driver or Executor?**

If you’re not familiar with Spark’s architecture, it’ll be hard to answer. But it’s still possible for an engineer with basic JVM knowledge to answer this question (and even better in this way to provide more solid evidence!). We can check driver’s and executor’s stderr and stdout log respectively. If there’s a lot of GC records in stdout log, we can confirm that the it’s under memory pressure. Let’s see: 

**Driver side stderr log**

```
ERROR: [2023-08-02 13:44:48,551][*******][:91]: pipeline failed
Traceback (most recent call last):
**********************************
py4j.protocol.Py4JJavaError: An error occurred while calling o15767.saveAsTable.
......
Caused by org.apache.spark.sql.execution.OutOfMemorySparkException: There is not enough memory to build the hash map
```

**Driver side stdout log**

Unfortunately, there’re no GC records before `13:44:48`. The most recent GC record before error was raised is at `13:35:22`. Besides, there’re a lot of memory free in ParOldGen. 

**Executor side stderr log**

```
23/08/02 13:43:57 INFO MemoryStore: Block rdd_4467_312 stored as values in memory (estimated size 3.1 MiB, free 3.9 GiB)
23/08/02 13:44:04 INFO TaskMemoryManager: 2112293120 bytes of memory are used for execution and 2458893004 bytes of memory are used for storage
23/08/02 13:44:05 ERROR Executor: Exception in task 312.0 in stage 189.0 (TID 2459)
org.apache.spark.sql.execution.OutOfMemorySparkException: There is not enough memory to build the hash map
23/08/02 13:44:38 ERROR Executor: Exception in task 311.0 in stage 189.0 (TID 2458)
org.apache.spark.sql.execution.OutOfMemorySparkException: There is not enough memory to build the hash map
23/08/02 13:44:45 INFO TaskMemoryManager: 2110837364 bytes of memory are used for execution and 2461804515 bytes of memory are used for storage
23/08/02 13:44:45 ERROR Executor: Exception in task 312.3 in stage 189.0 (TID 2483)
org.apache.spark.sql.execution.OutOfMemorySparkException: There is not enough memory to build the hash map
```

Okay, We see there’re a lot of errors and same `There is not enough memory to build the hash map` error raised. And it was just a minute before pipeline fail!

**Executor side stdout log**

```
2023-08-02T13:43:12.466+0000: [GC (Allocation Failure) [PSYoungGen: 2350233K-&gt;285749K(2410496K)] 4691094K-&gt;2626610K(6597632K), 0.0487498 secs] [Times: user=0.18 sys=0.00, real=0.05 secs]
2023-08-02T13:44:01.866+0000: [GC (Allocation Failure) [PSYoungGen: 2099028K-&gt;410926K(2380288K)] 4439889K-&gt;4259124K(6567424K), 0.3241892 secs] [Times: user=0.79 sys=0.47, real=0.32 secs]
2023-08-02T13:44:02.190+0000: [Full GC (Ergonomics) [PSYoungGen: 410926K-&gt;0K(2380288K)] [ParOldGen: 3848197K-&gt;3860421K(5868544K)] 4259124K-&gt;3860421K(8248832K), [Metaspace: 99036K-&gt;96953K(1157120K)], 0.4974180 secs] [Times: user=1.55 sys=0.04, real=0.50 secs]
2023-08-02T13:46:39.501+0000: [GC (Allocation Failure) [PSYoungGen: 1969152K-&gt;85311K(2463232K)] 5829573K-&gt;3945740K(8331776K), 0.0275780 secs] [Times: user=0.09 sys=0.00, real=0.03 secs]

```

There’re some GC records in executor side, right before   `13:44:48`, not enough memory left in both PSYoungGen and ParOldGen. We can confirm that the executor is under memory pressure.

### Root Cause

The root cause is that Executor Side Broadcast join (EBJ) is introduced in Databricks Runtime 11.3 TLS. For broadcast join, the memory pressure is moved from driver to executor. This post explains it very well: [Job fails with “not enough memory to build the hash map” error](https://kb.databricks.com/en_US/python/job-fails-with-not-enough-memory-to-build-the-hash-map-error#:~:text=Job%20fails%20with%20%60%60not%20enough%20memory%20to%20build%20the%20hash%20map%27%27%20error,-You%20should%20use%20adaptive%20query)

However, the post doesn’t explicitly suggest that we need to add more memory to executor, which may be confused for beginners. And for engineers who are familiar with Driver Side Broadcast Join before Databricks Runtime 11.3 TLS (eg, me, and my colleague), this post may be confusing. To confirm that it’s an executor memory issue, we’d better double check the stdout and GC log.

We fixed it by using i3en.3xlarge (96 GiB memory, 12 cores), seting spark.executor.cores to 3 and seting spark.executor.memory to 20g. It runs smoothly without issues after that.

### Learned
1. Use stdout/stderr to position where OOM happened

### Reference
- [databricks article](https://kb.databricks.com/en_US/python/job-fails-with-not-enough-memory-to-build-the-hash-map-error#:~:text=Job%20fails%20with%20%60%60not%20enough%20memory%20to%20build%20the%20hash%20map%27%27%20error,-You%20should%20use%20adaptive%20query)
