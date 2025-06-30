+++
date = '2023-01-13T12:00:00-07:00'
draft = false
title = 'Why signal file is a bad idea'
+++
### Observation

Signal file is widely used in Hadoop ecosystem. If you have experience with MapReduce, you’ll notice that by default MapReduce runtime writes an empty _SUCCESS file to mark successful completion of a job to the output folder. AWS DataPipeline and Databricks also support “file arrival” to trigger a downstream job.

### Question

Is signal file a good architecture design?

Can I use _SUCCESS created by MapReduce as signal file to trigger downstream job?

### Discussion

Generally speaking, signal file is a bad design comparing with direct api call.

1. Signal file introduce unnecessary dependency to the system.  
    - signal file: Job A -> signal file -> Job B
    - api call: Job A -> Job B
    Signal file introduces one more unnecessary dependency for the system and hurts overall maintainability. Signal file is perferred only when Job A can’t monitor when writing files to disk is complete.

2. Signal file is less standardized, its file format may change  
    To make things worse, you put some metadata inside signal file and downstream job has to implement an API to read data from it. This architecture design will be a pain if you have system migration (eg. python2 to python3). You have to solve API compatibility issues every time there’re infrastructure changes.

3. Signal file may be moved, renamed, or deleted  
    As long as user has the access to read/write signal file, its state can be mischanged by human errors.

4. Signal file may be created by a different process  
    There’ll be race conditions if multiple services can read and write signal files at the same time.

### Action

In January 2022, after our data pipeline migration from AWS EMR to Databricks, we eventually decided to replace ALL of the signal file architecture design with databricks api call to trigger downstream jobs. What I learned is that signal file is not a good idea for triggering another job comparing with direct api call. 