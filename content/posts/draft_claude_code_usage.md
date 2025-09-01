---
date: '2025-08-13'
draft: true
title: 'Claude Code'
tags: []
categories: []
---
## tricks
- 内置一个router，什么需要claude code，什么不需要
```
I want to move the knowledge base sample defined in run_sample.py to a json file called "knowledge_base_sample.json" ---> NO
and create another function in data_loader.py called "load_knowledge_base_from_sample" in data_loader.py to load it  ---> NO
The sample json path should be configurable in config.py and config.yaml                                             ---> YES
Add unittest function for "load_knowledge_base_from_sample" in test_data_loader.py                                   ---> YES
```
