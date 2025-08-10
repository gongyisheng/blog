---
date: '2025-08-09'
draft: true
title: '工作三年，许多想说'
tags: ["Career Notes"]
categories: ["Career Notes"]
---

- 异步工作
    - 文字 or 会议
    - 跨时区

- 加班
- 心态
    - 争取机会
    - 在工作外的学习，在工作中实践
    - 为自己的简历打工
    - 投资自己（plantegg的知识星球，树莓派，3060显卡）

tech sharing

- 团队
    - KPI压力小的地方，人际关系一般比较松弛
    - 建立信任：持续地信守承诺

我想以更有感情的方式来记录一些东西，而不是去讲一些简单的道理。

在工作外学习，在工作中实践：
在小公司的一个比较大的问题是技术栈比较简单，使用的技术一般比较成熟，甚至老旧。带来的问题就是视野和发展受限，“在工作中学习”是一种奢望，很容易处于一种“好像没什么再可以学的”状态，但是回归到人才市场时，却发现自己一无所长。我慢慢想出了一种变通解决的方法，那就是“在工作外学习，在工作中实践”，即选定一些自己感兴趣的知识和领域，慢慢了解它、学习它，并在工作中尝试运用它。一般来说，老板对新技术的采用抱开放态度（毕竟能提升性能或者省钱，都是看得见的成果），至少能申请一些做实验的机会，如果能上到生产环境那就是一次从0到1使用的经验，即使不采纳，也可以为以后的工作做铺垫。我发现在小公司里这样的机会还真不少，最终能上到生产环境的概率很高，比如这几年我做了：
- [N] (benchmark) valkey vs redis
- [N] (benchmark) postgresql vs mysql
- [Y] (full stack) 带计费功能的openai api proxy
- [Y] (redis - client tracking) redis客户端缓存
- [Y] (redis - redis stream) 基于redis的可靠消息队列
- [Y] (mysql - replication) 基于mysql binlog的变动数据捕获 (change data capture)
- [Y] (k8s - keda) 基于keda的k8s HPA
- [N] (python - cpp extension) 用cpp写的trietree包，加速字符串匹配

tech sharing
在公司做了不少tech sharing，大概一年2-4次，分享的话题从完整的项目实现到case study都有，做过面向全公司（包含非tech听众）和面向大团队（tech，但不熟悉技术栈）以及小团队的，时长不超过一小时。也许和大公司不同，在小公司做tech sharing压力不大，只要你懂得比同事多，就可以做tech sharing。我个人把tech sharing用作两个目的，一个是作为费曼学习法的实践，将自己学到的知识输出一遍，才算真正学会了，二是督促自己学习的方式，我常常会在自己有一个想share的idea，有所学习但不够深入的时候，就约一个两周或者一个月后的tech sharing，在这段时间鼓励自己深入动手学习。tech sharing的话题选择和内容安排，需要非常了解听众。tech sharing的idea，最好的是能引发听众讨论的。我个人做过讨论最积极的一次是关于RDS疑难杂症，像这种话题，团队里每个人或多或少都有使用RDS的经验，Q&A会很积极，很有深度。像这样容易引发讨论的sharing，我甚至不需要写深奥的内容，能起到抛砖引玉的效果就可以。如果share的话题听众并不熟悉，最好从听众熟悉的工作场景出发，帮助听众意识到介绍的新技术的优缺点。这几年大概做了：

- valkey vs redis
- databricks vs AWS EMR
- prompt engineering - 如何写prompt
- RDS deep dive - mysql RDS疑难杂症
- packet capture - 网络抓包
- change data capture - 模块介绍的Q&A

时间久了，也发现tech sharing的局限性。一场tech sharing的内容，听众事后能有30%能记得已经不错，如果不是自己遇到一次或者动手实践过，从sharing上学到的知识也会很快忘记，遇到过很多次同事遇到我曾经tech sharing上分享过的问题，仍然毫无头绪。另一种情况是听众知道分享的技术/工具/模块更好，但是由于采用过程需要引入重构迁移，被认为是重要但不紧急的工作，安排在backlog，最终遗忘。所以我认为做tech sharing并不太能帮助一个团队显著提升工作效率，更多的可能是对于自己个人的意义，以及讨论时抛砖引玉的效果，作为知识的记录。

