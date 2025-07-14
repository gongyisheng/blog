---
date: '2025-06-30'
draft: true
title: '一些常用prompt'
tags: []
categories: []
---

1. 多问AI可行性分析，在做之前调研好所有的选项，最后再开始动手做
2. prompt精修: 多轮对话修改prompt
    ```
    I want you to become my prompt engineer. Your goal is to help me craft the best possible prompt for my needs. 
    The prompt will be used by you, ChatGPT. You will follow the following process:
    1. Your first response will be to ask me what the prompt should be about. I will provide my answer, but we will 
    need to improve it through continual iterations by going through the next steps.
    2. Based on my input, you will generate 2 sections, a) Revised prompt (provide your rewritten prompt, it should 
    be clear, concise, and easily understood by you), b) Questions (ask any relevant questions pertaining to what 
    additional information is needed from me to improve the prompt).
    3. We will continue this iterative process with me providing additional information to you and you updating 
    the prompt in the Revised prompt section until I say we are done.
    ```
    (updated on July 7: 好像anthropic console里的prompt improver也很好)
3. AI很容易被prompt中带有倾向性的预设所影响，不擅长拒绝与反驳，最好的提问方式是“是否可行”，“是否合理”，“有没有其他选项”。如果需要AI拒绝/反驳，可以加入“convince me this is a bad idea”, "what's the hidden assumption I'm making".
4. Vibe coding, 有两种选择，一种是不需要长期维护的项目，那么短时间生成大量代码，面向测试/结果编程是可行的。另一种是需要长期维护的项目，那么最好能理解AI每一次的改动，牺牲速度换质量（这点claude code比较好，交互式，每次改动都需要人工确认，可以追问对代码进行解释）
5. Claude code web searching之后的结果会被truncated，建议直接复制内容丢到对话框里
6. CLAUDE.md 文件用于提供关键场外信息，尽量少且精简（必含：项目结构，关键文件和函数，workflow），权限在`.claude/settings.local.json`里管理
7. 多看论文，可以提前看到趋势，ReAct，SWE Agent，Agentic LLM都是重要的让我们看到未来的论文
8. 用https://r.jina.ai将html转化为markdown
9. 画流程图 https://dreampuf.github.io/GraphvizOnline