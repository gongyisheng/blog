+++
date = '2025-06-30T12:00:00-07:00'
draft = true
title = '一些常用prompt'
+++

1. 多问AI可行性分析，在做之前调研好所有的选项，最后再开始动手做
2. prompt之母: 多轮对话修改prompt
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
3. AI很容易被prompt中带有倾向性的预设所影响，不擅长拒绝与反驳，最好的提问方式是“是否可行”，“是否合理”，“有没有其他选项”。如果需要AI拒绝/反驳，可以加入“convince me this is a bad idea”, "what's the hidden assumption I'm making".