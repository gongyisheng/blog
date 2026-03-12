---
date: '2026-03-12'
draft: true
title: 'Context Engineering Is the Way to Go'
tags: [LLM, Agent, Engineering]
categories: [LLM]
---

## Part 1: The Attention Problem

### How Attention Actually Works
<!-- Brief recap: softmax over all tokens, finite budget analogy -->

### The Lost-in-the-Middle Problem
<!-- Models focus on the beginning and end of context, lose focus in the middle -->
<!-- Image: attention weight distribution curve — U-shaped, strong at edges, weak in the middle -->

### Why This Makes Context Engineering Critical
<!-- Bridge: if attention is biased and finite, what you put in the context window — and where — directly determines agent quality -->

## Part 2: When Context Goes Wrong

### Unrelated Context in the Middle
<!-- Model loses focus on the actual task, quality degrades -->
<!-- Example + image -->

### Contradictory Instructions
<!-- Conflicting information causes unstable or inconsistent responses -->
<!-- Example + image -->

### Negative Instructions That Backfire
<!-- "Don't do X" paradoxically increases the chance of X — model attends to the concept -->
<!-- Example + image -->

## Part 3: Context Management Toolbox

### CLAUDE.md: Static Persistent Context
<!-- System-level vs project-level instructions, always loaded, sets baseline behavior -->
<!-- Image: diagram showing system CLAUDE.md vs project CLAUDE.md layering -->

### Skills: Dynamic On-Load / Off-Load
<!-- Context loaded on demand when a task matches, unloaded when done -->
<!-- Image: before/after context window with skill loaded vs unloaded -->

### Subagents: Divide and Conquer
<!-- Break large tasks into pieces that fit a single context window -->
<!-- Image: fan-out diagram — main agent dispatches to subagents, each with focused context -->

### Compaction: Context Handoff
<!-- When context grows too large, summarize and hand off to a fresh agent -->
<!-- Image: context window filling up → compaction → clean handoff -->

### CLI vs MCP: Precise Context Control
<!-- MCP loads 20+ tool definitions upfront, bloating every request -->
<!-- CLI allows loading only the tools needed for the current step -->
<!-- Image: side-by-side context window comparison — MCP (cluttered) vs CLI (focused) -->

## Closing Thoughts
<!-- Context engineering is not optional — it's the core skill for agent builders -->
