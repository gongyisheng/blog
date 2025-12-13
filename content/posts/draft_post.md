---
date: '2025-12-12'
draft: true
title: 'Memorization vs Generalization'
tags: [learning, reasoning]
categories: [thoughts]
---

Today I revisit the classic LRU cache problem in my [playground](https://github.com/gongyisheng/playground/blob/main/algorithm/lru.py) repo. The standard answer comes to my mind immediately: HashMap + Doubly Linked List. Cool, I remember that. But then I ask myself, what about LFU, B+Tree, and other complex data structures? I can barely remember and have to look up the script I developed when I'm at school. 

Yeah, the real problem is that I'm memorizing the answer, not generalization. I suddenly realized that it's also where LLMs suffer from: they are so good at memorizing answer, but not able to generalize.

So I ask claude following questions:
```
To implement the LRU cache the optimal way is to use hashmap + doubly linked list, why use this combo? Why other combos are inefficient than this one? How would you derive it?
```

Here's claude's answer (claude-4.5-opus)
```
Great question. This is about working backwards from operation requirements to data structure choice.

  Step 1: List the Operations You Need

  For LRU cache, you need:
  1. Get by key → must be O(1)
  2. Move item to front → must be O(1)
  3. Evict from back → must be O(1)
  4. Delete arbitrary item (when updating existing key) → must be O(1)

  Step 2: Analyze Each Data Structure

  | Data Structure     | Get by key | Move to front | Evict back  | Delete arbitrary |
  |--------------------|------------|---------------|-------------|------------------|
  | Array              | O(n)       | O(n)          | O(1)        | O(n)             |
  | Singly Linked List | O(n)       | O(n)*         | O(n)**      | O(n)             |
  | Doubly Linked List | O(n)       | O(1)          | O(1)        | O(1)             |
  | HashMap            | O(1)       | ❌ no order   | ❌ no order | O(1)             |
  | HashMap + DLL      | O(1)       | O(1)          | O(1)        | O(1)             |

  *Singly linked: even if you have the node, you can't update prev.next without traversing
  **Singly linked: need to traverse to find second-to-last node

  Step 3: The Key Insight

  No single data structure gives O(1) for all operations. So you combine:

  - HashMap solves: "I need O(1) access by key"
  - Doubly Linked List solves: "I need O(1) positional operations (move/remove/insert)"

  The trick: HashMap stores pointers to DLL nodes, not values directly.

  HashMap                  Doubly Linked List
  ┌─────────────┐
  │ key1 → ─────┼────────► [Node1] <─────> [Node2] <─────> [Node3]
  │ key2 → ─────┼────────────────────────────┘               │
  │ key3 → ─────┼────────────────────────────────────────────┘
  └─────────────┘

  Why Not Other Combinations?

  HashMap + Array?
  - Moving to front = O(n) shift
  - Delete from middle = O(n) shift

  HashMap + Singly Linked List?
  - To remove a node, you need its previous node
  - Singly linked only has next, so you'd traverse O(n) to find prev

  Doubly Linked List alone?
  - Finding a key = O(n) traversal

  The Design Pattern

  When you see these requirements, think:

  | Requirement                | Data Structure     |
  |----------------------------|--------------------|
  | O(1) lookup by key         | HashMap            |
  | O(1) ordered insert/remove | Doubly Linked List |
  | O(1) min/max access        | Heap               |
  | O(1) both ends access      | Deque              |
  | O(log n) range queries     | BST / Skip List    |

  LRU needs both O(1) lookup AND O(1) ordered manipulation → HashMap + DLL

  The General Principle

  When one data structure can't satisfy all O(1) requirements, combine two that complement each other. Use one as the "index" into the other.

  This same pattern appears in:
  - LFU Cache: HashMap + HashMap + DLL (frequency buckets)
  - Database indexes: B-Tree + row storage
  - Redis sorted sets: HashMap + Skip List
```

This is basically what we want for generalization, the right reasoning trajectory in the middle of question and answer. Human also benefits a lot from theses because

So in the R1-like RL training, when we are only providing Q + A and apply GRPO training, we are trying to the training on Q + A → learning the trajectory between them. The trajectory is the real prize.

Same thing everywhere honestly:
- Math: deriving formulas vs memorizing them
- Chess: principles vs opening lines
- Coding: understanding the fix vs copy-pasting from SO

The goal isn't collecting answers. It's collecting reasoning patterns.

If you can't explain *why* the solution works, you only memorized the conclusion.