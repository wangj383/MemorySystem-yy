Memory System

## 核心功能
- sentence embedding
    - function: 把输入的str 做embedding，存起来
    - 输入：str
    - 输出：None
- query
    - function：基于cos-sim 来做搜索
    - 输入
        - str: 问题
        - top-k: 返回多少条
    - 输出: list(str)

## 实现步骤
### 设计
- 1. 用什么工具/package (embedding 模型的接入 - OpenAI SDK/本地部署)
    - 存储 - list(vector)
- 2. Class MemorySystem
    - 1. insert(str): embedding
    - 2. query(str, int)