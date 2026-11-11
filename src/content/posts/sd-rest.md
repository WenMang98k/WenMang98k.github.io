---
title: "REST：原理、源码与推理落地"
description: "把检索来的续写构造成候选树，让目标模型验证。"
date: 2026-11-11
tags: ["投机解码", "基础与检索"]
draft: false
series: speculative-decoding
seriesOrder: 4
---

> **本文是源码学习初稿，不是性能复现报告。**已读取内容、未确认的原始仓库和未核实的数字均明确区分。原作者结论只在相应实验条件下引用；本站未运行 GPU 加速基准。

## 这项技术解决什么问题

使用当前上下文的最长后缀检索 datastore；把检索文档的后续 token 组织为 Trie，剪去低频分支，再送入 target 的 tree attention 验证。检索负责便宜地提出多个续写，目标模型负责决定真正的输出。

## 原论文、仓库与版本

- 代码入口：[FasterDecoding/REST](https://github.com/FasterDecoding/REST)。以原项目版本与说明为准。
- 原论文：[arXiv:2311.08252](https://arxiv.org/abs/2311.08252)。论文全文本轮受访问限制；原文与框架实现的证据范围见下文。
- 已读取源码／文档：[`rest/model/rest_model.py`](https://github.com/FasterDecoding/REST/blob/154d0f0ae45de5d8a9cbd1c696f6ad5da10904bf/rest/model/rest_model.py)，提交 `154d0f0ae4`。

[统一源码阅读索引](/articles/sd-source-map/)记录可核实的文件版本和来源。链接至框架代码不表示论文作者已经将完整原始实现发布在该框架中。

## 训练流程与数据对齐

无需训练新的草稿 LM，但需要为目标 tokenizer 构建 datastore。构库数据、词表 ID 宽度、检索线程数和索引规模都影响部署成本。

阅读训练代码时，请记录 `input_ids`、`labels`、特征层、有效 `loss_mask` 与位置偏移。真实标签、教师分布和草稿生成 token 的来源必须分开；离线 teacher forcing 指标不能替代实际 rollout。

## 推理流程与状态提交

查询 datastore → 合并续写到 Trie → 剪枝 → 构造树掩码 → 验证合法路径 → 提交前缀并更新缓存。随机采样的具体树验证不能只用贪心匹配伪装成无损 sampler。

下面是**本站自写的机制伪代码**，省略批处理、EOS、张量并行和具体 API，不能当作可直接运行的作者实现：

```python
retrieved = datastore.lookup_longest_suffix(prefix)
trie = merge_continuations(retrieved)
trie = prune_low_frequency_branches(trie, node_budget)
mask, parent_ids = build_ancestor_only_attention(trie)
result = target_tree_verify(prefix, trie, mask, parent_ids)
commit_one_valid_path(result)
```

## 代码讲解：按调用链读，而不是按文件名猜

阅读 RestModel.rest_generate 与 baseline_generate，对比多出来的检索与候选整理阶段。继续检查 tokenizer 与 datastore 编码是否一致；作者 README 特别讨论过大词表 token ID 存储宽度。

一次完整阅读至少画出三份状态：候选 token、proposal 分布、KV／特征缓存。接受的 token 数、补采的校正 token 和缓存已经算到的位置不一定相同；必须以实现契约核对，不能凭长度变量名称推断。

## vLLM 与 SGLang 集成情况

属于独立作者实现。可在主框架的检索草稿机制上研究相同思想，但本轮未确认两个框架存在名为 REST 的当前默认入口。

上述判断基于本轮核对的主分支文档／代码，不代表任意稳定版都支持。启用前记录 target／draft revision、框架 commit、attention backend、并行模式和采样配置；[综述支持矩阵](/articles/speculative-decoding/#framework-support)提供横向对照。

## 加速收益：证据与适用条件

作者 README 展示 HumanEval 与 MT-Bench，在单张 NVIDIA A6000、96 CPU cores、batch=1 上评测，并给出不同温度／top-p。本文不从图片估算倍数，也不把该设置外推到高并发。

比较时固定 target、精度、任务、输入输出长度、温度／top-p 和负载。至少记录每轮有效输出、draft／verify／KV 管理耗时、TPOT、P99 和满足 SLO 的吞吐。接受率高不必然更快。

## 正确性、失败与降级

检索 CPU 开销、索引内存和候选树宽度可能抵消收益；支持任意模型的概念不等于代码对任意架构开箱可用。

“降级”优先选择保留目标采样语义的普通 target decode；不要通过放宽接受门槛伪造加速。可近似的质量策略应另行标注，不能继续称为严格无损。

## 学习实验怎么做

先固定小 datastore，再逐步扩大；比较检索延迟、Trie 节点数、接受长度与无检索 target 基线。

先验证正确性，再测性能。记录冷／热状态、版本和数据划分；图表没有实验条件时不用于方法排名。未跑的实验保持“计划”，不填占位通过结果。



## 延伸阅读

- [返回综述与完整系列](/articles/speculative-decoding/)
- [技术时间轴](/articles/speculative-decoding/timeline.html)
- [原始仓库与源码阅读索引](/articles/sd-source-map/)
- 上一篇：[Suffix Decoding](/articles/sd-suffix/)
- 下一篇：[Lookahead Decoding](/articles/sd-lookahead/)
