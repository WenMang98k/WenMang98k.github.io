---
title: "Token Recycling：原理、源码与推理落地"
description: "把已有计算的候选信息再利用，区别于只匹配文本后缀。"
date: 2026-11-11
tags: ["投机解码", "基础与检索"]
draft: false
series: speculative-decoding
seriesOrder: 6
---

> **本文是源码学习初稿，不是性能复现报告。**已读取内容、未确认的原始仓库和未核实的数字均明确区分。原作者结论只在相应实验条件下引用；本站未运行 GPU 加速基准。

## 这项技术解决什么问题

利用此前 target 计算得到的候选关联构造新的续写，而不另外运行一套完整草稿 LM。与 prompt lookup 的主要区别是复用的对象包含模型计算带来的候选信息，而不只是历史原文片段。

## 原论文、仓库与版本

- 本轮未确认作者官方实现仓库；不把第三方或同名项目标成原始实现。
- 原论文：[arXiv:2408.08696](https://arxiv.org/abs/2408.08696)。论文全文本轮受访问限制；原文与框架实现的证据范围见下文。

[统一源码阅读索引](/articles/sd-source-map/)记录可核实的文件版本和来源。链接至框架代码不表示论文作者已经将完整原始实现发布在该框架中。

## 训练流程与数据对齐

通常无需额外模型训练，但要维护候选关联／缓存结构。记录在什么时候创建、更新与清空；跨请求热缓存可能让评估与冷启动差异很大。

阅读训练代码时，请记录 `input_ids`、`labels`、特征层、有效 `loss_mask` 与位置偏移。真实标签、教师分布和草稿生成 token 的来源必须分开；离线 teacher forcing 指标不能替代实际 rollout。

## 推理流程与状态提交

读取候选关联 → 构造后续 token 树／链 → target 验证 → 用本轮计算更新缓存 → 提交接受前缀。关联表命中并不说明候选在当前语境成立。

下面是**本站自写的机制伪代码**，省略批处理、EOS、张量并行和具体 API，不能当作可直接运行的作者实现：

```python
candidate_links = recycled_candidates.lookup(prefix[-1])
tree = expand_candidate_links(candidate_links, budget)
verified, fresh_logits = target_verify(tree)
recycled_candidates.update_from_model_outputs(fresh_logits)
commit_verified_prefix(verified)
# Cache reset policy changes cold-start vs warm-cache measurements.
```

## 代码讲解：按调用链读，而不是按文件名猜

本轮实际读取的是 smpanaro/token-recycling，README 明确标 unofficial。其 adjacency matrix 与 cold-start 条件是值得阅读的线索，但不是作者官方源码审计。原仓库尚未核实，不能直接搬入标成作者实现。

一次完整阅读至少画出三份状态：候选 token、proposal 分布、KV／特征缓存。接受的 token 数、补采的校正 token 和缓存已经算到的位置不一定相同；必须以实现契约核对，不能凭长度变量名称推断。

## vLLM 与 SGLang 集成情况

当前默认两框架入口未核实；不要把支持 n-gram 推断成支持相同候选关联机制。

上述判断基于本轮核对的主分支文档／代码，不代表任意稳定版都支持。启用前记录 target／draft revision、框架 commit、attention backend、并行模式和采样配置；[综述支持矩阵](/articles/speculative-decoding/#framework-support)提供横向对照。

## 加速收益：证据与适用条件

该第三方 README 报告 Vicuna-7B-v1.3、单 A100 40GB、30 CPU cores、FP16、贪心、batch=1、单次运行约 2.08×；这是第三方特定设置，不是原作者或本站复现，也不直接与不同测试次数的榜单比较。

比较时固定 target、精度、任务、输入输出长度、温度／top-p 和负载。至少记录每轮有效输出、draft／verify／KV 管理耗时、TPOT、P99 和满足 SLO 的吞吐。接受率高不必然更快。

## 正确性、失败与降级

热缓存与冷缓存混用会夸大收益；关联信息失效时应裁剪或清空，跨请求复用需明确隔离。

“降级”优先选择保留目标采样语义的普通 target decode；不要通过放宽接受门槛伪造加速。可近似的质量策略应另行标注，不能继续称为严格无损。

## 学习实验怎么做

分别记录冷启动／热缓存，报告关联表大小、更新成本、平均接受长度及完整 latency。

先验证正确性，再测性能。记录冷／热状态、版本和数据划分；图表没有实验条件时不用于方法排名。未跑的实验保持“计划”，不填占位通过结果。

第三方阅读入口：[smpanaro/token-recycling](https://github.com/smpanaro/token-recycling)。该链接明确标注为非官方。

## 延伸阅读

- [返回综述与完整系列](/articles/speculative-decoding/)
- [技术时间轴](/articles/speculative-decoding/timeline.html)
- [原始仓库与源码阅读索引](/articles/sd-source-map/)
- 上一篇：[Lookahead Decoding](/articles/sd-lookahead/)
- 下一篇：[Blockwise Parallel Decoding](/articles/sd-blockwise/)
