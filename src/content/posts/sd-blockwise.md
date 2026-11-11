---
title: "Blockwise Parallel Decoding：原理、源码与推理落地"
description: "回看早期多未来预测与块验证，区分其与现代随机 SD 的关系。"
date: 2026-11-11
tags: ["投机解码", "基础与检索"]
draft: false
series: speculative-decoding
seriesOrder: 7
---

> **本文是源码学习初稿，不是性能复现报告。**已读取内容、未确认的原始仓库和未核实的数字均明确区分。原作者结论只在相应实验条件下引用；本站未运行 GPU 加速基准。

## 这项技术解决什么问题

多未来位置预测可以同时提出一块 token，再通过自回归模型核验连续一致部分。这是“用额外并行工作减少串行轮次”的早期思想，不等于后来完整的无损随机拒绝采样公式。

## 原论文、仓库与版本

- 本轮未确认作者官方实现仓库；不把第三方或同名项目标成原始实现。
- 原论文：[arXiv:1811.03115](https://arxiv.org/abs/1811.03115)。论文全文本轮受访问限制；原文与框架实现的证据范围见下文。

[统一源码阅读索引](/articles/sd-source-map/)记录可核实的文件版本和来源。链接至框架代码不表示论文作者已经将完整原始实现发布在该框架中。

## 训练流程与数据对齐

该路线需要多未来位置预测能力。本文按历史背景介绍，不在未阅读全文／原始实现的情况下断言具体头结构、loss 权重和共享方式。

阅读训练代码时，请记录 `input_ids`、`labels`、特征层、有效 `loss_mask` 与位置偏移。真实标签、教师分布和草稿生成 token 的来源必须分开；离线 teacher forcing 指标不能替代实际 rollout。

## 推理流程与状态提交

构造一块未来候选 → 自回归验证前缀 → 提交一致 token → 下一块。贪心的一致性检查与随机分布校正必须分开讨论。

下面是**本站自写的机制伪代码**，省略批处理、EOS、张量并行和具体 API，不能当作可直接运行的作者实现：

```python
future_predictions = multi_offset_heads(current_state)
verified = autoregressive_target_verify(future_predictions)
consistent_length = longest_prefix_matching_target(verified)
commit(future_predictions[:consistent_length])
# Greedy consistency alone does not implement stochastic correction.
```

## 代码讲解：按调用链读，而不是按文件名猜

本轮没有确认作者官方仓库，原论文是首要入口。源码学习任务是为每个预测偏移写出 label 下标，再检查块验证何时停止；可与 Medusa 的多头代码对照，但不要将后者当作原实现。

一次完整阅读至少画出三份状态：候选 token、proposal 分布、KV／特征缓存。接受的 token 数、补采的校正 token 和缓存已经算到的位置不一定相同；必须以实现契约核对，不能凭长度变量名称推断。

## vLLM 与 SGLang 集成情况

作为历史原理背景；当前两框架支持其他多步方法，并不证明提供同名原算法。

上述判断基于本轮核对的主分支文档／代码，不代表任意稳定版都支持。启用前记录 target／draft revision、框架 commit、attention backend、并行模式和采样配置；[综述支持矩阵](/articles/speculative-decoding/#framework-support)提供横向对照。

## 加速收益：证据与适用条件

早期任务、模型和硬件与现代 LLM serving 差异大，不使用其结果与 EAGLE／DFlash 排名。本篇没有数值收益结论。

比较时固定 target、精度、任务、输入输出长度、温度／top-p 和负载。至少记录每轮有效输出、draft／verify／KV 管理耗时、TPOT、P99 和满足 SLO 的吞吐。接受率高不必然更快。

## 正确性、失败与降级

只要候选后部依赖了未验证前部，首次拒绝之后就不能继续提交。

“降级”优先选择保留目标采样语义的普通 target decode；不要通过放宽接受门槛伪造加速。可近似的质量策略应另行标注，不能继续称为严格无损。

## 学习实验怎么做

用小型自回归模型比较块预测准确率、最长一致前缀和验证次数，分别说明训练代价与推理代价。

先验证正确性，再测性能。记录冷／热状态、版本和数据划分；图表没有实验条件时不用于方法排名。未跑的实验保持“计划”，不填占位通过结果。



## 延伸阅读

- [返回综述与完整系列](/articles/speculative-decoding/)
- [技术时间轴](/articles/speculative-decoding/timeline.html)
- [原始仓库与源码阅读索引](/articles/sd-source-map/)
- 上一篇：[Token Recycling](/articles/sd-recycling/)
- 下一篇：[早期 SpecDec / Seq2seq 投机执行](/articles/sd-early-specdec/)
