---
title: "Sequoia：原理、源码与推理落地"
description: "用接受率和硬件成本决定候选树形状。"
date: 2026-11-11
tags: ["投机解码", "训练与系统优化"]
draft: false
series: speculative-decoding
seriesOrder: 27
---

> **本文是源码学习初稿，不是性能复现报告。**已读取内容、未确认的原始仓库和未核实的数字均明确区分。原作者结论只在相应实验条件下引用；本站未运行 GPU 加速基准。

## 这项技术解决什么问题

候选树不只是模型概率问题。相同节点数、不同深宽结构在不同硬件和 target 上成本不同；Sequoia 用接受统计与成本信息搜索更适合硬件的树形配置。

## 原论文、仓库与版本

- 代码入口：[Infini-AI-Lab/Sequoia](https://github.com/Infini-AI-Lab/Sequoia)。以原项目版本与说明为准。
- 原论文：[arXiv:2402.12374](https://arxiv.org/abs/2402.12374)。论文全文本轮受访问限制；原文与框架实现的证据范围见下文。
- 已读取源码／文档：[`tree_search.py`](https://github.com/Infini-AI-Lab/Sequoia/blob/0ff3ff71475ca148f2dfb7545e0de5caf5c67b39/tree_search.py)，提交 `0ff3ff7147`。

[统一源码阅读索引](/articles/sd-source-map/)记录可核实的文件版本和来源。链接至框架代码不表示论文作者已经将完整原始实现发布在该框架中。

## 训练流程与数据对齐

无需把整套方法归为训练一个新 LM；可以使用现成 draft，并采集 acceptance rate vector 作为树搜索输入。校准数据和模型／温度变化会影响该向量。

阅读训练代码时，请记录 `input_ids`、`labels`、特征层、有效 `loss_mask` 与位置偏移。真实标签、教师分布和草稿生成 token 的来源必须分开；离线 teacher forcing 指标不能替代实际 rollout。

## 推理流程与状态提交

离线统计接受率与硬件 profile → 搜索 growmap → 在线按结构展开草稿 → target 树验证 → 提交合法路径。静态硬件感知树与 EAGLE-2 动态上下文树优化的是不同维度。

下面是**本站自写的机制伪代码**，省略批处理、EOS、张量并行和具体 API，不能当作可直接运行的作者实现：

```python
acceptance_vector = calibrate_draft_target_pair(data)
latency_model = profile_target_verification(hardware)
growmap = search_tree_shape(acceptance_vector, latency_model, budget)
proposal_tree = draft_following(growmap)
verified = target_tree_verify(proposal_tree)
commit_valid_path(verified)
```

## 代码讲解：按调用链读，而不是按文件名猜

作者 tree_search.py 是树结构搜索入口；README 的 test_accept.py 估计接受率，A100_growmaps／L40_growmaps 区分硬件配置。不要直接拿别的硬件 growmap 证明当前环境最优。

一次完整阅读至少画出三份状态：候选 token、proposal 分布、KV／特征缓存。接受的 token 数、补采的校正 token 和缓存已经算到的位置不一定相同；必须以实现契约核对，不能凭长度变量名称推断。

## vLLM 与 SGLang 集成情况

作者独立实现，README 推荐后续 UMbreLLa；本轮未确认最新两框架提供 Sequoia 同名默认配置。其思想可用于理解树预算。

上述判断基于本轮核对的主分支文档／代码，不代表任意稳定版都支持。启用前记录 target／draft revision、框架 commit、attention backend、并行模式和采样配置；[综述支持矩阵](/articles/speculative-decoding/#framework-support)提供横向对照。

## 加速收益：证据与适用条件

作者提供 A100／L40 与 offloading 运行入口，但本文未运行，不从预置 growmap 推出固定加速比。应在相同模型及负载比较固定链、规则树与搜索树。

比较时固定 target、精度、任务、输入输出长度、温度／top-p 和负载。至少记录每轮有效输出、draft／verify／KV 管理耗时、TPOT、P99 和满足 SLO 的吞吐。接受率高不必然更快。

## 正确性、失败与降级

旧校准向量过期会产生错误成本决策；目标精度或后端变更后要重新 profile。

“降级”优先选择保留目标采样语义的普通 target decode；不要通过放宽接受门槛伪造加速。可近似的质量策略应另行标注，不能继续称为严格无损。

## 学习实验怎么做

固定节点数和显存预算，比较 A100/L40 growmap 在当前硬件的适用性，并计入 profile 与树搜索成本。

先验证正确性，再测性能。记录冷／热状态、版本和数据划分；图表没有实验条件时不用于方法排名。未跑的实验保持“计划”，不填占位通过结果。



## 延伸阅读

- [返回综述与完整系列](/articles/speculative-decoding/)
- [技术时间轴](/articles/speculative-decoding/timeline.html)
- [原始仓库与源码阅读索引](/articles/sd-source-map/)
- 上一篇：[LiLiCorr](/articles/sd-lilicorr/)
- 下一篇：[SpecExec](/articles/sd-specexec/)
