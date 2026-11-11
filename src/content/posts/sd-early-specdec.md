---
title: "早期 SpecDec / Seq2seq 投机执行：原理、源码与推理落地"
description: "把 seq2seq 场景的投机执行与今天通用 LLM serving 区分开。"
date: 2026-11-11
tags: ["投机解码", "基础与检索"]
draft: false
series: speculative-decoding
seriesOrder: 8
---

> **本文是源码学习初稿，不是性能复现报告。**已读取内容、未确认的原始仓库和未核实的数字均明确区分。原作者结论只在相应实验条件下引用；本站未运行 GPU 加速基准。

## 这项技术解决什么问题

用便宜的预测先行提出序列，再由高质量模型核验，以减少昂贵串行调用。它体现投机执行在 seq2seq 生成中的应用，但研究任务与今天自由采样的通用 LLM 未必一致。

## 原论文、仓库与版本

- 本轮未确认作者官方实现仓库；不把第三方或同名项目标成原始实现。
- 原论文：[arXiv:2203.16487](https://arxiv.org/abs/2203.16487)。论文全文本轮受访问限制；原文与框架实现的证据范围见下文。

[统一源码阅读索引](/articles/sd-source-map/)记录可核实的文件版本和来源。链接至框架代码不表示论文作者已经将完整原始实现发布在该框架中。

## 训练流程与数据对齐

草稿质量与输入任务强相关。本文未完成原文训练配置核验，因此不编造数据规模、模型大小或训练预算；应先确认论文中 draft／target 的监督定义。

阅读训练代码时，请记录 `input_ids`、`labels`、特征层、有效 `loss_mask` 与位置偏移。真实标签、教师分布和草稿生成 token 的来源必须分开；离线 teacher forcing 指标不能替代实际 rollout。

## 推理流程与状态提交

输入编码／上下文 → 草稿续写 → 目标核验 → 接受前缀 → 新一轮。随机采样要求另外明确 proposal 与残差校正，不能从早期名称直接推导所有现代保证。

下面是**本站自写的机制伪代码**，省略批处理、EOS、张量并行和具体 API，不能当作可直接运行的作者实现：

```python
encoded_source = encoder(source_sequence)
proposal = cheap_seq2seq_draft(encoded_source, decoder_prefix)
verified = target_decoder_verify(encoded_source, proposal)
commit_verified_prefix(verified)
update_encoder_decoder_cache_contract(verified)
```

## 代码讲解：按调用链读，而不是按文件名猜

先从论文算法与作者代码链接核实 decoder、verification、cache 更新接口。本轮尚未确认原始仓库，所以保持来源空缺明确可见，而不是推荐同名但无关项目。

一次完整阅读至少画出三份状态：候选 token、proposal 分布、KV／特征缓存。接受的 token 数、补采的校正 token 和缓存已经算到的位置不一定相同；必须以实现契约核对，不能凭长度变量名称推断。

## vLLM 与 SGLang 集成情况

当前框架经典小模型草稿可用于学习相同范式，但并非对该原论文逐行实现的证明。

上述判断基于本轮核对的主分支文档／代码，不代表任意稳定版都支持。启用前记录 target／draft revision、框架 commit、attention backend、并行模式和采样配置；[综述支持矩阵](/articles/speculative-decoding/#framework-support)提供横向对照。

## 加速收益：证据与适用条件

早期 seq2seq 吞吐收益不直接代表多用户 LLM decode 收益。本篇不提供跨年代的倍率排名。

比较时固定 target、精度、任务、输入输出长度、温度／top-p 和负载。至少记录每轮有效输出、draft／verify／KV 管理耗时、TPOT、P99 和满足 SLO 的吞吐。接受率高不必然更快。

## 正确性、失败与降级

encoder-decoder 与 decoder-only 的缓存、前缀和批处理不同，迁移时不能只替换模型路径。

“降级”优先选择保留目标采样语义的普通 target decode；不要通过放宽接受门槛伪造加速。可近似的质量策略应另行标注，不能继续称为严格无损。

## 学习实验怎么做

先把输入编码、draft、verify 三部分分开计时，再报告输出长度、batch 和采样策略。

先验证正确性，再测性能。记录冷／热状态、版本和数据划分；图表没有实验条件时不用于方法排名。未跑的实验保持“计划”，不填占位通过结果。



## 延伸阅读

- [返回综述与完整系列](/articles/speculative-decoding/)
- [技术时间轴](/articles/speculative-decoding/timeline.html)
- [原始仓库与源码阅读索引](/articles/sd-source-map/)
- 上一篇：[Blockwise Parallel Decoding](/articles/sd-blockwise/)
- 下一篇：[Medusa](/articles/sd-medusa/)
