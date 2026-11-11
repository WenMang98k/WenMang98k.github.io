---
title: "LiLiCorr：原理、源码与推理落地"
description: "用小型 correlator 对各位置候选的组合评分。"
date: 2026-11-11
tags: ["投机解码", "块并行草稿"]
draft: false
series: speculative-decoding
seriesOrder: 26
---

> **本文是源码学习初稿，不是性能复现报告。**已读取内容、未确认的原始仓库和未核实的数字均明确区分。原作者结论只在相应实验条件下引用；本站未运行 GPU 加速基准。

## 这项技术解决什么问题

独立位置边缘分布都合理，并不保证拼起来合理。LiLiCorr 保留每槽 top-k 候选，用小型 Transformer correlator 关联整块候选，选择或采样更连贯的路径。目标验证机制仍负责正确性。

## 原论文、仓库与版本

- 代码入口：[sgl-project/sglang](https://github.com/sgl-project/sglang)。这是已核实的训练／服务框架实现，不冒充独立论文作者的原始仓库。
- 原论文：[arXiv:2608.20530](https://arxiv.org/abs/2608.20530)。论文全文本轮受访问限制；原文与框架实现的证据范围见下文。
- 已读取源码／文档：[`docs/docs/advanced_features/speculative_decoding.mdx`](https://github.com/sgl-project/sglang/blob/34dfaccc57d928f046317c2e3d9e16bd69837d9b/docs/docs/advanced_features/speculative_decoding.mdx)，提交 `34dfaccc57`。
- 已读取源码／文档：[`vllm/config/speculative.py`](https://github.com/vllm-project/vllm/blob/2ee8250eaf67059b4e8bc3c1bfb4cfb1c88892a7/vllm/config/speculative.py)，提交 `2ee8250eaf`。

[统一源码阅读索引](/articles/sd-source-map/)记录可核实的文件版本和来源。链接至框架代码不表示论文作者已经将完整原始实现发布在该框架中。

## 训练流程与数据对齐

需要兼容的已训练 correlator checkpoint；结构与草稿主干配置必须完整。当前所读 SGLang 文档注明尚无已发布 LiLiCorr 草稿，不把占位路径当成可下载权重。

阅读训练代码时，请记录 `input_ids`、`labels`、特征层、有效 `loss_mask` 与位置偏移。真实标签、教师分布和草稿生成 token 的来源必须分开；离线 teacher forcing 指标不能替代实际 rollout。

## 推理流程与状态提交

DFlash backbone → unary 候选格 → correlator 条件路径评分 → 贪心／概率提案 → 目标验证。概率提案要把实际使用的分布提交 rejection sampler。

下面是**本站自写的机制伪代码**，省略批处理、EOS、张量并行和具体 API，不能当作可直接运行的作者实现：

```python
hidden = dflash_backbone(context, slots)
lattice = unary_topk_candidates(shared_lm_head(hidden))
conditional_scores = small_correlator(lattice, hidden)
proposal, actual_q = walk_candidate_path(conditional_scores, temperature)
verified = target_verify(proposal, actual_q)
commit_verified_prefix(verified)
```

## 代码讲解：按调用链读，而不是按文件名猜

从 SGLang 指南 LiLiCorr 段落确认 architecture=LiLiCorrDraftModel 与 lilicorr_* 配置，再阅读 loader。vLLM 专门指南区分 draft_sample_method=greedy／probabilistic；检查配置缺失是否报错，避免有头权重却按普通 DFlash 加载。

一次完整阅读至少画出三份状态：候选 token、proposal 分布、KV／特征缓存。接受的 token 数、补采的校正 token 和缓存已经算到的位置不一定相同；必须以实现契约核对，不能凭长度变量名称推断。

## vLLM 与 SGLang 集成情况

两框架有相关文档，沿 dflash／DFLASH 路径，非独立算法枚举。所读 vLLM 文档明确与 adaptive verification 组合尚未端到端验证。

上述判断基于本轮核对的主分支文档／代码，不代表任意稳定版都支持。启用前记录 target／draft revision、框架 commit、attention backend、并行模式和采样配置；[综述支持矩阵](/articles/speculative-decoding/#framework-support)提供横向对照。

## 加速收益：证据与适用条件

有效性应拆成候选覆盖、条件路径质量和 correlator 额外耗时。文档提醒 CPU seq_lens 同步及 eager fallback 可能吞掉吞吐；本轮没有 GPU 倍数。

比较时固定 target、精度、任务、输入输出长度、温度／top-p 和负载。至少记录每轮有效输出、draft／verify／KV 管理耗时、TPOT、P99 和满足 SLO 的吞吐。接受率高不必然更快。

## 正确性、失败与降级

无公开权重时不能复现完整端到端结果。attention backend 和 CUDA graph folding 不匹配可能保留正确性却使性能退化。

“降级”优先选择保留目标采样语义的普通 target decode；不要通过放宽接受门槛伪造加速。可近似的质量策略应另行标注，不能继续称为严格无损。

## 学习实验怎么做

固定候选覆盖，分别测 argmax、条件概率采样、correlator 开关；核对实际 q 和是否有 device-to-host 同步。

先验证正确性，再测性能。记录冷／热状态、版本和数据划分；图表没有实验条件时不用于方法排名。未跑的实验保持“计划”，不填占位通过结果。



## 延伸阅读

- [返回综述与完整系列](/articles/speculative-decoding/)
- [技术时间轴](/articles/speculative-decoding/timeline.html)
- [原始仓库与源码阅读索引](/articles/sd-source-map/)
- 上一篇：[Domino](/articles/sd-domino/)
- 下一篇：[Sequoia](/articles/sd-sequoia/)
