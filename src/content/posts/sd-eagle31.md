---
title: "EAGLE3.1 实现追踪：原理、源码与推理落地"
description: "记录框架中的后续变体，明确已核实范围与未核实细节。"
date: 2026-11-11
tags: ["投机解码", "多头与特征"]
draft: false
series: speculative-decoding
seriesOrder: 16
---

> **本文是源码学习初稿，不是性能复现报告。**已读取内容、未确认的原始仓库和未核实的数字均明确区分。原作者结论只在相应实验条件下引用；本站未运行 GPU 加速基准。

## 这项技术解决什么问题

所读 SpecForge 方法表把 EAGLE3.1 描述为带 attention drift 的 feature-based autoregressive drafting。它是具体框架变体入口，不宜凭“3.1”把结构、损失、性能当成 EAGLE-3 的自动升级。

## 原论文、仓库与版本

- 代码入口：[sgl-project/SpecForge](https://github.com/sgl-project/SpecForge)。这是已核实的训练／服务框架实现，不冒充独立论文作者的原始仓库。
- 已读取源码／文档：[`docs/features/speculative_decoding/README.md`](https://github.com/vllm-project/vllm/blob/2ee8250eaf67059b4e8bc3c1bfb4cfb1c88892a7/docs/features/speculative_decoding/README.md)，提交 `2ee8250eaf`。

[统一源码阅读索引](/articles/sd-source-map/)记录可核实的文件版本和来源。链接至框架代码不表示论文作者已经将完整原始实现发布在该框架中。

## 训练流程与数据对齐

应以对应 draft config、strategy 与训练 checkpoint 为准。当前证据只足以确认方法索引存在，尚未完成其完整训练目标逐函数复核，因此本文提供阅读任务和边界，不虚构公式。

阅读训练代码时，请记录 `input_ids`、`labels`、特征层、有效 `loss_mask` 与位置偏移。真实标签、教师分布和草稿生成 token 的来源必须分开；离线 teacher forcing 指标不能替代实际 rollout。

## 推理流程与状态提交

沿特征采集、草稿展开、验证、状态更新四个接口检查 attention drift 如何影响缓存与可见上下文。它仍需正确 target 验证；不能直接跳过验证。

下面是**本站自写的机制伪代码**，省略批处理、EOS、张量并行和具体 API，不能当作可直接运行的作者实现：

```python
config = load_exact_eagle31_configuration()
assert_training_and_serving_attention_contracts_match(config)
proposal = configured_feature_draft(verified_target_state, config)
result = target_verify(proposal)
commit_verified_prefix(result)
# Resolve attention-drift semantics from the selected implementation.
```

## 代码讲解：按调用链读，而不是按文件名猜

从 SpecForge README 的 EAGLE3.1 配置进入，找 architecture、attention layout 和训练策略；将漂移配置逐项对齐服务端 loader。这里提供的是实现追踪文章，而非已完成全部源码审计的结论。

一次完整阅读至少画出三份状态：候选 token、proposal 分布、KV／特征缓存。接受的 token 数、补采的校正 token 和缓存已经算到的位置不一定相同；必须以实现契约核对，不能凭长度变量名称推断。

## vLLM 与 SGLang 集成情况

已确认 SpecForge 方法索引；没有据此确认 vLLM／SGLang 任意稳定版自动支持。集成需查实际 config 与 loader。

上述判断基于本轮核对的主分支文档／代码，不代表任意稳定版都支持。启用前记录 target／draft revision、框架 commit、attention backend、并行模式和采样配置；[综述支持矩阵](/articles/speculative-decoding/#framework-support)提供横向对照。

## 加速收益：证据与适用条件

未核实到同条件性能数字。评估时尤其比较长前缀、长 rollout 的接受长度和误差积累，而不是只看第一步准确率。

比较时固定 target、精度、任务、输入输出长度、温度／top-p 和负载。至少记录每轮有效输出、draft／verify／KV 管理耗时、TPOT、P99 和满足 SLO 的吞吐。接受率高不必然更快。

## 正确性、失败与降级

不要将命名相似当成权重兼容。无法确认配置语义时，使用已验证的 EAGLE-3 路线。

“降级”优先选择保留目标采样语义的普通 target decode；不要通过放宽接受门槛伪造加速。可近似的质量策略应另行标注，不能继续称为严格无损。

## 学习实验怎么做

锁定含该方法配置的 commit，记录 drift 开关、mask 差异，并逐步消融后测 acceptance 与缓存行为。

先验证正确性，再测性能。记录冷／热状态、版本和数据划分；图表没有实验条件时不用于方法排名。未跑的实验保持“计划”，不填占位通过结果。

本篇是**资料范围明确的实现追踪初稿**；后续补核内容会在文首更新，不以空缺数据制造成熟度。

## 延伸阅读

- [返回综述与完整系列](/articles/speculative-decoding/)
- [技术时间轴](/articles/speculative-decoding/timeline.html)
- [原始仓库与源码阅读索引](/articles/sd-source-map/)
- 上一篇：[P-EAGLE](/articles/sd-peagle/)
- 下一篇：[LSTM / RNN 轻量草稿](/articles/sd-lstm/)
