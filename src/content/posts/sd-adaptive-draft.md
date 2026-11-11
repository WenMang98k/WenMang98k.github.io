---
title: "自适应草稿步数：SGLang：原理、源码与推理落地"
description: "按接受行为与 batch 范围切换草稿步数，避免固定 K 的负收益。"
date: 2026-11-11
tags: ["投机解码", "训练与系统优化"]
draft: false
series: speculative-decoding
seriesOrder: 32
---

> **本文是源码学习初稿，不是性能复现报告。**已读取内容、未确认的原始仓库和未核实的数字均明确区分。原作者结论只在相应实验条件下引用；本站未运行 GPU 加速基准。

## 这项技术解决什么问题

最佳草稿步数会随任务、batch 和接受行为变化。SGLang 专门指南描述按 batch-size 区间维护 EMA 等状态，在预设档位之间切换草稿步数与候选 token 数。

## 原论文、仓库与版本

- 代码入口：[sgl-project/sglang](https://github.com/sgl-project/sglang)。这是已核实的训练／服务框架实现，不冒充独立论文作者的原始仓库。
- 已读取源码／文档：[`docs/docs/advanced_features/speculative_decoding.mdx`](https://github.com/sgl-project/sglang/blob/34dfaccc57d928f046317c2e3d9e16bd69837d9b/docs/docs/advanced_features/speculative_decoding.mdx)，提交 `34dfaccc57`。
- 已读取源码／文档：[`python/sglang/srt/speculative/spec_info.py`](https://github.com/sgl-project/sglang/blob/34dfaccc57d928f046317c2e3d9e16bd69837d9b/python/sglang/srt/speculative/spec_info.py)，提交 `34dfaccc57`。

[统一源码阅读索引](/articles/sd-source-map/)记录可核实的文件版本和来源。链接至框架代码不表示论文作者已经将完整原始实现发布在该框架中。

## 训练流程与数据对齐

不是重新训练 drafter；需要设置候选档位、阈值、滞回与批量分区。不同 batch-size 的统计应隔离，否则低负载高接受结果会污染高负载策略。

阅读训练代码时，请记录 `input_ids`、`labels`、特征层、有效 `loss_mask` 与位置偏移。真实标签、教师分布和草稿生成 token 的来源必须分开；离线 teacher forcing 指标不能替代实际 rollout。

## 推理流程与状态提交

读当前 batch 与历史统计 → 决定运行档位 → 使用该档位的模型／图状态 → 正常草稿与验证 → 记录接受结果更新 EMA。

下面是**本站自写的机制伪代码**，省略批处理、EOS、张量并行和具体 API，不能当作可直接运行的作者实现：

```python
batch_range = locate_batch_size_range(batch_size)
stats = acceptance_ema[batch_range]
tier = choose_steps_with_hysteresis(stats, supported_tiers)
runtime = prepared_runtime_states[tier]
result = runtime.draft_and_verify(prefix)
acceptance_ema[batch_range].update(result)
```

## 代码讲解：按调用链读，而不是按文件名猜

官方专门文档列出 AdaptiveSpeculativeParams、SpecRuntimeState、AdaptiveController 三部分。运行时切换通常利用预先准备的状态，不应误解为每步重新捕获 CUDA graph。

一次完整阅读至少画出三份状态：候选 token、proposal 分布、KV／特征缓存。接受的 token 数、补采的校正 token 和缓存已经算到的位置不一定相同；必须以实现契约核对，不能凭长度变量名称推断。

## vLLM 与 SGLang 集成情况

所读专门指南支持 EAGLE／EAGLE3 且 speculative-eagle-topk=1；不满足时回退静态设置。不要将其等同于 vLLM 的 per-slot 验证预算。

上述判断基于本轮核对的主分支文档／代码，不代表任意稳定版都支持。启用前记录 target／draft revision、框架 commit、attention backend、并行模式和采样配置；[综述支持矩阵](/articles/speculative-decoding/#framework-support)提供横向对照。

## 加速收益：证据与适用条件

收益是减少不值得生成的草稿，而非让相同草稿更准确。应比较流量变化时的 goodput 与 p99，不能只在一个稳定低并发点得出结论。

比较时固定 target、精度、任务、输入输出长度、温度／top-p 和负载。至少记录每轮有效输出、draft／verify／KV 管理耗时、TPOT、P99 和满足 SLO 的吞吐。接受率高不必然更快。

## 正确性、失败与降级

档位切换过频可能抖动；EMA 迟滞又可能在突发流量下反应太慢。策略关闭后要仍能正常静态 decode。

“降级”优先选择保留目标采样语义的普通 target decode；不要通过放宽接受门槛伪造加速。可近似的质量策略应另行标注，不能继续称为严格无损。

## 学习实验怎么做

构造 batch 与任务域变化的分段负载；比较固定最优、固定保守与 adaptive 策略，记录切换次数和延迟尾部。

先验证正确性，再测性能。记录冷／热状态、版本和数据划分；图表没有实验条件时不用于方法排名。未跑的实验保持“计划”，不填占位通过结果。



## 延伸阅读

- [返回综述与完整系列](/articles/speculative-decoding/)
- [技术时间轴](/articles/speculative-decoding/timeline.html)
- [原始仓库与源码阅读索引](/articles/sd-source-map/)
- 上一篇：[FR-Spec](/articles/sd-frspec/)
- 下一篇：[自适应验证预算：vLLM](/articles/sd-adaptive-verify/)
