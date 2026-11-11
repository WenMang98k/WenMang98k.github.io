---
title: "FR-Spec：原理、源码与推理落地"
description: "从大词表输出头入手，减少 EAGLE 草稿的 lm_head 计算。"
date: 2026-11-11
tags: ["投机解码", "训练与系统优化"]
draft: false
series: speculative-decoding
seriesOrder: 31
---

> **本文是源码学习初稿，不是性能复现报告。**已读取内容、未确认的原始仓库和未核实的数字均明确区分。原作者结论只在相应实验条件下引用；本站未运行 GPU 加速基准。

## 这项技术解决什么问题

即使草稿 Transformer 很轻，大词表 LM head 仍可能是主要耗时。FR-Spec 利用频率排名的 token 子集／映射，减少草稿输出头需要计算的词表范围。目标模型仍验证，因此候选没有覆盖到某 token 时可以由 target 校正。

## 原论文、仓库与版本

- 代码入口：[sgl-project/sglang](https://github.com/sgl-project/sglang)。这是已核实的训练／服务框架实现，不冒充独立论文作者的原始仓库。
- 原论文：[arXiv:2502.14856](https://arxiv.org/abs/2502.14856)。论文全文本轮受访问限制；原文与框架实现的证据范围见下文。
- 已读取源码／文档：[`docs/docs/advanced_features/speculative_decoding.mdx`](https://github.com/sgl-project/sglang/blob/34dfaccc57d928f046317c2e3d9e16bd69837d9b/docs/docs/advanced_features/speculative_decoding.mdx)，提交 `34dfaccc57`。

[统一源码阅读索引](/articles/sd-source-map/)记录可核实的文件版本和来源。链接至框架代码不表示论文作者已经将完整原始实现发布在该框架中。

## 训练流程与数据对齐

需要与草稿词表一致的 token map 与相应准备过程。原文训练细节未全文核验，框架集成中的 map 文件不是任意截取词表数组。

阅读训练代码时，请记录 `input_ids`、`labels`、特征层、有效 `loss_mask` 与位置偏移。真实标签、教师分布和草稿生成 token 的来源必须分开；离线 teacher forcing 指标不能替代实际 rollout。

## 推理流程与状态提交

特征草稿 → 受限词表输出 logits → 映射回 target token ID → 验证 → 回退／接受。实际 proposal 的归一化与 ID 映射必须正确。

下面是**本站自写的机制伪代码**，省略批处理、EOS、张量并行和具体 API，不能当作可直接运行的作者实现：

```python
subset_logits = selected_lm_head_rows(draft_hidden, token_map)
small_vocab_candidate = sample(subset_logits)
target_token_id = token_map[small_vocab_candidate]
verified = target_verify(target_token_id, mapped_actual_proposal)
commit_or_correct(verified)
```

## 代码讲解：按调用链读，而不是按文件名猜

在 SGLang 指南的 Frequency-Ranked Speculative Sampling 段，检查 --speculative-token-map 的加载、映射方向与 lm_head 子集。该文档指出 EAGLE-3 会忽略此参数，不能把 EAGLE-2 配置不加说明搬过去。

一次完整阅读至少画出三份状态：候选 token、proposal 分布、KV／特征缓存。接受的 token 数、补采的校正 token 和缓存已经算到的位置不一定相同；必须以实现契约核对，不能凭长度变量名称推断。

## vLLM 与 SGLang 集成情况

SGLang 文档有 EAGLE-2 的 token-map 入口；本轮未确认 vLLM 独立 FR-Spec 开关。

上述判断基于本轮核对的主分支文档／代码，不代表任意稳定版都支持。启用前记录 target／draft revision、框架 commit、attention backend、并行模式和采样配置；[综述支持矩阵](/articles/speculative-decoding/#framework-support)提供横向对照。

## 加速收益：证据与适用条件

收益取决于 LM head 在 draft 中的占比。词表更小可能更快，也可能损害候选覆盖；本文未核实原论文倍率，建议输出头耗时与接受率一起看。

比较时固定 target、精度、任务、输入输出长度、温度／top-p 和负载。至少记录每轮有效输出、draft／verify／KV 管理耗时、TPOT、P99 和满足 SLO 的吞吐。接受率高不必然更快。

## 正确性、失败与降级

tokenizer 更新后旧 token map 不可盲用；覆盖变差时恢复全词表，而非继续压缩候选。

“降级”优先选择保留目标采样语义的普通 target decode；不要通过放宽接受门槛伪造加速。可近似的质量策略应另行标注，不能继续称为严格无损。

## 学习实验怎么做

扫保留词表大小，测 output projection 时间、草稿 top-k 覆盖、接受长度和端到端 TPOT。

先验证正确性，再测性能。记录冷／热状态、版本和数据划分；图表没有实验条件时不用于方法排名。未跑的实验保持“计划”，不填占位通过结果。



## 延伸阅读

- [返回综述与完整系列](/articles/speculative-decoding/)
- [技术时间轴](/articles/speculative-decoding/timeline.html)
- [原始仓库与源码阅读索引](/articles/sd-source-map/)
- 上一篇：[DistillSpec](/articles/sd-distillspec/)
- 下一篇：[自适应草稿步数：SGLang](/articles/sd-adaptive-draft/)
