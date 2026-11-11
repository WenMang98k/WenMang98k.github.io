---
title: "UNO：原理、源码与推理落地"
description: "复用目标 Transformer 与专门 LoRA，分离草稿与 base-only 验证。"
date: 2026-11-11
tags: ["投机解码", "MTP 与自投机"]
draft: false
series: speculative-decoding
seriesOrder: 21
---

> **本文是源码学习初稿，不是性能复现报告。**已读取内容、未确认的原始仓库和未核实的数字均明确区分。原作者结论只在相应实验条件下引用；本站未运行 GPU 加速基准。

## 这项技术解决什么问题

SGLang 指南中的 UNO 让 target Transformer 同时承担 draft pass 与 verify pass。draft 的首行使用 base 权重，其余候选行使用训练的 UNO LoRA；target verify 保持 base-only。

## 原论文、仓库与版本

- 代码入口：[sgl-project/sglang](https://github.com/sgl-project/sglang)。这是已核实的训练／服务框架实现，不冒充独立论文作者的原始仓库。
- 已读取源码／文档：[`docs/docs/advanced_features/speculative_decoding.mdx`](https://github.com/sgl-project/sglang/blob/34dfaccc57d928f046317c2e3d9e16bd69837d9b/docs/docs/advanced_features/speculative_decoding.mdx)，提交 `34dfaccc57`。
- 已读取源码／文档：[`python/sglang/srt/speculative/spec_info.py`](https://github.com/sgl-project/sglang/blob/34dfaccc57d928f046317c2e3d9e16bd69837d9b/python/sglang/srt/speculative/spec_info.py)，提交 `34dfaccc57`。

[统一源码阅读索引](/articles/sd-source-map/)记录可核实的文件版本和来源。链接至框架代码不表示论文作者已经将完整原始实现发布在该框架中。

## 训练流程与数据对齐

需要为精确 target checkpoint 训练适配器，不是任意已有 LoRA 都能用。本轮核实集成文档，未把缺少原文审阅的 adapter objective 说成确定算法。

阅读训练代码时，请记录 `input_ids`、`labels`、特征层、有效 `loss_mask` 与位置偏移。真实标签、教师分布和草稿生成 token 的来源必须分开；离线 teacher forcing 指标不能替代实际 rollout。

## 推理流程与状态提交

按 B/K/V 配置生成候选：B 是草稿前向宽度，K 是每次扩展保留候选数，V 是验证宽度。linear 模式构造一条链，tree 模式使用 EAGLE 树验证路径；verify 不能错误混入 LoRA。

下面是**本站自写的机制伪代码**，省略批处理、EOS、张量并行和具体 API，不能当作可直接运行的作者实现：

```python
draft_rows = make_uno_rows(prefix, B)
base_row, adapter_rows = split_first_row(draft_rows)
proposal = forward_base_and_uno_lora(base_row, adapter_rows)
proposal = select_linear_or_tree_candidates(proposal, K, V)
verified = base_only_target_verify(proposal)
commit_verified_prefix(verified)
```

## 代码讲解：按调用链读，而不是按文件名猜

先从 SpeculativeAlgorithm.UNO 找 worker，再核对 guide 的 B/K/V 到 CLI 参数映射。UNO 的 K 是树扩展宽度，不是用户 sampling top_k。测试 base-only verify 是必要正确性检查。

一次完整阅读至少画出三份状态：候选 token、proposal 分布、KV／特征缓存。接受的 token 数、补采的校正 token 和缓存已经算到的位置不一定相同；必须以实现契约核对，不能凭长度变量名称推断。

## vLLM 与 SGLang 集成情况

SGLang 有 UNO。所读指南验证范围包含 Qwen3-8B，并列出多项采样／grammar／deterministic inference 限制；vLLM 对应入口本轮未确认。

上述判断基于本轮核对的主分支文档／代码，不代表任意稳定版都支持。启用前记录 target／draft revision、框架 commit、attention backend、并行模式和采样配置；[综述支持矩阵](/articles/speculative-decoding/#framework-support)提供横向对照。

## 加速收益：证据与适用条件

省去独立 draft Transformer 不等于省去草稿计算；LoRA 行混合、eager 路径和宽度都影响延迟。当前没有本文实测数字。

比较时固定 target、精度、任务、输入输出长度、温度／top-p 和负载。至少记录每轮有效输出、draft／verify／KV 管理耗时、TPOT、P99 和满足 SLO 的吞吐。接受率高不必然更快。

## 正确性、失败与降级

用户请求使用不支持的 logit processor／grammar 时应拒绝或关闭 UNO，而非静默忽略请求约束。

“降级”优先选择保留目标采样语义的普通 target decode；不要通过放宽接受门槛伪造加速。可近似的质量策略应另行标注，不能继续称为严格无损。

## 学习实验怎么做

对 linear 与 tree 的相同验证预算比较，逐轮确认 verify 未加 LoRA，记录 eager 与 CUDA graph 的覆盖。

先验证正确性，再测性能。记录冷／热状态、版本和数据划分；图表没有实验条件时不用于方法排名。未跑的实验保持“计划”，不填占位通过结果。



## 延伸阅读

- [返回综述与完整系列](/articles/speculative-decoding/)
- [技术时间轴](/articles/speculative-decoding/timeline.html)
- [原始仓库与源码阅读索引](/articles/sd-source-map/)
- 上一篇：[LayerSkip](/articles/sd-layerskip/)
- 下一篇：[DFlash](/articles/sd-dflash/)
