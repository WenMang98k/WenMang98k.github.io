---
title: "P-EAGLE：原理、源码与推理落地"
description: "在特征草稿路线中研究并行 proposal，而非把所有 EAGLE 都看作串行递归。"
date: 2026-11-11
tags: ["投机解码", "多头与特征"]
draft: false
series: speculative-decoding
seriesOrder: 15
---

> **本文是源码学习初稿，不是性能复现报告。**已读取内容、未确认的原始仓库和未核实的数字均明确区分。原作者结论只在相应实验条件下引用；本站未运行 GPU 加速基准。

## 这项技术解决什么问题

P-EAGLE 是并行 EAGLE 路线。所读 SpecForge 代码中 PEagleDraftModel 继承 EAGLE3 家族的模型接口，但另建 first layer、standard layer 和对应 backbone；这说明共享接口并不代表内部注意力结构相同。

## 原论文、仓库与版本

- 代码入口：[sgl-project/SpecForge](https://github.com/sgl-project/SpecForge)。这是已核实的训练／服务框架实现，不冒充独立论文作者的原始仓库。
- 原论文：[arXiv:2602.01469](https://arxiv.org/abs/2602.01469)。论文全文本轮受访问限制；原文与框架实现的证据范围见下文。
- 已读取源码／文档：[`specforge/modeling/draft/peagle.py`](https://github.com/sgl-project/SpecForge/blob/ce3ef8fa05ea397881634590768d0b0150ec24f9/specforge/modeling/draft/peagle.py)，提交 `ce3ef8fa05`。

[统一源码阅读索引](/articles/sd-source-map/)记录可核实的文件版本和来源。链接至框架代码不表示论文作者已经将完整原始实现发布在该框架中。

## 训练流程与数据对齐

在冻结目标特征条件下训练草稿，必须遵守模型定义的输入投影、位置与 mask 约定。本文核实模型结构入口，不把没有逐项检查的原论文目标或训练预算写成确定配方。

阅读训练代码时，请记录 `input_ids`、`labels`、特征层、有效 `loss_mask` 与位置偏移。真实标签、教师分布和草稿生成 token 的来源必须分开；离线 teacher forcing 指标不能替代实际 rollout。

## 推理流程与状态提交

提取 target 状态 → project_hidden_states／embed_input_ids → 并行草稿 backbone → compute_logits → target 验证。候选并行提出不意味着接受可忽略前缀依赖。

下面是**本站自写的机制伪代码**，省略批处理、EOS、张量并行和具体 API，不能当作可直接运行的作者实现：

```python
target_features = capture_selected_layers(prefix)
inputs = project_features_and_embed_slots(target_features)
parallel_hidden = peagle_backbone(inputs, valid_mask)
proposal = sample_from_projected_logits(parallel_hidden)
verified = target.verify(proposal)
commit_verified_prefix(verified)
```

## 代码讲解：按调用链读，而不是按文件名猜

对照 PEagleFirstLayer 与 PEagleStandardLayer 的输入维度，随后看 project_hidden_states、backbone、forward。与 EAGLE-3 对比时，先比较张量语义，再比较层数，不能只换 architecture 名称。

一次完整阅读至少画出三份状态：候选 token、proposal 分布、KV／特征缓存。接受的 token 数、补采的校正 token 和缓存已经算到的位置不一定相同；必须以实现契约核对，不能凭长度变量名称推断。

## vLLM 与 SGLang 集成情况

SpecForge README 有独立训练项；训练支持不自动等于两推理框架发布版都支持。需核对导出模型 architecture 与对应 serving loader。

上述判断基于本轮核对的主分支文档／代码，不代表任意稳定版都支持。启用前记录 target／draft revision、框架 commit、attention backend、并行模式和采样配置；[综述支持矩阵](/articles/speculative-decoding/#framework-support)提供横向对照。

## 加速收益：证据与适用条件

并行草稿减少串行调用的潜力需要由接受长度和单次更宽前向成本共同检验。本轮没有核实可直接比较的作者倍数，不填数值。

比较时固定 target、精度、任务、输入输出长度、温度／top-p 和负载。至少记录每轮有效输出、draft／verify／KV 管理耗时、TPOT、P99 和满足 SLO 的吞吐。接受率高不必然更快。

## 正确性、失败与降级

并行槽位掩码错误可能泄漏未来监督或让离线指标虚高；缺少 loader 支持时切回兼容 EAGLE-3，而不是强制加载。

“降级”优先选择保留目标采样语义的普通 target decode；不要通过放宽接受门槛伪造加速。可近似的质量策略应另行标注，不能继续称为严格无损。

## 学习实验怎么做

用相同 target、草稿参数预算、K 和目标验证预算对比 EAGLE-3；分开测主干延迟和连续接受长度。

先验证正确性，再测性能。记录冷／热状态、版本和数据划分；图表没有实验条件时不用于方法排名。未跑的实验保持“计划”，不填占位通过结果。



## 延伸阅读

- [返回综述与完整系列](/articles/speculative-decoding/)
- [技术时间轴](/articles/speculative-decoding/timeline.html)
- [原始仓库与源码阅读索引](/articles/sd-source-map/)
- 上一篇：[EAGLE-3](/articles/sd-eagle3/)
- 下一篇：[EAGLE3.1 实现追踪](/articles/sd-eagle31/)
