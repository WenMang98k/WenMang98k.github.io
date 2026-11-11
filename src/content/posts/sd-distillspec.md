---
title: "DistillSpec：原理、源码与推理落地"
description: "将草稿训练目标从通用能力转向与目标模型对齐。"
date: 2026-11-11
tags: ["投机解码", "训练与系统优化"]
draft: false
series: speculative-decoding
seriesOrder: 30
---

> **本文是源码学习初稿，不是性能复现报告。**已读取内容、未确认的原始仓库和未核实的数字均明确区分。原作者结论只在相应实验条件下引用；本站未运行 GPU 加速基准。

## 这项技术解决什么问题

相同尺寸的小模型不一定有相同接受率；投机收益依赖 draft 在 target 会访问的上下文中与其预测分布接近。DistillSpec 属于用知识蒸馏改善 draft／target 对齐的路线。

## 原论文、仓库与版本

- 本轮未确认作者官方实现仓库；不把第三方或同名项目标成原始实现。
- 原论文：[arXiv:2310.08461](https://arxiv.org/abs/2310.08461)。论文全文本轮受访问限制；原文与框架实现的证据范围见下文。

[统一源码阅读索引](/articles/sd-source-map/)记录可核实的文件版本和来源。链接至框架代码不表示论文作者已经将完整原始实现发布在该框架中。

## 训练流程与数据对齐

训练需要从目标模型和相关上下文获得监督，并选择适合目标采样场景的分布对齐策略。本轮未读取原论文全文且没有核实唯一官方仓库，因此不擅自指定 KL 方向、温度或 on-policy 比例为原文配方。

阅读训练代码时，请记录 `input_ids`、`labels`、特征层、有效 `loss_mask` 与位置偏移。真实标签、教师分布和草稿生成 token 的来源必须分开；离线 teacher forcing 指标不能替代实际 rollout。

## 推理流程与状态提交

蒸馏后的 draft 可接入经典 proposal／verification 流程；改变的是 q 的质量，不是省略 p 的验证。分布距离降低后仍需正确残差重采样。

下面是**本站自写的机制伪代码**，省略批处理、EOS、张量并行和具体 API，不能当作可直接运行的作者实现：

```python
contexts = collect_contexts_relevant_to_target_distribution()
teacher_distribution = target(contexts).detach()
student_distribution = draft(contexts)
loss = chosen_alignment_objective(teacher_distribution, student_distribution)
update_draft_only(loss)
# This sketch does not assert the paper's KL direction or data recipe.
```

## 代码讲解：按调用链读，而不是按文件名猜

源码学习入口应先确认论文作者链接，再核对数据生成、target logits、loss 和 serving exporter。未确认作者代码时，只提供原理论阅读与本站自写参考采样器，不把搜索到的同名第三方库冠以官方。

一次完整阅读至少画出三份状态：候选 token、proposal 分布、KV／特征缓存。接受的 token 数、补采的校正 token 和缓存已经算到的位置不一定相同；必须以实现契约核对，不能凭长度变量名称推断。

## vLLM 与 SGLang 集成情况

训练后的通用小 LM 在结构兼容时可作为 vLLM draft_model／SGLang STANDALONE；这不代表二者内置 DistillSpec 训练流程。

上述判断基于本轮核对的主分支文档／代码，不代表任意稳定版都支持。启用前记录 target／draft revision、框架 commit、attention backend、并行模式和采样配置；[综述支持矩阵](/articles/speculative-decoding/#framework-support)提供横向对照。

## 加速收益：证据与适用条件

预期改善接受率，但额外训练成本与跨域失配要计入。没有核实到同条件数值，本篇不报告倍率；比较时必须固定 draft 大小与 runtime。

比较时固定 target、精度、任务、输入输出长度、温度／top-p 和负载。至少记录每轮有效输出、draft／verify／KV 管理耗时、TPOT、P99 和满足 SLO 的吞吐。接受率高不必然更快。

## 正确性、失败与降级

离线数据只覆盖易接受任务可能高估收益；与 target 一起升级时应重新校准，而非复用旧蒸馏结果。

“降级”优先选择保留目标采样语义的普通 target decode；不要通过放宽接受门槛伪造加速。可近似的质量策略应另行标注，不能继续称为严格无损。

## 学习实验怎么做

同一个 draft 架构，比较原 checkpoint、业务 SFT 与分布蒸馏，在域内／域外分别测 acceptance、成本和质量。

先验证正确性，再测性能。记录冷／热状态、版本和数据划分；图表没有实验条件时不用于方法排名。未跑的实验保持“计划”，不填占位通过结果。



## 延伸阅读

- [返回综述与完整系列](/articles/speculative-decoding/)
- [技术时间轴](/articles/speculative-decoding/timeline.html)
- [原始仓库与源码阅读索引](/articles/sd-source-map/)
- 上一篇：[SpecInfer](/articles/sd-specinfer/)
- 下一篇：[FR-Spec](/articles/sd-frspec/)
