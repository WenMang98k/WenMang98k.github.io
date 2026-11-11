---
title: "DFlash：原理、源码与推理落地"
description: "用轻量块扩散草稿减少串行提案主干调用。"
date: 2026-11-11
tags: ["投机解码", "块并行草稿"]
draft: false
series: speculative-decoding
seriesOrder: 22
---

> **本文是源码学习初稿，不是性能复现报告。**已读取内容、未确认的原始仓库和未核实的数字均明确区分。原作者结论只在相应实验条件下引用；本站未运行 GPU 加速基准。

## 这项技术解决什么问题

目标模型仍是自回归 LLM，扩散／块预测发生在轻量 drafter。以目标前缀特征为条件，在多个未来槽位上并行产生候选，减少每个候选都串行运行草稿 Transformer 的成本。

## 原论文、仓库与版本

- 代码入口：[z-lab/dflash](https://github.com/z-lab/dflash)。以原项目版本与说明为准。
- 原论文：[arXiv:2602.06036](https://arxiv.org/abs/2602.06036)。论文全文本轮受访问限制；原文与框架实现的证据范围见下文。
- 已读取源码／文档：[`specforge/modeling/draft/dflash.py`](https://github.com/sgl-project/SpecForge/blob/ce3ef8fa05ea397881634590768d0b0150ec24f9/specforge/modeling/draft/dflash.py)，提交 `ce3ef8fa05`。

[统一源码阅读索引](/articles/sd-source-map/)记录可核实的文件版本和来源。链接至框架代码不表示论文作者已经将完整原始实现发布在该框架中。

## 训练流程与数据对齐

冻结 target，采集指定层特征与 token，采样 anchor 构造块，依配置使用 masked slots 和注意力布局监督后续位置；有效 token mask 过滤 padding。SpecForge 可选 CE、分布对齐和位置加权，不将所有选项当成原论文默认。

阅读训练代码时，请记录 `input_ids`、`labels`、特征层、有效 `loss_mask` 与位置偏移。真实标签、教师分布和草稿生成 token 的来源必须分开；离线 teacher forcing 指标不能替代实际 rollout。

## 推理流程与状态提交

extract_context_feature → 草稿 forward → _sample_draft_tokens → target 验证 → 截取连续前缀 → 更新状态。一次块级特征提案不代表未来 token 完全独立时也能得到长前缀。

下面是**本站自写的机制伪代码**，省略批处理、EOS、张量并行和具体 API，不能当作可直接运行的作者实现：

```python
context = collect_target_features(prefix)
slots = make_anchor_and_masked_proposal_slots(block_size)
draft_hidden = block_draft(context, slots, attention_layout)
proposal = project_and_sample(draft_hidden)
verified = target_verify(prefix, proposal)
commit_and_refresh_context(verified)
```

## 代码讲解：按调用链读，而不是按文件名猜

DFlashDraftModel.forward 关注前缀与 proposal 的注意力契约；_prepare_attention_mask 决定可见上下文；_sample_draft_tokens 连接特征与 token；spec_generate 展示本地参考循环。训练库里的 spec_generate 不等于服务框架的完整调度实现。

一次完整阅读至少画出三份状态：候选 token、proposal 分布、KV／特征缓存。接受的 token 数、补采的校正 token 和缓存已经算到的位置不一定相同；必须以实现契约核对，不能凭长度变量名称推断。

## vLLM 与 SGLang 集成情况

vLLM: dflash；SGLang: DFLASH。需要配套 target 与 draft checkpoint，并核对 block_size、特征层、mask token 和 attention backend。

上述判断基于本轮核对的主分支文档／代码，不代表任意稳定版都支持。启用前记录 target／draft revision、框架 commit、attention backend、并行模式和采样配置；[综述支持矩阵](/articles/speculative-decoding/#framework-support)提供横向对照。

## 加速收益：证据与适用条件

收益来自降低草稿串行主干调用，但更宽前向和远端候选低接受会抵消它。官方 README 的最新 benchmark 命令是可复现入口，不是本站实际结果。

比较时固定 target、精度、任务、输入输出长度、温度／top-p 和负载。至少记录每轮有效输出、draft／verify／KV 管理耗时、TPOT、P99 和满足 SLO 的吞吐。接受率高不必然更快。

## 正确性、失败与降级

固定 block_size 很大时不一定高效；量化或更换后端会影响数值与接受率。权重／层配置不符应启动时报错，而非静默回退成另一结构。

“降级”优先选择保留目标采样语义的普通 target decode；不要通过放宽接受门槛伪造加速。可近似的质量策略应另行标注，不能继续称为严格无损。

## 学习实验怎么做

与 EAGLE-3 对比草稿主干时间、输出头时间、前缀长度和完整轮次；扫块大小与并发，不只测 batch=1。

先验证正确性，再测性能。记录冷／热状态、版本和数据划分；图表没有实验条件时不用于方法排名。未跑的实验保持“计划”，不填占位通过结果。



## 延伸阅读

- [返回综述与完整系列](/articles/speculative-decoding/)
- [技术时间轴](/articles/speculative-decoding/timeline.html)
- [原始仓库与源码阅读索引](/articles/sd-source-map/)
- 上一篇：[UNO](/articles/sd-uno/)
- 下一篇：[DFlash2](/articles/sd-dflash2/)
