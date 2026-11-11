---
title: "Domino：原理、源码与推理落地"
description: "在并行草稿后加入便宜 GRU logit correction。"
date: 2026-11-11
tags: ["投机解码", "块并行草稿"]
draft: false
series: speculative-decoding
seriesOrder: 25
---

> **本文是源码学习初稿，不是性能复现报告。**已读取内容、未确认的原始仓库和未核实的数字均明确区分。原作者结论只在相应实验条件下引用；本站未运行 GPU 加速基准。

## 这项技术解决什么问题

Domino 在 DFlash 类块草稿后，用轻量 GRU 校正后继 token 的预测。昂贵 Transformer 特征可以并行算，便宜的循环校正负责建模局部序列依赖，二者不是互相排斥的方案。

## 原论文、仓库与版本

- 代码入口：[sgl-project/SpecForge](https://github.com/sgl-project/SpecForge)。这是已核实的训练／服务框架实现，不冒充独立论文作者的原始仓库。
- 原论文：[arXiv:2605.29707](https://arxiv.org/abs/2605.29707)。论文全文本轮受访问限制；原文与框架实现的证据范围见下文。
- 已读取源码／文档：[`specforge/modeling/draft/domino.py`](https://github.com/sgl-project/SpecForge/blob/ce3ef8fa05ea397881634590768d0b0150ec24f9/specforge/modeling/draft/domino.py)，提交 `ce3ef8fa05`。

[统一源码阅读索引](/articles/sd-source-map/)记录可核实的文件版本和来源。链接至框架代码不表示论文作者已经将完整原始实现发布在该框架中。

## 训练流程与数据对齐

冻结 target，训练块草稿和校正头；具体 lambda 调度与目标需以配置为准。本轮核实 DominoDraftModel 模型头，不将所有训练框架可选损失当作论文配方。

阅读训练代码时，请记录 `input_ids`、`labels`、特征层、有效 `loss_mask` 与位置偏移。真实标签、教师分布和草稿生成 token 的来源必须分开；离线 teacher forcing 指标不能替代实际 rollout。

## 推理流程与状态提交

块特征提案 → 从合法前序信息初始化 GRU → correction logits 与 base logits 合并 → 产生候选 → target 验证。训练与推理使用的前序 token 来源必须对应。

下面是**本站自写的机制伪代码**，省略批处理、EOS、张量并行和具体 API，不能当作可直接运行的作者实现：

```python
hidden = parallel_block_draft(context, slots)
base_logits = shared_lm_head(hidden)
gru_state = initial_prefix_state(context)
for slot in proposal_slots:
    correction, gru_state = gru_correct(previous_token, gru_state)
    previous_token = propose(base_logits[slot] + correction)
verified = target_verify(proposal)
commit_and_restore_valid_gru_state(verified)
```

## 代码讲解：按调用链读，而不是按文件名猜

阅读 _init_draft_head 建立参数结构；suffix_start 确定从何处校正；_run_gru 处理循环，compute_correction_logits 与 apply_logits_head 组合输出。查看 _sample_draft_tokens 是否与训练校正路径一致。

一次完整阅读至少画出三份状态：候选 token、proposal 分布、KV／特征缓存。接受的 token 数、补采的校正 token 和缓存已经算到的位置不一定相同；必须以实现契约核对，不能凭长度变量名称推断。

## vLLM 与 SGLang 集成情况

SpecForge 提供模型／训练实现；所读 SGLang DFlash DP attention 小节明确该组合不支持 Domino 头。该限制不能泛化为所有模式，也不能忽略后强行宣称全面支持。vLLM 同名默认入口未核实。

上述判断基于本轮核对的主分支文档／代码，不代表任意稳定版都支持。启用前记录 target／draft revision、框架 commit、attention backend、并行模式和采样配置；[综述支持矩阵](/articles/speculative-decoding/#framework-support)提供横向对照。

## 加速收益：证据与适用条件

增加 GRU 时间换接受长度，净收益由两者比例决定。本轮没有同条件数据，不把“GRU 很小”当成零成本。

比较时固定 target、精度、任务、输入输出长度、温度／top-p 和负载。至少记录每轮有效输出、draft／verify／KV 管理耗时、TPOT、P99 和满足 SLO 的吞吐。接受率高不必然更快。

## 正确性、失败与降级

不支持的架构／并行后端应禁用或换受支持草稿，而非忽略 GRU 权重继续跑。

“降级”优先选择保留目标采样语义的普通 target decode；不要通过放宽接受门槛伪造加速。可近似的质量策略应另行标注，不能继续称为严格无损。

## 学习实验怎么做

对校正前后 logits、teacher-forced 与自生成前序两种条件做消融，记录 GRU 时间与前缀接受变化。

先验证正确性，再测性能。记录冷／热状态、版本和数据划分；图表没有实验条件时不用于方法排名。未跑的实验保持“计划”，不填占位通过结果。



## 延伸阅读

- [返回综述与完整系列](/articles/speculative-decoding/)
- [技术时间轴](/articles/speculative-decoding/timeline.html)
- [原始仓库与源码阅读索引](/articles/sd-source-map/)
- 上一篇：[DSpark](/articles/sd-dspark/)
- 下一篇：[LiLiCorr](/articles/sd-lilicorr/)
