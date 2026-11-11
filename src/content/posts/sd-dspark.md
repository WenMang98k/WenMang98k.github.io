---
title: "DSpark：原理、源码与推理落地"
description: "通过 Markov 校正和置信度头连接候选依赖、接受概率与验证预算。"
date: 2026-11-11
tags: ["投机解码", "块并行草稿"]
draft: false
series: speculative-decoding
seriesOrder: 24
---

> **本文是源码学习初稿，不是性能复现报告。**已读取内容、未确认的原始仓库和未核实的数字均明确区分。原作者结论只在相应实验条件下引用；本站未运行 GPU 加速基准。

## 这项技术解决什么问题

所读框架实现基于 DFlash 主干，加可选 Markov 头和 confidence head。Markov 校正让后继 logits 感知前序 token；confidence 估计接受潜力，供服务系统选择验证预算。这里的原始论文与框架移植需区分，未确认统一的独立作者代码仓库。

## 原论文、仓库与版本

- 代码入口：[sgl-project/SpecForge](https://github.com/sgl-project/SpecForge)。这是已核实的训练／服务框架实现，不冒充独立论文作者的原始仓库。
- 原论文：[arXiv:2607.05147](https://arxiv.org/abs/2607.05147)。论文全文本轮受访问限制；原文与框架实现的证据范围见下文。
- 已读取源码／文档：[`specforge/modeling/draft/dspark.py`](https://github.com/sgl-project/SpecForge/blob/ce3ef8fa05ea397881634590768d0b0150ec24f9/specforge/modeling/draft/dspark.py)，提交 `ce3ef8fa05`。
- 已读取源码／文档：[`specforge/modeling/draft/heads/markov.py`](https://github.com/sgl-project/SpecForge/blob/ce3ef8fa05ea397881634590768d0b0150ec24f9/specforge/modeling/draft/heads/markov.py)，提交 `ce3ef8fa05`。

[统一源码阅读索引](/articles/sd-source-map/)记录可核实的文件版本和来源。链接至框架代码不表示论文作者已经将完整原始实现发布在该框架中。

## 训练流程与数据对齐

代码实现可组合 token CE、教师／草稿分布 L1 和 confidence BCE。单步平均接受概率等于分布重叠 $1-\frac12\lVert p-q\rVert_1$；训练用它作为 detached soft target 校准置信度。teacher-forced 值不是线上 rollout 的保证。

阅读训练代码时，请记录 `input_ids`、`labels`、特征层、有效 `loss_mask` 与位置偏移。真实标签、教师分布和草稿生成 token 的来源必须分开；离线 teacher forcing 指标不能替代实际 rollout。

## 推理流程与状态提交

并行草稿特征 → Markov 条件 logits／采样 → 置信度估计 → 可选验证预算选择 → target 真正验证与校正 → KV 提交。confidence 绝不替代接受判定。

下面是**本站自写的机制伪代码**，省略批处理、EOS、张量并行和具体 API，不能当作可直接运行的作者实现：

```python
hidden = parallel_draft(context, proposal_slots)
base_logits = shared_lm_head(hidden)
previous = committed_anchor_token
for slot in proposal_slots:
    logits = markov_correct(base_logits[slot], previous, hidden[slot])
    previous = sample_and_record_actual_q(logits)
confidence = predict_acceptance(hidden)
verified = target_verify_with_supported_budget(confidence)
commit_verified_prefix(verified)
```

## 代码讲解：按调用链读，而不是按文件名猜

build_markov_head 支持 vanilla／gated／rnn；DSparkDraftModel.apply_logits_head 负责校正，predict_confidence 负责预测；MarkovHead.sample_block_tokens 揭示前一 token 如何影响下一槽。L1 与 confidence loss 在 SpecForge dflash_family_model.py，需与上面模型头一起读。

一次完整阅读至少画出三份状态：候选 token、proposal 分布、KV／特征缓存。接受的 token 数、补采的校正 token 和缓存已经算到的位置不一定相同；必须以实现契约核对，不能凭长度变量名称推断。

### 从分布到 confidence 的训练链

在所读 SpecForge 训练代码中，教师与草稿的分布 L1 同时用于对齐损失和构造 confidence 软标签：

$$a=1-\tfrac12\lVert p-q\rVert_1.$$

先将 a 裁剪到 [0,1]，再 detach，作为 BCE-with-logits 的软监督；不能把 confidence 的输出 logits 当成已经归一化的概率。用于统计或调度前还需 sigmoid 等实现约定。

```python
# 概念性训练分解，略去位置权重与有效 token 分母。
l1 = (teacher_prob - draft_prob).abs().sum(dim=-1)
accept_target = (1 - 0.5 * l1).clamp(0, 1).detach()
loss_conf = binary_cross_entropy_with_logits(confidence_logits, accept_target)
loss = ce_weight * token_ce + l1_weight * l1 + conf_weight * loss_conf
```

这个 a 是在相同条件分布下的单步理论重叠，并不是任意真实 rollout 上的保证。测试还要看条件前缀变化和 confidence 校准。

## vLLM 与 SGLang 集成情况

vLLM 配置含 dspark；SGLang 算法枚举含 DSPARK。两侧调度机制不同：不能把 vLLM adaptive verification 当作 DSpark 论文全部内容。

上述判断基于本轮核对的主分支文档／代码，不代表任意稳定版都支持。启用前记录 target／draft revision、框架 commit、attention backend、并行模式和采样配置；[综述支持矩阵](/articles/speculative-decoding/#framework-support)提供横向对照。

## 加速收益：证据与适用条件

潜在收益一部分来自更好候选，一部分来自少验证低价值槽位。本站不提供未经复现的固定倍数；必须消融 Markov、confidence 和 adaptive budget 三个组件。

比较时固定 target、精度、任务、输入输出长度、温度／top-p 和负载。至少记录每轮有效输出、draft／verify／KV 管理耗时、TPOT、P99 和满足 SLO 的吞吐。接受率高不必然更快。

## 正确性、失败与降级

置信筛选会改变选择过程，应检查实际 proposal 概率契约，不能声称任意裁剪都无损。Markov 头是可选结构，RNN 配置不自动表示 LSTM。

“降级”优先选择保留目标采样语义的普通 target decode；不要通过放宽接受门槛伪造加速。可近似的质量策略应另行标注，不能继续称为严格无损。

## 学习实验怎么做

查看置信校准误差、预测生存长度与真实接受长度，跨 batch 测有效吞吐、p99 和关闭调度时的基线。

先验证正确性，再测性能。记录冷／热状态、版本和数据划分；图表没有实验条件时不用于方法排名。未跑的实验保持“计划”，不填占位通过结果。



## 延伸阅读

- [返回综述与完整系列](/articles/speculative-decoding/)
- [技术时间轴](/articles/speculative-decoding/timeline.html)
- [原始仓库与源码阅读索引](/articles/sd-source-map/)
- 上一篇：[DFlash2](/articles/sd-dflash2/)
- 下一篇：[Domino](/articles/sd-domino/)
