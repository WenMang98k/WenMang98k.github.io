---
title: "Medusa：原理、源码与推理落地"
description: "从同一隐藏状态预测多个未来位置，读懂多头候选树与接受规则。"
date: 2026-11-11
tags: ["投机解码", "多头与特征"]
draft: false
series: speculative-decoding
seriesOrder: 9
---

> **本文是源码学习初稿，不是性能复现报告。**已读取内容、未确认的原始仓库和未核实的数字均明确区分。原作者结论只在相应实验条件下引用；本站未运行 GPU 加速基准。

## 这项技术解决什么问题

在基座隐藏状态上附加多个未来 token 头。每个头预测不同偏移的候选，组合后形成树。其草稿很便宜，但远位置头未必知道前面实际选了什么 token，因此候选的联合一致性是薄弱点。

## 原论文、仓库与版本

- 代码入口：[FasterDecoding/Medusa](https://github.com/FasterDecoding/Medusa)。以原项目版本与说明为准。
- 原论文：[arXiv:2401.10774](https://arxiv.org/abs/2401.10774)。论文全文本轮受访问限制；原文与框架实现的证据范围见下文。
- 已读取源码／文档：[`medusa/model/utils.py`](https://github.com/FasterDecoding/Medusa/blob/e2a5d20c048a9b0a4092e6933c34313687422518/medusa/model/utils.py)，提交 `e2a5d20c04`。
- 已读取源码／文档：[`medusa/train/train_legacy.py`](https://github.com/FasterDecoding/Medusa/blob/e2a5d20c048a9b0a4092e6933c34313687422518/medusa/train/train_legacy.py)，提交 `e2a5d20c04`。

[统一源码阅读索引](/articles/sd-source-map/)记录可核实的文件版本和来源。链接至框架代码不表示论文作者已经将完整原始实现发布在该框架中。

## 训练流程与数据对齐

Medusa-1 冻结基座，只优化新增头；compute_loss 按头的偏移对齐 logits 和 label。Medusa-2 可更新基座，需重新明确验证所对应的 target checkpoint。真实标签缺失和 padding 不能参与损失。

阅读训练代码时，请记录 `input_ids`、`labels`、特征层、有效 `loss_mask` 与位置偏移。真实标签、教师分布和草稿生成 token 的来源必须分开；离线 teacher forcing 指标不能替代实际 rollout。

## 推理流程与状态提交

基座状态 → 各头 top-k → generate_candidates → tree_decoding → evaluate_posterior → update_inference_inputs。提交必须是合法路径的连续前缀。

下面是**本站自写的机制伪代码**，省略批处理、EOS、张量并行和具体 API，不能当作可直接运行的作者实现：

```python
state = target_state(committed_prefix)
future_logits = [head(state) for head in medusa_heads]
tree = combine_head_topk(future_logits, fixed_tree_choices)
verified = target_tree_forward(tree, ancestor_mask(tree))
path, count = posterior_rule(verified, tree, sampling_mode)
commit_path_and_reindex_kv(path, count)
# Typical acceptance and exact sampling need different guarantees.
```

## 代码讲解：按调用链读，而不是按文件名猜

generate_medusa_buffers 构建树索引与 mask；generate_candidates 把多头候选整理成验证布局；evaluate_posterior 决定接受长度；update_inference_inputs 同步 token 和 KV。train_legacy.py 的 CustomizedTrainer.compute_loss 展示标签偏移，而不是为每个头复制一套基座。

一次完整阅读至少画出三份状态：候选 token、proposal 分布、KV／特征缓存。接受的 token 数、补采的校正 token 和缓存已经算到的位置不一定相同；必须以实现契约核对，不能凭长度变量名称推断。

### 标签偏移的具体读法

所读 `train_legacy.py` 对第 i 个新增头使用 `2+i` 的位置偏移，而非一律 next-token 的 `1`。这是因为基座头已经承担紧邻下一 token。设序列长 T，第 i 个头的有效监督长为 `T-(2+i)`；logits 与 label 两边裁剪后才能展平计算 CE。

```python
# 本站对原实现思想的缩略表达，不是直接复制作者代码。
offset = head_index + 2
prediction = head_logits[:, :-offset, :]
future_label = sequence_labels[:, offset:]
loss = cross_entropy_on_nonignored_positions(prediction, future_label)
```

拿五个 token 手画一遍：头 0 的位置 0 预测 token 2；头 1 的位置 0 预测 token 3。最容易出错的是把每个头都对齐到相同 label，从而训练出多个重复 next-token 头。

## vLLM 与 SGLang 集成情况

作为历史重要机制保留。当前核实的两框架默认方法索引未列出统一 Medusa 入口；历史版本或外部集成不能替代当前版本核验。

上述判断基于本轮核对的主分支文档／代码，不代表任意稳定版都支持。启用前记录 target／draft revision、框架 commit、attention backend、并行模式和采样配置；[综述支持矩阵](/articles/speculative-decoding/#framework-support)提供横向对照。

## 加速收益：证据与适用条件

官方 README 报告更新版在多种模型上约 2.2–3.6×，该摘要没有同时给齐硬件、任务与采样条件，不能进入本站同条件排名。本站没有 GPU 复现。

比较时固定 target、精度、任务、输入输出长度、温度／top-p 和负载。至少记录每轮有效输出、draft／verify／KV 管理耗时、TPOT、P99 和满足 SLO 的吞吐。接受率高不必然更快。

## 正确性、失败与降级

作者实现包含 typical acceptance；“任务质量接近”不等于严格目标分布一致。使用概率采样时，必须确认接受规则，而不是只看头的准确率。

“降级”优先选择保留目标采样语义的普通 target decode；不要通过放宽接受门槛伪造加速。可近似的质量策略应另行标注，不能继续称为严格无损。

## 学习实验怎么做

分别测各偏移头 token accuracy、连续接受长度和树节点预算；将独立多头与依赖多头在相同 target／树预算下比较。

先验证正确性，再测性能。记录冷／热状态、版本和数据划分；图表没有实验条件时不用于方法排名。未跑的实验保持“计划”，不填占位通过结果。



## 延伸阅读

- [返回综述与完整系列](/articles/speculative-decoding/)
- [技术时间轴](/articles/speculative-decoding/timeline.html)
- [原始仓库与源码阅读索引](/articles/sd-source-map/)
- 上一篇：[早期 SpecDec / Seq2seq 投机执行](/articles/sd-early-specdec/)
- 下一篇：[Hydra](/articles/sd-hydra/)
