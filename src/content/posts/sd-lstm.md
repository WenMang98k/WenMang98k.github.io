---
title: "LSTM / RNN 轻量草稿：原理、源码与推理落地"
description: "循环网络可以成为便宜 drafter，但架构名字本身不定义完整投机采样算法。"
date: 2026-11-11
tags: ["投机解码", "多头与特征"]
draft: false
series: speculative-decoding
seriesOrder: 17
---

> **本文是源码学习初稿，不是性能复现报告。**已读取内容、未确认的原始仓库和未核实的数字均明确区分。原作者结论只在相应实验条件下引用；本站未运行 GPU 加速基准。

## 这项技术解决什么问题

用便宜的循环状态融合前序 token 与目标特征，输出后续 token 或对基础 logits 做校正。LSTM、GRU 与普通 RNN 是不同结构；ReDrafter 的 recurrent、DSpark 的 rnn 选项都不能未经代码核实就统称 LSTM。

## 原论文、仓库与版本

- 本轮未确认作者官方实现仓库；不把第三方或同名项目标成原始实现。
- 已读取源码／文档：[`specforge/modeling/draft/heads/markov.py`](https://github.com/sgl-project/SpecForge/blob/ce3ef8fa05ea397881634590768d0b0150ec24f9/specforge/modeling/draft/heads/markov.py)，提交 `ce3ef8fa05`。

[统一源码阅读索引](/articles/sd-source-map/)记录可核实的文件版本和来源。链接至框架代码不表示论文作者已经将完整原始实现发布在该框架中。

## 训练流程与数据对齐

若使用真正 LSTM，需要明确输入特征、hidden／cell 状态、teacher forcing 与 rollout 的切换，以及 token／教师分布监督。校正头与完整草稿 LM 的训练目标不同。本轮未核实一篇统一命名为 LSTM speculative decoding 的作者原始实现，因此不虚构专有论文或仓库。

阅读训练代码时，请记录 `input_ids`、`labels`、特征层、有效 `loss_mask` 与位置偏移。真实标签、教师分布和草稿生成 token 的来源必须分开；离线 teacher forcing 指标不能替代实际 rollout。

## 推理流程与状态提交

初始化循环状态 → 根据已知前序提出候选 → 记录实际 proposal → target 验证 → 更新被接受前缀对应的循环状态和 KV。拒绝之后的 hidden／cell 同样必须回退，不能只回退 target KV。

下面是**本站自写的机制伪代码**，省略批处理、EOS、张量并行和具体 API，不能当作可直接运行的作者实现：

```python
h, c = state_at_committed_boundary(prefix)
checkpoint = (h.clone(), c.clone())
for token in candidate_steps:
    h, c = lstm_step(embed(previous_token), (h, c))
    previous_token = sample_and_record_actual_q(lm_head(h))
verified = target_verify(candidate_tokens, actual_q)
h, c = state_at_verified_boundary(checkpoint, verified)
commit_tokens_and_both_recurrent_states(verified)
```

## 代码讲解：按调用链读，而不是按文件名猜

SpecForge markov.py 中可读取 RNNMarkovHead、_rnn_step、apply_block_logits 与 sample_block_tokens，作为循环校正的实际例子；这是 RNN 实现，不冒充 LSTM。真正 LSTM 还需跟踪 h 与 c 两份状态，逐轮核对哪个 token 消费后才更新。

一次完整阅读至少画出三份状态：候选 token、proposal 分布、KV／特征缓存。接受的 token 数、补采的校正 token 和缓存已经算到的位置不一定相同；必须以实现契约核对，不能凭长度变量名称推断。

## vLLM 与 SGLang 集成情况

当前两框架不因支持 DSpark／MTP 就等于支持任意 LSTM draft。作为独立草稿接入时需实现对应模型和 KV／循环状态协议；作为头接入时需 loader 与 worker 支持。

上述判断基于本轮核对的主分支文档／代码，不代表任意稳定版都支持。启用前记录 target／draft revision、框架 commit、attention backend、并行模式和采样配置；[综述支持矩阵](/articles/speculative-decoding/#framework-support)提供横向对照。

## 加速收益：证据与适用条件

便宜循环单元降低主干成本的潜力，与较低候选质量存在权衡。本站没有 LSTM GPU 对比；参数少不能直接推出服务加速。

比较时固定 target、精度、任务、输入输出长度、温度／top-p 和负载。至少记录每轮有效输出、draft／verify／KV 管理耗时、TPOT、P99 和满足 SLO 的吞吐。接受率高不必然更快。

## 正确性、失败与降级

回退后继续使用拒绝候选的循环状态会污染后续 proposal。模型切换、batch 合并和请求取消都需要对应状态生命周期。

“降级”优先选择保留目标采样语义的普通 target decode；不要通过放宽接受门槛伪造加速。可近似的质量策略应另行标注，不能继续称为严格无损。

## 学习实验怎么做

在相同 target 与候选预算下比较无校正、普通 RNN、GRU、LSTM；测状态回退、草稿耗时与前缀接受长度。

先验证正确性，再测性能。记录冷／热状态、版本和数据划分；图表没有实验条件时不用于方法排名。未跑的实验保持“计划”，不填占位通过结果。



## 延伸阅读

- [返回综述与完整系列](/articles/speculative-decoding/)
- [技术时间轴](/articles/speculative-decoding/timeline.html)
- [原始仓库与源码阅读索引](/articles/sd-source-map/)
- 上一篇：[EAGLE3.1 实现追踪](/articles/sd-eagle31/)
- 下一篇：[Multi-Token Prediction / MTP](/articles/sd-mtp/)
