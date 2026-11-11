---
title: "Draft & Verify：原理、源码与推理落地"
description: "跳过部分层提出候选，再使用完整模型完成自验证。"
date: 2026-11-11
tags: ["投机解码", "MTP 与自投机"]
draft: false
series: speculative-decoding
seriesOrder: 19
---

> **本文是源码学习初稿，不是性能复现报告。**已读取内容、未确认的原始仓库和未核实的数字均明确区分。原作者结论只在相应实验条件下引用；本站未运行 GPU 加速基准。

## 这项技术解决什么问题

从同一个 target 中跳过部分层得到便宜的近似 drafter，再由完整层集合验证。与独立小模型相比，参数共享明显；与 LayerSkip 相比，该路线探索不重新训练就利用跳层的可能性。

## 原论文、仓库与版本

- 作者代码仓库：[dilab-zju/self-speculative-decoding](https://github.com/dilab-zju/self-speculative-decoding)。
- 已读取固定提交 [`decoding.py`](https://github.com/dilab-zju/self-speculative-decoding/blob/688db8b74a782af21777b7ad5aeb399d7eb26faf/decoding.py) 与 [`searching.py`](https://github.com/dilab-zju/self-speculative-decoding/blob/688db8b74a782af21777b7ad5aeb399d7eb26faf/searching.py)。
- 原论文：[arXiv:2309.08168](https://arxiv.org/abs/2309.08168)。论文全文本轮受访问限制；原文与框架实现的证据范围见下文。
- 已读取源码／文档：[`docs/features/speculative_decoding/README.md`](https://github.com/vllm-project/vllm/blob/2ee8250eaf67059b4e8bc3c1bfb4cfb1c88892a7/docs/features/speculative_decoding/README.md)，提交 `2ee8250eaf`。

[统一源码阅读索引](/articles/sd-source-map/)记录可核实的文件版本和来源。链接至框架代码不表示论文作者已经将完整原始实现发布在该框架中。

## 训练流程与数据对齐

典型讨论是训练无关，但要选择合适的 skip pattern；寻找模式可能需要校准或搜索成本，这不等价于训练新 LM。不能假设任意模型任意去掉一半层都有效。

阅读训练代码时，请记录 `input_ids`、`labels`、特征层、有效 `loss_mask` 与位置偏移。真实标签、教师分布和草稿生成 token 的来源必须分开；离线 teacher forcing 指标不能替代实际 rollout。

## 推理流程与状态提交

按跳层模式连续提出候选 → 完整 target 批验证 → 更新合法 KV；被省略层在验证时需要建立正确缓存，不能复用根本没有算过的状态。

下面是**本站自写的机制伪代码**，省略批处理、EOS、张量并行和具体 API，不能当作可直接运行的作者实现：

```python
model.set_skip_layers(attention_skip, mlp_skip)
with model.self_draft():
    candidate, q = draft_steps(prefix, max_steps)
verified = full_model_forward(prefix, candidate)
accepted = validate_with_matching_sampling_mode(verified, q)
trim_rejected_layer_kv(accepted)
commit_verified_prefix(accepted)
```

## 代码讲解：按调用链读，而不是按文件名猜

作者实现的 `LayerSkippingSearching` 用 BayesianOptimization 搜索 attention／MLP 跳层集合。`decoding.py` 分别提供 `exact_self_speculative_generate` 和 `self_speculative_sample`：后者在 `model.self_draft()` 内提出候选，记录 draft 概率，在完整模型中验证，再裁剪 KV。阅读时必须分别核对 `do_sample`、`do_sample_draft`、温度／截断与实际 proposal 的一致性，不能因函数名字带 exact 就跳过模式审计。

一次完整阅读至少画出三份状态：候选 token、proposal 分布、KV／特征缓存。接受的 token 数、补采的校正 token 和缓存已经算到的位置不一定相同；必须以实现契约核对，不能凭长度变量名称推断。

## vLLM 与 SGLang 集成情况

当前两框架方法表未据此证明同名内置入口；自投机不能仅用一个通用 draft_model 配置自动实现。

上述判断基于本轮核对的主分支文档／代码，不代表任意稳定版都支持。启用前记录 target／draft revision、框架 commit、attention backend、并行模式和采样配置；[综述支持矩阵](/articles/speculative-decoding/#framework-support)提供横向对照。

## 加速收益：证据与适用条件

减少草稿成本与降低预测质量存在权衡。本文未核实同条件作者数值，不以层数减半推导 2×收益。

比较时固定 target、精度、任务、输入输出长度、温度／top-p 和负载。至少记录每轮有效输出、draft／verify／KV 管理耗时、TPOT、P99 和满足 SLO 的吞吐。接受率高不必然更快。

## 正确性、失败与降级

缓存按层错误复用、position 递增不一致会破坏正确性。模型不适合跳层时应关闭而非不断加大 K。

“降级”优先选择保留目标采样语义的普通 target decode；不要通过放宽接受门槛伪造加速。可近似的质量策略应另行标注，不能继续称为严格无损。

## 学习实验怎么做

固定层数预算比较不同 skip pattern，单列搜索成本与推理成本，覆盖首次拒绝和全接受两类 KV 路径。

先验证正确性，再测性能。记录冷／热状态、版本和数据划分；图表没有实验条件时不用于方法排名。未跑的实验保持“计划”，不填占位通过结果。



## 延伸阅读

- [返回综述与完整系列](/articles/speculative-decoding/)
- [技术时间轴](/articles/speculative-decoding/timeline.html)
- [原始仓库与源码阅读索引](/articles/sd-source-map/)
- 上一篇：[Multi-Token Prediction / MTP](/articles/sd-mtp/)
- 下一篇：[LayerSkip](/articles/sd-layerskip/)
