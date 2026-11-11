---
title: "Multi-Token Prediction / MTP：原理、源码与推理落地"
description: "从多未来训练目标到原生推理草稿，区分训练模块数与实际发布权重。"
date: 2026-11-11
tags: ["投机解码", "MTP 与自投机"]
draft: false
series: speculative-decoding
seriesOrder: 18
---

> **本文是源码学习初稿，不是性能复现报告。**已读取内容、未确认的原始仓库和未核实的数字均明确区分。原作者结论只在相应实验条件下引用；本站未运行 GPU 加速基准。

## 这项技术解决什么问题

MTP 是训练设计家族，不是固定的一种 sampler。原生模块共享主干特征、嵌入或输出头，以预测更远未来；DeepSeek 风格模块顺序连接，训练时真实 token 与推理候选 token 的来源必须区分。

## 原论文、仓库与版本

- 代码入口：[deepseek-ai/DeepSeek-V3](https://github.com/deepseek-ai/DeepSeek-V3)。以原项目版本与说明为准。
- 原论文：[arXiv:2412.19437](https://arxiv.org/abs/2412.19437)。论文全文本轮受访问限制；原文与框架实现的证据范围见下文。
- 已读取源码／文档：[`vllm/config/speculative.py`](https://github.com/vllm-project/vllm/blob/2ee8250eaf67059b4e8bc3c1bfb4cfb1c88892a7/vllm/config/speculative.py)，提交 `2ee8250eaf`。
- 已读取源码／文档：[`docs/docs/advanced_features/speculative_decoding.mdx`](https://github.com/sgl-project/sglang/blob/34dfaccc57d928f046317c2e3d9e16bd69837d9b/docs/docs/advanced_features/speculative_decoding.mdx)，提交 `34dfaccc57`。

[统一源码阅读索引](/articles/sd-source-map/)记录可核实的文件版本和来源。链接至框架代码不表示论文作者已经将完整原始实现发布在该框架中。

## 训练流程与数据对齐

主 next-token 目标外添加未来预测项；不同模块可能共享或串行连接。DeepSeek-V3 官方权重说明开源版本包含 1 个 MTP 模块，不能拿训练框图中的任意预测深度当成权重里已有层数。

阅读训练代码时，请记录 `input_ids`、`labels`、特征层、有效 `loss_mask` 与位置偏移。真实标签、教师分布和草稿生成 token 的来源必须分开；离线 teacher forcing 指标不能替代实际 rollout。

## 推理流程与状态提交

主干状态 → 辅助模块提出候选 → 必要时递归复用模块 → 主目标验证 → 提交与回退。MTP 训练本身不授权推理阶段跳过验证直接输出多个 token。

下面是**本站自写的机制伪代码**，省略批处理、EOS、张量并行和具体 API，不能当作可直接运行的作者实现：

```python
state, next_token = main_model_step(prefix)
proposal = [next_token]
for depth in range(num_speculations):
    state = mtp_module(state, embed(proposal[-1]))
    proposal.append(propose_from_shared_head(state))
verified = main_model_verify(proposal)
commit_verified_prefix_and_caches(verified)
```

## 代码讲解：按调用链读，而不是按文件名猜

先读 README_WEIGHTS.md 的 num_nextn_predict_layers、共享 embedding／head、eh_proj 与 norm；再检查框架 MTP loader 选择的 model_type。接口名 EAGLE／NEXTN 不代表模型内部就是某一代 EAGLE。

一次完整阅读至少画出三份状态：候选 token、proposal 分布、KV／特征缓存。接受的 token 数、补采的校正 token 和缓存已经算到的位置不一定相同；必须以实现契约核对，不能凭长度变量名称推断。

## vLLM 与 SGLang 集成情况

vLLM 有 mtp 及模型专用类型；SGLang 指南常由 EAGLE 路径启用原生 MTP，相关别名与模型识别随版本变化。

上述判断基于本轮核对的主分支文档／代码，不代表任意稳定版都支持。启用前记录 target／draft revision、框架 commit、attention backend、并行模式和采样配置；[综述支持矩阵](/articles/speculative-decoding/#framework-support)提供横向对照。

## 加速收益：证据与适用条件

无需独立完整小模型可能减少额外驻留成本，但 MTP 层依然消耗算力、内存与 KV。本文未统一实测，不能把“原生支持”写成固定倍数收益。

比较时固定 target、精度、任务、输入输出长度、温度／top-p 和负载。至少记录每轮有效输出、draft／verify／KV 管理耗时、TPOT、P99 和满足 SLO 的吞吐。接受率高不必然更快。

## 正确性、失败与降级

主模型经过 MTP 训练后的分布是正确性参照；它不承诺与训练前 checkpoint 相同。辅助模块缺失或版本不符时回退主模型。

“降级”优先选择保留目标采样语义的普通 target decode；不要通过放宽接受门槛伪造加速。可近似的质量策略应另行标注，不能继续称为严格无损。

## 学习实验怎么做

比较 K=1 和多步递归，记录实际加载模块数、额外显存、目标调用次数与不同 batch 下的服务收益。

先验证正确性，再测性能。记录冷／热状态、版本和数据划分；图表没有实验条件时不用于方法排名。未跑的实验保持“计划”，不填占位通过结果。



## 延伸阅读

- [返回综述与完整系列](/articles/speculative-decoding/)
- [技术时间轴](/articles/speculative-decoding/timeline.html)
- [原始仓库与源码阅读索引](/articles/sd-source-map/)
- 上一篇：[LSTM / RNN 轻量草稿](/articles/sd-lstm/)
- 下一篇：[Draft & Verify](/articles/sd-draftverify/)
