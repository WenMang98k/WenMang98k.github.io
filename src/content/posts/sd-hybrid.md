---
title: "混合与级联投机：原理、源码与推理落地"
description: "组合检索、模型草稿与动态路由时，重新定义预算和概率契约。"
date: 2026-11-11
tags: ["投机解码", "训练与系统优化"]
draft: false
series: speculative-decoding
seriesOrder: 35
---

> **本文是源码学习初稿，不是性能复现报告。**已读取内容、未确认的原始仓库和未核实的数字均明确区分。原作者结论只在相应实验条件下引用；本站未运行 GPU 加速基准。

## 这项技术解决什么问题

混合投机不是把方法列表全部同时开启。可以在 n-gram 高命中时优先检索，在未命中时使用模型草稿；也可以按请求域路由或形成小模型—中模型—大模型级联。

## 原论文、仓库与版本

- 本轮未确认作者官方实现仓库；不把第三方或同名项目标成原始实现。
- 已读取源码／文档：[`docs/features/speculative_decoding/README.md`](https://github.com/vllm-project/vllm/blob/2ee8250eaf67059b4e8bc3c1bfb4cfb1c88892a7/docs/features/speculative_decoding/README.md)，提交 `2ee8250eaf`。
- 已读取源码／文档：[`python/sglang/srt/speculative/spec_info.py`](https://github.com/sgl-project/sglang/blob/34dfaccc57d928f046317c2e3d9e16bd69837d9b/python/sglang/srt/speculative/spec_info.py)，提交 `34dfaccc57`。

[统一源码阅读索引](/articles/sd-source-map/)记录可核实的文件版本和来源。链接至框架代码不表示论文作者已经将完整原始实现发布在该框架中。

## 训练流程与数据对齐

可不训练路由，也可对代价／接受预测器校准；若混合多个草稿，应独立记录每条路径的 proposal 和选择规则。不能把两个单独正确的 sampler 拼接后自动宣布整体无损。

阅读训练代码时，请记录 `input_ids`、`labels`、特征层、有效 `loss_mask` 与位置偏移。真实标签、教师分布和草稿生成 token 的来源必须分开；离线 teacher forcing 指标不能替代实际 rollout。

## 推理流程与状态提交

估计负载与候选可得性 → 选择草稿来源／级联层次 → 记录实际 q 与选择概率语义 → 验证 → 提交 → 根据负收益降级。最稳健的第一版是请求级选择一种已验证路径。

下面是**本站自写的机制伪代码**，省略批处理、EOS、张量并行和具体 API，不能当作可直接运行的作者实现：

```python
route = choose_supported_route(request_domain, load, cache_hits)
if route == 'lookup' and not lookup_hit(prefix):
    route = 'model_draft'
proposal = routes[route].propose_with_distribution(prefix)
verified = target_verify(proposal.tokens, proposal.actual_q)
commit(verified)
release_or_rebuild_state_before_switching_routes()
```

## 代码讲解：按调用链读，而不是按文件名猜

先定义接口：draft 输出 token、proposal 分布、来源、缓存句柄；verify 输出接受前缀、校正 token 与状态；router 不应修改已承诺的采样语义。下面伪代码是本站机制示意，不是框架原始代码。

一次完整阅读至少画出三份状态：候选 token、proposal 分布、KV／特征缓存。接受的 token 数、补采的校正 token 和缓存已经算到的位置不一定相同；必须以实现契约核对，不能凭长度变量名称推断。

## vLLM 与 SGLang 集成情况

两框架有多种方法和动态机制，不等于任意方法可在一个请求内随意混合；需查 worker、调度、KV 与模型同时驻留限制。

上述判断基于本轮核对的主分支文档／代码，不代表任意稳定版都支持。启用前记录 target／draft revision、框架 commit、attention backend、并行模式和采样配置；[综述支持矩阵](/articles/speculative-decoding/#framework-support)提供横向对照。

## 加速收益：证据与适用条件

组合可能改善长尾任务，也可能增加调度和多模型内存。本站未测，必须对每个单方法基线、混合策略与关闭投机做消融。

比较时固定 target、精度、任务、输入输出长度、温度／top-p 和负载。至少记录每轮有效输出、draft／verify／KV 管理耗时、TPOT、P99 和满足 SLO 的吞吐。接受率高不必然更快。

## 正确性、失败与降级

错误的在线切换会复用不兼容草稿 KV。切换时重建状态或在安全轮次边界进行，负收益时回退普通 target decode。

“降级”优先选择保留目标采样语义的普通 target decode；不要通过放宽接受门槛伪造加速。可近似的质量策略应另行标注，不能继续称为严格无损。

## 学习实验怎么做

先实现请求级路由并统计决策成本；再考虑轮次级切换，验证缓存隔离、分布正确性和故障降级。

先验证正确性，再测性能。记录冷／热状态、版本和数据划分；图表没有实验条件时不用于方法排名。未跑的实验保持“计划”，不填占位通过结果。



## 延伸阅读

- [返回综述与完整系列](/articles/speculative-decoding/)
- [技术时间轴](/articles/speculative-decoding/timeline.html)
- [原始仓库与源码阅读索引](/articles/sd-source-map/)
- 上一篇：[D-PACE 训练目标](/articles/sd-dpace/)
