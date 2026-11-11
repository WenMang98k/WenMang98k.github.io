---
title: "LayerSkip：原理、源码与推理落地"
description: "先训练早退能力，再共享浅层计算开展自投机。"
date: 2026-11-11
tags: ["投机解码", "MTP 与自投机"]
draft: false
series: speculative-decoding
seriesOrder: 20
---

> **本文是源码学习初稿，不是性能复现报告。**已读取内容、未确认的原始仓库和未核实的数字均明确区分。原作者结论只在相应实验条件下引用；本站未运行 GPU 加速基准。

## 这项技术解决什么问题

利用浅层网络生成草稿，深层完成验证，并复用浅层计算和缓存。与训练无关的跳层不同，LayerSkip 通过专门配方让中间层可用于预测，减少直接早退的质量损失。

## 原论文、仓库与版本

- 代码入口：[facebookresearch/LayerSkip](https://github.com/facebookresearch/LayerSkip)。以原项目版本与说明为准。
- 原论文：[arXiv:2404.16710](https://arxiv.org/abs/2404.16710)。论文全文本轮受访问限制；原文与框架实现的证据范围见下文。

[统一源码阅读索引](/articles/sd-source-map/)记录可核实的文件版本和来源。链接至框架代码不表示论文作者已经将完整原始实现发布在该框架中。

## 训练流程与数据对齐

配方涉及层 dropout 与中间层早退监督；作者 README 强调需要使用按 LayerSkip 配方训练的 checkpoint 才能观察预期收益。它通常是对训练后模型定义完整 target 基线。

阅读训练代码时，请记录 `input_ids`、`labels`、特征层、有效 `loss_mask` 与位置偏移。真实标签、教师分布和草稿生成 token 的来源必须分开；离线 teacher forcing 指标不能替代实际 rollout。

## 推理流程与状态提交

到 exit_layer 的浅层连续提案 → 后续深层验证 → 接受前缀的浅／深层缓存对齐 → 拒绝后回退。浅层与全层 position 要一致。

下面是**本站自写的机制伪代码**，省略批处理、EOS、张量并行和具体 API，不能当作可直接运行的作者实现：

```python
lower_state = cached_lower_layers(prefix)
proposal = generate_until_exit_layer(lower_state, exit_layer, K)
verified = run_remaining_layers(proposal, reuse_lower_kv=True)
accepted = verify_against_full_target(verified, proposal)
commit_lower_and_upper_kv(accepted)
rollback_both_cache_ranges_after_rejection(accepted)
```

## 代码讲解：按调用链读，而不是按文件名猜

从官方 generate.py 的 generation_strategy=self_speculative、exit_layer、num_speculations 参数进入；本轮没有逐函数读取具体策略文件，所以不猜测一个失效文件路径或函数名。阅读时重点追踪浅层 KV 是否被深层正确消费。

一次完整阅读至少画出三份状态：候选 token、proposal 分布、KV／特征缓存。接受的 token 数、补采的校正 token 和缓存已经算到的位置不一定相同；必须以实现契约核对，不能凭长度变量名称推断。

## vLLM 与 SGLang 集成情况

作者仓库记录 Transformers 集成；这不是 vLLM／SGLang 默认自投机入口的证明，应按框架版本单独核验。

上述判断基于本轮核对的主分支文档／代码，不代表任意稳定版都支持。启用前记录 target／draft revision、框架 commit、attention backend、并行模式和采样配置；[综述支持矩阵](/articles/speculative-decoding/#framework-support)提供横向对照。

## 加速收益：证据与适用条件

早退过浅节省计算却降低接受率，过深又使 draft 太贵。作者提供 exit_layer 与 speculation 数的扫描方案；本站未跑 GPU，不报告自测倍数。

比较时固定 target、精度、任务、输入输出长度、温度／top-p 和负载。至少记录每轮有效输出、draft／verify／KV 管理耗时、TPOT、P99 和满足 SLO 的吞吐。接受率高不必然更快。

## 正确性、失败与降级

比较对象应是完整 LayerSkip checkpoint 的普通 decode，不能把重新训练前后的质量差异归咎于验证算法。

“降级”优先选择保留目标采样语义的普通 target decode；不要通过放宽接受门槛伪造加速。可近似的质量策略应另行标注，不能继续称为严格无损。

## 学习实验怎么做

二维扫描 exit_layer 与 K，统计低／高并发、长短输入的最佳点，并测 extra memory 与缓存回退。

先验证正确性，再测性能。记录冷／热状态、版本和数据划分；图表没有实验条件时不用于方法排名。未跑的实验保持“计划”，不填占位通过结果。



## 延伸阅读

- [返回综述与完整系列](/articles/speculative-decoding/)
- [技术时间轴](/articles/speculative-decoding/timeline.html)
- [原始仓库与源码阅读索引](/articles/sd-source-map/)
- 上一篇：[Draft & Verify](/articles/sd-draftverify/)
- 下一篇：[UNO](/articles/sd-uno/)
