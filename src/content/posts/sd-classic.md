---
title: "经典 SD / Speculative Sampling：原理、源码与推理落地"
description: "从 proposal、接受概率到残差重采样，建立投机解码的正确性地基。"
date: 2026-11-11
tags: ["投机解码", "基础与检索"]
draft: false
series: speculative-decoding
seriesOrder: 1
---

> **本文是源码学习初稿，不是性能复现报告。**已读取内容、未确认的原始仓库和未核实的数字均明确区分。原作者结论只在相应实验条件下引用；本站未运行 GPU 加速基准。

## 这项技术解决什么问题

草稿模型 q 先提出一条候选链；目标模型 p 在已知这条链后，一次前向计算多个位置的分布。只有连续接受的前缀可以提交。其目的不是让小模型替代大模型，而是让大模型每次调用推进更多有效 token。

候选 y 的接受概率为 $\min(1,p(y)/q(y))$。拒绝后必须从 $[p-q]_+$ 归一化后的残差分布抽样。接受贡献 $\min(p(v),q(v))$，拒绝分支补回正差，最终恢复 p。全部候选接受时可以再从目标模型末位置分布抽一个 bonus token，停止条件除外。

## 原论文、仓库与版本

- 本轮未确认作者官方实现仓库；不把第三方或同名项目标成原始实现。
- 原论文：[arXiv:2211.17192](https://arxiv.org/abs/2211.17192)。论文全文本轮受访问限制；原文与框架实现的证据范围见下文。
- 已读取源码／文档：[`vllm/config/speculative.py`](https://github.com/vllm-project/vllm/blob/2ee8250eaf67059b4e8bc3c1bfb4cfb1c88892a7/vllm/config/speculative.py)，提交 `2ee8250eaf`。
- 已读取源码／文档：[`python/sglang/srt/speculative/spec_info.py`](https://github.com/sgl-project/sglang/blob/34dfaccc57d928f046317c2e3d9e16bd69837d9b/python/sglang/srt/speculative/spec_info.py)，提交 `34dfaccc57`。

[统一源码阅读索引](/articles/sd-source-map/)记录可核实的文件版本和来源。链接至框架代码不表示论文作者已经将完整原始实现发布在该框架中。

## 训练流程与数据对齐

不一定需要训练：可直接选同 tokenizer 的现成小模型。需要训练时，可以用目标业务 token 或目标分布蒸馏草稿；目标模型通常冻结。应让训练覆盖 target 实际生成的上下文，而不只比较通用 benchmark 分数。

阅读训练代码时，请记录 `input_ids`、`labels`、特征层、有效 `loss_mask` 与位置偏移。真实标签、教师分布和草稿生成 token 的来源必须分开；离线 teacher forcing 指标不能替代实际 rollout。

## 推理流程与状态提交

草稿顺序生成 K 个候选并保存实际 q；目标 teacher-forcing 式验证；逐位置接受／拒绝；首次拒绝时残差采样并截断；保留已接受位置的 KV，清理无效后缀，再开始下一轮。

下面是**本站自写的机制伪代码**，省略批处理、EOS、张量并行和具体 API，不能当作可直接运行的作者实现：

```python
proposal, q = draft(prefix, K)
p = target.verify(prefix, proposal)
for i, token in enumerate(proposal):
    if uniform() <= min(1, p[i][token] / q[i][token]):
        commit(token)
    else:
        commit(sample(normalize(clamp_min(p[i] - q[i], 0))))
        rollback_unaccepted_kv()
        break
else:
    commit(sample(p[len(proposal)]))
```

## 代码讲解：按调用链读，而不是按文件名猜

从框架配置识别独立 draft_model／STANDALONE 路径，再顺着 draft worker → target verify → rejection sampler → KV commit 阅读。这里最重要的变量是“实际采样 q”，不是未经过温度与截断的原始 logits。本站提供独立的有限词表参考实现，可在下方源码索引找到。

一次完整阅读至少画出三份状态：候选 token、proposal 分布、KV／特征缓存。接受的 token 数、补采的校正 token 和缓存已经算到的位置不一定相同；必须以实现契约核对，不能凭长度变量名称推断。

## vLLM 与 SGLang 集成情况

vLLM: draft_model；SGLang: STANDALONE。模型、tokenizer、采样处理器与并行配置都需要兼容。

上述判断基于本轮核对的主分支文档／代码，不代表任意稳定版都支持。启用前记录 target／draft revision、框架 commit、attention backend、并行模式和采样配置；[综述支持矩阵](/articles/speculative-decoding/#framework-support)提供横向对照。

## 加速收益：证据与适用条件

经典方法减少昂贵 target 的 decode 次数，但增加 draft、验证宽度和缓存管理开销。本站没有 GPU 倍数数据；应分别测 draft 时间、每轮有效推进长度与 target 验证成本。

比较时固定 target、精度、任务、输入输出长度、温度／top-p 和负载。至少记录每轮有效输出、draft／verify／KV 管理耗时、TPOT、P99 和满足 SLO 的吞吐。接受率高不必然更快。

## 正确性、失败与降级

拒绝后直接从 p 抽样是常见错误；在第一处拒绝之后继续提交后面的 token 同样错误。随机数消耗路径不同，所以同 seed 文本不同不等于分布不一致。

“降级”优先选择保留目标采样语义的普通 target decode；不要通过放宽接受门槛伪造加速。可近似的质量策略应另行标注，不能继续称为严格无损。

## 学习实验怎么做

先用二元词表 p=(0.7,0.3)、q=(0.4,0.6) 检查最终质量，再在同 target 上扫 K=1/2/4/8 与并发。

先验证正确性，再测性能。记录冷／热状态、版本和数据划分；图表没有实验条件时不用于方法排名。未跑的实验保持“计划”，不填占位通过结果。

另一篇奠基论文：[Chen 等 Speculative Sampling](https://arxiv.org/abs/2302.01318)。本轮未确认这两篇经典论文的统一作者代码仓库，不将第三方实现标成原始仓库。

## 延伸阅读

- [返回综述与完整系列](/articles/speculative-decoding/)
- [技术时间轴](/articles/speculative-decoding/timeline.html)
- [原始仓库与源码阅读索引](/articles/sd-source-map/)
- 下一篇：[N-gram / Prompt Lookup](/articles/sd-ngram/)
