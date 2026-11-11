---
title: "N-gram / Prompt Lookup：原理、源码与推理落地"
description: "不用额外神经网络，从输入和已生成上下文寻找候选续写。"
date: 2026-11-11
tags: ["投机解码", "基础与检索"]
draft: false
series: speculative-decoding
seriesOrder: 2
---

> **本文是源码学习初稿，不是性能复现报告。**已读取内容、未确认的原始仓库和未核实的数字均明确区分。原作者结论只在相应实验条件下引用；本站未运行 GPU 加速基准。

## 这项技术解决什么问题

把当前输出后缀作为查询，在 prompt 或历史 token 中找到相同片段，直接拷贝其后续 token 作为草稿。候选来源是文本重用，不是额外学习到的语言模型；token 匹配也不等于语义检索。

## 原论文、仓库与版本

- 代码入口：[apoorvumang/prompt-lookup-decoding](https://github.com/apoorvumang/prompt-lookup-decoding)。以原项目版本与说明为准。
- 已读取源码／文档：[`README.md`](https://github.com/apoorvumang/prompt-lookup-decoding/blob/cada4fe6ec7368d41c386ec3fb94fa11268971eb/README.md)，提交 `cada4fe6ec`。
- 已读取源码／文档：[`vllm/config/speculative.py`](https://github.com/vllm-project/vllm/blob/2ee8250eaf67059b4e8bc3c1bfb4cfb1c88892a7/vllm/config/speculative.py)，提交 `2ee8250eaf`。
- 已读取源码／文档：[`docs/docs/advanced_features/speculative_decoding.mdx`](https://github.com/sgl-project/sglang/blob/34dfaccc57d928f046317c2e3d9e16bd69837d9b/docs/docs/advanced_features/speculative_decoding.mdx)，提交 `34dfaccc57`。

[统一源码阅读索引](/articles/sd-source-map/)记录可核实的文件版本和来源。链接至框架代码不表示论文作者已经将完整原始实现发布在该框架中。

## 训练流程与数据对齐

无需神经网络训练。要构造匹配索引并决定 n、搜索窗口、候选最大长度与冲突处理规则；所有索引只应来自请求有权使用的数据。

阅读训练代码时，请记录 `input_ids`、`labels`、特征层、有效 `loss_mask` 与位置偏移。真实标签、教师分布和草稿生成 token 的来源必须分开；离线 teacher forcing 指标不能替代实际 rollout。

## 推理流程与状态提交

匹配后缀 → 找到历史位置 → 截取续写 → target 批验证 → 接受连续前缀 → 更新历史。匹配不到或候选过短时，回到常规 decode，不应为了保持投机率而硬造候选。

下面是**本站自写的机制伪代码**，省略批处理、EOS、张量并行和具体 API，不能当作可直接运行的作者实现：

```python
match = longest_token_suffix_match(prefix, prompt_history)
if match is None:
    return target_decode_one(prefix)
candidate = continuation_after(match)[:max_draft_tokens]
result = target_verify(prefix, candidate)
commit_verified_prefix(result)
```

## 代码讲解：按调用链读，而不是按文件名猜

先读 PLD README 的候选构造与任务例子，再读框架 ngram 方法入口。检查候选从哪里来、是否跨请求共享、检索是在 CPU 还是 GPU、最大 n 与 K 如何限制。不要将 ngram_gpu 仅因名字相似就当成所有 CPU 路径参数的无缝替代。

一次完整阅读至少画出三份状态：候选 token、proposal 分布、KV／特征缓存。接受的 token 数、补采的校正 token 和缓存已经算到的位置不一定相同；必须以实现契约核对，不能凭长度变量名称推断。

## vLLM 与 SGLang 集成情况

vLLM 配置包含 ngram／ngram_gpu；SGLang 提供 NGRAM，所读指南标注 CUDA-only。具体设备和 sampler 支持按锁定版本检查。

上述判断基于本轮核对的主分支文档／代码，不代表任意稳定版都支持。启用前记录 target／draft revision、框架 commit、attention backend、并行模式和采样配置；[综述支持矩阵](/articles/speculative-decoding/#framework-support)提供横向对照。

## 加速收益：证据与适用条件

代码、长文摘取、结构重复任务更可能受益；开放式闲聊未必高命中。衡量 lookup 命中率与命中后的接受率，二者分母不同；零训练成本不等于零检索成本。

比较时固定 target、精度、任务、输入输出长度、温度／top-p 和负载。至少记录每轮有效输出、draft／verify／KV 管理耗时、TPOT、P99 和满足 SLO 的吞吐。接受率高不必然更快。

## 正确性、失败与降级

反复返回相同但无效的续写会浪费验证预算。跨租户共享历史文本还会引入数据隔离问题。

“降级”优先选择保留目标采样语义的普通 target decode；不要通过放宽接受门槛伪造加速。可近似的质量策略应另行标注，不能继续称为严格无损。

## 学习实验怎么做

对代码改写、原文摘取、自由对话分别测试命中率；关闭 lookup 与开启 lookup 在相同负载下比较 TPOT。

先验证正确性，再测性能。记录冷／热状态、版本和数据划分；图表没有实验条件时不用于方法排名。未跑的实验保持“计划”，不填占位通过结果。



## 延伸阅读

- [返回综述与完整系列](/articles/speculative-decoding/)
- [技术时间轴](/articles/speculative-decoding/timeline.html)
- [原始仓库与源码阅读索引](/articles/sd-source-map/)
- 上一篇：[经典 SD / Speculative Sampling](/articles/sd-classic/)
- 下一篇：[Suffix Decoding](/articles/sd-suffix/)
