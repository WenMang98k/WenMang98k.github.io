---
title: "ReDrafter / Recurrent Drafter：原理、源码与推理落地"
description: "用轻量循环状态提出多条候选，并通过动态树减少重复验证。"
date: 2026-11-11
tags: ["投机解码", "多头与特征"]
draft: false
series: speculative-decoding
seriesOrder: 11
---

> **本文是源码学习初稿，不是性能复现报告。**已读取内容、未确认的原始仓库和未核实的数字均明确区分。原作者结论只在相应实验条件下引用；本站未运行 GPU 加速基准。

## 这项技术解决什么问题

循环 drafter 利用 target 隐藏状态与候选 token 递归展开，beam search 形成多个续写。共享前缀可以在验证时合并，避免每条 beam 都完整重算相同路径。名字中的 recurrent 不自动表示 LSTM。

## 原论文、仓库与版本

- 代码入口：[apple/ml-recurrent-drafter](https://github.com/apple/ml-recurrent-drafter)。以原项目版本与说明为准。
- 原论文：[arXiv:2403.09919](https://arxiv.org/abs/2403.09919)。论文全文本轮受访问限制；原文与框架实现的证据范围见下文。
- 已读取源码／文档：[`recurrent_drafting/recurrent_drafting.py`](https://github.com/apple/ml-recurrent-drafter/blob/0ae66cb657b847c3279feb4fa58f57e233ca9c6c/recurrent_drafting/recurrent_drafting.py)，提交 `0ae66cb657`。

[统一源码阅读索引](/articles/sd-source-map/)记录可核实的文件版本和来源。链接至框架代码不表示论文作者已经将完整原始实现发布在该框架中。

## 训练流程与数据对齐

冻结基座，仅训练草稿；作者仓库使用 ShareGPT 训练并提供评估脚本。目标是低成本地拟合 target 附近的续写分布；本轮未成功获取训练脚本的猜测路径，因此不虚构其精确损失函数。

阅读训练代码时，请记录 `input_ids`、`labels`、特征层、有效 `loss_mask` 与位置偏移。真实标签、教师分布和草稿生成 token 的来源必须分开；离线 teacher forcing 指标不能替代实际 rollout。

## 推理流程与状态提交

初始化循环状态 → beam 候选 → 共享前缀树处理 → _verify_candidates → 接受路径选择 → _update_kv_cache_and_input_ids → 开启下一轮。

下面是**本站自写的机制伪代码**，省略批处理、EOS、张量并行和具体 API，不能当作可直接运行的作者实现：

```python
state = target_state(prefix)
beams = recurrent_drafter.beam_search(state, beam_width, depth)
packed_tree, restore_indices = merge_shared_prefixes(beams)
verified = target_verify_tree(packed_tree)
chosen, accepted = choose_candidate(verified, sampling_mode)
update_kv_and_input_ids(chosen, accepted, restore_indices)
```

## 代码讲解：按调用链读，而不是按文件名猜

从 RecurrentDrafting.generate／_generate 进入；对比 _greedy_accept_candidate_tokens 与 _accept_candidate_tokens；最后阅读 _update_kv_cache_and_input_ids。缓存更新把“选择哪条 beam”变成真实内存索引，是学习本实现最有价值的一段。

一次完整阅读至少画出三份状态：候选 token、proposal 分布、KV／特征缓存。接受的 token 数、补采的校正 token 和缓存已经算到的位置不一定相同；必须以实现契约核对，不能凭长度变量名称推断。

### KV 更新为什么比 beam 数更值得先读

候选搜索输出的是多条路径；目标验证张量里可能有压缩后的共享节点；下一轮需要的却只是一条已提交路径。`_update_kv_cache_and_input_ids` 将这三种布局联系起来。

先画一个例子：路径 `[a,b,c]` 与 `[a,b,d]` 共享 a、b。选中第二条并接受到 d 时，提交的是其祖先链，而不是原始 beam 张量里的前四个内存槽。若拒绝在 b 之后，c、d 的缓存都不能被下一轮当成真实前缀。

这一映射还要与 left padding、batch 内各请求接受长度对应。先把索引验证清楚，再增加 beam width，否则更快地复制错误缓存只会让问题难定位。

## vLLM 与 SGLang 集成情况

主要依据 Apple 原始实现；最新两框架默认索引未确认同名入口。可与 EAGLE 类实现比较，但不能替换模型配置后认为语义一致。

上述判断基于本轮核对的主分支文档／代码，不代表任意稳定版都支持。启用前记录 target／draft revision、框架 commit、attention backend、并行模式和采样配置；[综述支持矩阵](/articles/speculative-decoding/#framework-support)提供横向对照。

## 加速收益：证据与适用条件

收益来自便宜草稿和共享前缀压缩两部分。作者 README 的训练耗时不是推理加速倍数；本文没有引用未完整核实的推理倍数。

比较时固定 target、精度、任务、输入输出长度、温度／top-p 和负载。至少记录每轮有效输出、draft／verify／KV 管理耗时、TPOT、P99 和满足 SLO 的吞吐。接受率高不必然更快。

## 正确性、失败与降级

作者 README Known Issues 明确指出非贪心实现可能不能做到精确分布匹配；贪心不受该条限制。生产随机采样不能直接忽略这一说明。

“降级”优先选择保留目标采样语义的普通 target decode；不要通过放宽接受门槛伪造加速。可近似的质量策略应另行标注，不能继续称为严格无损。

## 学习实验怎么做

扫描 beam_width、beam_length，记录压缩前后节点数、验证时间与接受长度；单列贪心和非贪心结果。

先验证正确性，再测性能。记录冷／热状态、版本和数据划分；图表没有实验条件时不用于方法排名。未跑的实验保持“计划”，不填占位通过结果。



## 延伸阅读

- [返回综述与完整系列](/articles/speculative-decoding/)
- [技术时间轴](/articles/speculative-decoding/timeline.html)
- [原始仓库与源码阅读索引](/articles/sd-source-map/)
- 上一篇：[Hydra](/articles/sd-hydra/)
- 下一篇：[EAGLE](/articles/sd-eagle/)
