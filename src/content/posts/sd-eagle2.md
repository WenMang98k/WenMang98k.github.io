---
title: "EAGLE-2：原理、源码与推理落地"
description: "把固定候选树改成上下文相关的动态扩展预算。"
date: 2026-11-11
tags: ["投机解码", "多头与特征"]
draft: false
series: speculative-decoding
seriesOrder: 13
---

> **本文是源码学习初稿，不是性能复现报告。**已读取内容、未确认的原始仓库和未核实的数字均明确区分。原作者结论只在相应实验条件下引用；本站未运行 GPU 加速基准。

## 这项技术解决什么问题

EAGLE-2 的核心不是重新定义概率校正，而是使用草稿置信分数动态扩展更有价值的分支。不同上下文可以对应不同树形状，将有限验证节点分配给更可能被接受的路径。

## 原论文、仓库与版本

- 代码入口：[SafeAILab/EAGLE](https://github.com/SafeAILab/EAGLE)。以原项目版本与说明为准。
- 原论文：[arXiv:2406.16858](https://arxiv.org/abs/2406.16858)。论文全文本轮受访问限制；原文与框架实现的证据范围见下文。
- 已读取源码／文档：[`eagle/model/ea_model.py`](https://github.com/SafeAILab/EAGLE/blob/cb7e0841fe0c206c6ed74a197ad5e2a1f13f5a2b/eagle/model/ea_model.py)，提交 `cb7e0841fe`。
- 已读取源码／文档：[`eagle/model/utils.py`](https://github.com/SafeAILab/EAGLE/blob/cb7e0841fe0c206c6ed74a197ad5e2a1f13f5a2b/eagle/model/utils.py)，提交 `cb7e0841fe`。

[统一源码阅读索引](/articles/sd-source-map/)记录可核实的文件版本和来源。链接至框架代码不表示论文作者已经将完整原始实现发布在该框架中。

## 训练流程与数据对齐

依赖已有 EAGLE 类草稿能力；不应因为版本号改变就虚构一套完全不同的监督目标。应校验草稿分数在业务域内与实际接受率是否相关，必要时重新训练或校准。

阅读训练代码时，请记录 `input_ids`、`labels`、特征层、有效 `loss_mask` 与位置偏移。真实标签、教师分布和草稿生成 token 的来源必须分开；离线 teacher forcing 指标不能替代实际 rollout。

## 推理流程与状态提交

获取目标特征 → 对草稿扩展打分 → 在节点／深度预算内选分支 → 建树 mask 与验证布局 → target 验证并选择路径 → 更新 KV。

下面是**本站自写的机制伪代码**，省略批处理、EOS、张量并行和具体 API，不能当作可直接运行的作者实现：

```python
frontier = initial_feature_candidates(prefix)
while tree_node_count < budget:
    node = choose_high_value_frontier(frontier)
    children = extend_with_eagle_draft(node)
    frontier.update(children)
verified = target_tree_verify(pack(frontier))
commit_valid_path_and_refresh_features(verified)
```

## 代码讲解：按调用链读，而不是按文件名猜

EaModel 负责模型轮次；utils 的树索引、retrieve indices 和 posterior 完成候选到路径的映射。动态树的置信评分属于 proposal 预算策略，最终接受规则仍必须独立检查。

一次完整阅读至少画出三份状态：候选 token、proposal 分布、KV／特征缓存。接受的 token 数、补采的校正 token 和缓存已经算到的位置不一定相同；必须以实现契约核对，不能凭长度变量名称推断。

## vLLM 与 SGLang 集成情况

SGLang EAGLE 对应其 EAGLE-2 指南，可配合 FR-Spec 与受限条件下的 adaptive speculative decoding。vLLM 不以独立 eagle2 枚举证明所有 EAGLE-2 结构均可直接加载，需核对 checkpoint。

上述判断基于本轮核对的主分支文档／代码，不代表任意稳定版都支持。启用前记录 target／draft revision、框架 commit、attention backend、并行模式和采样配置；[综述支持矩阵](/articles/speculative-decoding/#framework-support)提供横向对照。

## 加速收益：证据与适用条件

官方 README 在上述 Vicuna-13B／2×RTX3090／FP16 图示环境报告约 4×普通解码、约 1.4×EAGLE-1；本文没有同硬件服务复现。

比较时固定 target、精度、任务、输入输出长度、温度／top-p 和负载。至少记录每轮有效输出、draft／verify／KV 管理耗时、TPOT、P99 和满足 SLO 的吞吐。接受率高不必然更快。

## 正确性、失败与降级

草稿 confidence 不是天然校准的概率。高负载时应减少树预算，不要只看 offline 接受长度挑树。

“降级”优先选择保留目标采样语义的普通 target decode；不要通过放宽接受门槛伪造加速。可近似的质量策略应另行标注，不能继续称为严格无损。

## 学习实验怎么做

固定验证节点数，对比固定树与动态树；按任务域观察分数校准、节点利用率和 P99 时延。

先验证正确性，再测性能。记录冷／热状态、版本和数据划分；图表没有实验条件时不用于方法排名。未跑的实验保持“计划”，不填占位通过结果。



## 延伸阅读

- [返回综述与完整系列](/articles/speculative-decoding/)
- [技术时间轴](/articles/speculative-decoding/timeline.html)
- [原始仓库与源码阅读索引](/articles/sd-source-map/)
- 上一篇：[EAGLE](/articles/sd-eagle/)
- 下一篇：[EAGLE-3](/articles/sd-eagle3/)
