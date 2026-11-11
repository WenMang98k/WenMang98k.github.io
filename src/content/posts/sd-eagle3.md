---
title: "EAGLE-3：原理、源码与推理落地"
description: "多层特征融合与 training-time testing，让训练更贴近实际草稿 rollout。"
date: 2026-11-11
tags: ["投机解码", "多头与特征"]
draft: false
series: speculative-decoding
seriesOrder: 14
---

> **本文是源码学习初稿，不是性能复现报告。**已读取内容、未确认的原始仓库和未核实的数字均明确区分。原作者结论只在相应实验条件下引用；本站未运行 GPU 加速基准。

## 这项技术解决什么问题

融合目标模型低、中、高层特征，去掉原版必须精确预测未来特征的约束，让草稿更直接服务于 token 预测。training-time testing 让训练过程暴露于更接近递归生成的状态，而不是把测试集拿来训练。

## 原论文、仓库与版本

- 代码入口：[SafeAILab/EAGLE](https://github.com/SafeAILab/EAGLE)。以原项目版本与说明为准。
- 原论文：[arXiv:2503.01840](https://arxiv.org/abs/2503.01840)。论文全文本轮受访问限制；原文与框架实现的证据范围见下文。
- 已读取源码／文档：[`eagle/model/ea_model.py`](https://github.com/SafeAILab/EAGLE/blob/cb7e0841fe0c206c6ed74a197ad5e2a1f13f5a2b/eagle/model/ea_model.py)，提交 `cb7e0841fe`。
- 已读取源码／文档：[`eagle/train/main.py`](https://github.com/SafeAILab/EAGLE/blob/cb7e0841fe0c206c6ed74a197ad5e2a1f13f5a2b/eagle/train/main.py)，提交 `cb7e0841fe`。

[统一源码阅读索引](/articles/sd-source-map/)记录可核实的文件版本和来源。链接至框架代码不表示论文作者已经将完整原始实现发布在该框架中。

## 训练流程与数据对齐

冻结 target，提取指定层特征；融合并输入轻量草稿；按配置在训练中展开多个预测步骤，对 token／教师分布监督。读取旧训练入口时必须注意该脚本可能属于特定代际，生产训练建议结合 SpecForge 当前策略与模型定义。

阅读训练代码时，请记录 `input_ids`、`labels`、特征层、有效 `loss_mask` 与位置偏移。真实标签、教师分布和草稿生成 token 的来源必须分开；离线 teacher forcing 指标不能替代实际 rollout。

## 推理流程与状态提交

prefill 采集多层辅助状态 → 草稿递归提案 → 一次目标验证 → 取得被接受前缀对应的真实特征 → 修正草稿状态与缓存。

下面是**本站自写的机制伪代码**，省略批处理、EOS、张量并行和具体 API，不能当作可直接运行的作者实现：

```python
features = target.capture(prefix, layers=selected_layers)
state = fuse_multilevel_features(features)
proposal = rollout_draft(state, known_tokens, depth=K)
verified = target.verify(proposal)
commit_verified_prefix(verified)
refresh_with_verified_multilevel_features(verified)
# Training rollout must not silently consume unknown future tokens.
```

## 代码讲解：按调用链读，而不是按文件名猜

先从 EaModel 的草稿入口确认特征张量，再追踪 training-time 流程是否用预测状态继续展开。对训练文件里的 compute_loss、loss_mask 与位移做记录；“不回归未来特征”不等于“不使用 target 特征”。

一次完整阅读至少画出三份状态：候选 token、proposal 分布、KV／特征缓存。接受的 token 数、补采的校正 token 和缓存已经算到的位置不一定相同；必须以实现契约核对，不能凭长度变量名称推断。

### 一次 rollout 应记录哪些张量

| 状态 | 需要记录的语义 | 常见误读 |
|---|---|---|
| target 多层特征 | 来自哪个已知 token、哪些层 | 把预测特征当作已验证真特征 |
| 草稿输入 token | 当前步骤实际可见 token | 训练偷看未来 token 后宣称 rollout 很准 |
| token label／教师分布 | 与预测位置的 shift | 把所有步骤对齐到同一标签 |
| 有效 mask | padding、无监督区域、rollout 可达位置 | 用 padding 数量改变损失缩放 |

实际训练时，记录 rollout 的前两步输入来源，比只记录一个总 loss 更容易发现错误。不同训练入口的特征协议有代际差异，先核对模型定义再复用脚本。

## vLLM 与 SGLang 集成情况

vLLM: eagle3；SGLang: EAGLE3。SpecForge 提供训练到 SGLang serving 的路径。支持某算法不等于支持任意 feature layer 配置。

上述判断基于本轮核对的主分支文档／代码，不代表任意稳定版都支持。启用前记录 target／draft revision、框架 commit、attention backend、并行模式和采样配置；[综述支持矩阵](/articles/speculative-decoding/#framework-support)提供横向对照。

## 加速收益：证据与适用条件

官方 README 在 Vicuna-13B／FP16／2×RTX3090 的图示设置报告约 5.6×普通 decode。这是原作者特定基线数据，不等于相对最新框架同配置有 5.6×。

比较时固定 target、精度、任务、输入输出长度、温度／top-p 和负载。至少记录每轮有效输出、draft／verify／KV 管理耗时、TPOT、P99 和满足 SLO 的吞吐。接受率高不必然更快。

## 正确性、失败与降级

target 更新、量化或 chat template 改变，都可能损伤草稿对齐。无法获得约定层特征或导出不兼容时，应拒绝加载或关闭草稿。

“降级”优先选择保留目标采样语义的普通 target decode；不要通过放宽接受门槛伪造加速。可近似的质量策略应另行标注，不能继续称为严格无损。

## 学习实验怎么做

比较 teacher-forced accuracy 与真实 rollout 前缀长度；报告不同上下文长度、并发和采样温度下的收益。

先验证正确性，再测性能。记录冷／热状态、版本和数据划分；图表没有实验条件时不用于方法排名。未跑的实验保持“计划”，不填占位通过结果。



## 延伸阅读

- [返回综述与完整系列](/articles/speculative-decoding/)
- [技术时间轴](/articles/speculative-decoding/timeline.html)
- [原始仓库与源码阅读索引](/articles/sd-source-map/)
- 上一篇：[EAGLE-2](/articles/sd-eagle2/)
- 下一篇：[P-EAGLE](/articles/sd-peagle/)
