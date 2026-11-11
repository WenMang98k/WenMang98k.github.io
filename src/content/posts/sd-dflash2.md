---
title: "DFlash2：原理、源码与推理落地"
description: "动态分组卷积与低秩候选路径选择，改善块内关联。"
date: 2026-11-11
tags: ["投机解码", "块并行草稿"]
draft: false
series: speculative-decoding
seriesOrder: 23
---

> **本文是源码学习初稿，不是性能复现报告。**已读取内容、未确认的原始仓库和未核实的数字均明确区分。原作者结论只在相应实验条件下引用；本站未运行 GPU 加速基准。

## 这项技术解决什么问题

在 DFlash 草稿 attention／MLP 子层周围加入分组动态卷积，并在各槽位 unary top-k 上做依赖前序 token 的低秩候选选择。将昂贵主干并行化，同时用便宜结构补 token 间关系。

## 原论文、仓库与版本

- 代码入口：[z-lab/dflash](https://github.com/z-lab/dflash)。以原项目版本与说明为准。
- 已读取源码／文档：[`specforge/modeling/draft/dflash2.py`](https://github.com/sgl-project/SpecForge/blob/ce3ef8fa05ea397881634590768d0b0150ec24f9/specforge/modeling/draft/dflash2.py)，提交 `ce3ef8fa05`。

[统一源码阅读索引](/articles/sd-source-map/)记录可核实的文件版本和来源。链接至框架代码不表示论文作者已经将完整原始实现发布在该框架中。

## 训练流程与数据对齐

在基础块预测目标外，selector 用严格 unary top-k 集合上的分类交叉熵；若正确 token 不在候选集，该位置不贡献 selector CE。卷积与 selector 有初始 no-op 设计，避免新增模块一开始破坏基础草稿。

阅读训练代码时，请记录 `input_ids`、`labels`、特征层、有效 `loss_mask` 与位置偏移。真实标签、教师分布和草稿生成 token 的来源必须分开；离线 teacher forcing 指标不能替代实际 rollout。

## 推理流程与状态提交

并行主干与局部卷积 → 每位置候选池 → 轻量路径选择 → target 验证 → 前缀提交。路径选择可能含便宜顺序部分；不能把“主干并行”说成“没有任何依赖”。

下面是**本站自写的机制伪代码**，省略批处理、EOS、张量并行和具体 API，不能当作可直接运行的作者实现：

```python
hidden = parallel_backbone_with_grouped_dynamic_conv(context, slots)
unary = shared_lm_head(hidden)
candidates = strict_topk_per_position(unary, selector_top_k)
path = low_rank_conditional_path_selector(candidates, hidden)
verified = target_verify(path)
commit_verified_prefix(verified)
```

## 代码讲解：按调用链读，而不是按文件名猜

先看 DFlashGroupedConv.prepare／finish 的输入输出，再看 Qwen3DFlash2DecoderLayer.forward；随后跟 DFlash2DraftModel._init_draft_head、_sample_draft_tokens。conv_group_size 与 hidden_size 的整除关系和 selector_top_k 是必须检查的配置。

一次完整阅读至少画出三份状态：候选 token、proposal 分布、KV／特征缓存。接受的 token 数、补采的校正 token 和缓存已经算到的位置不一定相同；必须以实现契约核对，不能凭长度变量名称推断。

### 选择器的覆盖上限

设每槽候选集为 C_i，selector 只能从 C_i 里选择。在所读实现中，真实 token 不属于 C_i 时，该位置不参加 selector CE。应同时记录三种量：unary top-k 覆盖率、在已覆盖位置上的 selector 准确率、整条路径的连续存活率。

```python
candidates = unary_topk(base_logits, k)
covered = candidates.contains(gold_token)
selector_loss = categorical_ce_on_candidate_set(selector_scores, gold_token)
selector_loss = reduce_with_valid_weight(selector_loss, covered & loss_mask)
```

若只汇报已覆盖子集的高准确率，却不汇报覆盖率，容易产生“选择器很好，接受前缀却很短”的假象。扩大 k 能改善覆盖，也会增加候选评分成本。

## vLLM 与 SGLang 集成情况

两框架沿 DFlash 路径加载兼容 checkpoint；SGLang 算法名仍为 DFLASH，不必存在 DFLASH2 独立枚举。是否有最新 loader／卷积 kernel 取决于 commit。

上述判断基于本轮核对的主分支文档／代码，不代表任意稳定版都支持。启用前记录 target／draft revision、框架 commit、attention backend、并行模式和采样配置；[综述支持矩阵](/articles/speculative-decoding/#framework-support)提供横向对照。

## 加速收益：证据与适用条件

候选池覆盖率决定 selector 的上限；若 unary top-k 不包含正确 token，再强 selector 也无法恢复。本文没有同条件倍数，建议把 unary recall 与 selector conditional accuracy 单列。

比较时固定 target、精度、任务、输入输出长度、温度／top-p 和负载。至少记录每轮有效输出、draft／verify／KV 管理耗时、TPOT、P99 和满足 SLO 的吞吐。接受率高不必然更快。

## 正确性、失败与降级

导出漏掉卷积／selector 配置可能让模型表现像普通 DFlash。需要严格检查权重键与结构；不一致时拒绝启动。

“降级”优先选择保留目标采样语义的普通 target decode；不要通过放宽接受门槛伪造加速。可近似的质量策略应另行标注，不能继续称为严格无损。

## 学习实验怎么做

对基础 DFlash、只加卷积、只加 selector、全部启用做消融，保持 target、采样与节点预算相同。

先验证正确性，再测性能。记录冷／热状态、版本和数据划分；图表没有实验条件时不用于方法排名。未跑的实验保持“计划”，不填占位通过结果。



## 延伸阅读

- [返回综述与完整系列](/articles/speculative-decoding/)
- [技术时间轴](/articles/speculative-decoding/timeline.html)
- [原始仓库与源码阅读索引](/articles/sd-source-map/)
- 上一篇：[DFlash](/articles/sd-dflash/)
- 下一篇：[DSpark](/articles/sd-dspark/)
