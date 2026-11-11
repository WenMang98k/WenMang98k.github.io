---
title: "自适应验证预算：vLLM：原理、源码与推理落地"
description: "以候选前缀存活概率与硬件成本模型分配验证槽位。"
date: 2026-11-11
tags: ["投机解码", "训练与系统优化"]
draft: false
series: speculative-decoding
seriesOrder: 33
---

> **本文是源码学习初稿，不是性能复现报告。**已读取内容、未确认的原始仓库和未核实的数字均明确区分。原作者结论只在相应实验条件下引用；本站未运行 GPU 加速基准。

## 这项技术解决什么问题

即使草稿已经提出较长候选，也不必全部送 target 验证。vLLM 文档描述用每请求／位置的 survival probability 估计收益，再结合启动时 profile 的成本模型分配全局验证预算。

## 原论文、仓库与版本

- 代码入口：[vllm-project/vllm](https://github.com/vllm-project/vllm)。这是已核实的训练／服务框架实现，不冒充独立论文作者的原始仓库。
- 已读取源码／文档：[`docs/features/speculative_decoding/README.md`](https://github.com/vllm-project/vllm/blob/2ee8250eaf67059b4e8bc3c1bfb4cfb1c88892a7/docs/features/speculative_decoding/README.md)，提交 `2ee8250eaf`。
- 已读取源码／文档：[`vllm/config/speculative.py`](https://github.com/vllm-project/vllm/blob/2ee8250eaf67059b4e8bc3c1bfb4cfb1c88892a7/vllm/config/speculative.py)，提交 `2ee8250eaf`。

[统一源码阅读索引](/articles/sd-source-map/)记录可核实的文件版本和来源。链接至框架代码不表示论文作者已经将完整原始实现发布在该框架中。

## 训练流程与数据对齐

不一定训练新模型。DSpark 可用专用 confidence head；其他模型草稿可用在线校准估计器，suffix 用历史匹配统计。置信度来源不同，不应使用同一种校准假设。

阅读训练代码时，请记录 `input_ids`、`labels`、特征层、有效 `loss_mask` 与位置偏移。真实标签、教师分布和草稿生成 token 的来源必须分开；离线 teacher forcing 指标不能替代实际 rollout。

## 推理流程与状态提交

草稿 → 估计每位置条件接受率 → 连乘形成前缀存活价值 → 在成本预算内选择槽位 → target 正确验证 → 反馈统计。confidence 是价值预测，不是接受授权。

下面是**本站自写的机制伪代码**，省略批处理、EOS、张量并行和具体 API，不能当作可直接运行的作者实现：

```python
conditional_acceptance = estimate_from_confidence_or_history(draft)
prefix_survival = cumulative_product(conditional_acceptance)
budget = cost_model.choose_budget(prefix_survival, current_load)
slots = admit_supported_prefix_slots(prefix_survival, budget)
result = exact_target_verify(draft, slots, actual_proposal)
update_online_calibration(result)
```

## 代码讲解：按调用链读，而不是按文件名猜

从 enable_adaptive_verification 参数进入配置与专门文档。检查 profile 使用的 KV 长度、是否支持设备端变化的 query length、CUDA graph 条件，以及 sampler 收到的实际 proposal。

一次完整阅读至少画出三份状态：候选 token、proposal 分布、KV／特征缓存。接受的 token 数、补采的校正 token 和缓存已经算到的位置不一定相同；必须以实现契约核对，不能凭长度变量名称推断。

## vLLM 与 SGLang 集成情况

vLLM 专门文档列出 attention backend、full CUDA graph、LoRA／PP 等约束。不同主分支文档可能不同步，运行前以锁定版本检查为准。SGLang 的自适应草稿不是这一机制的同名替代。

上述判断基于本轮核对的主分支文档／代码，不代表任意稳定版都支持。启用前记录 target／draft revision、框架 commit、attention backend、并行模式和采样配置；[综述支持矩阵](/articles/speculative-decoding/#framework-support)提供横向对照。

## 加速收益：证据与适用条件

低并发可保留长块，高负载减少低价值验证。本文不提供固定倍数；应在 SLO 下比较 goodput，拆出 profiling 的一次性成本。

比较时固定 target、精度、任务、输入输出长度、温度／top-p 和负载。至少记录每轮有效输出、draft／verify／KV 管理耗时、TPOT、P99 和满足 SLO 的吞吐。接受率高不必然更快。

## 正确性、失败与降级

若后端不能处理设备端 query 长度，应关闭机制或更换兼容配置。依据具体候选做筛选还要检查选择导致的 proposal 条件变化，不能泛化为任意裁剪无损。

“降级”优先选择保留目标采样语义的普通 target decode；不要通过放宽接受门槛伪造加速。可近似的质量策略应另行标注，不能继续称为严格无损。

## 学习实验怎么做

在并发1/8/32与流量突发下测接受长度、被验证槽位数、profile 误差、吞吐和 p99；普通 decode 也是必选基线。

先验证正确性，再测性能。记录冷／热状态、版本和数据划分；图表没有实验条件时不用于方法排名。未跑的实验保持“计划”，不填占位通过结果。



## 延伸阅读

- [返回综述与完整系列](/articles/speculative-decoding/)
- [技术时间轴](/articles/speculative-decoding/timeline.html)
- [原始仓库与源码阅读索引](/articles/sd-source-map/)
- 上一篇：[自适应草稿步数：SGLang](/articles/sd-adaptive-draft/)
- 下一篇：[D-PACE 训练目标](/articles/sd-dpace/)
