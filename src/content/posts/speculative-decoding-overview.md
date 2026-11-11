---
title: "LLM 投机解码综述"
description: "一张机制地图，35 篇方法文章：从概率校正与检索草稿，到 EAGLE、MTP、DFlash、DSpark 与动态调度。"
date: 2026-11-11
tags: ["投机解码", "推理加速"]
draft: false
series: speculative-decoding
seriesRole: overview
seriesOrder: 0
---
## 先理解一轮投机，而不是先背方法名

低并发 decode 经常没有用满 GPU 计算能力。投机解码用便宜的方式先提出候选，再由昂贵目标模型批量核验，将一次 target 调用从“推进一个 token”变成“可能推进多个有效 token”。

<div class="mechanism-flow" aria-label="投机解码的四个步骤"><div><span>01 / PROPOSE</span><strong>草稿提案</strong><small>小模型、检索、多头或块预测</small></div><b aria-hidden="true">→</b><div><span>02 / VERIFY</span><strong>目标验证</strong><small>一次前向，核验多个位置</small></div><b aria-hidden="true">→</b><div><span>03 / CORRECT</span><strong>接受与校正</strong><small>提交连续前缀，拒绝时修正</small></div><b aria-hidden="true">→</b><div><span>04 / COMMIT</span><strong>状态提交</strong><small>更新合法 KV，开启下一轮</small></div></div>

验证能并行，是因为候选已经成为已知输入；最终输出仍有从左到右的条件依赖。第二个候选被拒绝后，不能继续无条件提交第三个。

> **本专题是学习记录，不是加速排行榜。**核心实现的源码链接锁定了读取提交；部分原论文全文仍受访问限制。没有跑过的 GPU 实验不会标为“本站实测”，没有核实的官方仓库也不会用第三方仓库代替。

## 五条路线，分别优化什么

| 路线 | 代表方法 | 优化目标 | 应先问的问题 |
|---|---|---|---|
| 基础与检索 | 经典 SD、n-gram、suffix、REST、Lookahead | 便宜地产生候选 | 候选来源可靠且足够便宜吗？ |
| 多头与特征 | Medusa、Hydra、ReDrafter、EAGLE 系列 | 提高低成本候选的质量 | 后续位置是否感知了前序候选？ |
| MTP 与自投机 | 原生 MTP、LayerSkip、Draft & Verify、UNO | 利用已有主干和训练能力 | 草稿与验证共享什么状态？ |
| 块并行草稿 | DFlash、DFlash2、DSpark、Domino、LiLiCorr | 减少草稿主干的串行步骤 | 并行化后如何补回块内依赖？ |
| 训练与系统优化 | 蒸馏、Sequoia、SpecExec、FR-Spec、自适应预算 | 优化训练、树与资源分配 | 单轮推进收益能否覆盖额外计算？ |

它们不是严格的替代链。检索可以与模型草稿互补，动态预算可以叠加到多种方法上；MTP 也不是 EAGLE 的“下一版本”。[技术时间轴](/articles/speculative-decoding/timeline.html)分开记录论文公开、代码发布与框架支持。

## 正确性地基：接受与残差重采样

设目标分布为 $p$、实际提案分布为 $q$，先采样 $y\sim q$。接受概率为：

$$
\alpha(y)=\min\left(1,\frac{p(y)}{q(y)}\right).
$$

拒绝时，从归一化残差分布采样替代 token：

$$
r(v)=\frac{\max(0,p(v)-q(v))}{\sum_u\max(0,p(u)-q(u))}.
$$

接受贡献 $\min(p(v),q(v))$，拒绝分支补回正差，所以最终恢复 $p(v)$。当两分布相同，拒绝概率为零，不需要计算分母为零的残差。

**贪心一致、随机分布一致、任务指标接近是三种不同保证。**Medusa 的 typical acceptance、ReDrafter 原实现的非贪心限制，以及训练后 target 与原 checkpoint 的差别，都在分篇中单独说明。

[经典 SD 分篇](/articles/sd-classic/)提供完整机制和错误检查；[本站有限词表参考代码](https://github.com/WenMang98k/WenMang98k.github.io/tree/main/research/speculative-decoding)用于验证概率恒等式，不是 GPU 服务实现。

## 收益模型：接受率高，不一定更快

设候选数为 $K$，接受候选数为 $A$，在未触发停止条件时通常额外产生校正或 bonus token。近似加速比：

$$
S\approx\frac{(\mathbb E[A]+1)T_{\mathrm{AR}}}{T_{\mathrm{draft}}(K)+T_{\mathrm{verify}}(K)+T_{\mathrm{manage}}}.
$$

管理成本包含采样、树索引、KV 提交与回退、通信和调度。EOS 与最大输出长度会改变实际推进量。

草稿更准但更贵，可能负收益；候选更长但后段很少被接受，也可能负收益。低并发有计算余量，高并发时验证候选可能挤占真实请求的算力，所以必须报告 TPOT、P99 和满足 SLO 的吞吐，而不是只报告接受长度。

本文引用的倍数只来自原项目说明，并保留硬件／模型等已知条件；未完整披露条件的数字不用于横向排名。[源码与收益证据索引](/articles/sd-source-map/)说明证据等级。

<h2 id="framework-support">框架支持：接口存在不等于任意版本都可用</h2>

以下为 **2026-11-11 核对的公开主分支**，不是稳定发行版兼容性承诺。

| 方法 | vLLM | SGLang | 主要前提 |
|---|---|---|---|
| 独立小模型 | `draft_model` | `STANDALONE` | 模型、tokenizer 与采样兼容 |
| EAGLE 类 | `eagle`、`eagle3` | `EAGLE`、`EAGLE3` | 对应 target 特征与草稿权重 |
| MTP | `mtp` 与模型专用类型 | 常沿 `EAGLE`／相关别名 | 原生模块与 loader 匹配 |
| n-gram | `ngram`／`ngram_gpu` | `NGRAM` | 设备与实现约束；SGLang 文档注明 CUDA-only |
| suffix | `suffix` | 独立同名入口未核实 | 来源、索引与历史统计 |
| DFlash 家族 | `dflash` | `DFLASH` | checkpoint 架构与 attention backend |
| DSpark | `dspark` | `DSPARK` | Markov／confidence 配置与服务支持 |
| 自适应机制 | Dynamic SD／Adaptive Verification | Adaptive Speculative Decoding | 两者优化位置和限制不同 |
| 扩展 | LiLiCorr 文档 | FR-Spec、UNO、LiLiCorr 文档 | 部分缺公开权重或有功能限制 |

Medusa、Hydra、ReDrafter 等仍是理解机制的重要背景，但没有因历史上出现过集成就将其标为当前默认可用。训练框架支持也不等于 serving loader 和 kernel 全部支持。

## 阅读源码：先沿调用链，再看 kernel

建议依次标出 `draft → verify → accept → commit` 四个边界，再画 token、proposal 分布、KV／feature 三种状态的下标。训练侧另画 label shift 与 loss mask。

- **基础路线**：[经典 SD](/articles/sd-classic/) → [n-gram](/articles/sd-ngram/)／[suffix](/articles/sd-suffix/)。
- **理解依赖**：[Medusa](/articles/sd-medusa/) → [Hydra](/articles/sd-hydra/) → [ReDrafter](/articles/sd-redrafter/)。
- **特征路线**：[EAGLE](/articles/sd-eagle/) → [EAGLE-2](/articles/sd-eagle2/) → [EAGLE-3](/articles/sd-eagle3/)。
- **并行路线**：[MTP](/articles/sd-mtp/) → [DFlash](/articles/sd-dflash/) → [DFlash2](/articles/sd-dflash2/)／[DSpark](/articles/sd-dspark/)。
- **系统路线**：[SGLang 自适应步数](/articles/sd-adaptive-draft/)与[vLLM 自适应验证](/articles/sd-adaptive-verify/)。

[完整 35 篇目录](#series-directory)按机制分组；[源码阅读索引](/articles/sd-source-map/)将文章、原仓库和实际读取文件关联起来。官方代码保留在上游，不将本站伪代码伪装成作者实现。

## 一份可信的实验报告应该包含什么

1. **锁定版本**：target／draft revision、tokenizer、框架 commit、设备、精度、TP／PP 与 attention backend。
2. **固定负载**：任务、输入／输出长度、并发／到达率、温度、top-p、停止条件。
3. **验证正确性**：贪心序列、随机概率分布、首位拒绝、中途拒绝、全接受、EOS 与缓存回退。
4. **拆解成本**：draft、target verify、输出头／selector、KV 操作与通信。
5. **报告完整效果**：有效输出 token、接受长度、TPOT、P99、吞吐与 SLO goodput。
6. **提供降级**：负收益、不支持的采样约束或权重不匹配时，安全回到普通 target decode。

本轮完成的是来源核对、教学参考实现与网站构建检查，没有在 GPU 上运行这些方法的性能对比。

## 资料与后续更新

- [经典 SD 论文](https://arxiv.org/abs/2211.17192)／[Speculative Sampling](https://arxiv.org/abs/2302.01318)。
- [EAGLE 官方仓库](https://github.com/SafeAILab/EAGLE)、[Medusa](https://github.com/FasterDecoding/Medusa)、[Hydra](https://github.com/zankner/Hydra)、[Apple ReDrafter](https://github.com/apple/ml-recurrent-drafter)。
- [DFlash 官方仓库](https://github.com/z-lab/dflash)、[SpecForge](https://github.com/sgl-project/SpecForge)、[vLLM](https://github.com/vllm-project/vllm)、[SGLang](https://github.com/sgl-project/sglang)。
- [原始长版综述与参考文献](https://github.com/WenMang98k/WenMang98k.github.io/blob/main/articles/speculative-decoding/README.md)、[资料核对清单](/articles/speculative-decoding/source-notes.md)。

各分篇区分作者原始实现、框架移植与第三方复现。对尚未核实的论文细节，保留明确边界并继续补充，不以同名仓库、搜索摘要或伪造实验替代证据。
