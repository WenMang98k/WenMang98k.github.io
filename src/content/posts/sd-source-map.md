---
title: "投机解码：原始仓库与源码阅读索引"
description: "把 35 篇方法文章、论文、原始仓库、框架移植和已读取源码关联起来，保留版本、证据范围与学习入口。"
date: 2026-11-11
tags: ["投机解码", "源码阅读"]
draft: false
series: speculative-decoding
seriesRole: resource
seriesOrder: 100
---

## 如何使用这个索引

先从[综述](/articles/speculative-decoding/)选路线，再进入单篇的“代码讲解”。这里把原始实现、框架移植和未确认来源分开列出。**链接到代码不等于已审计整个仓库；“官方框架”也不等于“该论文作者原始实现”。**

本站不整仓镜像官方项目，以免丢失版本与许可证关系或把大文件带入网站。已读取的关键文件提供固定 commit 链接与校验记录，方便回到同一版本学习；教学伪代码和参考采样器为本站原创。

## 35 篇文章与原始资料

| 方法文章 | 代码入口与来源性质 | 论文入口 |
|---|---|---|
| [LSTM / RNN 轻量草稿](/articles/sd-lstm/) | 统一作者实现未确认；循环架构不等于独立算法 | 未猜测论文链接 |
| [经典 SD / Speculative Sampling](/articles/sd-classic/) | 作者仓库未确认；详见文章限制 | [arXiv:2211.17192](https://arxiv.org/abs/2211.17192) |
| [N-gram / Prompt Lookup](/articles/sd-ngram/) | [apoorvumang/prompt-lookup-decoding](https://github.com/apoorvumang/prompt-lookup-decoding) · 原项目入口 | 以项目文档为准，未猜测论文链接 |
| [Suffix Decoding](/articles/sd-suffix/) | [vllm-project/vllm](https://github.com/vllm-project/vllm) · 框架实现，非独立作者仓库 | 以项目文档为准，未猜测论文链接 |
| [REST](/articles/sd-rest/) | [FasterDecoding/REST](https://github.com/FasterDecoding/REST) · 原项目入口 | [arXiv:2311.08252](https://arxiv.org/abs/2311.08252) |
| [Lookahead Decoding](/articles/sd-lookahead/) | [hao-ai-lab/LookaheadDecoding](https://github.com/hao-ai-lab/LookaheadDecoding) · 原项目入口 | [arXiv:2402.02057](https://arxiv.org/abs/2402.02057) |
| [Token Recycling](/articles/sd-recycling/) | 作者仓库未确认；详见文章限制 | [arXiv:2408.08696](https://arxiv.org/abs/2408.08696) |
| [Blockwise Parallel Decoding](/articles/sd-blockwise/) | 作者仓库未确认；详见文章限制 | [arXiv:1811.03115](https://arxiv.org/abs/1811.03115) |
| [早期 SpecDec / Seq2seq 投机执行](/articles/sd-early-specdec/) | 作者仓库未确认；详见文章限制 | [arXiv:2203.16487](https://arxiv.org/abs/2203.16487) |
| [Medusa](/articles/sd-medusa/) | [FasterDecoding/Medusa](https://github.com/FasterDecoding/Medusa) · 原项目入口 | [arXiv:2401.10774](https://arxiv.org/abs/2401.10774) |
| [Hydra](/articles/sd-hydra/) | [zankner/Hydra](https://github.com/zankner/Hydra) · 原项目入口 | [arXiv:2402.05109](https://arxiv.org/abs/2402.05109) |
| [ReDrafter / Recurrent Drafter](/articles/sd-redrafter/) | [apple/ml-recurrent-drafter](https://github.com/apple/ml-recurrent-drafter) · 原项目入口 | [arXiv:2403.09919](https://arxiv.org/abs/2403.09919) |
| [EAGLE](/articles/sd-eagle/) | [SafeAILab/EAGLE](https://github.com/SafeAILab/EAGLE) · 原项目入口 | [arXiv:2401.15077](https://arxiv.org/abs/2401.15077) |
| [EAGLE-2](/articles/sd-eagle2/) | [SafeAILab/EAGLE](https://github.com/SafeAILab/EAGLE) · 原项目入口 | [arXiv:2406.16858](https://arxiv.org/abs/2406.16858) |
| [EAGLE-3](/articles/sd-eagle3/) | [SafeAILab/EAGLE](https://github.com/SafeAILab/EAGLE) · 原项目入口 | [arXiv:2503.01840](https://arxiv.org/abs/2503.01840) |
| [P-EAGLE](/articles/sd-peagle/) | [sgl-project/SpecForge](https://github.com/sgl-project/SpecForge) · 框架实现，非独立作者仓库 | [arXiv:2602.01469](https://arxiv.org/abs/2602.01469) |
| [EAGLE3.1 实现追踪](/articles/sd-eagle31/) | [sgl-project/SpecForge](https://github.com/sgl-project/SpecForge) · 框架实现，非独立作者仓库 | 以项目文档为准，未猜测论文链接 |
| [Multi-Token Prediction / MTP](/articles/sd-mtp/) | [deepseek-ai/DeepSeek-V3](https://github.com/deepseek-ai/DeepSeek-V3) · 原项目入口 | [arXiv:2412.19437](https://arxiv.org/abs/2412.19437) |
| [Draft & Verify](/articles/sd-draftverify/) | [dilab-zju/self-speculative-decoding](https://github.com/dilab-zju/self-speculative-decoding) · 作者实现 | [arXiv:2309.08168](https://arxiv.org/abs/2309.08168) |
| [LayerSkip](/articles/sd-layerskip/) | [facebookresearch/LayerSkip](https://github.com/facebookresearch/LayerSkip) · 原项目入口 | [arXiv:2404.16710](https://arxiv.org/abs/2404.16710) |
| [UNO](/articles/sd-uno/) | [sgl-project/sglang](https://github.com/sgl-project/sglang) · 框架实现，非独立作者仓库 | 以项目文档为准，未猜测论文链接 |
| [DFlash](/articles/sd-dflash/) | [z-lab/dflash](https://github.com/z-lab/dflash) · 原项目入口 | [arXiv:2602.06036](https://arxiv.org/abs/2602.06036) |
| [DFlash2](/articles/sd-dflash2/) | [z-lab/dflash](https://github.com/z-lab/dflash) · 原项目入口 | 以项目文档为准，未猜测论文链接 |
| [DSpark](/articles/sd-dspark/) | [sgl-project/SpecForge](https://github.com/sgl-project/SpecForge) · 框架实现，非独立作者仓库 | [arXiv:2607.05147](https://arxiv.org/abs/2607.05147) |
| [Domino](/articles/sd-domino/) | [sgl-project/SpecForge](https://github.com/sgl-project/SpecForge) · 框架实现，非独立作者仓库 | [arXiv:2605.29707](https://arxiv.org/abs/2605.29707) |
| [LiLiCorr](/articles/sd-lilicorr/) | [sgl-project/sglang](https://github.com/sgl-project/sglang) · 框架实现，非独立作者仓库 | [arXiv:2608.20530](https://arxiv.org/abs/2608.20530) |
| [Sequoia](/articles/sd-sequoia/) | [Infini-AI-Lab/Sequoia](https://github.com/Infini-AI-Lab/Sequoia) · 原项目入口 | [arXiv:2402.12374](https://arxiv.org/abs/2402.12374) |
| [SpecExec](/articles/sd-specexec/) | [yandex-research/specexec](https://github.com/yandex-research/specexec) · 原项目入口 | [arXiv:2406.02532](https://arxiv.org/abs/2406.02532) |
| [SpecInfer](/articles/sd-specinfer/) | [flexflow/FlexFlow](https://github.com/flexflow/FlexFlow) · 原项目入口 | [arXiv:2305.09781](https://arxiv.org/abs/2305.09781) |
| [DistillSpec](/articles/sd-distillspec/) | 作者仓库未确认；详见文章限制 | [arXiv:2310.08461](https://arxiv.org/abs/2310.08461) |
| [FR-Spec](/articles/sd-frspec/) | [sgl-project/sglang](https://github.com/sgl-project/sglang) · 框架实现，非独立作者仓库 | [arXiv:2502.14856](https://arxiv.org/abs/2502.14856) |
| [自适应草稿步数：SGLang](/articles/sd-adaptive-draft/) | [sgl-project/sglang](https://github.com/sgl-project/sglang) · 框架实现，非独立作者仓库 | 以项目文档为准，未猜测论文链接 |
| [自适应验证预算：vLLM](/articles/sd-adaptive-verify/) | [vllm-project/vllm](https://github.com/vllm-project/vllm) · 框架实现，非独立作者仓库 | 以项目文档为准，未猜测论文链接 |
| [D-PACE 训练目标](/articles/sd-dpace/) | [sgl-project/SpecForge](https://github.com/sgl-project/SpecForge) · 框架实现，非独立作者仓库 | [arXiv:2605.18810](https://arxiv.org/abs/2605.18810) |
| [混合与级联投机](/articles/sd-hybrid/) | 作者仓库未确认；详见文章限制 | 以项目文档为准，未猜测论文链接 |

## 已读取并锁定的源码入口

以下每个链接固定到实际读取的上游提交。SHA256 和失败路径另存[完整来源记录](https://github.com/WenMang98k/WenMang98k.github.io/blob/main/research/speculative-decoding/sources.lock.json)，不要把一次失败的猜测路径当成算法不存在的证据。

| 项目 | 已读取文件 | 固定提交 |
|---|---|---|
| FasterDecoding/Medusa | [`medusa/model/utils.py`](https://github.com/FasterDecoding/Medusa/blob/e2a5d20c048a9b0a4092e6933c34313687422518/medusa/model/utils.py) | `e2a5d20c04` |
| FasterDecoding/Medusa | [`medusa/train/train_legacy.py`](https://github.com/FasterDecoding/Medusa/blob/e2a5d20c048a9b0a4092e6933c34313687422518/medusa/train/train_legacy.py) | `e2a5d20c04` |
| SafeAILab/EAGLE | [`eagle/model/utils.py`](https://github.com/SafeAILab/EAGLE/blob/cb7e0841fe0c206c6ed74a197ad5e2a1f13f5a2b/eagle/model/utils.py) | `cb7e0841fe` |
| SafeAILab/EAGLE | [`eagle/model/ea_model.py`](https://github.com/SafeAILab/EAGLE/blob/cb7e0841fe0c206c6ed74a197ad5e2a1f13f5a2b/eagle/model/ea_model.py) | `cb7e0841fe` |
| SafeAILab/EAGLE | [`eagle/train/main.py`](https://github.com/SafeAILab/EAGLE/blob/cb7e0841fe0c206c6ed74a197ad5e2a1f13f5a2b/eagle/train/main.py) | `cb7e0841fe` |
| zankner/Hydra | [`hydra/model/utils.py`](https://github.com/zankner/Hydra/blob/3f39fbd2c22552e788b62dfd29a12b07ad63abfe/hydra/model/utils.py) | `3f39fbd2c2` |
| apple/ml-recurrent-drafter | [`recurrent_drafting/recurrent_drafting.py`](https://github.com/apple/ml-recurrent-drafter/blob/0ae66cb657b847c3279feb4fa58f57e233ca9c6c/recurrent_drafting/recurrent_drafting.py) | `0ae66cb657` |
| FasterDecoding/REST | [`rest/model/rest_model.py`](https://github.com/FasterDecoding/REST/blob/154d0f0ae45de5d8a9cbd1c696f6ad5da10904bf/rest/model/rest_model.py) | `154d0f0ae4` |
| hao-ai-lab/LookaheadDecoding | [`README.md`](https://github.com/hao-ai-lab/LookaheadDecoding/blob/eed010da9c7b1867912675d480feec5629e0c2d0/README.md) | `eed010da9c` |
| Infini-AI-Lab/Sequoia | [`tree_search.py`](https://github.com/Infini-AI-Lab/Sequoia/blob/0ff3ff71475ca148f2dfb7545e0de5caf5c67b39/tree_search.py) | `0ff3ff7147` |
| sgl-project/SpecForge | [`specforge/modeling/draft/dflash.py`](https://github.com/sgl-project/SpecForge/blob/ce3ef8fa05ea397881634590768d0b0150ec24f9/specforge/modeling/draft/dflash.py) | `ce3ef8fa05` |
| sgl-project/SpecForge | [`specforge/modeling/draft/dspark.py`](https://github.com/sgl-project/SpecForge/blob/ce3ef8fa05ea397881634590768d0b0150ec24f9/specforge/modeling/draft/dspark.py) | `ce3ef8fa05` |
| sgl-project/SpecForge | [`specforge/modeling/draft/heads/markov.py`](https://github.com/sgl-project/SpecForge/blob/ce3ef8fa05ea397881634590768d0b0150ec24f9/specforge/modeling/draft/heads/markov.py) | `ce3ef8fa05` |
| sgl-project/SpecForge | [`specforge/modeling/draft/peagle.py`](https://github.com/sgl-project/SpecForge/blob/ce3ef8fa05ea397881634590768d0b0150ec24f9/specforge/modeling/draft/peagle.py) | `ce3ef8fa05` |
| sgl-project/SpecForge | [`specforge/modeling/draft/domino.py`](https://github.com/sgl-project/SpecForge/blob/ce3ef8fa05ea397881634590768d0b0150ec24f9/specforge/modeling/draft/domino.py) | `ce3ef8fa05` |
| sgl-project/SpecForge | [`specforge/modeling/draft/dflash2.py`](https://github.com/sgl-project/SpecForge/blob/ce3ef8fa05ea397881634590768d0b0150ec24f9/specforge/modeling/draft/dflash2.py) | `ce3ef8fa05` |
| sgl-project/sglang | [`python/sglang/srt/speculative/spec_info.py`](https://github.com/sgl-project/sglang/blob/34dfaccc57d928f046317c2e3d9e16bd69837d9b/python/sglang/srt/speculative/spec_info.py) | `34dfaccc57` |
| vllm-project/vllm | [`vllm/config/speculative.py`](https://github.com/vllm-project/vllm/blob/2ee8250eaf67059b4e8bc3c1bfb4cfb1c88892a7/vllm/config/speculative.py) | `2ee8250eaf` |
| sgl-project/sglang | [`docs/docs/advanced_features/speculative_decoding.mdx`](https://github.com/sgl-project/sglang/blob/34dfaccc57d928f046317c2e3d9e16bd69837d9b/docs/docs/advanced_features/speculative_decoding.mdx) | `34dfaccc57` |
| vllm-project/vllm | [`docs/features/speculative_decoding/README.md`](https://github.com/vllm-project/vllm/blob/2ee8250eaf67059b4e8bc3c1bfb4cfb1c88892a7/docs/features/speculative_decoding/README.md) | `2ee8250eaf` |
| apoorvumang/prompt-lookup-decoding | [`README.md`](https://github.com/apoorvumang/prompt-lookup-decoding/blob/cada4fe6ec7368d41c386ec3fb94fa11268971eb/README.md) | `cada4fe6ec` |

补核 Draft & Verify 作者源码：

- [`README.md`](https://github.com/dilab-zju/self-speculative-decoding/blob/688db8b74a782af21777b7ad5aeb399d7eb26faf/README.md)
- [`decoding.py`](https://github.com/dilab-zju/self-speculative-decoding/blob/688db8b74a782af21777b7ad5aeb399d7eb26faf/decoding.py)
- [`searching.py`](https://github.com/dilab-zju/self-speculative-decoding/blob/688db8b74a782af21777b7ad5aeb399d7eb26faf/searching.py)

## 建议的代码阅读顺序

1. **Medusa／Hydra**：候选缓冲、树掩码、候选生成、posterior、token／KV 更新。
2. **EAGLE**：生成轮次、特征／token shift、树验证与更新；main 与 v1 分支的代际差异不能忽略。
3. **ReDrafter**：beam 验证、贪心／随机接受函数、KV 选取和压缩。
4. **DFlash／DFlash2**：前缀特征、proposal mask、并行草稿、动态卷积、selector。
5. **DSpark／Domino**：Markov／GRU 校正与 confidence；区分“生成候选”和“决定验证预算”。
6. **框架**：方法枚举只是入口，还要继续看 loader、worker、采样器和调度约束。

## 可以直接运行的本站参考代码

[下载 exact_sampling.py](/learning/speculative-decoding/exact_sampling.py)，或在[网站 GitHub 源码](https://github.com/WenMang98k/WenMang98k.github.io/tree/main/research/speculative-decoding)阅读。

```bash
python3 exact_sampling.py
```

它实现单位置有限词表的 proposal、接受与残差重采样，运行二元词表 Monte Carlo 示例以及零支持／相同分布边界。只需 Python 标准库。它不包含 Transformer、树采样或 GPU kernel，不伪装成高性能框架。

## 加速收益的证据规则

- **作者报告**：给出原仓库与能查到的硬件、target、任务、batch、精度和采样条件。条件不齐时明确注明，不能进入统一排名。
- **第三方复现**：单独标识；例如 Token Recycling 的非官方实现数字不标为作者或本站实测。
- **本站已运行**：有限词表数值检查、参考程序、网站类型检查／构建／浏览器验证；不包含任何 GPU 性能实验。
- **尚未核实**：不填写倍率、参数量和延迟，不从方法名字推测算法。

## 版本、许可证与后续收录

目前只发布原创学习文章、教学代码和上游链接，没有复制上游整仓。链接不改变上游许可证；未来实际收录文件时，应逐 revision 检查许可、保留版权和 LICENSE，记录修改，并避免把演示图片、训练数据、模型权重一并搬来。

部分 arXiv 正文受环境网络限制。系列清楚标注哪些结论来自实际代码／README，哪些还需要论文全文复核；对没有找到官方实现的工作，保留“不确定”，不拿同名第三方仓库充数。

## 同名与相近项目的辨别

本轮也读到 google-research/google-research 的 speculative_kd README，其名称为 Speculative Knowledge Distillation（SKD）；不能仅凭同属蒸馏路线就把它当作 DistillSpec 原论文官方实现。未确认的对应关系保持明确，不拿相近项目填空。
