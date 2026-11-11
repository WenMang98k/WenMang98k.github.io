# LLM 投机解码技术时间轴

[返回综述](README.md) · [网页阅读](index.html)

整理日期：2026-11-11。按“论文登记年月／有据可查的公开发布”排列；**不是优先权裁决，也不是框架支持日期表**。论文年月主要依据 arXiv 编号的 YYMM，未逐项访问 submission history 核对具体日期；若官方仓库给出代码／博客发布时间，另行注明。早期工作可能首先针对机器翻译、seq2seq，不能全部等同于今天的通用 LLM 服务。

## 一、研究主线：先解决正确性，再改善候选与系统效率

| 时间 | 方法／事件 | 主要变化 | 与后续技术的关系 |
|---|---|---|---|
| 2018-11 | [Blockwise Parallel Decoding](https://arxiv.org/abs/1811.03115) | 用多个未来位置预测与验证减少串行解码步骤 | 多 token 候选与块验证的早期思想；不是现代无损随机投机采样公式的同义词 |
| 2022-03 | [Speculative Decoding：面向 seq2seq 的投机执行](https://arxiv.org/abs/2203.16487) | 借助便宜预测与验证加速生成 | 体现投机执行思路向序列生成迁移的早期路线 |
| 2022-11 | [Leviathan 等：Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192) | 小模型提案、大模型批验，结合概率校正保持目标分布 | 现代经典 SD 的基础；论文发表于 ICML 2023 |
| 2023-02 | [Chen 等：Speculative Sampling](https://arxiv.org/abs/2302.01318) | 系统阐述草稿、接受／拒绝与残差重采样 | 与经典 SD 共同奠定无损随机采样路线 |
| 2023-05 | [SpecInfer](https://arxiv.org/abs/2305.09781) | 候选树验证与推理系统协同 | 后续树验证、批处理和系统集成的重要背景 |
| 2023-09 | [Draft & Verify](https://arxiv.org/abs/2309.08168) | 用跳层构造便宜草稿，完整模型验证 | 自投机路线，减少独立小模型需求 |
| 2023-10 | [DistillSpec](https://arxiv.org/abs/2310.08461) | 通过蒸馏改善 draft／target 对齐 | 开始更明确地优化“容易被目标接受”，而非只选现成小模型 |
| 2023-11 | [REST](https://arxiv.org/abs/2311.08252)；[Lookahead 公开博客](https://lmsys.org/blog/2023-11-21-lookahead-decoding/) | 检索续写／复用 Jacobi 轨迹中的 n-gram | “草稿不一定来自额外神经网络”成为重要分支 |
| 2024-01 | [Medusa 论文](https://arxiv.org/abs/2401.10774)；[EAGLE 论文](https://arxiv.org/abs/2401.15077) | Medusa 用多个未来预测头；EAGLE 利用目标特征预测后续 | 多头路线与特征草稿路线并行发展；Medusa 存在更早演示，本表此处记论文年月 |
| 2024-02 | [Hydra](https://arxiv.org/abs/2402.05109)；[Sequoia](https://arxiv.org/abs/2402.12374)；[Lookahead 论文](https://arxiv.org/abs/2402.02057) | 依赖多头、硬件感知树、并行轨迹复用 | 不同层次分别改善候选相关性、树预算与候选来源 |
| 2024-03 | [Recurrent Drafter／ReDrafter](https://arxiv.org/abs/2403.09919) | 轻量循环草稿与动态树处理 | 在草稿质量、串行成本和共享前缀计算之间折中 |
| 2024-04 | [Multi-token Prediction](https://arxiv.org/abs/2404.19737)；[LayerSkip](https://arxiv.org/abs/2404.16710) | 前者引入多未来位置训练目标；后者训练早退能力 | 将“适合投机”的能力做进训练，目标模型与草稿不再总是两个独立 LM |
| 2024-06 | [EAGLE-2](https://arxiv.org/abs/2406.16858)；[SpecExec](https://arxiv.org/abs/2406.02532) | 动态候选树；大规模并行候选与验证 | 从固定候选结构走向上下文／硬件相关的预算分配 |
| 2024-08 | [Token Recycling](https://arxiv.org/abs/2408.08696) | 复用已有计算产生的候选信息 | 无额外训练／模型方向继续发展；与 prompt lookup 并非完全相同 |
| 2024-12 | [DeepSeek-V3 技术报告](https://arxiv.org/abs/2412.19437) | 将 MTP 融入大规模模型训练设计 | 原生 MTP 成为后续部署的重要候选能力；具体发布模块数需查权重说明 |
| 2025-02 | [FR-Spec](https://arxiv.org/abs/2502.14856) | 用高频词表等机制减少草稿输出头成本 | 草稿瓶颈不只有 Transformer 主干，也可能是大词表 LM head |
| 2025-03 | [EAGLE-3](https://arxiv.org/abs/2503.01840) | 多层特征融合、移除原特征回归约束、training-time testing | 更关注 rollout 状态与训练分布；官方仓库另记代码发布于 2025-03-19 |
| 2026-02 | [P-EAGLE](https://arxiv.org/abs/2602.01469)；[DFlash](https://arxiv.org/abs/2602.06036) | 特征草稿的并行化／块并行草稿 | 减少轻量 drafter 仍然存在的逐 token 串行调用 |
| 2026-05 | [D-PACE](https://arxiv.org/abs/2605.18810)；[Domino](https://arxiv.org/abs/2605.29707) | 训练目标优化／GRU logit 校正等扩展 | 分别关注接受收益与块内依赖；本轮仅核对框架索引，待原文详读 |
| 2026-07 | [DSpark](https://arxiv.org/abs/2607.05147) | 置信度调度的半自回归生成；实现包含 Markov／confidence 头 | 将草稿质量、可接受性预测和验证资源分配联系起来 |
| 2026-08 | [DFlash2 官方发布资料](https://github.com/z-lab/dflash)；[LiLiCorr](https://arxiv.org/abs/2608.20530) | 动态分组卷积、候选路径选择／候选关联 | 继续改善块并行草稿中的 token 间相关性；DFlash2 日期按官方 README BibTeX 的 August 2026，不臆造 arXiv 首发日 |

**这是一张分支图，不是一条替代链。**检索草稿、自投机、MTP、特征草稿、块并行草稿长期并存；动态树与负载调度也可叠加到多种草稿模型上。

## 二、用五条支线理解演进

```text
概率校正：经典 SD / Sampling ───────────────────→ 各种正确验证与采样实现
候选来源：独立小 LM → 检索 / n-gram / suffix → 候选复用与混合路由
依赖建模：Medusa → Hydra / ReDrafter → 块内校正、路径选择、相关性建模
模型训练：EAGLE → EAGLE-2 → EAGLE-3；MTP / LayerSkip 是相邻的训练分支
并行系统：树验证 / Sequoia / SpecExec → 块并行草稿 → 置信度与负载自适应
```

上图的箭头表示问题延续与阅读顺序，不意味着方法间存在严格继承或作者归属关系。特别是 EAGLE-2 主要改进动态树，EAGLE-3 主要改进特征与训练；MTP 不是 EAGLE 的下一版本。

## 三、框架时间线：有证据的节点与当前快照

| 时间／口径 | 核实到的事件 | 能说明什么，不能说明什么 |
|---|---|---|
| 2025-07-23，EAGLE 官方仓库公告 | 作者推荐用 SpecForge 进行 EAGLE-3 训练并衔接 SGLang | 是可核实的工具链推荐节点，不是 SGLang 首次支持 EAGLE-3 的日期 |
| 2026-01，SpecForge README 自述 | 记录 DFlash 块并行在线训练与 SGLang serving 支持 | 属仓库维护者的历史记录；早于其论文编号月份，不据此推断论文首发或所有版本已发布 |
| 2026-06，SpecForge README 自述 | 记录加入 D-PACE 可选损失、Domino 在线训练 | 是训练工具支持节点，不自动等于 vLLM／SGLang 推理端全功能支持 |
| 2026-07，SpecForge README 自述 | 记录 DSpark 在线训练及相关草稿支持 | 不能直接当作两推理框架的首个稳定 release 日期 |
| 2026-08，SpecForge README 自述 | 记录 DFlash2 在线训练 | 同样要区分训练代码、权重发布和 serving loader／kernel 集成 |
| 截至本轮文档核对，2026-11-11 | vLLM／SGLang 当前主分支已包含经典双模型、EAGLE 类、MTP、n-gram、DFlash、DSpark 等入口；部分方法只在一侧核实 | 证明当前主分支存在相关实现／文档，不证明最早集成时间、生产成熟度或任意 release 都支持 |

SGLang 指南的 UNO、LiLiCorr 等入口，本轮没有进一步考证首次合入日期，因此不在表中硬填日期。vLLM adaptive verification 与 SGLang adaptive speculative decoding 也应分别追溯，不归并成一次共同发布。

要做“正式框架版本演进表”，下一步需要逐方法记录首个合入 PR、merge commit、首次包含该 commit 的 release tag、对应模型权重和功能限制，不能拿 README 当前内容反推历史。

## 四、建议的精读顺序

1. 经典 SD／Sampling：先弄清接受公式、拒绝残差与 KV 回退。
2. Medusa → Hydra → ReDrafter：理解未来 token 之间为什么需要依赖。
3. EAGLE → EAGLE-2 → EAGLE-3：分清特征、树结构和训练 rollout 三个问题。
4. MTP、LayerSkip：理解训练赋予的多步／早退能力如何变成推理加速。
5. DFlash → DFlash2、DSpark：理解块并行和便宜依赖校正的分工。
6. vLLM／SGLang 调度：最后看负载、验证预算与真实 SLO。

## 来源与限制

方法细节及已读来源见[综述参考资料](README.md#参考资料与核对说明)和[核对清单](source-notes.md)。2018／2022 早期路线作为背景补充，仅列原论文入口，尚未做全文评审。arXiv 正文访问限制仍然存在；表中 arXiv 年月由编号可判断，不冒充已核实到日的 submission history。没有使用网页最后更新时间替代论文提出时间，也没有按发布日期排列加速比高低。
