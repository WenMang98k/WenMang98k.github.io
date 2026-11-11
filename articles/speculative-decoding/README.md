# LLM 投机解码综述：从经典双模型，到 EAGLE、MTP、DFlash 与 DSpark

> [返回主页](../../index.html) · [网页版](index.html) · [资料核对说明](source-notes.md)

> 作者：文氓（WenMang） · 研究初稿 v1 · 资料核对日期：2026-11-11
> 适合读者：希望理解投机解码机制，并进一步开展训练、推理实现和服务优化的算法／AI Infra 工程师。  
> 阅读路线：先读第一部分建立全景，再读第二部分理解单点技术，最后用第三部分设计实验。

## 阅读前：范围与证据

本文讨论 LLM 自回归生成中的 speculative decoding／speculative sampling。为方便表述，统一简称“投机解码”；涉及严格随机采样算法时使用“投机采样”。**SD 在本文中表示 speculative decoding，不表示 Stable Diffusion，也不默认指某个独立方法。**“drafter”是草稿生成器的通称；Apple 的 Recurrent Drafter／ReDrafter 是其中一种具体实现。用户提到的“dlfash”按 DFlash 处理。

本文核对了 EAGLE、Medusa、Hydra、ReDrafter、Lookahead、LayerSkip 的作者仓库，以及 SpecForge、vLLM 的公开文档与部分训练代码。经典概率校正公式给出推导；DFlash2、DSpark 的细节明确以所读取的框架实现为依据。**当前环境无法访问 arXiv 正文，不能把论文链接的存在当作已经通读论文。**文末列出原论文入口、实际读取的材料与待复核项；本文是一份可继续审校的研究初稿，不是对所有论文版本的穷尽评审，也没有运行 GPU 性能实验。

---

# 第一部分：把技术谱系放在同一张地图上

## 1. 为什么“大模型少跑几次”可能比“小模型更准”重要？

自回归 LLM 每生成一个 token，通常要执行一次目标模型的 decode。低并发下，这个过程经常受权重与 KV cache 的访存、kernel launch 和同步开销限制，GPU 的计算能力未必用满。

投机解码利用这部分余量：

1. 用便宜的方式提出一段候选 token；
2. 把这些候选作为已知输入，让目标模型一次前向并行计算多个位置的条件分布；
3. 按验证规则接受一段前缀，遇到拒绝时校正并回退；
4. 提交已接受 token 和对应 KV，开始下一轮。

这里的“并行”有一个必要条件：**验证阶段已经知道整段候选，所以可以像 teacher forcing 一样计算各位置；最终提交仍要满足从左到右的依赖。**不能在第二个候选被拒绝后，继续无条件提交第三个候选。

一句话理解：把“昂贵模型每次只推进一步”，改成“便宜系统先猜，昂贵模型成批核验”。

### 一个比接受率更有用的性能模型

设一轮提出 K 个候选，接受的候选数为 A，校正／bonus token 在没有触发停止条件时通常再带来一个输出 token。近似有：

$$
S \approx \frac{(\mathbb E[A]+1)\,T_{\mathrm{AR}}}
{T_{\mathrm{draft}}(K)+T_{\mathrm{verify}}(K)+T_{\mathrm{manage}}}.
$$

其中管理开销包括采样、树构建、KV 提交／回退、调度和通信。EOS、输出长度上限等会改变实际推进量，不能机械加一。

这解释了两个常见反例：

- 更大的 drafter 虽然更准，但生成太慢，整体反而变慢；
- 更长的候选提高单轮输出量，却让验证计算、KV 占用和回退浪费增长，超过收益。

**优化目标是单位时间提交的有效 token，不是孤立的 acceptance rate。**

## 2. 一张方法地图：草稿来自哪里，依赖关系如何建模？

| 路线 | 代表方法 | 草稿生成方式 | 是否需要额外训练 | 关键代价／限制 |
|---|---|---|---|---|
| 独立小模型 | 经典 speculative sampling | 小型自回归 LM 逐步生成 | 可直接用现成小模型，也可蒸馏 | 草稿串行、额外模型与 KV；常需匹配 tokenizer |
| 无模型检索 | Prompt lookup、n-gram、suffix、REST | 从输入、历史输出或语料检索续写 | 通常不需要 | 任务相关性强，匹配不到时收益低 |
| 并行多头 | Medusa | 目标模型特征上的多个未来预测头 | 需要 | 跨未来位置依赖较弱；验证规则决定质量保证 |
| 依赖多头／循环草稿 | Hydra、ReDrafter | 后续候选显式依赖前序 token 或循环状态 | 需要 | 增加轻量串行依赖，但通常改善候选质量 |
| 特征草稿 | EAGLE、EAGLE-2、EAGLE-3 | 利用目标模型隐藏特征做轻量扩展 | 需要 | 目标特征耦合、训练／推理状态分布差异 |
| 原生多 token 预测 | MTP、DeepSeek 式 MTP | 主干共享或原生辅助模块预测后续 token | 通常需模型已有相关训练 | 不同模型的 MTP 结构并不统一 |
| 自投机 | Draft & Verify、LayerSkip | 跳层／浅层作为 drafter，完整模型验证 | 有无训练两类 | 浅层质量、特征与 KV 复用、基线定义 |
| 迭代并行候选 | Jacobi、Lookahead、后续块解码 | 并行迭代构造候选再验证 | 部分不需要 | 额外迭代、候选复用和注意力开销 |
| 块并行草稿 | DFlash、DFlash2、P-EAGLE | 一次或少量轻量前向产生一块候选 | 需要 | 块内 token 依赖建模和长前缀存活率 |
| 半自回归草稿与置信调度 | DSpark | 块并行底座＋便宜的依赖校正＋置信估计 | 需要 | 草稿训练、置信校准与服务调度共同决定收益 |

这些类别会交叉。动态树、动态草稿长度、动态验证预算是可叠加的系统策略，不一定是新的 drafter 架构。EAGLE-2 主要改进“树怎么长”；DFlash2 在所核对的实现中主要改进“并行草稿如何表达块内依赖”。

### 技术演进的五个问题

- 经典双模型：能否让小模型的候选被大模型批量验证？
- Medusa／Hydra／ReDrafter：能否省掉独立小模型的完整计算？
- EAGLE 系列：能否利用目标模型特征提高草稿质量？
- MTP：能否把多步预测直接做进模型训练？
- DFlash／DSpark 等：能否在减少草稿串行计算的同时，保留足够的 token 间依赖，并按负载决定验证预算？

这不是一条“新方法完全淘汰旧方法”的直线。重复性强的代码／文档任务中，几乎零训练成本的检索草稿仍可能很有竞争力。

## 技术时间轴：论文、代码与框架支持分开看

[独立 Markdown 时间轴](timeline.md)。下表按月梳理研究主线，后表单列工具链支持节点。

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


## 面向落地：优先研究 vLLM 与 SGLang 已集成的路线

**本节核对日期：2026-11-11；范围为当时公开主分支的文档与代码，不代表任意稳定发行版都支持，也不代表实测采用率排名。**“框架支持”还需满足目标模型、草稿权重、attention backend、并行策略和采样模式的组合约束。

核对时记录到的上游 main：vLLM `01fba62494f4fd61c6f36c0dafbe4eae13d90833`，SGLang `2c5b680aa6294bcd246366aba197a42f876b75f7`。主分支随时更新，复现时应固定提交并重新检查。

| 方法／机制 | vLLM 已核实入口 | SGLang 已核实入口 | 本文建议的阅读优先级 |
|---|---|---|---|
| 独立小模型经典 SD | `method: draft_model` | `STANDALONE` | 基础必读；理解 proposal、验证和 KV 回退 |
| EAGLE／EAGLE-3 | `eagle`、`eagle3` | `EAGLE` 对应指南中的 EAGLE-2；`EAGLE3` | 核心路线；适合系统学习训练式草稿 |
| 原生 MTP | `mtp` 及模型专用 MTP 类型 | 官方指南以 `EAGLE` 路径展示 MTP；`NEXTN` 为相关别名 | 核心路线；优先利用模型已有能力，不把接口名当架构名 |
| n-gram／Prompt Lookup | `ngram`，另有 `ngram_gpu` 配置类型 | `NGRAM`，指南标注 CUDA-only | 核心基线；无需训练草稿，适合重复性任务 |
| Suffix Decoding | `suffix`，有专门文档 | 本轮未核实独立 suffix 入口 | 值得补充；不要把它与所有 n-gram 实现视为完全相同 |
| DFlash／DFlash2 | `dflash`；具体结构由兼容 checkpoint 决定 | `DFLASH`；DFlash2 复用该入口 | 重点进阶；比较块并行主干和块内依赖建模 |
| DSpark | 配置枚举包含 `dspark`，adaptive verification 文档给出示例 | 算法枚举包含 `DSPARK`，验证文档也点名该族 | 重点进阶；需核对模型权重与调度支持 |
| 动态草稿／自适应验证 | Dynamic Speculative Decoding、Adaptive Verification | Adaptive Speculative Decoding | 系统必读；它们是机制组合，不是同一个算法 |
| FR-Spec | 本轮未确认对应独立开关 | EAGLE-2 的 `--speculative-token-map` | 补充优化；降低 draft LM head 的词表计算成本 |
| UNO | 本轮未确认内置对应入口 | `UNO`；官方指南提供 linear／tree 模式 | 新近集成路线；目标模型及功能限制较多 |
| LiLiCorr | 兼容 correlator checkpoint 走 `dflash` | 兼容 checkpoint 走 `DFLASH` | 跟踪路线；已写入集成文档不等于有可直接下载的公开草稿权重 |

### 本轮最值得新增的四个主题

**1. Suffix Decoding：复用已有续写，而不只是当前 prompt。**vLLM 文档说明它是无额外草稿模型的候选来源，可以利用自身历史及既有响应等后缀匹配信息。重点看候选来源、匹配长度、历史统计与缓存成本。与训练式方法相比，它可能更适合快速试验业务是否存在可复用模式。

**2. FR-Spec：优化 drafter 的输出头。**对于词表很大的目标模型，轻量草稿模块不代表整个 draft 都便宜，LM head 仍可能占显著时间。SGLang 指南中 FR-Spec 使用高频 token 子集与 token map 降低这部分计算。它是 EAGLE 路径的优化维度，不是完全独立的验证算法；被核对文档明确说明 EAGLE-3 会忽略该 token-map 参数。

**3. 自适应调度：vLLM 与 SGLang 要分开讲。**所读 SGLang 专门指南以按 batch-size 范围维护的 EMA 等机制切换草稿步数，支持条件是 `EAGLE`／`EAGLE3` 且 `topk=1`，不满足时退回静态设置。vLLM 专门指南描述的 adaptive verification 则以请求／位置的存活概率与成本模型分配验证预算。两者优化位置、约束与运行成本不同，不能用一个“动态 SD”框图覆盖。

**4. UNO：复用目标 Transformer，加上专门训练的 LoRA 生成候选。**SGLang 指南说明 draft pass 的部分行使用 UNO LoRA，目标验证仅使用 base 权重；支持链与树两种候选形态。它省的是独立 draft Transformer，而非完全没有额外训练或计算。当前指南的验证模型范围有限，并明确列出 grammar、部分 sampling 处理及 deterministic inference 等限制，不建议作为初学者第一条复现路线。

### 应放在历史脉络／后续阅读里的方法

Medusa、Hydra、ReDrafter、Lookahead、LayerSkip 依然有解释机制的价值，但本轮核对的当前框架方法索引没有把它们全部列成默认内置入口。应按具体版本查实现，不能因历史版本出现过就宣称“最新版本一键可用”。

另补充 **Sequoia、SpecExec、SpecInfer、DistillSpec、Token Recycling** 作为研究路线：

- **Sequoia**：候选树形状与硬件开销共同优化。作者仓库提供 acceptance vector 估计与 growmap 生成入口，说明树预算不是与硬件无关的抽象超参。[16]
- **SpecExec**：大规模并行候选构造与验证，作者仓库同时提供普通与 offloading 评测入口。适合对比目标访存开销很高时，较大候选预算为何可能有价值。[17]
- **SpecInfer**：树式验证与推理系统协同的重要先行工作；当前 FlexFlow 仓库已拆分，旧链接和旧运行步骤需重新核实，本文不提供未经确认的现代框架启动命令。
- **DistillSpec**：通过蒸馏提升 draft 与 target 的对齐，属于“草稿如何训练得更易接受”的路线。原文及官方实现本轮尚未完成核对，先列阅读入口，不写未经验证的训练配方。[18]
- **Token Recycling**：复用既往目标计算中得到的候选信息，减少独立草稿的需求。本轮仅读取明确标注 unofficial 的复现说明，不据此宣称主框架已内置或采纳其加速比。[19]

**建议本系列的落地顺序：经典 SD → n-gram／suffix → EAGLE-3 → 原生 MTP → DFlash／DSpark → 自适应调度。**Medusa 等用于解释技术演进，UNO、FR-Spec、LiLiCorr 作为专项扩展。优先级反映本文对可操作性的判断，不是下载量、生产采用率或统一性能排行榜。


## 3. “无损”必须先说清楚是哪一种

### 3.1 贪心一致

temperature=0 时，验证每个候选是否等于目标模型在对应前缀上的 argmax。接受连续匹配的前缀，首次不匹配时输出目标模型的 argmax。

在相同模型、logit 处理规则及理想数值条件下，可与逐 token 贪心解码一致。现实中的并行 kernel、精度和归约顺序可能让近似平局 token 出现差异，因此还要区分算法保证与实现中的 bitwise 一致。

### 3.2 随机采样分布一致

随机采样下，不是“目标模型觉得这个 token 还行就接受”。要用正确的 proposal 概率、接受率和拒绝后的残差分布，保证最终输出服从目标采样分布。

### 3.3 任务指标基本不变

某些近似接受机制在任务评测中质量接近基线，但这不等价于输出分布严格相同。Medusa 的 typical acceptance 就需要与经典无损拒绝采样区分。[3]

还有一个基线问题：如果为了获得 MTP／早退能力而更新了主模型，那么“验证无损”通常是相对于**更新后的目标模型**，不是承诺和原 checkpoint 分布一致。

---

# 第二部分：逐项讲清训练与推理

## 4. 经典投机采样：正确性地基

### 训练：可以完全没有，也可以让小模型向目标模型靠近

最简单的办法是选一个现成小模型，不改变目标模型。若要训练 drafter，可在目标业务的 prompt 和续写上做监督训练，或蒸馏目标模型分布。

独立草稿模型的目标不是一般意义上“足够聪明”，而是**在目标模型真正会走到的上下文中，以很低成本提出容易被接受的候选**。同规模但分词器、对齐风格、推理模板和业务数据都不匹配的 drafter，未必好用。

### 推理：接受公式与残差重采样

令当前前缀为 x，目标分布为 p，草稿分布为 q。草稿先提出候选 y：

$$y\sim q(\cdot\mid x).$$

接受概率是：

$$\alpha(y)=\min\left(1,\frac{p(y\mid x)}{q(y\mid x)}\right).$$

若拒绝，从残差分布采样替代 token：

$$r(v)=\frac{\max\{0,p(v\mid x)-q(v\mid x)\}}
{\sum_u\max\{0,p(u\mid x)-q(u\mid x)\}}.$$

对 K 个候选，目标模型一次前向获得沿候选链的各位置分布。然后从左到右执行上述判定，首次拒绝即终止这条链。如果全部通过，则可从最后一个候选之后的目标分布再采一个 bonus token。[1][2]

**为什么分布正确？**候选 v 被接受的概率质量是 min(p(v), q(v))；拒绝分支补回正差 p(v)−q(v)。二者相加就是 p(v)。当 p=q 时拒绝概率为零，不需要构造分母为零的残差分布。

这里 q 的零概率 token 不会被 q 采到；实现仍需正确处理零值、精度和归一化。

### 最小数值例子

若词表只有 a、b，p=(0.7, 0.3)，q=(0.4, 0.6)：

- a 总被接受，贡献 0.4；
- b 以 0.3/0.6=0.5 的概率接受，贡献 0.3；
- 总拒绝质量为 0.3，残差分布只在 a 上补回这 0.3；
- 最终仍为 (0.7, 0.3)。

### 容易写错的地方

- 实际从 q′ 采样，却在接受公式中代入截断前的 q；
- 拒绝后直接从 p 重采样，漏掉残差校正；
- 忽略温度、top-k／top-p、重复惩罚、语法约束对 p、q 的变换；
- 首次拒绝后继续提交后续位置，或保留它们的无效 KV。

独立小模型之外，检索、树、块并行方法也可以使用正确的验证机制，但不能把链式公式不加修改地套进任意树搜索或并行采样程序。

## 5. 无训练路线：n-gram、suffix、REST 与 Lookahead

### 5.1 Prompt lookup／n-gram／suffix／REST

**训练：**通常不训练额外神经网络。区别主要在候选池：输入 prompt、当前输出、历史请求续写，或者外部数据存储。

**推理：**用当前后缀查询候选池，取匹配位置之后的 token 作为草稿；目标模型批量验证并提交有效前缀；未命中则回到普通解码或其他 drafter。

**适合：**代码补全、重复模板、拷贝式任务、与上下文高度重合的生成。**不适合盲目乐观：**检索命中不代表续写被接受，语义相近也不代表 token 序列相同。

REST 将外部检索作为草稿来源，可参考作者仓库与论文入口。[10] 若把历史请求放进共享候选池，还要处理租户隔离、保留策略和敏感内容边界。

### 5.2 Lookahead：不训练草稿模型，但仍然有额外计算

**训练：**不需要训练新的 drafter。

**推理：**Lookahead 使用 Jacobi 迭代的并行轨迹构造 n-gram 池；lookahead 分支持续生成候选，verification 分支从池中选择候选并用目标模型验证。关键不在“每次 Jacobi 都已经得到了正确答案”，而在于收集、缓存并复用其中有价值的片段。[8]

它以额外并行计算换更少串行推进轮次。不要把“无需额外训练”和“无需额外推理开销”混为一谈。

## 6. Medusa 与 Hydra：从独立多头到依赖多头

### 6.1 Medusa：一个状态，预测多个未来位置

**结构：**在目标模型隐藏状态上附加多个预测头。第 j 个头预测更远的未来 token。各头产生 top-k 候选，通过组合形成候选树。

**训练流程：**

1. 准备训练序列，使用基座模型得到当前位置隐藏状态；
2. 将未来不同偏移位置的 token 作为各头监督标签；
3. 用多头交叉熵等目标优化；
4. Medusa-1 冻结主模型，只训练新增头；Medusa-2 联合更新主模型与头，需另外看它相对哪个目标模型定义质量。[3]

抽象损失可以写为：

$$\mathcal L_{\mathrm{heads}}=\sum_j\lambda_j\,\mathrm{CE}(H_j(h_t),y_{t+j+1}),$$

其中 j 表示未来偏移；具体索引随实现的 label shift 约定变化。

**推理流程：**目标状态 → 多头并行候选 → 构造候选树 → 目标模型 tree attention 验证 → 选择被接受路径 → 提交 KV。

**关键边界：**多头从同一个状态预测不同未来位置，远端头没有天然拿到整条候选续写，因此块内依赖不足。原实现介绍的 typical acceptance 不应直接写成严格的随机采样分布一致。

### 6.2 Hydra：让后一个头知道前面猜了什么

**结构：**后续头显式条件于候选前序 token／续写，缓解 Medusa 头间独立造成的候选不协调。[4]

**训练流程：**冻结基座，训练草稿头；根据具体配置，以目标 token 和／或教师分布提供监督，让各偏移头学习在前序 token 条件下预测后继。作者训练示例明确只训练 Hydra 头；不同 Hydra／Hydra++ 配置不可混用损失描述。

**推理流程：**生成前几个候选 → 后续头利用候选续写生成更远 token → 形成树 → 目标模型验证。

**权衡：**条件更充分通常有助于长前缀存活，但增加轻量依赖。它要优化的是“候选质量提升能否覆盖新增串行成本”。

## 7. ReDrafter：循环状态复用与动态树压缩

ReDrafter 全称 Recurrent Drafter，来自 Apple；不是所有 drafter 的统称。[5]

### 训练

冻结目标模型，仅训练轻量循环 drafter。以目标模型隐藏状态作为上下文条件，让 drafter 在循环展开中吸收前序 token／内部状态，预测一段续写。作者仓库提供 ShareGPT 训练和评估入口；具体监督损失应继续对照对应训练实现，不能因为名字里有 recurrent 就直接断言使用 LSTM。

### 推理

1. 从目标模型当前状态初始化循环草稿过程；
2. 使用 beam search 扩展多条候选；
3. 对共享前缀进行动态树处理／压缩，减少重复验证位置；
4. 目标模型一次树验证，提交接受路径；
5. 更新 drafter 和目标模型缓存。

其系统价值不仅来自草稿模型更小，也来自**候选共享前缀的重复计算被压缩**。

### 一个必须保留的实现限定

所读取的官方 README 明确提示：当前非贪心生成实现可能无法做到精确分布匹配；贪心解码不受该项限制。因此不能仅凭“投机解码”四个字给整套代码盖上随机采样无损的章。

## 8. EAGLE 系列：三个版本分别解决什么问题？

### 8.1 EAGLE：利用目标模型特征做草稿

**思路：**利用目标模型接近输出层的上下文特征，训练轻量网络预测后续特征；同时条件于适当移位的 token 信息，减少下一 token 不确定性带来的歧义，再经输出头得到候选分布。[6]

**训练：**冻结目标模型，准备 token 和隐藏特征；训练轻量特征预测网络，结合特征回归与 token 层监督。训练时还需缓解“只看真实目标特征，推理却递归使用预测特征”的偏移。具体噪声与损失权重依版本实现核对。

**推理：**目标特征＋已知 token → 轻量草稿递归展开 → 形成候选树 → 目标模型树验证 → 用验证后的真实特征开启下一轮。

EAGLE 仍然有轻量递归步骤，不应描述为“所有未来位置完全一次并行预测”。

### 8.2 EAGLE-2：从固定树，走向上下文自适应树

EAGLE-2 的主要改进是利用草稿置信度估计候选价值，动态调整树结构与扩展预算。它主要回答“哪些分支值得猜、猜多深”，不是简单多训练了一个更大 drafter。[6]

**训练：**沿用 EAGLE 类草稿能力，重点区别不在新建一套与前代完全无关的监督体系。

**推理：**从当前上下文出发，按草稿分数动态扩展候选，组织成受预算约束的树；再由目标模型进行一致的树验证。

需要注意：draft confidence 是启发式价值估计，不自动等于校准后的真实接受概率。

### 8.3 EAGLE-3：面向 rollout 训练，减少不必要的特征约束

作者说明的两个核心变化是：去掉 EAGLE 原有的特征预测约束；融合目标模型低、中、高层信息，并通过 training-time testing 模拟推理时的过程。[6]

**训练流程：**

1. 在业务数据或通用数据上运行冻结的目标模型，提取所选层隐藏状态；
2. 融合多层特征，输入草稿模块；
3. 在训练中模拟草稿展开，使后续步骤暴露于更接近实际 rollout 的状态；
4. 主要围绕 token 预测／教师分布进行监督，而非要求逐维精确恢复目标未来特征；
5. 导出与目标 checkpoint、tokenizer 和特征层约定匹配的草稿权重。

**推理：**目标模型提供多层辅助特征 → 轻量 drafter 递归生成候选 → 验证 → 更新特征和缓存。

**不要误解：**“去掉特征回归损失”不等于“完全不依赖目标模型特征”；“训练时模拟测试”也不是把测试集用于训练。

### 8.4 后续扩展：P-EAGLE、EAGLE3.1

所读取的 SpecForge 主分支列出 P-EAGLE 与 EAGLE3.1。前者被标为 Parallel EAGLE，后者标注 attention drift。它们表明特征草稿内部也在继续探索并行化与分布对齐。本文不依据名称推导完整训练配方；本轮只把它们列为已发现的后续阅读入口。[12]

## 9. MTP：多 token 训练目标，不是一个固定的推理算法

MTP（Multi-Token Prediction）首先是一类训练设计：在标准下一 token 目标之外，让模型预测更远未来。不同工作可以采用并行预测头、顺序相连的模块、共享嵌入／输出头等结构；不能把所有 MTP 都画成一张图。[9]

### 训练

可以概括为：

$$\mathcal L=\mathcal L_{\mathrm{next}}+\sum_{j=1}^{D}\lambda_j\mathcal L_{\mathrm{future},j}.$$

DeepSeek-V3 一类设计使用顺序连接的 MTP 模块，将上一预测阶段的表示与相应未来 token 嵌入结合，再预测更远未来。训练时可用真实序列做监督；推理草稿需要使用真实可获得的 token／候选，不能偷看未来标签。

MTP 常在主模型训练阶段引入，也可以针对支持的架构训练相关模块。原生已有 MTP 的模型不一定需要独立小型 LM，但辅助模块依然有参数、算力与缓存成本。DeepSeek-V3 的官方权重说明还明确：该次开源权重包含 1 个 MTP 模块；训练概念中的多偏移能力，不代表发布权重一定附带同样数量的预测模块。

### 推理

1. 主干生成下一 token 所需状态；
2. 原生 MTP 模块产生额外候选，可按实现递归展开；
3. 主目标模型批量验证；
4. 按验证规则提交并回退缓存。

**结论：**“训练时能预测多个 token”和“推理时可以不验证直接输出多个 token”是两件事。后者若要保持原目标分布，仍需相应算法保证。

## 10. 自投机与 LayerSkip：把目标模型自己当 drafter

### 训练

需区分两类：

- Draft & Verify 类方法探索通过跳层形成便宜草稿，可不重新训练目标模型，但选层本身是一个质量／成本问题；
- LayerSkip 通过专门的训练配方，让中间层更适合早退；其思路包括层 dropout 与中间层早退预测监督。作者仓库明确指出，想观察预期加速，应使用按 LayerSkip 配方训练过的 checkpoint。[7]

### 推理

浅层网络先生成候选；较深层对候选进行批量验证，复用能复用的浅层计算与 KV；接受后提交对应缓存，拒绝后回退。

它降低了同时驻留独立草稿模型的需求，但早退太浅时接受率可能很低。自投机的“无损”相对于用于完整验证的那个模型；经过持续预训练的 LayerSkip checkpoint 不应与原始 checkpoint 的分布混为一谈。

## 11. DFlash 与 DFlash2：把昂贵的草稿主干并行化

### 11.1 DFlash：块扩散草稿，目标模型仍是自回归验证器

DFlash 官方将其定位为用于投机解码的轻量 block diffusion 模型。[11]

**关键直觉：**不要为了得到 K 个候选，让草稿 Transformer 主干串行运行 K 次。利用目标前缀特征，给未来位置提供掩码／待预测槽位，通过轻量块级计算同时提出一段候选。

这不等于把目标 LLM 改造成扩散模型，也不等于每一轮都需要执行传统扩散生成中的大量去噪步数。

**训练流程（按本轮核对的 SpecForge 实现抽象）：**

1. 用冻结目标模型获得所选层隐藏特征和 token 序列；
2. 从序列选取多个 anchor，构造固定长度草稿块；
3. 根据配置的掩码与注意力规则，在可见前缀条件下预测后续 token；
4. 用有效 token mask 过滤 padding 与不参与监督区域，对草稿块计算预测损失；
5. 可使用位置权重及实现支持的教师分布对齐目标；不能把所有可选损失说成原始 DFlash 论文的默认配方。

**推理流程：**目标前缀特征 → 一次块草稿前向 → 得到候选链／块 → 自回归目标模型并行验证 → 接受连续前缀并修正 → 更新缓存。

**主要矛盾：**草稿主干更并行，不代表未来 token 的相关性自动解决。多个位置各自看起来合理，不保证拼在一起后仍有长的可接受前缀。

### 11.2 DFlash2：并行主干＋局部关联＋候选路径选择

DFlash2 官方代码仓库提供独立入口，SpecForge 的专门说明给出可核对的实现细节：[11][12]

- 在草稿 attention／MLP 子层周围加入分组动态卷积，增强局部依赖；
- 在各位置 unary top-k 候选中，利用依赖前序 token 的低秩转移分数选择更连贯路径；
- 草稿结构变化发生在 drafter 内，目标特征采集协议与 DFlash 路线复用。

**训练：**保留底层块预测目标，增加 selector 的分类交叉熵；所读实现中，selector 只在真实标签落入严格 unary top-k 候选集合时计算该项损失，另有 warmup／ramp 等可选调度。

**推理：**并行主干产生特征和候选 → 轻量路径选择 → 目标模型验证。更便宜的依赖校正仍可能含顺序部分；“keep drafting parallel”不能误读成整条解码链没有任何串行依赖。

**可观测指标：**unary top-k recall、selector 条件准确率、连续前缀存活率、草稿主干／选择器耗时。候选池没有覆盖正确 token 时，再强的 selector 也救不回来。

### 11.3 相关延伸：Domino、D-PACE

所读 SpecForge README 将 Domino 描述为带 GRU logit correction 的 DFlash 扩展，将 D-PACE 列为 DFlash 可选训练目标。它们分别提示两个研究方向：用便宜循环结构补块内依赖，以及让训练更贴近真正的前缀收益。完整算法和原论文结论仍需后续全文核对，不在此编造独立的性能排名。

## 12. DSpark：用依赖校正提高候选质量，用置信度决定验证预算

本节依据 SpecForge 的 `DSparkDraftModel`、训练目标实现和 vLLM adaptive verification 文档。原论文入口为 arXiv:2607.05147；本文未读取其全文，因此以下不冒充逐句复述论文。[12][13]

### 结构

所读代码中，DSpark 复用 DFlash 类块草稿主干，并允许增加：

- Markov 依赖校正头：让后继 token 的 logits 依赖前序 token；支持 vanilla、gated、RNN 等配置；
- confidence head：估计候选位置的可接受程度，供后续调度参考。

这意味着“并行生成草稿特征”和“轻量建立 token 间依赖”可以分工，不必把完整 Transformer 主干重新串行跑一遍。也不能将某个 RNN 配置直接写成所有 DSpark 都使用 LSTM。

### 训练：不只监督对错，还监督分布重叠

所读训练实现包含三项可配置目标：

$$\mathcal L=\lambda_{\mathrm{CE}}\mathcal L_{\mathrm{CE}}+
\lambda_{\mathrm{L1}}\lVert p-q\rVert_1+
\lambda_{\mathrm{conf}}\mathcal L_{\mathrm{conf}}.$$

其中 p、q 是相同条件下、经对应输出头构造的目标／草稿分布。经典采样的单步平均接受概率为：

$$\sum_v q(v)\min(1,p(v)/q(v))
=\sum_v\min(p(v),q(v))
=1-\tfrac12\lVert p-q\rVert_1.$$

因此分布 L1 对齐直接联系到理论单步接受概率。所读实现将这一数值作为 detached soft target，通过 BCE-with-logits 训练 confidence head。

**训练流程：**采集目标层特征与所需教师末层特征 → 构造 anchor 和块 → 并行草稿主干前向 → 用前序 token 做 Markov 校正 → token CE／分布 L1／confidence 联合优化 → 导出含 projector、head 配置的草稿。

训练中的固定数据、teacher-forced 前序 token 与真实推理中的生成前缀存在偏移，离线置信准确率不应直接充当线上接受率。

### 推理：草稿与验证预算是两个环节

1. 块并行主干生成 proposal 特征；
2. 若启用 Markov 头，按实际前序 token 校正并产生候选；
3. confidence head 给出位置级估计；
4. 服务系统可据此截短待验证块、分配全局验证预算；
5. 目标模型进行真正的验证与采样校正；
6. 提交有效前缀，回退无效 KV。

**重要区分：**置信度调度不替代接受判定。只要按正确的 proposal 与验证规则处理，少验证一些候选可以只改变性能；但如果根据已经采出的具体 token 选择性改变候选使用方式，仍需检查实现是否正确处理了由选择机制带来的条件分布变化，不能泛化为“任意裁剪都无损”。

### 从位置置信度到前缀收益

若 c_j 近似表示“前面均存活时，第 j 个候选被接受的条件概率”，则：

$$P(A\ge i)\approx\prod_{j=1}^{i}c_j,\qquad
\mathbb E[A]\approx\sum_{i=1}^{K}\prod_{j=1}^{i}c_j.$$

如果这些条件概率是准确的链式条件概率，连乘对应链式法则；真实预测器与实际前缀不完全匹配时，这是估计而非事实。

vLLM 当前 adaptive verification 文档描述了按前缀 survival probability 在请求间分配验证 token 预算，并结合启动时 profiling 的成本模型选取预算。该能力属于服务系统，不应全部归为 DSpark 架构本身；没有专门 confidence head 的方法也可以使用其他在线估计器。[13]

## 13. 树验证、链验证与混合投机：系统层不能缺席

### 链与树的区别

链只验证一条续写；树同时保留多个可能分支，提高命中连续前缀的机会。tree attention 必须保证节点只能读取自己的合法祖先与共享前缀，不能看到兄弟分支。

树越宽不一定越快：节点占用验证算力、KV、attention metadata 和图捕获预算。尤其在高并发时，更多候选可能挤占其他请求的有效 decode。

### 混合投机不应只是“把所有方法叠上去”

可考虑的组合包括：

- n-gram 高命中时优先用检索，未命中再用模型草稿；
- 根据负载动态调整 K，负收益时切回普通 decode；
- 块并行主干＋轻量循环／Markov 校正；
- 按任务域、请求长度或在线接受统计路由不同 drafter；
- 级联：极便宜的草稿先被中等模型检查，再由大模型验证。

组合后需重新定义 proposal、验证与缓存契约。不能把多个单独正确的方法拼起来，就默认采样分布仍然正确。

### 为什么高并发可能让加速消失？

低并发时目标模型常有空闲计算，额外验证较划算；高并发时普通 decode 已经有很好的批处理利用率，候选验证开始与真实 token 竞争计算和 KV。

因此同一方法可能降低单请求 TPOT，却降低整体吞吐，或者使 P99 延迟恶化。投机解码首先是一个工作负载相关的系统优化，而不是一个可以脱离负载发布的固定倍数。

---

# 第三部分：如何设计可信的训练与推理实验

## 14. 统一训练流程：先确保“训练的东西”和“上线用的东西”一致

1. **冻结实验定义。**记录 target／draft 权重版本、tokenizer、chat template、特征层、精度、mask 与 label shift。
2. **构造业务代表性数据。**覆盖真实 prompt 分布、长短输入、代码／数学／对话等目标域，隔离训练与评估样本；不要只看容易高接受的任务。
3. **生成监督。**可用真实 token、目标模型续写、soft logits 或隐藏特征。记录是离线 teacher forcing、在线目标采集，还是带 rollout 的训练。
4. **按机制选择损失。**独立草稿做 token／分布蒸馏；Medusa 用多偏移监督；EAGLE 类考虑特征与 rollout；DSpark 类额外对齐接受概率与置信度。
5. **评估前缀，而非仅评估 token。**逐位置 accuracy 高，不代表连续前缀长；离线 teacher-forced 指标不等于真实 rollout 指标。
6. **导出并实际加载。**保证训练与服务的投影层、Markov／selector 头、LM head、tokenizer 和层索引一致。

本文不提供“通吃所有方法”的训练命令。上述项目接口变化较快，直接抄当前 main 分支命令而不锁定 commit，复现性很差。

## 15. 推理验收：正确性、性能和稳定性分开测

### 正确性

- 贪心：对相同目标模型比较输出 token 序列，覆盖 EOS、长度上限、长上下文、重复惩罚与约束解码；记录数值精度。
- 随机：先用可枚举小词表验证概率质量与拒绝残差，再做统计分布检验；不能以“同 seed 输出文本不一样”直接判断算法有损，因为随机数消耗路径不同。
- 缓存：专门覆盖第一个候选就拒绝、全部接受、中途拒绝、多分支共享前缀、连续多轮回退。
- 混合策略：核对实际采样 q 与验证使用 q 完全一致；明确置信筛选、树遍历和 bonus token 的实现契约。

### 性能

至少报告：

| 维度 | 应记录的内容 |
|---|---|
| 模型与软件 | target／draft checkpoint、参数量、精度、服务框架版本／commit、kernel 配置 |
| 硬件 | GPU／NPU 型号与数量、TP／PP、互联、显存、是否同卡部署 |
| 负载 | 并发／到达率、输入输出长度分布、任务域、warmup、缓存命中策略 |
| 采样 | temperature、top-k／top-p、停止条件、结构化输出约束 |
| 草稿 | K、树节点预算、块大小、动态调度策略、额外内存 |
| 效果 | 平均接受长度、逐位置条件接受率、拒绝位置分布、目标调用次数 |
| 延迟 | TTFT、TPOT／ITL、端到端延迟，至少 p50／p95／p99 |
| 吞吐与成本 | 有效 output tokens/s、满足 SLO 的 goodput、GPU 成本／千 token |
| 时间分解 | 草稿、验证、采样／selector、KV 操作、通信、调度 |

本文没有统一复现以上方法，因此不制作跨论文加速比排行榜。不同论文里的“3×”“5×”没有相同目标模型、负载、精度和基线时，不具有直接排序意义。

### 稳定性

关注 OOM、KV 泄漏、动态形状和 CUDA graph 覆盖、草稿掉线／加载失败后的降级、负载突增时的自动关闭、量化 target 更新后 drafter 的兼容性。

对于 AIGC／多模态服务，还要明确图像编码、prefill 与 decode 各占多少：若瓶颈主要是视觉编码或排队，文本 decode 加速不会按相同比例改善端到端时延。本文的概率算法也不能直接迁移为 DiT 的通用加速方案。

## 16. 工程选型建议

| 你的条件 | 优先尝试 | 理由与验证重点 |
|---|---|---|
| 先验证收益，没有训练预算 | n-gram／suffix 或匹配的小型 draft | 最快获得业务接受长度和端到端成本曲线 |
| 有原生 MTP checkpoint | 框架匹配的 MTP 路径 | 复用模型能力；确认当前版本实际支持该模型及采样模式 |
| 能获得目标特征并训练草稿 | EAGLE-3 与受支持的块并行方案 | 比较草稿成本、rollout 接受率与服务兼容性 |
| 目标是降低草稿串行开销 | DFlash／DFlash2／P-EAGLE 等 | 单独测主干、选择器与验证耗时，不只看总接受率 |
| 服务负载变化大 | 动态 K／adaptive verification／负收益降级 | 与普通 decode 在相同 SLO 下比较 goodput |
| 显存很紧，想减少独立模型 | 有合适 checkpoint 的自投机／LayerSkip | 考察早退质量、缓存复用和模型许可 |
| 输出高度重复 | 检索草稿 | 验证命中率、缓存隔离和近乎零训练成本的收益 |

建议的落地顺序是：普通 decode 基线 → 最小成本草稿 → 目标业务评测 → 一个训练式 drafter → 动态调度。先把“什么时候不该投机”做出来，通常比无条件扩大草稿更有价值。

## 17. 后续单篇深挖的阅读顺序

这篇综述已经给出每类方法的训练／推理骨架。若拆成知乎系列，可按下面顺序继续，每篇都包含结构图、张量维度、训练标签、损失、一次完整推理轨迹、KV 回退与实验：

1. **经典 SD：从接受公式到无损性证明。**实现小词表验证器，检查 top-p 与拒绝残差。
2. **Medusa → Hydra → ReDrafter：未来 token 依赖怎么表达？**比较多头、条件头与循环草稿。
3. **EAGLE 1／2／3：特征预测、动态树与训练时 rollout。**重点核对 token／feature shift。
4. **MTP 与自投机：训练目标怎样变成推理能力？**分开讲不同 MTP 结构与基线问题。
5. **DFlash → DFlash2：块并行草稿与轻量路径关联。**比较 unary 覆盖率、selector 与前缀存活。
6. **DSpark：从分布重叠到置信度调度。**逐项解释 CE、L1、confidence 和验证预算。
7. **服务实战：投机解码为何在低并发快、高并发不一定快？**测量 SLO、goodput 与动态降级。

---

# 参考资料与核对说明

以下编号用于对应正文。论文入口用于进一步核验；“已读取”仅指明确列出的仓库、文档或代码，不代表论文全文已读取。

**[1] 经典投机解码。**Leviathan et al., *Fast Inference from Transformers via Speculative Decoding*（ICML 2023）。[论文页面](https://proceedings.mlr.press/v202/leviathan23a.html)；[arXiv](https://arxiv.org/abs/2211.17192)。本轮原论文正文未读取；公式在文中独立推导，并用小词表数值校验。

**[2] 经典投机采样。**Chen et al., *Accelerating Large Language Model Decoding with Speculative Sampling*。[论文](https://arxiv.org/abs/2302.01318)。已读取[第三方复现说明](https://github.com/romsto/Speculative-Decoding)作为交叉线索，不将其当作者官方实现。

**[3] Medusa。**[作者仓库](https://github.com/FasterDecoding/Medusa)；[论文](https://arxiv.org/abs/2401.10774)。已读取 README 中的结构、Medusa-1／2、训练说明和 typical acceptance 描述。

**[4] Hydra。**[作者仓库](https://github.com/zankner/Hydra)；[论文](https://arxiv.org/abs/2402.05109)。已读取 README 的 sequentially dependent heads、训练配置示例和冻结基座说明。

**[5] Recurrent Drafter。**[Apple 官方仓库](https://github.com/apple/ml-recurrent-drafter)；[论文](https://arxiv.org/abs/2403.09919)。已读取 README 与文档索引，包含非贪心分布匹配的 Known Issues。

**[6] EAGLE 系列。**[作者仓库](https://github.com/SafeAILab/EAGLE)；[EAGLE](https://arxiv.org/abs/2401.15077)、[EAGLE-2](https://arxiv.org/abs/2406.16858)、[EAGLE-3](https://arxiv.org/abs/2503.01840)。已读取 README 和 `eagle/train/main.py`，并交叉核对 SpecForge 训练说明。

**[7] 自投机与 LayerSkip。**[LayerSkip 官方仓库](https://github.com/facebookresearch/LayerSkip)；[论文](https://arxiv.org/abs/2404.16710)；Draft & Verify [论文入口](https://arxiv.org/abs/2309.08168)。本轮已读 LayerSkip README；后者列为进一步阅读，未完整核对原实现。

**[8] Lookahead。**[作者仓库](https://github.com/hao-ai-lab/LookaheadDecoding)；[论文](https://arxiv.org/abs/2402.02057)。已读取 README 的 Jacobi、候选池和双分支验证说明。

**[9] MTP。**[DeepSeek-V3 官方仓库](https://github.com/deepseek-ai/DeepSeek-V3)；[技术报告](https://arxiv.org/abs/2412.19437)；Gloeckle et al. *Better & Faster Large Language Models via Multi-token Prediction* [论文](https://arxiv.org/abs/2404.19737)；[vLLM MTP 文档](https://github.com/vllm-project/vllm/blob/main/docs/features/speculative_decoding/mtp.md)。本轮已读 DeepSeek README、官方权重结构说明和 vLLM 文档；详细模块公式须继续对照技术报告全文，不把不同 MTP 架构合并。

**[10] 检索路线。**REST [作者仓库入口](https://github.com/FasterDecoding/REST)、[论文](https://arxiv.org/abs/2311.08252)；[vLLM 方法索引](https://github.com/vllm-project/vllm/blob/main/docs/features/speculative_decoding/README.md)。本轮已读框架方法索引与 REST 官方 README 的检索、Trie 剪枝及树验证说明；具体采样实现待进一步逐文件核对。

**[11] DFlash 系列。**[作者仓库](https://github.com/z-lab/dflash)；[DFlash 论文](https://arxiv.org/abs/2602.06036)；[DFlash2 官方博客入口](https://inco.ai/blog/dflash2/)。本轮已读取官方仓库 README 与 SpecForge DFlash2 文档；博客全文未读取，不独立引用其性能宣称。

**[12] SpecForge。**[官方仓库](https://github.com/sgl-project/SpecForge)；[训练文档](https://github.com/sgl-project/SpecForge/blob/main/docs/sections/basic_usage/training.md)；[DFlash2 文档](https://github.com/sgl-project/SpecForge/blob/main/docs/sections/concepts/DFlash2.md)；[DSpark 草稿结构](https://github.com/sgl-project/SpecForge/blob/main/specforge/modeling/draft/dspark.py)；[DSpark 训练目标所在文件](https://github.com/sgl-project/SpecForge/blob/main/specforge/algorithms/common/dflash_family_model.py)。已读取上述文档与相关代码。README 给出 [DSpark 论文入口](https://arxiv.org/abs/2607.05147)、[P-EAGLE](https://arxiv.org/abs/2602.01469)、[D-PACE](https://arxiv.org/abs/2605.18810)、[Domino](https://arxiv.org/abs/2605.29707)；这些论文全文本轮未读。

**[13] 服务调度。**[vLLM Adaptive Verification](https://github.com/vllm-project/vllm/blob/main/docs/features/speculative_decoding/adaptive_verification.md)。已读取该文档。其与同仓库 README 的支持范围描述存在不同步，本文对具体调度机制优先使用专门文档，并要求实验锁定实际部署版本。

资料获取与代码快照的本地记录见 [资料核对说明](source-notes.md)。框架 main 分支持续变化，实际实验应固定 commit；本文不将“当前主分支存在”表述为“任意已发布版本都支持”。


**[14] SGLang 投机解码指南。**[主指南](https://github.com/sgl-project/sglang/blob/main/docs/docs/advanced_features/speculative_decoding.mdx)；[自适应指南](https://github.com/sgl-project/sglang/blob/main/docs/docs/advanced_features/adaptive_speculative_decoding.mdx)；[算法枚举](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/speculative/spec_info.py)。本轮已读取；接口和限制以对应提交为准。

**[15] vLLM 当前接口。**[配置枚举](https://github.com/vllm-project/vllm/blob/main/vllm/config/speculative.py)；[Suffix](https://github.com/vllm-project/vllm/blob/main/docs/features/speculative_decoding/suffix.md)；[N-gram](https://github.com/vllm-project/vllm/blob/main/docs/features/speculative_decoding/n_gram.md)；[LiLiCorr](https://github.com/vllm-project/vllm/blob/main/docs/features/speculative_decoding/lilicorr.md)。本轮已读取，实际模型／采样约束需实验核验。

**[16] Sequoia。**[作者仓库](https://github.com/Infini-AI-Lab/Sequoia)；[论文入口](https://arxiv.org/abs/2402.12374)。已读作者 README 的硬件 growmap 与接受率向量构造说明。

**[17] SpecExec。**[作者仓库](https://github.com/yandex-research/specexec)；[论文入口](https://arxiv.org/abs/2406.02532)。已读 README 的预算和 offloading 评测入口。

**[18] DistillSpec。**[论文入口](https://arxiv.org/abs/2310.08461)。仅列为待全文核实的草稿蒸馏路线；本轮不引用训练细节或性能数字。

**[19] Token Recycling。**[论文入口](https://arxiv.org/abs/2408.08696)；[第三方复现说明](https://github.com/smpanaro/token-recycling)。已读取第三方 README；不是官方实现核验。
