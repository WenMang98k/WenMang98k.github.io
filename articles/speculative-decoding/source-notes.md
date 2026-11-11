# 投机解码调研：资料核对与后续清单

研究日期：2026-11-11。主文是综述与单点流程说明的初稿；现整理入个人网站仓库供公开阅读，尚未发布到知乎。

## 本轮交付

- `README.md`：可编辑正文，包含方法地图、正确性、训练与推理流程、工程选型和参考资料。
- `index.html`：带目录、表格、数学公式的独立阅读版。
- 原始抓取资料留在研究工作目录；仓库不批量转载上游 README／源码。正文保留实际读取来源与限制。
- `scripts/verify_probabilities.py`：有限词表概率恒等式验证，不是 LLM 正确性或性能测试。

## 证据层级

| 内容 | 已核对的材料 | 仍需补的工作 |
|---|---|---|
| 经典 SD／sampling | 公式独立推导、小词表校验、复现仓库引文 | 原论文全文和更复杂 tree／parallel proposal 的证明 |
| Medusa | 作者 README，训练方案与 typical acceptance | sampler 实现与不同接受策略的精确保证 |
| Hydra | 作者 README、训练示例、冻结基座说明 | 每个结构变体的目标函数与输入张量 |
| ReDrafter | Apple README、文档索引、Known Issues | 训练损失代码、循环结构、树压缩和 sampling 实现逐函数核对 |
| EAGLE 1／2／3 | 作者 README、训练入口、SpecForge 文档 | 按版本锁定 commit，对齐 feature/token shift 与 tree posterior |
| MTP | DeepSeek-V3 README、README_WEIGHTS，vLLM MTP 文档 | 原技术报告全文，训练结构与发布模块／服务递归的区别 |
| LayerSkip | 官方 README 与 checkpoint 使用要求 | 训练配方全文和缓存复用实现 |
| Lookahead | 作者 README 的 Jacobi、n-gram 池与双分支 | 多候选随机采样的详细实现 |
| REST | 官方 README 的检索、Trie 剪枝与 tree mask | sampler 与数据存储隔离细节 |
| DFlash | 官方 README，SpecForge 草稿结构／训练说明 | 原论文全文；训练 masking 与推理实际语义逐项匹配 |
| DFlash2 | 官方 README，SpecForge 专门文档和训练描述 | 官方博客全文及原作者版本与框架实现的差异 |
| DSpark | SpecForge 草稿模型、CE／L1／confidence 训练代码、vLLM 调度文档 | 原论文全文、官方参考实现与框架移植的差异 |
| P-EAGLE／EAGLE3.1／Domino／D-PACE | SpecForge 方法索引 | 作为后续阅读入口，未做完整单点评审 |

## 环境限制

arxiv.org 请求返回代理 403，未绕过策略；官方 GitHub 与 raw.githubusercontent.com 可访问。因此正文将原论文链接与实际已读内容分开标注。未将搜索结果摘要当作方法细节依据。

框架 main 分支文档会变化。本地 SHA256 可标识采集内容，但不等于固定了上游 commit；正式复现实验前必须固定各仓库 commit、权重 revision 和依赖版本。

发现 vLLM 索引与专门 adaptive verification 文档的支持范围文字不同步。正文对机制采用专门文档，不把当前 main 分支能力泛化到任意 release。

## 建议下一轮优先补核

1. 开放 arxiv.org 的阅读访问后，核对经典 SD、EAGLE-3、DFlash、DSpark 原文中的算法和假设。
2. 为 EAGLE-3、DFlash2、DSpark 各选一个固定 commit，写出张量维度、掩码图、label shift、rollout 和一次验证后的 KV 状态。
3. 对经典 SD 写可运行的概率参考实现；对树／并行 proposal 不沿用未经证明的链式采样逻辑。
4. 在同一个 target、同一硬件、同一 sampling config、同一数据与负载下运行基线与 2—3 个方法；再谈加速比排名。
5. 发布前整理知乎公式排版、图表、引文，并确认篇幅和作者署名；本轮没有自动发布。

## 数值验证结果

`python3 scripts/verify_probabilities.py`：1000 组随机离散分布与 3 组边界／示例通过。

验证内容：接受概率质量＋拒绝残差＝目标分布；分布重叠＝1−TV。它不证明任意工程实现的正确性，也不代表已执行 GPU 性能评测。

## 框架支持补充核对

已读取 vLLM 的 speculative 配置枚举、suffix／n-gram／LiLiCorr／adaptive verification 指南，以及 SGLang 的投机解码主指南、adaptive 指南与算法枚举。新增框架支持矩阵，区分当前 main 分支接口、模型权重要求与待复核路线。未安装这两个 GPU 服务框架，未执行 GPU 加速实验。
