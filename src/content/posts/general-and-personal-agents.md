---
title: "通用 Agent 与个人 Agent：国内外产品版图和趋势"
description: "对比国内外大厂、创业公司与开源项目的 Agent 路线，梳理跨任务执行、个人常驻、浏览器、办公与编程产品的现状。"
date: 2026-11-20
section: tech
tags: ["Agent", "产品调研", "个人助手", "工作流"]
draft: false
---

通用 Agent 的卖点是**一次给目标，跨工具交付成果**。个人 Agent 的卖点是**持续理解同一用户，在适当的时间帮他接手任务**。一个产品可以兼具两者，不过用户愿不愿意首次委托复杂任务，和愿不愿意长期授权邮箱、文件与日历，是两道不同的产品门槛。

截至 **2026 年 11 月 20 日**，公开材料里可以看到三条比较清楚的路线。大厂把 Agent 放进已有办公、搜索、聊天和云平台入口；创业公司把长任务交付或浏览器、代码等具体执行场景做深；个人常驻助手则在探索本地部署、跨渠道和长期记忆的组合。本文选取代表产品，逐项区分公开源码、商业产品定位和实际可用范围。部分商业官网在本次环境中无法访问，渠道、地区、套餐与配额没有统一复核。仓库更新与商业普及度使用不同证据判断。

## 两条产品轴：任务广度与关系持续时间

| 产品方向 | 用户交给它什么 | 交付形态 | 需要重点验证 |
| --- | --- | --- | --- |
| 通用任务 Agent | 调研竞品、整理附件、比较报价并出报告 | 来源可追溯的文档、表格、网页或任务结果 | 长任务的完成率、工件质量、失败恢复、单位成功成本 |
| 个人 Agent | 我的日程、偏好、资料、周期性订阅和提醒 | 延续上下文的帮助、定时摘要、授权内代办 | 记忆能否查看和删除、是否会误触发、退出后能否续跑 |
| 垂直 Agent | 修一个仓库的缺陷、处理一张工单或审核报销单 | 补丁与测试、工单状态、审计记录 | 领域正确性、异常交接、数据与权限边界 |
| Agent 平台 | 工具、知识、组织规则、发布与运维入口 | 工作流、SDK、运行环境、追踪与治理 | 集成成本、权限隔离、可观测性、部署方式 |

例如，给系统一件“整理三家供应商报价”的任务，通用 Agent 可以产出可编辑表格；个人 Agent 还可以记住比较偏好并每周更新价格；接入采购系统后，企业 Agent 还要按组织规则审批并留下记录。

## 海外：大厂争入口，创业公司争执行体验

| 公司或项目 | 产品领域与代表入口 | 能从公开材料读到的发展状态 |
| --- | --- | --- |
| OpenAI | ChatGPT agent 面向跨网页、文件与工具的任务；Agents SDK 面向开发者 | ChatGPT agent 有公开发布材料。已核实的 SDK 提供沙箱任务、语音与运行追踪，适合分析从消费端任务到开发者运行时的两条线。消费端账户权限与额度需另查。 [ChatGPT agent](https://openai.com/index/introducing-chatgpt-agent/) · [Agents SDK](https://github.com/openai/openai-agents-python/blob/6c4e4a29cd08eef454db6980d74da6afc8939d24/README.md) |
| Anthropic | Claude Cowork 面向桌面知识工作；Claude Code 面向代码；Agent Skills 提供可复用能力的组织方式 | Cowork 的公开材料以研究预览为历史起点，当前套餐未复核。Claude Code 与 Skills 仓库可以核实，显示产品正在复用文件、工具和能力声明这套工作方式。 [Cowork 发布材料](https://claude.com/blog/cowork-research-preview) · [Claude Code](https://github.com/anthropics/claude-code/tree/688120e365aff198e348965cc115f7dd79275dff) · [Skills](https://github.com/anthropics/skills/tree/e4bb5d960d87126ff3b3a8af1a104b7db67560ba) |
| Google | Gemini Deep Research 面向研究，ADK 面向 Agent 开发和部署 | 已核实 ADK 支持图式工作流、人工介入，以及容器化部署到 Cloud Run 等环境。Gemini 对应功能的地区与套餐需看产品页。 [Gemini Deep Research](https://gemini.google/overview/deep-research/) · [ADK](https://github.com/google/adk-python/blob/c81876a49911745248f2b774c02770ed211ff460/README.md) |
| Microsoft | Microsoft 365 Copilot、Copilot Studio 面向办公协作和内部流程 | 主要产品领域是企业文档、协作和业务工作流。现阶段比较宜围绕 Microsoft 365 里的身份、连接器、组织数据和审批，本文未复核具体 Agent 的许可或租户开放范围。 [Microsoft 365 Copilot](https://www.microsoft.com/en-us/microsoft-365/copilot) · [Copilot Studio](https://learn.microsoft.com/en-us/microsoft-copilot-studio/) |
| Apple | Siri 与 Apple Intelligence 面向系统级个人助理 | 代表设备与系统数据入口的路线。已公开的产品方向包括个人上下文与跨 App 操作，具体功能落地应逐项核对系统版本、设备、语言和地区。本文未复核发布范围。 [Apple Intelligence](https://www.apple.com/apple-intelligence/) |
| Perplexity、Genspark | Comet 从浏览器进入阅读和网页任务；Genspark 从研究与内容产物进入知识工作 | 两者代表不同的入口选择。Comet 把上下文放在用户正在看的网页附近；Genspark 的公开定位更靠近交报告、幻灯片等成果。当前地区、套餐与动作权限未复核。 [Comet](https://www.perplexity.ai/comet) · [Genspark](https://www.genspark.ai/) |
| Browser Use、OpenHands | 前者聚焦浏览器执行，后者聚焦软件工程任务与持续自动化 | 已核实 Browser Use 有开源库与云服务入口。OpenHands Agent Canvas 文档包含本地、远程、云后端以及定时或事件自动化，展示了垂直产品怎么延长任务生命周期。 [Browser Use](https://github.com/browser-use/browser-use/blob/ccbdb8095fbaf903c748880a600054a34c7b6191/README.md) · [OpenHands](https://github.com/OpenHands/OpenHands/blob/428b3dc2a0e8d3d0f689a909e2607e33d973da76/README.md) |

## 国内：从模型和办公入口走向执行层

| 公司或项目 | 产品领域与代表入口 | 能从公开材料读到的发展状态 |
| --- | --- | --- |
| 阿里巴巴及通义生态 | Qwen-Agent 面向开发者；AgentScope 的 QwenPaw 面向个人常驻助理 | 原 CoPaw 仓库当前以 **QwenPaw** 为名，源码文档列出了本地或云端部署、记忆、定时技能、插件与钉钉、飞书等渠道。这个方向适合观察用户是否愿意把长期资料和日常任务交给一个自管助手。 [Qwen-Agent](https://github.com/QwenLM/Qwen-Agent/blob/31a4d36d123688581a9e9744427272b33ce940e0/README.md) · [QwenPaw](https://github.com/agentscope-ai/QwenPaw/blob/a6549d749a648bbe88312c477fa4149f2f0aef1e/README.md) |
| 字节跳动及扣子生态 | Coze Studio 面向可视化搭建；DeerFlow 面向研究和长任务执行 | Coze Studio 源码列出工作流、知识库、插件与发布管理。DeerFlow 2.0 的源码文档显示它已经从研究工作流扩展到包含沙箱、文件、长期记忆与子 Agent 的执行框架。两者分别面向流程配置和任务执行。 [Coze Studio](https://github.com/coze-dev/coze-studio/blob/fefb05ff27be1da939612fbf9faf5db62583b8ae/README.md) · [DeerFlow](https://github.com/bytedance/deer-flow/blob/8113420c8fb5f1ab76b27e64f6edcb0c72d3c1ab/README.md) |
| 腾讯 | Youtu-Agent 面向通用工作流，Youtu Tip 探索本地桌面入口 | 源码文档提供资料调研、数据分析、文件整理、演示稿等用例，并列出 macOS 和离线模型方向的 Youtu Tip。当前商业渠道和开放范围未复核。 [Youtu-Agent](https://github.com/TencentCloudADP/youtu-agent/blob/631ee33fcb0796e8ff4327251d56929f0cd675a5/README.md) |
| 百度 | 文心智能体平台面向搭建和分发 | 作为搜索和应用入口侧的跟踪样本，适合关注智能体能否从获客、问答走进商家流程。平台当期权限、活跃和付费数据本轮未复核。 [文心智能体平台](https://agents.baidu.com/) |
| 月之暗面 | Kimi 面向个人知识工作；Kimi Code 面向终端编程 | 原 Python 版 Kimi CLI 已归档，并明确由 **Kimi Code CLI** 接替。项目更名与迁移有固定源码记录。编程产品的状态应与 Kimi 消费端助手分别比较。 [Kimi](https://www.kimi.com/) · [旧版迁移说明](https://github.com/MoonshotAI/kimi-cli/blob/9ab1286b8fe4e6bcd116949a27ce5e0ac3389c82/README.md) |
| 智谱、MiniMax | 智谱 Open-AutoGLM 探索移动端 GUI 执行；MiniMax Agent 面向通用任务和工件交付 | Open-AutoGLM 公开了移动设备交互项目，适合验证 App 界面变化、确认点和失败交接。MiniMax 的商业产品属于跟踪样本，本轮未复核地区、套餐或个人记忆范围。 [Open-AutoGLM](https://github.com/zai-org/Open-AutoGLM/blob/86f55382982fb054e8fc98ca80609dff8a2cdc3c/README.md) · [MiniMax Agent](https://agent.minimax.io/) |
| Manus | 由中国创业团队起步的全球化产品样本，面向异步通用任务 | 作为云端执行和交付件路线的跟踪样本，应按实际任务完成率、后台执行、导出和额度验证。公司归属及经营数据随时间变化，本轮未复核。 [Manus](https://manus.im/) |

开源个人助手也值得单列。**OpenClaw** 属于自托管、跨消息渠道的个人助手路线，用户或运营方要承担实例运行、授权和维护；**QwenPaw** 则提供了本地模型和国内聊天工具接入的公开样本。常驻助理的体验受在线状态、触发机制和数据管理共同影响，单次演示很难把这些差异测出来。[OpenClaw](https://github.com/openclaw/openclaw/tree/b93fc4052da08ddd486fec42dd3b9e3151d38fe0) · [QwenPaw](https://github.com/agentscope-ai/QwenPaw/blob/a6549d749a648bbe88312c477fa4149f2f0aef1e/README.md)

## 接下来值得跟踪的四个变化

1. **交付从回答变成可检查的成果。** 报告需要来源，表格需要可编辑，代码需要能运行测试。用户真正节省的是审查与返工后的时间。创业产品即使覆盖任务较少，也可以靠某一种成果的质量积累复用。
2. **任务从聊天窗口走向后台。** 沙箱、持久工作区、任务队列与恢复支持更长的执行。个人产品还要处理跨设备提醒、定时运行与状态回顾。把“关闭页面后是否继续执行”作为可实际测试的能力。
3. **个人关系围绕记忆与授权展开。** 记住写作偏好和读取邮件是两类权限。把记忆的来源、编辑、删除、导出和同步做成产品能力，再逐步开放发信、提交表单等有外部影响的动作。
4. **企业和个人的分发路径分化。** 企业场景已有组织身份、业务系统与审批，预算通常来自可追踪的流程收益。个人场景更多从浏览器、桌面和消息入口进入，需要更轻的首次授权与足够高的复用频率。工具和技能接口可以降低集成成本，实际差异仍会体现在流程完成、权限与运维上。

## 怎么做下一轮横评

- **通用任务。** 给产品同样的三份报价单与相同交付要求，观察是否能输出可编辑表格、引用原始页码、正确处理缺列。记录成功率、人工接管次数、耗时和每次可用交付成本。
- **个人任务。** 用隔离测试账户连续一周生成日程与订阅摘要，检查定时是否可靠、换会话能否保留偏好、撤销权限后是否停止读取，以及删除记忆后是否继续沿用旧偏好。
- **组织任务。** 用测试组织跑审批前的草稿流程，让管理员改一次权限，验证操作历史能否追到操作者、来源数据与审核节点。把是否正式开放、所在地区、席位或按量价格、失败回退写进同一张测试记录。

如果准备做一个新产品，可以先选一个每周反复发生、结果容易核验的任务，再决定需要多广的工具链和多深的个人上下文。个人助理可以从只读摘要或资料整理开始验证复用；团队侧可以从一个需要留痕的部门流程测交付与审批成本。这些是选型假设，本篇未做付费账号实测，也未用融资、流量或仓库热度替代真实采用数据。
