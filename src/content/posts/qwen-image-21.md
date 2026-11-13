---
title: "Qwen-Image-2.1：从模型结构到产品落地，怎么看懂 Turbo 与 Lightning 加速"
description: "32 层单流 DiT、7B 视觉生成组件、Qwen3-VL 8B、RGBA VAE 与前缀 KV 复用；结合 Lightning LoRA，梳理应用场景、性能口径及产品经理的上线判断。"
date: 2026-11-12
tags: ["AIGC", "Qwen-Image", "DiT", "模型加速", "产品设计"]
draft: false
---
## 先看结论：它是怎样一款模型

**Qwen-Image-2.1 是统一文本生图与图像编辑的开放权重模型。**值得关注的不是“又多一个生图模型”，而是它把中文／英文排版、多参考图编辑、原生透明图像与相对紧凑的视觉生成主干放在同一条流水线里。

| 一眼了解 | 已核实信息 | 不应误读成 |
|---|---|---|
| 视觉生成 | 约 7B、32 层 Single-Stream DiT | 整条服务只有 7B |
| 条件编码 | Qwen3-VL 8B | 只有很小的 CLIP 文本编码器 |
| 图像表示 | 64 通道 RGBA VAE、16×空间压缩 | PNG 后缀自动等于真实透明图 |
| 编辑能力 | 官方称最多 10 张参考图；支持局部标注与主体保持 | 10 张高分辨率图的成本与单图相同 |
| 缓存机制 | 条件前缀在去噪步之间复用 KV | 缓存整张正在生成的图，或者任意跨请求命中 |
| 加速交付 | 官方 Turbo 为 8 步；另有用户提供的 Lightning LoRA | Turbo 与 Lightning 是同一模型／同一采样配方 |
| 授权 | 官方仓库为 Qwen Research License，商用需另行许可 | 旧版或框架的 Apache 许可自动覆盖 2.1 权重 |

> **证据范围。**本文实际读取了 2.1 官方仓库、许可证、Diffusers 专用 Transformer／Pipeline，以及 LightX2V 的 2.1 指南、配置与 LoRA 代码；关键文件固定到文末提交。用户提供的 [Qwen-Image-2.1-Lightning 模型卡](https://huggingface.co/lightx2v/Qwen-Image-2.1-Lightning)被当前环境代理拦截，尚未读取其文件名、采样配方与许可。本文不借用旧版 Lightning 的倍数冒充 2.1 数据，也没有运行 GPU 性能实验。

## 版本先分清：Base、Turbo、Lightning 与 Pro API

| 名称 | 是什么 | 本轮确认到哪里 |
|---|---|---|
| `Qwen/Qwen-Image-2.1` | 统一生图／编辑基座 | 官方仓库与实现可读；示例通常为 40 步、无 CFG |
| `Qwen/Qwen-Image-2.1-Turbo` | 官方加速 checkpoint，沿用 7B 视觉架构 | 官方说明为 8 步，携带推荐 sampling sigmas |
| `lightx2v/Qwen-Image-2.1-Lightning` | 用户给出的第三方加速 LoRA 仓库 | 模型卡未读取；不能确认实际 4／8 步、rank、sigma、CFG、收益与许可 |
| Qwen-Image-2.1 Pro／Turbo API | 云平台提供的服务产品 | 官方 README 给出入口；定价、配额与服务条款需按 API 当前文档核对 |
| 旧 Qwen-Image／Edit／2512 Lightning | 其他代际的基座与加速适配器 | 不能因名称相近就与 2.1 互换 |

官方仓库记录：2.1 于 **2026-09-20**公开，Turbo checkpoint 与 Pro／Turbo API 的公告日期为 **2026-10-09**。这不意味着开源权重的所有能力、质量与商用 API 产品完全等价。

尤其要避免两个混淆：**官方 Turbo 不是“基座随便改成 8 步”**；第三方 Lightning LoRA 也不是 Turbo 的别名。二者的训练、采样调度、权重和许可要分别管理。

## 模型结构：7B 主干之外还有什么

先把结构分成两层看：左侧是 **Backbone 的输入、联合序列与输出头**，右侧展开一个 DiT block。绿色代表条件与缓存，蓝色代表目标计算，琥珀色代表时间调制。

<figure class="model-diagram">
  <div class="model-diagram-scroll" tabindex="0" role="region" aria-label="Backbone 结构图，可横向滚动">
    <a href="/images/qwen-image21/architecture.svg" target="_blank" rel="noopener"><img src="/images/qwen-image21/architecture.svg" width="1120" height="1180" alt="Qwen-Image-2.1 Backbone：txt_in 与 img_in 构造联合序列，经过 32 层单流 DiT；右侧展开 LayerNorm、调制、block-causal attention、SwiGLU 和两次门控残差。" /></a>
  </div>
  <figcaption>图 1 · Backbone 与单层算子。手机端可横向滑动，点击图片可打开 SVG 原图。</figcaption>
</figure>

**读图时留意三个细节。**参考图不只是拼接在文本后面，而是其 latent 替换预留图像位置；共享 modulation 给每层提供两组 scale / gate；缓存保存各层条件前缀的 K/V，而非缓存所有目标计算。图中序列块是示意，真实布局由输入指令与图像占位决定。

### 1. Qwen3-VL 8B：理解指令与参考图

文本指令和参考图共同进入 Qwen3-VL。它输出模型理解的条件表示，处理“把第一张图的人放进第二张图的场景”“保留商品形状，只换背景”等语义关系。

因此参考图不仅经过 VAE 压缩，也进入视觉语言理解。把预先算好的 VAE latent 直接当作参考图输入，不能替代这一路图像语义编码。Diffusers 的 Pipeline 也对这种用法作了检查。

可选的 Prompt Enhancer 则是另一层产品／推理工具，不是 7B DiT 的固定组成部分；开启它会改变输入内容、耗时和成本，需要单独评估。

### 2. 32 层单流 DiT：文本与图像在同一序列里协同

官方明确 2.1 使用 **Single-Stream DiT**。Diffusers 实现把条件图像 latent 放入编码器预留的图像位置，再附加目标图像 token，用同一串序列建模。

所读取的 `QwenImage21Transformer2DModel` 构造默认值如下；**这是实现默认值，不冒充已直接读取 Hugging Face checkpoint 的 config.json**：

| 字段 | 默认值 | 技术意义 |
|---|---:|---|
| `num_layers` | 32 | 单流 Transformer 层数 |
| `num_attention_heads` | 32 | 注意力头数 |
| `attention_head_dim` | 128 | 每头维度，主干宽度为 4096 |
| `context_in_dim` | 4096 | 条件特征投影输入维度 |
| `mlp_ratio` | 3 | SwiGLU 中间宽度系数 |
| `in_channels`／`out_channels` | 64／64 | latent 通道，而非最终 RGBA 的 4 个通道 |
| `patch_size` | 1 | 消费未再次分块的 latent token |
| `axes_dims_rope` | `(16,56,56)` | 多轴旋转位置编码维度 |
| `causal_condition` | `true` | 条件不随去噪时间变化，是精确前缀缓存的前提 |

每层包含归一化、注意力、SwiGLU 和带门控的残差；模型还共享一份 modulation 投影，而不是每层各持一套同类调制参数。它不能按旧版 20B MMDiT 的双流结构直接解释。

### 3. 64 通道 RGBA VAE：透明度进入模型表示

VAE 支持 RGBA，空间压缩 16×。最终图像的 alpha 不是把 RGB 图片保存成 PNG 时自动“补出来”的。真正透明素材需要检查 alpha 是否有有效变化，以及毛发、半透明边缘和阴影在不同底色上的表现。

例如，将全不透明的 RGB 图片转成 RGBA，alpha 全是 255，文件仍然不是产品意义上的透明背景素材。

### 4. Flow Matching 与 Euler 调度

目标图像从噪声 latent 出发，DiT 预测更新方向，调度器按时间／sigma 网格迭代更新，最后交给 VAE 解码。

示意写成：

$$
x_{t+\Delta t}=x_t+\Delta t\,v_\theta(x_t,t,c).
$$

这里是帮助理解的 Euler 形式，不替代实际 scheduler 的符号、时间方向与 shifting 实现。模型推荐的 sigma 网格属于配方的一部分，不是可以随意替换的装饰参数。

## 为什么条件前缀能够跨步复用 KV

**每次去噪都要重新计算目标图像，但不必重新计算固定条件。**Qwen-Image-2.1 让条件前缀在每个 Transformer 层的表示不依赖目标噪声和去噪时间，因而可以把它的 K/V 保存下来，供后续去噪步读取。

这里缓存的是 **DiT 各层的文本／参考图条件 K/V**，不是 Qwen3-VL 编码器内部的生成缓存，不是 VAE latent 本身，也不是最终生成图像。

### 1. 复用成立需要两个条件，缺一不可

把联合序列记作 `[c, x_t]`：c 包含交错的文本和参考图 token，x_t 是当前目标图像 token。目标图像在去噪中变化，但条件前缀需要保持不变。

**第一，注意力不能让条件读取目标。**2.1 使用 block-causal 规则：

$$
\mathrm{allow}(q,k)=(\mathrm{pos}(q)\ge\mathrm{pos}(k))\;\lor\;\mathrm{same\_image\_block}(q,k).
$$

文本按因果规则读取历史，同一个图像块内部允许双向注意；后面的目标块可以读取条件前缀，前面的条件却不能读取后面的目标。相邻参考图仍是独立图像块，不能误合成一个双向块。

**第二，条件调制不能随去噪时间变化。**当 `causal_condition=true` 时，条件 token 使用固定的 **t=0 调制**，目标 token 使用当前 t。这里的 t=0 是条件调制选用的时间输入，不代表参考图经过了“一次零噪声去噪”。

只有第一个条件还不够：即使 c 看不到 x_t，如果每步仍用不同 t 调制 c，条件 K/V 依然会变化。反过来，只有固定 t=0 也不够：如果条件能读取变化的目标，它的表示仍会被目标影响。

### 2. 为什么能逐层证明条件 K/V 不变

输入条件表示固定，且每层只读取合法的前缀上下文；条件使用的时间调制、模型权重和位置编码也固定。因此可以逐层推导：第一层条件输出不随去噪步变化，第二层的条件输入便不变，依此类推到全部 32 层。

用第 ℓ 层表示写成：

$$
h_c^{(\ell+1)}=F_c^{(\ell)}\big(h_c^{(\ell)},t=0\big),
\qquad
(K_c^{(\ell)},V_c^{(\ell)})=G^{(\ell)}\big(h_c^{(\ell)}\big).
$$

右侧没有当前目标 x_t，也没有变化的目标时间 t。这就是前缀能够**精确复用**的原因。这里“精确”指模型依赖关系没有被近似；不同 kernel、分块和浮点运算顺序仍可能造成数值差异，不保证逐 bit 相同。

### 3. 缓存到底存什么，为什么不用保存 Q

每个 DiT block 有自己的注意力投影，因此必须保存**每层各自的 K/V**，不能只缓存一份条件 embedding，或者把第一层 K/V 给所有层使用。

在固定版本 Diffusers 实现中，Q/K 先做 RMSNorm，再做 RoPE；缓存的 K 已经过这两步，V 则是 value 投影结果，**V 不做 RoPE**。缓存的典型布局为 `[batch, prefix_tokens, local_heads, head_dim]`；张量并行时只保存当前 rank 的头。

后续步只提出目标 query：

$$
\operatorname{Attn}\!\left(Q_{x,t}^{(\ell)},
[K_c^{(\ell)};K_{x,t}^{(\ell)}],
[V_c^{(\ell)};V_{x,t}^{(\ell)}]\right).
$$

目标仍需读取条件，因此需要 K_c、V_c；条件本身不再提出 query，也不再生成新的条件输出，所以不必缓存 Q_c。目标图像块内部是双向注意，目标 query 也会读取当前目标块全部合法 K/V。

### 4. 首步 extract 与后续 cached，执行范围有什么不同

| 阶段 | 首步 `extract` | 后续 `cached` |
|---|---|---|
| block 输入 | 条件前缀 + 当前目标 | 仅当前目标 token |
| Q 投影 | 条件与目标 | 仅目标 |
| K/V 投影 | 条件与目标；保存每层条件 K/V | 仅目标；与已存条件 K/V 拼接 |
| Attention 输出 | 联合序列，遵守 block-causal 规则 | 仅目标 query 的输出 |
| MLP 与残差 | 条件与目标 | 仅目标 |
| scheduler 更新 | 取末尾目标预测，更新目标 latent | 同左；条件缓存保持不变 |

Pipeline 在首步选择 `kv_cache_mode="extract"`，之后选择 `"cached"`。Transformer 的 cached 分支在进入 block 前裁掉条件前缀；因此节省的不只有条件 K/V 投影，也包括条件 query、注意力输出、MLP 等主体计算。

但当前实现仍会准备部分输入和布局，不能把“block 只重算目标”理解为所有前处理都消失。关闭缓存时，Pipeline 每步重算联合序列。

对应 attention 内部的骨架如下。变量经过简化，完整源码入口见后文映射表：

```python
q, k, v = project_qkv(hidden_states)
q, k = rmsnorm_q(q), rmsnorm_k(k)
q, k = apply_rope(q), apply_rope(k)  # V 不旋转

if mode == "extract":
    # 每层各存一份；只保存条件前缀，不包含目标。
    cache[layer] = (k[:, :prefix_len].clone(),
                    v[:, :prefix_len].clone())
elif mode == "cached":
    # 此时 hidden_states 已只剩当前目标。
    kc, vc = cache[layer]
    k = concat(kc, k, dim=1)
    v = concat(vc, v, dim=1)

output = attention(q, k, v, mask)
```

这里 `clone()` 也有工程意义：切片可能与完整 prefill 张量共享底层存储。源码特意复制前缀，避免“小切片”把整份含目标的 K/V 显存一直占住；只写 `contiguous()` 不保证产生独立存储。

### 5. 它能省多少计算，又要付出多少显存

设条件长度为 P，目标长度为 T，去噪步数为 N。不开缓存时，条件分支主体计算大致重复 N 次；开启后，这部分主要在首步做一次。但目标每步仍需要读取前缀，因此不会把注意力里的 T×P 交互全部消掉。

如果把完整条件处理成本记作 A，把每步目标处理成本（包含目标读取条件）记作 B，可以用下面的**解释性近似**理解收益：

$$
C_{\rm no\ cache}\approx N(A+B),\qquad
C_{\rm cache}\approx A+NB+C_{\rm cache\ overhead}.
$$

这不是性能预测公式：kernel、padding、并行、访存以及首步／后续步的执行形态都会改变真实耗时。条件越长、去噪步数越多，越有机会摊薄首步成本；少步 Turbo 的可摊销步数较少，不能直接沿用 Base 的加速倍数。

K/V 缓存则占用显存。若所有层头数一致，忽略 padding 和管理开销，其总大小约为：

$$
M_{\rm KV}\approx 2\times L\times B_{\rm batch}\times P\times H_{\rm KV}\times D_{\rm head}\times s_{\rm bytes}.
$$

按 32 层、batch=1、P=4096、32 个 KV 头、head_dim=128、BF16 两字节作**算例**，全模型前缀缓存约为 **2 GiB**。这里 P 是实际展开后的条件序列长度，不是用户输入的汉字数；4096 是举例，不能当成固定前缀长度。张量并行通常将头分摊到不同 rank；真实单卡占用还要计入临时拼接、padding、目标激活和框架开销。

启用 true CFG 时，正条件和负条件路径分别有 `cond_cache`、`neg_cache`。两条路径的长度可能不同，缓存与算力开销需要分别计算，不能机械地把所有成本视作固定翻倍。

### 6. 缓存什么时候失效，能不能跨请求复用

当前阅读的 Pipeline 在一次调用内创建缓存，**没有自动提供跨请求的持久缓存服务**。下面应区分模型上的可复用性与服务端已经实现的功能。

| 变化 | 如何处理 |
|---|---|
| 目标 x_t 随去噪更新、当前 t 改变 | 正常复用条件缓存；这是设计目标 |
| prompt、实际编码内容、参考图内容或顺序改变 | 重建；条件表示已经不同 |
| 图像预处理、条件 token 布局、mask 或位置编码改变 | 重建；不仅要比较原始图片文件 |
| 模型 revision、LoRA 权重／strength 改变 | 重建；各层投影或条件表示可能变化 |
| dtype、量化、并行切分、设备或缓存格式改变 | 不直接沿用旧缓存；需要显式兼容处理与验证 |
| 只改变随机 seed 或目标调度 | 不天然改变前缀依赖；仍需核对布局、位置与 batch 协议，当前 Pipeline 会重新建缓存 |
| 目标尺寸改变 | 检查是否联动条件预处理、位置或布局；不能简单认定一定命中 |

如果以后做跨请求缓存，key 应覆盖**实际条件张量及其布局、权重版本和执行配置**，并设计租户隔离、容量上限、逐出与并发只读协议；只用 prompt 字符串作 key 不够。

### 7. 上线前应该怎样验证它

先固定模型、输入、seed、调度和执行配置，分别运行 `use_kv_cache=False / True`，逐步对比目标预测／latent，再比较最终结果。浮点容差应依据 dtype 与 backend 设定，避免只看“最后两张图差不多”。

性能测试至少分开记录：条件编码时间、首步 prefill、后续每步耗时、端到端 P50／P95、峰值显存，以及多参考图与 true CFG 下的变化。**本文尚未执行这组 GPU 对照实验，以上是验证方法，不是已测收益。**

与 LLM KV Cache 相比，这里的复用轴是**去噪步**：条件长度不随每步增长，目标块整体重算；与近似复用目标特征的缓存方法相比，它依赖固定条件的独立性，不以“目标变化很小”为前提。多参考图可能首步更贵、缓存更大，后续则避免重复处理条件；这既不意味着参考图免费，也不意味着两个用户的请求天然能共享缓存。

## 一次请求怎么走：从上传到 RGBA 结果

下面按 `QwenImage21Pipeline.__call__` 的调用顺序画推理过程。中间左右两栏是**同一个模型在不同去噪步的执行方式**，不是两套 Backbone；首步建立缓存，后续步读取缓存。

<figure class="model-diagram">
  <div class="model-diagram-scroll" tabindex="0" role="region" aria-label="推理流程图，可横向滚动">
    <a href="/images/qwen-image21/inference.svg" target="_blank" rel="noopener"><img src="/images/qwen-image21/inference.svg" width="1120" height="1250" loading="lazy" alt="Qwen-Image-2.1 推理流程：联合条件编码与目标初始化，设置时间网格；首步 extract 写入每层条件 KV，后续 cached 只重算目标；scheduler 更新 latent 并循环，最后 VAE 解码。" /></a>
  </div>
  <figcaption>图 2 · 从输入到输出，以及条件 KV 的写入、读取与失效边界。虚线表示缓存读取或下一步回环。</figcaption>
</figure>

1. **校验输入**：文本、参考图数量与顺序、尺寸、输出格式；区分生成与编辑场景。
2. **预处理**：按面积与长宽比调整参考图。LightX2V 显式 `size` 使用 `HEIGHT WIDTH`；示例要求尺寸按 32 对齐。
3. **联合条件编码**：Qwen3-VL 编码文本与参考图；VAE 提供条件图像 latent。
4. **初始化目标噪声**：准备待生成图的 latent、seed 与时间网格。
5. **首步与循环**：首步建立条件 KV，后续只更新目标部分并读取条件缓存。
6. **VAE 解码**：得到 RGB／RGBA 输出。
7. **产品后处理**：透明度检查、必要的 OCR／素材校验、存储、结果回传与计费。

参考图顺序是语义的一部分。前端若支持拖动重排，必须把最终顺序和 prompt 中的“第一张、第二张”对应起来。局部标注／mask 能力也不等于每个后端都有相同 `mask_image` 参数，接入时要核对真实接口。

### 图中的模块，对应哪段推理代码

以下链接固定到本文读取的 Diffusers 提交，方便从图跳到实际实现。图中的「只重算目标」指 Transformer block 的主体计算；Pipeline 仍会准备输入布局，不能理解成所有前处理都消失。

| 图中位置 | 代码入口 | 关注什么 |
|---|---|---|
| 图 1：文本与图像投影 | [`txt_in` / `img_in`](https://github.com/huggingface/diffusers/blob/f9316b0d4d69555c5e3f625585262e79cd299d34/src/diffusers/models/transformers/transformer_qwenimage21.py#L188-L200) | 文本投影包含 RMSNorm 与两层 Linear；图像 latent 走独立输入投影 |
| 图 1：单层算子 | [`QwenImage21TransformerBlock.forward`](https://github.com/huggingface/diffusers/blob/f9316b0d4d69555c5e3f625585262e79cd299d34/src/diffusers/models/transformers/transformer_qwenimage21.py#L615-L656) | 归一化、scale 调制、attention、SwiGLU，以及两次 `tanh(gate)` 残差 |
| 图 1：联合序列与时间调制 | [`QwenImage21Transformer2DModel.forward`](https://github.com/huggingface/diffusers/blob/f9316b0d4d69555c5e3f625585262e79cd299d34/src/diffusers/models/transformers/transformer_qwenimage21.py#L918-L953) | 图像槽位展开替换；目标追加；条件用 t=0，目标用当前 t |
| 图 2：后续步的目标裁剪 | [`kv_cache_mode == "cached"`](https://github.com/huggingface/diffusers/blob/f9316b0d4d69555c5e3f625585262e79cd299d34/src/diffusers/models/transformers/transformer_qwenimage21.py#L969-L999) | 进入 block 前切掉条件前缀，保留目标 query 和对应位置 |
| 图 2：K/V 缓存读写 | [attention QKV 准备函数](https://github.com/huggingface/diffusers/blob/f9316b0d4d69555c5e3f625585262e79cd299d34/src/diffusers/models/transformers/transformer_qwenimage21.py#L328-L369) | 保存 post-RoPE prefix K/V；后续拼接 cached K/V 与新目标 K/V |
| 图 2：时间网格与去噪循环 | [`QwenImage21Pipeline.__call__`](https://github.com/huggingface/diffusers/blob/f9316b0d4d69555c5e3f625585262e79cd299d34/src/diffusers/pipelines/qwenimage21/pipeline_qwenimage21.py#L731-L816) | sigmas 优先级、首步 extract / 后续 cached、CFG 分支与 scheduler 更新 |
| 图 2：最终解码 | [unpack / VAE decode](https://github.com/huggingface/diffusers/blob/f9316b0d4d69555c5e3f625585262e79cd299d34/src/diffusers/pipelines/qwenimage21/pipeline_qwenimage21.py#L839-L855) | latent 反归一化后解码；随后处理输出格式 |

把执行骨架缩到最短，可以这样阅读。下面是**省略 padding、并行、回调和可选 CFG 的伪代码**，不替代实际 Pipeline：

```python
condition = encode_prompt(prompt, reference_images)
x, reference_latents = prepare_latents(seed, size, reference_images)
timesteps = configure_schedule(explicit_sigmas, checkpoint_sigmas, steps)
cache = make_per_layer_prefix_cache()  # use_kv_cache && causal_condition

for i, t in enumerate(timesteps):
    prediction = transformer(
        hidden_states=concat(reference_latents, x),
        encoder_hidden_states=condition,
        timestep=t / 1000,
        kv_cache=cache,
        kv_cache_mode="extract" if i == 0 else "cached",
    )
    # cached 分支会在 block 前裁掉条件前缀，而不是重复算全部 token。
    target_prediction = prediction[:, -x.shape[1]:]
    x = scheduler.step(target_prediction, t, x)

image = vae.decode(restore_latent_statistics(unpack(x)))
```

图的[可复现绘制脚本](/learning/qwen-image21/build_diagrams.py)也随文章保留，便于后续跟随源码修订。图是本站根据代码重新绘制的，不是官方原图；算子尺寸采用构造默认值，未宣称完成 GPU 实测。

### 分辨率为什么会明显改变成本

按 16×空间压缩与 `patch_size=1` 估算，目标图像 token 数为：

$$
L_{\mathrm{image}}\approx\frac H{16}\frac W{16}.
$$

1024×1024 对应约 4096 个目标 token；2048×2048 对应约 16384 个，是 4 倍。注意力计算还与序列长度、缓存和 kernel 有关，因此图像面积变 4 倍，不保证延迟也只变 4 倍。参考图还会贡献额外条件 token。

同理，不能用 7B×BF16 算完显存就结束。若粗略把 7B 视觉主干与 8B 编码器都按 BF16 全驻留，权重本身就是约 30 GB 的十进制量级，另有 VAE、激活和 KV。**这不是显存最低要求**：offload、分阶段驻留、量化与并行会改变实际占用，必须测目标部署配置。

## Lightning LoRA：应该怎样理解与验证

用户提供的入口为：[lightx2v/Qwen-Image-2.1-Lightning](https://huggingface.co/lightx2v/Qwen-Image-2.1-Lightning)。本轮无法直接读取该模型卡，以下区分“已确认的加载能力”与“这个权重仍待确认的配方”。

### LoRA 小，不代表基座被压成小模型

通常 LoRA 将权重更新写作：

$$
W'=W+s\frac\alpha r BA.
$$

r 是低秩维度，α 是适配器缩放，s 是使用强度；确切语义还要以适配器 metadata 和实现为准。LoRA 文件小，是因为它保存增量，不意味着推理时不需要原基座。

对加速 LoRA，重要的往往是让模型适应更少的采样步骤；**低秩参数化本身不自动减少每步主干 FLOPs**。也不能仅凭 Lightning 名字断言这份 2.1 权重使用某一种特定蒸馏损失。

### 加载能力已经核实，但采样配方尚未核实

Diffusers 的 `QwenImage21Pipeline` 继承 `QwenImageLoraLoaderMixin`。LightX2V 也存在 2.1 专用 LoRA adapter，并检查权重配对、形状、alpha 和并行切分。

但这只能证明“存在加载路径”，不能证明任意 LoRA 与任意量化基座、步数、sigma 都兼容。

| 上线前需要核实 | 为什么必须查模型卡／权重 |
|---|---|
| 基座 model ID 与 revision | 2.1 Base、Turbo、旧 2512／Edit 不可按名字混用 |
| 完整 weight filename、版本与 hash | 不凭旧版目录猜文件名 |
| 推荐步数与 sigma／scheduler／shift | “设为 4 步”不能替代蒸馏配方 |
| CFG／`true_cfg_scale`／negative prompt | 少步模型可能有不同训练与使用约束 |
| LoRA rank、alpha、strength | 影响增量缩放和加载兼容性 |
| 精度、量化、merge 与动态加载支持 | 能单独运行不代表组合可运行 |
| 训练目标、数据与质量评估范围 | 不能把旧版的能力与代价照搬过来 |
| adapter 与基座的许可 | adapter 的许可不覆盖基座材料的使用条件 |

因此本文**不提供猜测出来的 2.1-Lightning 文件名、rank、步数或加速倍数**。待模型卡可读后，再补“固定 revision 的完整配方”，而不是留下看起来可运行、实际上可能错配的代码。

### LightX2V 的一个实际兼容性边界

本轮读取的 `validate_lora_config` 在存在 `lora_configs` 时，会拒绝 `dit_quantized`、`weight_auto_quant`、`shared_cpu_weights`、`dit_disk_streaming` 等组合；动态加载还要求恰好一个适配器，并对 rsLoRA／DoRA／alpha pattern 等作检查。

这限制的是**本轮固定提交中的这条加载路径**。不能由此断言整个生态永远不支持“LoRA＋量化”；也不能反过来把 5090 的 FP8 benchmark 与另一个 BF16 LoRA 配方直接拼起来。预先合并再量化、其他框架、后续版本都需要分别核验。

## 官方 Turbo：已核实的少步选择

官方说明 Turbo 沿用 7B 视觉生成架构，以 **8 个去噪步骤**完成生成和编辑。当前 Diffusers 可以从 checkpoint 的 `sample_sigmas` 读取推荐网格。

这里有一个容易忽略的接口细节：显式 `sigmas` 优先于 checkpoint 的 `sample_sigmas`，这两者都会决定实际步数，而不是盲目服从 `num_inference_steps`。不传自定义网格时才按对应逻辑使用步数生成调度。

下面是按官方用法整理的参考代码，**未在本站 GPU 上运行**；调用前还需满足模型访问与许可条件：

```python
import torch
from diffusers import QwenImage21Pipeline

pipe = QwenImage21Pipeline.from_pretrained(
    "Qwen/Qwen-Image-2.1-Turbo",
    dtype=torch.bfloat16,
).to("cuda")

# 不覆盖 checkpoint 自带的 sample_sigmas。
image = pipe(
    prompt="一张简洁的咖啡店海报，醒目的中文标题“周末咖啡”，自然光，留白构图。",
    height=1024,
    width=1024,
    generator=torch.Generator("cuda").manual_seed(42),
).images[0]
image.save("qwen21-turbo.png")
```

少步配方不应被随意改成原基座的强 CFG 配方。Diffusers 专门文档还说明，其 **Turbo modular blocks**没有 guider，不接受 negative prompt；这条限制要与普通 Pipeline 的参数行为分开看。

[本站参考脚本](https://github.com/WenMang98k/WenMang98k.github.io/tree/main/research/qwen-image21)提供 Base／Turbo 的输入与尺寸校验，以及无需下载模型的 dry-run。它不包含未经核实的 Lightning 配方。

## 加速收益怎么读：先对齐测量口径

### 一组可追溯的公开数据

LightX2V 的 2.1 指南给出以下结果，属于**项目维护者报告，非本站实测，也不是用户这份 Lightning LoRA 的成绩**：

| GPU | 任务 | 输出尺寸 | 报告的端到端延迟 |
|---|---|---|---:|
| RTX 5090 ×1 | 文生图 | 1024×1024 | 5.586 s |
| RTX 5090 ×1 | 图像编辑 | 1024×1024 | 6.804 s |
| RTX 5090 ×1 | 文生图 | 2048×2048 | 30.333 s |
| RTX 5090 ×1 | 图像编辑 | 2048×2048 | 42.787 s |
| RTX 5090 ×2 | 文生图 | 2048×2048 | 16.868 s |
| RTX 5090 ×2 | 图像编辑 | 2048×2048 | 25.212 s |

报告条件：PyTorch 2.11.0+cu130、**40 步、seed=42、关闭 CFG**、示例 prompt；编辑使用指定 `girl.png` 并按对应分辨率预处理。5090 路线包含指南列出的 FP8 权重、优化 attention／融合算子、编码器 offload 等；双卡使用相应并行配置。

统计口径为：同一实例首次请求之后连续三次请求的中位数；双卡取较慢 rank。包含图片保存，不包含模型加载与一次性初始化。它没有给出生产队列、高并发 P95 或完整质量盲测，不能直接写成产品 SLA。

### 为什么 40 步变 8 步，不等于端到端必然快 5 倍

可以按下面的成本拆解：

$$
T_{\mathrm{request}}=T_{\mathrm{queue}}+T_{\mathrm{prompt}}+T_{\mathrm{condition}}+T_{\mathrm{denoise}}+T_{\mathrm{VAE}}+T_{\mathrm{IO}}.
$$

只有其中部分随去噪步骤近似线性缩小。即使单步成本相同，40→8 的 **5 倍**也首先是迭代次数比，不是端到端实测结果；首步条件计算、VAE 与排队不会按相同比例消失。

还要分开 **steps、NFE 与 CFG 分支**。当前普通 Diffusers Pipeline 开启 true CFG 后会计算条件与无条件两路；等效工作量增加，其他实现可能合批或并行，不能一概按相同“函数调用次数”比较。2.1 基准默认已经无 CFG，不能再重复把“去掉 CFG”的收益加一次。

旧版 Lightning README 的 12–25×等数字、Edit-2511 的其他加速报告，都不能转记到这份 2.1 LoRA 上。更不能把不同硬件、精度、步数、输入尺寸的各个倍率直接相乘。

## 应用场景：什么时候值得进入候选名单

| 场景 | 2.1 的相关能力 | 产品验收重点 | 不宜直接承诺 |
|---|---|---|---|
| 中文海报、活动素材 | 文本渲染、图文布局 | 文案逐字正确、层级清楚、品牌位置 | 金额、时间与小字永远准确 |
| 电商商品换背景 | 参考图理解与主体保持 | SKU 外观、logo、材质、比例不漂移 | 不经校验自动发布全部商品图 |
| 多参考图合成 | 最多 10 张参考图、统一条件建模 | 人物／商品归属关系、顺序与冲突处理 | 参考越多一定越好、耗时不变 |
| 局部修图 | 圈选、涂画、mask 条件能力 | 修改区域生效、非目标区域保真 | 生成模型天然逐像素保护原图 |
| 透明素材与贴纸 | 原生 RGBA、主体提取 | alpha 边缘、半透明、阴影与底色适配 | 仅换成 PNG 就得到可用抠图 |
| 设计探索与快速预览 | Turbo／可评估的 Lightning 配方 | 首次有用结果、重试率、后续可编辑性 | 演示质量等于商业交付通过率 |

对于价格、日期、法定说明、密集数据表等硬约束内容，更稳健的产品路线往往是：模型生成视觉素材，程序化排版叠加确定性文字，必要时再结合 OCR 与人工确认。**能把文字画出来，不代表它是结构化排版引擎。**

对图像编辑，也可以用外部 mask／合成逻辑保障明确的像素保护范围；不要只依赖一句“其他地方不变”的 prompt。

## 产品经理应优先问的七个问题

### 1. 我们真正卖的是哪一种“成功”？

不要只问“图好不好看”。应定义可验收标准：主体一致、指定文字正确、编辑区域符合要求、未编辑区域是否保留、透明边缘是否可用、最终尺寸与格式是否一致。

CLIP 相似度或单一美学分数不足以代替这些标准。对文案做 OCR／人工核对，对商品／人脸一致性做专门评测，对 alpha 做棋盘格与多底色检查。

### 2. 许可允许我们用哪一种交付方式？

当前官方 2.1 仓库的 **Qwen Research License** 第 2 条将材料使用授权限定为研究／评估等非商业用途，商用需单独申请许可，文中给出联系地址 `model-business@notice.qwencloud.com`。

因此“能下载权重”不能直接作为商业上线批准。基座材料、LoRA、推理框架、云 API 条款及上传参考图的使用权需要分别核对；不能因 LightX2V／Diffusers 的软件许可较宽松，就推导基座可以直接商用。

这不等于一句话裁定所有输出图片的权利。这里应区分模型材料使用、API 服务、输出作品与参考图授权，不作混为一谈的推断。

### 3. 关注的是平均延迟，还是用户的等待体验？

需要分别量化：排队、首次有用预览、最终结果、取消后资源释放、失败恢复。warm median 不等于高并发 P95；移动端上传 10 张参考图、Prompt Enhancer 与对象存储时间也要进入完整链路。

建议交付档位显式分开：快速预览、标准生成、精修导出。步数和分辨率变更会影响质量，应符合用户选定的服务约定，而非静默切低。

### 4. 一张“可用图片”到底花多少钱？

成本不应只除以请求数：

$$
C_{\mathrm{usable}}=\frac{C_{\mathrm{compute}}+C_{\mathrm{prompt}}+C_{\mathrm{storage}}+C_{\mathrm{postprocess}}}{N_{\mathrm{accepted\ images}}}.
$$

少步模型可能单次更便宜，但如果重试和人工修图明显增加，可用结果成本不一定降低。在近似独立、同分布的重试假设下，首次成功率 p 对应约 1/p 次尝试；真实编辑流程有上下文相关性，需要用实际日志测量。

### 5. 我们的输入交互是否真的匹配模型？

参考图的顺序、作用、局部标注、mask 与输出比例应显式可控。前端可以让用户标明“主体图”“风格图”“背景图”，但后端仍要将其转换成与模型协议匹配的有序条件，而不是以为 UI 标签本身就能被模型理解。

对透明图，应保留原始 alpha；若为了展示加白底，不能把展示用白底副本误当成编辑源图。

### 6. 基座、LoRA 与缓存能否安全切换？

至少记录：model revision、LoRA 文件 hash、strength、sigma recipe、尺寸、参考图摘要与精度。相同 seed 不保证跨版本、跨 kernel、跨精度完全一致。

适配器或条件变化后，缓存需要正确失效。LightX2V 当前代码在移除 LoRA 时清理条件 KV，这是产品侧“切风格／切适配器”需要端到端覆盖的行为，不只是一项 UI 开关。

### 7. 上线回退方案是什么？

当某个 LoRA、量化或 attention 后端组合不可用时，回到经过验证的基座或官方 Turbo 配方；不能只卸载 LoRA 却保留其专属少步 sigma 后宣称已恢复基座。

服务过载时，还应有排队上限、幂等、超时、取消和失败重试策略。它们决定真实 SLA，而不是只由 DiT 单次耗时决定。

## 框架选择：比较实际图像执行路径

| 路线 | 本轮已核实 | 适合先做什么 |
|---|---|---|
| Diffusers | `QwenImage21Pipeline`、专用 Transformer、RGBA VAE、LoRA loader、前缀缓存、Turbo schedule | 建立可读性强的功能／质量基线 |
| LightX2V | 2.1 专用 runner、prefill／infer、融合算子、量化／并行配置、LoRA 校验 | 结合目标硬件做延迟与驻留优化 |
| vLLM-Omni | 2.1 官方 README 指向专用图像 recipe 与 step-wise 路径 | 面向服务调度、连续批处理与多卡评估 |
| SGLang-Diffusion | 2.1 官方 README 指向 cookbook 与集成 PR | 多卡、缓存、offload 与服务路径评估 |
| ComfyUI | 官方 2.1 README 指向原生支持和工作流模板 | 设计／产品快速体验与可视化工作流 |

“vLLM 支持”这里指 **vLLM-Omni 的图像执行路径**，不能直接照搬普通 LLM 的启动与投机解码设置；SGLang 也要区分 Diffusion 路径。官方生态列表说明有支持方向，不替代每个后端的端到端实测。

Diffusers 对 `flex_attention` 另有具体提醒：切换 flex processor 时应结合 compile；未编译的路径可能实体化很大的 FP32 attention 分数矩阵，高分辨率下会慢甚至 OOM。配置越“高级”不代表默认越快。

## 建议的验证计划：先质量，再看每张可用图成本

下面是**实验计划**，不是已经跑出的结果：

| 配置组 | 目的 | 当前状态 |
|---|---|---|
| Base 40 步、同精度，缓存开／关 | 检查前缀缓存收益及数值／质量一致性 | 未运行 |
| 官方 Turbo 默认 sigma、8 步 | 测量少步质量、延迟与重试率 | 未运行 |
| 用户 Lightning LoRA 的固定 revision 配方 | 验证 LoRA 与基座、步数、sigma 的完整匹配 | 模型卡待读取，不猜配方 |
| 目标部署的 FP8／offload／并行路径 | 测硬件收益与内存预算 | 先排除与 LoRA 加载的已知冲突，再运行 |

建立覆盖中文文字、多图关系、人物／商品保持、局部修改、透明边缘、写实纹理和困难提示词的测试集，隔离调参与验收集。固定随机种子集合，做盲评和多次采样，不用一张“幸运样例”代表通过率。

每组至少报告：模型／LoRA hash、软件 commit、GPU／显存、精度、尺寸、参考图数量、CFG、sigma、实际 NFE、冷／热耗时、P50／P95、失败与重试、可用率、峰值显存和可用图成本。并发与请求取消要单独测。

产品输出可以选择“快预览→标准确认→精修导出”的分层路径，但是否保留 Lightning，应由同任务下质量—延迟—成本的 Pareto 前沿决定，而不是由单个倍数决定。

## 资料、版本与还需要补核的部分

本文没有下载基座、LoRA 或训练数据，没有执行生图和 GPU benchmark。已经核实的是公开文档、实现和许可证；示例代码只做静态语法与 dry-run 检查。

- [Qwen-Image-2.1 官方仓库](https://github.com/QwenLM/Qwen-Image-2.1)：发布记录、架构、Turbo 与生态入口。
- [官方 LICENSE](https://github.com/QwenLM/Qwen-Image-2.1/blob/main/LICENSE)：研究许可与单独商用授权条件。
- [用户给定 Lightning 模型卡](https://huggingface.co/lightx2v/Qwen-Image-2.1-Lightning)：本轮访问被代理阻断；实际权重、配方、许可与收益仍待核实。
- [Diffusers 2.1 专用指南](https://github.com/huggingface/diffusers/blob/main/docs/source/en/api/pipelines/qwenimage21.md)：pipeline、缓存、CFG、sigma 与 flex processor。
- [LightX2V 2.1 指南](https://github.com/ModelTC/LightX2V/blob/main/scripts/qwen_image_21/README.md)：部署、尺寸规则与本文引用的 5090 测量条件。

下面的固定版本索引由本轮实际读取记录生成。若模型卡后续可读，优先补齐 Lightning 的配方和许可，再做同条件实验；不把“资料尚未读到”写成“模型不存在或效果不好”。

| 已读取资料 | 固定版本入口 |
|---|---|
| qwen21-readme | [`README.md`](https://github.com/QwenLM/Qwen-Image-2.1/blob/ab96caa944157ca28a4f36b695c3047549f48f98/README.md) · `ab96caa944` |
| qwen21-license | [`LICENSE`](https://github.com/QwenLM/Qwen-Image-2.1/blob/ab96caa944157ca28a4f36b695c3047549f48f98/LICENSE) · `ab96caa944` |
| transformer | [`src/diffusers/models/transformers/transformer_qwenimage21.py`](https://github.com/huggingface/diffusers/blob/f9316b0d4d69555c5e3f625585262e79cd299d34/src/diffusers/models/transformers/transformer_qwenimage21.py) · `f9316b0d4d` |
| pipeline | [`src/diffusers/pipelines/qwenimage21/pipeline_qwenimage21.py`](https://github.com/huggingface/diffusers/blob/f9316b0d4d69555c5e3f625585262e79cd299d34/src/diffusers/pipelines/qwenimage21/pipeline_qwenimage21.py) · `f9316b0d4d` |
| pipeline-guide | [`docs/source/en/api/pipelines/qwenimage21.md`](https://github.com/huggingface/diffusers/blob/f9316b0d4d69555c5e3f625585262e79cd299d34/docs/source/en/api/pipelines/qwenimage21.md) · `f9316b0d4d` |
| lightx-guide | [`scripts/qwen_image_21/README.md`](https://github.com/ModelTC/LightX2V/blob/5c5a8c8e98ebf8a6e3cfac746cb2fdfd8d03ff4f/scripts/qwen_image_21/README.md) · `5c5a8c8e98` |
| lightx-config | [`configs/qwen_image_21/qwen_image_21.json`](https://github.com/ModelTC/LightX2V/blob/5c5a8c8e98ebf8a6e3cfac746cb2fdfd8d03ff4f/configs/qwen_image_21/qwen_image_21.json) · `5c5a8c8e98` |
| lightx-runner | [`lightx2v/models/runners/qwen_image_21/qwen_image_21_runner.py`](https://github.com/ModelTC/LightX2V/blob/5c5a8c8e98ebf8a6e3cfac746cb2fdfd8d03ff4f/lightx2v/models/runners/qwen_image_21/qwen_image_21_runner.py) · `5c5a8c8e98` |
| lightx-model | [`lightx2v/models/networks/qwen_image_21/model.py`](https://github.com/ModelTC/LightX2V/blob/5c5a8c8e98ebf8a6e3cfac746cb2fdfd8d03ff4f/lightx2v/models/networks/qwen_image_21/model.py) · `5c5a8c8e98` |
| lightx-lora | [`lightx2v/models/networks/qwen_image_21/lora.py`](https://github.com/ModelTC/LightX2V/blob/5c5a8c8e98ebf8a6e3cfac746cb2fdfd8d03ff4f/lightx2v/models/networks/qwen_image_21/lora.py) · `5c5a8c8e98` |
| lightx-infer | [`lightx2v/models/networks/qwen_image_21/infer/transformer_infer.py`](https://github.com/ModelTC/LightX2V/blob/5c5a8c8e98ebf8a6e3cfac746cb2fdfd8d03ff4f/lightx2v/models/networks/qwen_image_21/infer/transformer_infer.py) · `5c5a8c8e98` |
| lightning-legacy | [`README.md`](https://github.com/ModelTC/Qwen-Image-Lightning/blob/115d41afa959f0bc858d25d3b3fddd44054767ad/README.md) · `115d41afa9` |

[完整来源与 hash 记录](https://github.com/WenMang98k/WenMang98k.github.io/blob/main/research/qwen-image21/sources.lock.json) · [下载产品验收模板](/learning/qwen-image21/evaluation-template.csv) · [下载参考脚本](/learning/qwen-image21/reference_inference.py)

