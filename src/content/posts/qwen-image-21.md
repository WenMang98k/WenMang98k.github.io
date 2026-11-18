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

### 先沿一次编辑请求走完整数据流

用“给参考图中的人物换一件蓝色外套”作为例子。先把三个不同的“固定”区分开：参考图文件不变、编码后的条件输入不变、每个 DiT 层的条件 K/V 不变，是三个层次。前两个能在准备输入时得到，最后一个还需要块因果注意力和固定条件调制来保证。

下面按一次请求展开。图中绿色是固定条件及其缓存，蓝色是每步变化的目标，橙色强调复用边界。节点中的具体 shape 使用本节 F 的统一假设：输出1024×1024、参考1024×1204（默认缩放至960×1120）、有效非图像token为2048，B=1、无TP。**图中的第 1 层／第 2 层是网络深度，首步／后续步是去噪时间，两者不能混为一谈。**

<figure>
  <div style="overflow-x:auto">
    <a href="/images/qwen-image21/kv-dataflow.svg" target="_blank" rel="noopener"><img src="/images/qwen-image21/kv-dataflow.svg" width="1120" height="2320" loading="lazy" alt="Qwen-Image-2.1 条件KV完整数据流：参考图分别经过VLM与VAE；首步联合序列进入每层并写入该层前缀KV；后续从更新后的目标latent重新进入第1层，每层读取自己的前缀KV、拼接新目标KV；最后只更新目标latent并在去噪结束后解码。" /></a>
  </div>
  <figcaption>KV 数据流图（shape 已直接标在节点中）· 按后文具体算例，条件前缀6248、目标4096；首步与后续步的 Q/K/V、缓存及输出形状可沿箭头对照。2048 指有效非图像 token；完整假设与模板差异见本节 F。箭头表示数据依赖，不代表实测耗时；点击可查看原图。</figcaption>
</figure>

#### A. 循环开始前：算出固定条件，初始化目标噪声

1. **文字与参考图 → VLM。**Qwen3-VL 提取指令和参考图的联合条件表示，输出不是最终图像，也不是后面要缓存的逐层 K/V。
2. **参考图 → VAE。**同一参考图另经 RGBA VAE 编码为参考 latent，用来提供视觉细节。参考图有两条路径，不能只保留其中一条。
3. **组装条件前缀 c。**Transformer 将条件 embedding 投影，展开图像占位并用参考 latent 的投影替换相应位置，得到交错的文本／参考图 token。
4. **随机噪声 → 目标 latent x。**目标 latent 对应要生成的整张图片，后面每一步 scheduler 都会更新它。目标 token 附加在条件前缀后，形成 `[c, x]`。

位置编码、图像块 ID、有效 key 标记和目标 token 标记一起决定合法的读取关系。这里为了讲解将组装结果统一记作 c；实现里仍会准备和传入若干输入张量与布局信息，并非只传一个预先算好的 `c` 对象。

#### B. 首个去噪步：同时完成一次预测和整套缓存的建立

假设首个 scheduler 时间为 $t_0$；它通常不是条件调制用的零时间。`extract` 的意思是**在正常前向中提取前缀 K/V**，不是额外生成一次图片。

先进入第 1 层：

```text
输入：条件 h_c^0 + 当前目标 h_x^0
          ↓ LayerNorm + 调制（条件用固定0，目标用 t₀）
          ↓ 本层 Q/K/V 投影；Q/K norm；Q/K RoPE
得到：Q_c、K_c、V_c，以及 Q_x、K_x、V_x
          ├─ 保存 K_c、V_c → cache[第1层]
          └─ 做块因果注意力 → 残差 → FFN → 下一层
```

保存的是**本层注意力实际使用的前缀 key/value**：K 已经过 QK norm 与 RoPE，V 是 value 投影结果、不做 RoPE。写入发生在该层注意力输出之前，而不是等该层 FFN 结束后才取输出当 K/V。

首步仍然要计算条件 query 的输出和条件 FFN，因为这些结果会成为第 2 层的条件输入。第 2 层有自己的投影权重，又从它接收到的条件状态产生另一份 K/V，存入 `cache[第2层]`。依次走完所有层，得到：

```text
cache[0] = 第1层的 (K_c, V_c)
cache[1] = 第2层的 (K_c, V_c)
...
cache[L-1] = 第L层的 (K_c, V_c)
```

每份缓存都属于自己的网络层，不能把第 1 层 K/V 交给第 2 层使用。缓存也不需要保存条件 Q：后续不再计算条件输出，只需要供目标查询的 K/V。

首步最终经过输出头，Pipeline 取末尾目标 token 的预测，再由 scheduler 更新目标 latent。**条件 latent 不随 scheduler 更新，已建立的前缀缓存也不随这次更新而改变。**

#### C. 后续去噪步：新目标重新过所有层，每层读取自己的缓存

scheduler 给出更新后的目标 latent 和下一时间 $t_1$。这张目标 latent 会重新经过输入投影，从第 1 层进入网络；不是把上一步最后一层的隐藏状态直接送回第 1 层。

`cached` 分支在 block 之前裁掉条件 query 对应的隐藏状态。现在第 ℓ 层接收的只有**当前步、当前层的目标状态**：

```python
# 教学伪代码；布局 [batch, tokens, heads, head_dim]。
# h_x 是本步流到第 ℓ 层的目标状态，不是上一去噪步的目标 K/V。
u_x = layer_norm_and_modulate(h_x, timestep=current_t)
q_x, k_x, v_x = project_qkv_for_layer(u_x, layer=ell)
q_x, k_x = qk_norm_then_rope(q_x, k_x)

k_c, v_c = cache[ell]               # 同一层，来自首步
k_all = cat([k_c, k_x], dim=1)     # [固定条件 | 当前目标]
v_all = cat([v_c, v_x], dim=1)
attn_x = attention(q_x, k_all, v_all, padding_mask)
h_x = target_residual_and_ffn(h_x, attn_x, current_t)
# h_x 继续进入第 ℓ+1 层；cache[ell] 不追加、不覆盖。
```

这里重用的是绿色的 `k_c/v_c`；蓝色的 `q_x/k_x/v_x` 每个去噪步、每个网络层都重新计算。目标 query 仍要读取所有有效的条件 K/V，因此**缓存省去条件侧的重算，却没有省去目标对条件的注意力读取**。

后续只有目标 query，而目标本来就可以看全部有效前缀和整个目标块，因此不再需要首步完整的块因果 mask；若存在 padding，无效 key 仍须屏蔽。具体 mask 代码见下一节。

用长度表示会更直观。设条件 P 个 token、目标 T 个 token、每层 H 个本地注意力头、头维 D：

| 张量 | 首步 `extract` | 后续 `cached` |
| --- | --- | --- |
| 本层新算的 Q | `[B, P+T, H, D]` | `[B, T, H, D]` |
| 本层新算的 K/V | 各为 `[B, P+T, H, D]` | 各为 `[B, T, H, D]` |
| 本层持久保存的条件 K/V | 各为 `[B, P, H, D]`，从首步切出并独立保存 | 只读这两份张量 |
| 目标注意力实际读取的 K/V | 包含条件与本步目标 | 拼接后仍各为 `[B, P+T, H, D]` |
| 经过注意力、FFN 传播到下一层的 token | 条件与目标 | 仅目标 |

例如 P=6、T=4，首步每层新算 10 个 query，后续新算 4 个 query，但后续每个目标 query 仍会读取 10 个 key/value。这是长度算例，不是模型固定尺寸，也不是速度倍率。

#### D. 一个后续步结束：只带走目标 latent，保留条件缓存

本步目标经过全部层和输出头后，scheduler 再更新目标 latent。下一步重复 C：重新进入第 1 层、依层读取缓存、产生新的目标预测。**跨去噪步传递的主要对象是目标 latent 与固定前缀缓存，不是持续增长的目标 K/V 历史。**

最后一次 scheduler 更新结束，目标 latent 才经还原、反归一化与 VAE 解码成为 RGBA 图片。VAE 不读取这套 DiT 前缀缓存，也不是每个去噪步都把完整目标解码成 PNG。

这与自回归 LLM 的常见追加式 KV Cache 不同：这里缓存的前缀长度 P 在一次请求内固定，目标块每步整体重算，不会每步把新目标 K/V 追加成更长的历史。

#### E. 为什么旧缓存可以配新的目标：依赖链与生命周期

只要输入、权重和布局不变，首步与后续步在每个层的条件计算都相同：条件看不到变化的目标，且总使用固定零时间调制。由第 1 层到第 L 层逐层推导，前缀 K/V 才能保持不变。这是模型依赖结构允许的复用，不是认为“相邻两步差别小”而近似跳算。

| 对象 | 一次请求内怎样流动 |
| --- | --- |
| 文字／参考图条件输入 | 循环前准备，在布局一致时保持固定 |
| 每层条件 K/V | 首步写一次，后续按层读取；不随 scheduler 更新 |
| 当前目标 Q/K/V、注意力输出和 FFN 状态 | 每步、每层新算，不作为跨步目标缓存 |
| 目标 latent | 初始为噪声，每步由 scheduler 更新 |
| 正条件与负条件缓存 | true CFG 启用时分别建立，不能串用 |
| 请求结束后的缓存 | 此 Pipeline 没有自动建立跨请求持久缓存服务；不能默认下次请求命中 |

#### F. 代入具体尺寸：输出 1024×1024、参考图 1024×1204、文本 2048 token

下面按你给定的尺寸，把数据流中的 shape 展开。图片尺寸统一用 **宽 W × 高 H** 表示，张量中的空间轴则明确写成 H、W，避免横竖反过来。

**算例假设：**batch=1、一张参考图、单卡且不做 TP、BF16、先看正条件单分支；采用源码默认的 32 层、32 个注意力头、head_dim=128，模型宽度为 4096。这里将“2048 token”定义为 Pipeline 去掉系统前缀后、进入 DiT 的**有效非图像 token 总数**，包含保留的文本标记，不包含参考图图像槽，也不含 padding。它不是 `prompt_embeds.shape[1]`。

如果 2048 仅指用户原始 prompt 分词结果，模板、图像起止等特殊 token 会额外贡献长度。设这些净增加量为 δ，则下文 `3098、6248、10344` 分别变为 `3098+δ、6248+δ、10344+δ`；δ 需要实际 processor/tokenizer 执行后取得，不能凭字符数猜测。如果你说的 2048 是整个 `prompt_embeds` 长度，则也不能再次把全部 2048 当成纯文本：应先减掉其中图像槽数。

**第一步：参考图先缩放，而不是直接对 1204 除以 16。**本节固定的 Diffusers Pipeline 参数名为 `output_resolution=1024`；它用参考图原始宽高比与约 1024² 的目标面积，分别将宽高四舍五入到 32 的倍数：

```python
ratio = 1024 / 1204
width_f = sqrt(1024**2 * ratio)
width  = round(width_f / 32) * 32            # 960
height = round((width_f / ratio) / 32) * 32  # 1120
```

因此，原图 **W=1024、H=1204 → W=960、H=1120**，VAE latent 的空间尺寸为 **H=70、W=60**。四舍五入后面积不是严格的 1024²，这是正常结果。该缩放发生在 Pipeline 的参考图预处理；VLM 还会经自己的 processor，不能把 VAE 像素张量误当成 VLM 的 patch 张量。

**第二步：算参考 token、图像槽与联合长度。**匹配模型的 processor 与 VAE 布局需要满足源码协议：一个 VLM 图像槽对应 2×2=4 个参考 latent token。本例参考图有 4200 个 DiT token，对应 1050 个 VLM 图像槽。真实部署应读取 `image_pad_mask` 并核对该槽数，不能假定任意自定义 processor 都保持这个关系。

```text
有效非图像 token                  U = 2048
参考 VAE / DiT token             R = 70 × 60 = 4200
VLM 参考图像槽                    S = R / 4 = 1050
prompt_embeds 的序列长度          E = U + S = 3098
目标 token                       T = 64 × 64 = 4096
目标占位槽                       A = T / 4 = 1024
条件前缀展开后长度                P = U + R = 6248
首步联合序列长度                  N = P + T = 10344
```

注意 `prompt_embeds` 中的 1050 个图像槽会被**四倍展开并替换**，不能在 3098 后面再直接追加 4200，造成重复计数。

| 输入准备阶段 | 张量 shape | 含义 |
| --- | --- | --- |
| VAE 参考像素 | `[1, 4, 1, 1120, 960]` | B,C,F,H,W；RGBA，F=1 |
| 参考 latent | `[1, 64, 1, 70, 60]` | VAE 压缩 16×，64 通道 |
| 展平参考 latent | `[1, 4200, 64]` | 尚未 `img_in` 投影 |
| 展平目标噪声／目标 latent | `[1, 4096, 64]` | 目标 H=W=64 个 latent |
| 拼接的参考与目标 latent | `[1, 8296, 64]` | 4200+4096，**不含文本** |
| `prompt_embeds` | `[1, 3098, 4096]` | 2048 非图像 token + 1050 参考图像槽 |
| 追加目标槽后的 `img_mask` | `[1, 4122]` | 3098+1024，尚未展开为 DiT 长度 |
| `img_in` 后的图像表示 | `[1, 8296, 4096]` | 参考与目标统一投影到模型宽度 |
| 占位展开与替换后的联合表示 | `[1, 10344, 4096]` | 6248 条件 + 4096 目标 |
| 展开后的目标／图像位置标记 | 各为 `[10344]` | 对应 `target_token_mask`／`image_pad_mask` 等，非像素 mask |

本例 `img_shapes` 为 `[[ (1,70,60), (1,64,64) ]]`。VAE 的 F=1 只是保留的单帧轴，不表示正在生成视频。目标噪声在代码初始构造时使用 `[B,1,C,H,W]`，此处统一展示展平后的目标形状；解码前则还原为 `[B,C,1,H,W]`。

**第三步：首步每层建缓存。**每层联合输入 `[1,10344,4096]` 经过归一化、调制和投影，再将 4096 拆为 32×128：

```text
首步每层新算 Q、K、V：各 [1,10344,32,128]
                          ↓ 按 token 维取前 6248
cache[ℓ].K：                [1,6248,32,128]
cache[ℓ].V：                [1,6248,32,128]

首步注意力输出合头：         [1,10344,4096]
首步 FFN 后送下一层：        [1,10344,4096]
```

第 ℓ 层缓存存第 ℓ 层自己的前缀 K/V，各层 shape 一样、数值不同；不是把 32 层压进一份共享 K/V。首步最后的输出头可产生 `[1,10344,64]`，Pipeline 只取末尾目标 `[1,4096,64]` 更新目标 latent。

**第四步：后续步只新算目标，但拼接后读取长度不变。**

| 后续第 ℓ 层的对象 | Shape |
| --- | --- |
| 裁掉条件后的目标隐藏状态 | `[1,4096,4096]` |
| 本步新算的目标 Q／K／V | 各 `[1,4096,32,128]` |
| 从 `cache[ℓ]` 读取的条件 K／V | 各 `[1,6248,32,128]` |
| 沿 token 轴拼接后的 K／V | 各 `[1,10344,32,128]` |
| 逻辑目标 attention score | `[1,32,4096,10344]` |
| attention 输出，尚未合头 | `[1,4096,32,128]` |
| 合头、输出投影与 FFN 后 | `[1,4096,4096]` |
| 最终目标预测、scheduler 更新后的目标 | 各 `[1,4096,64]` |

上述 Q/K/V 布局按缓存代码采用的 `[B,tokens,heads,head_dim]` 书写；某些底层 attention 内核会转成 `[B,heads,tokens,head_dim]`。逻辑 score 的形状说明“谁查询谁”，**不表示 FlashAttention/Flex 等内核会实际物化完整 score 张量**。

无 padding 时 cached attention mask 可以为 `None`；有无效 key 时，可广播的 key mask 为 `[B,1,1,10344]`。首步块因果关系逻辑上覆盖 10344×10344，但 Flex `BlockMask` 是专用结构，不能按这个大小推断实际 mask 存储；固定实现还会按 128-token block 对齐到 10368，再裁回真实长度。默认分段处理器则不需要构造这份完整 mask。

**第五步：最后才解码。**所有 scheduler 更新完成后，目标 `[1,4096,64]` 还原为 `[1,64,1,64,64]`，反归一化并经 VAE 得到 RGBA 像素张量 `[1,4,1,1024,1024]`，去掉单帧轴后为 `[1,4,1024,1024]`；交付的 PIL 图像尺寸为 1024×1024，模式为 RGBA。

**这组尺寸的缓存成本。**BF16 每元素 2 字节，每层 K+V 为 `2×1×6248×32×128×2 = 102367232` 字节，约 **97.625 MiB**；32 层合计约 **3.051 GiB**。这是正条件单分支的纯 KV 数据，不含模型权重、目标激活、临时拼接、内核工作区和内存分配开销。true CFG 的负条件长度应另算；不能默认同长度并机械翻倍。

<figure>
  <div style="overflow-x:auto">
    <a href="/images/qwen-image21/kv-shapes.svg" target="_blank" rel="noopener"><img src="/images/qwen-image21/kv-shapes.svg" width="1120" height="1700" loading="lazy" alt="具体尺寸的Qwen-Image-2.1张量流：参考图缩放为960乘1120，产生4200参考token；2048非图像token组成6248条件前缀，目标4096token；首步10344个query，后续4096个query读取10344个key/value，展示每层缓存shape和最终RGBA输出。" /></a>
  </div>
  <figcaption>带 shape 的实例图 · B=1、无 TP、32 头×128；2048 指有效非图像 token。参考图的像素宽高与张量 H/W 顺序已分开标注。</figcaption>
</figure>

可运行 [shape_example.py](/learning/qwen-image21/shape_example.py) 复核这些算术关系。它不会分配模型张量或下载权重，不代表已实测该输入的模型兼容性、显存峰值或质量；原始 prompt 的模板开销与实际 processor 输出仍应在真实请求中记录。


下面再从 mask、调制和源码角度验证这条数据流，并讨论显存与失效条件。 图的[标准库绘制脚本](/learning/qwen-image21/build_kv_dataflow.py)也随文提供。


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

## 结合代码理解 attn_mask：它在限制什么

`attn_mask` 决定一个 query 可以读取哪些 key。对 Qwen-Image-2.1 而言，它既表达联合文本／图像的块因果结构，也可以屏蔽 padding；它不是编辑时圈选修改位置的像素 mask。

本节沿用本文固定的 Diffusers 提交 `f9316b0d4d69555c5e3f625585262e79cd299d34`。代码片段保留关键语句，省略类型、并行与部分元数据准备，不是可独立运行的完整推理实现。

### 1. 源码规则：文本因果，图像块内部双向

[`build_qwenimage21_block_causal_mask`](https://github.com/huggingface/diffusers/blob/f9316b0d4d69555c5e3f625585262e79cd299d34/src/diffusers/models/transformers/transformer_qwenimage21.py#L257-L305) 用 `image_ids` 标记联合序列：文本位置为 `-1`，每张参考图与目标图分别使用不同的非负 ID。

```python
# 源码关键语句，q_idx 是 query 位置，kv_idx 是 key 位置。
q_image_id, kv_image_id = image_ids[q_idx], image_ids[kv_idx]
same_image_block = (q_image_id == kv_image_id) & (q_image_id >= 0)
allowed = ((q_idx >= kv_idx) | same_image_block) & key_valid[batch_idx, kv_idx]
return allowed & ~is_padding
```

逐句看：

- `q_idx >= kv_idx`：允许读取当前位置及历史位置，包括文本的自身位置。
- `same_image_block`：同一张图内部额外开放双向读取；`>= 0` 防止全部文本因共享 `-1` 被误判成一个图像块。
- `key_valid`：排除 prompt 中无效的 padding key。
- `~is_padding`：排除为 FlexAttention 内核对齐而补出的 query／key 位置；它与原始 prompt 的右侧 padding 是不同层次的处理。

参考图 token 会替换预留图像槽，因此真实联合序列可能是文本与参考图交错，而不是简单的三个连续大块。下面用一个七 token 的例子理解：

```text
序列：T0 T1 R0a R0b T2 Xa Xb
类型：文本   参考图0  文本  目标图1

行 = query，列 = key；1 表示允许读取，0 表示禁止。

       T0 T1 R0a R0b T2 Xa Xb
T0      1  0   0   0  0  0  0
T1      1  1   0   0  0  0  0
R0a     1  1   1   1  0  0  0
R0b     1  1   1   1  0  0  0
T2      1  1   1   1  1  0  0
Xa      1  1   1   1  1  1  1
Xb      1  1   1   1  1  1  1
```

`R0a` 能读取后面的 `R0b`，因为它们属于同一张参考图；它不能读取更后面的文本 `T2` 或目标图。最后两行则能读取所有有效条件和整个目标块。两张参考图即使相邻，也要用不同 ID，不能合并成同一双向块。

注意：这是二维 token 可见性矩阵，不是图片像素空间中的黑白选区。

### 2. 为什么不能直接去掉 mask，或者换成普通 causal

若把首步联合序列改成完全双向注意力，条件 query 就会读取目标图的 K/V。目标随去噪变化，第一层条件输出便会变化，下一层条件 K/V 也会变化。此时直接复用首步条件缓存，会改变原本的模型计算。

这里有一个层间细节：本层 K/V 是在本层注意力输出之前生成的。因此并非“只要看见目标，所有层的 K/V 当场都变化”，而是目标依赖通过条件输出传播到后续层，使**整套逐层前缀缓存不再成立**。

反过来，简单使用全序列下三角 `is_causal=True`，会禁止同一图像块里的前 token 读取后 token，也不符合模型的图像块双向语义。所需的是块因果结构，不是普通语言模型的逐 token 因果结构。

### 3. 源码里有两种实现：完整 mask 与按段切片

首步联合序列包含条件 query 和目标 query，需要保留上述规则。代码提供两条处理器路径：

| 路径 | 如何表达限制 | `attn_mask` 的含义 |
| --- | --- | --- |
| `QwenImage21FlexAttnProcessor` | 构造 `BlockMask`，交给 flex 后端 | 包含完整块因果规则 |
| `QwenImage21AttnProcessor` | 每段只传入合法范围的 K/V；文本段额外加因果 mask | 图像段可能无需显式 mask；边界已由切片限制 |

默认分段路径的[关键调用](https://github.com/huggingface/diffusers/blob/f9316b0d4d69555c5e3f625585262e79cd299d34/src/diffusers/models/transformers/transformer_qwenimage21.py#L505-L544)可概括为：

```python
# 每段 query 只访问到本段结束的 key/value。
# 文本段的 seg_mask = [对历史全开放 | 本段内部下三角]。
# 图像段内部双向，若没有 padding，seg_mask 可以为 None。
out = dispatch_attention_fn(
    query[:, start:end],
    key[:, :end],
    value[:, :end],
    attn_mask=seg_mask,
    dropout_p=0.0,
    backend=None,
    parallel_config=self._parallel_config,
)
```

对参考图段而言，后面的目标 K/V 根本没有传进该次调用，所以 `attn_mask=None` 仍然可以正确隔离目标。**调试时应同时看传入的 Q/K/V 范围与 mask，不能只搜索参数是否为 None。**

Flex 路径则把同样的结构写入 `BlockMask`，由一次 flex 调用处理；这是执行组织不同，不是两套模型语义。`BlockMask` 是专用对象，也不等同于手工分配一个完整的 $N\times N$ 布尔矩阵。固定版本还提示，未编译的 flex 路径在高分辨率下可能产生很大的中间显存开销，因此启用时要核对处理器与编译配置。

### 4. 为什么后续 cached 步反而可以不带完整 mask

[`forward` 的 cached 分支](https://github.com/huggingface/diffusers/blob/f9316b0d4d69555c5e3f625585262e79cd299d34/src/diffusers/models/transformers/transformer_qwenimage21.py#L967-L995)先裁掉前缀 query：

```python
joint_hidden_states = joint_hidden_states[:, prefix_len:]
rotary_emb = rotary_emb[prefix_len:]
modulation_mask = modulation_mask[prefix_len:]
attention_mask = (
    None if joint_key_valid is None
    else joint_key_valid[:, None, None, :]
)
```

之后每层将已缓存的条件 K/V 与新目标 K/V 拼接。此时张量范围是：

```text
Q：     [当前目标]
K/V：   [缓存条件前缀 | 当前目标]
mask：  [原矩阵最后的目标 query 行]
```

观察前面矩阵，最后两行都是 1，因此没有无效 key 时，剩下的结构就是完全可见，`attention_mask=None` 合法；有 padding 时仍要保留 key 有效性掩码。

后续步可以省略完整块因果 mask，**是因为条件 query 已经不参加计算**，而不是取消了建立缓存时所需的依赖约束。普通 `is_causal=True` 也不能替代这里的目标对完整前缀及整个目标块的读取。

### 5. mask 与固定时间调制要一起看

只屏蔽目标还不够。源码在 `causal_condition=True` 时，[追加固定的 t=0 输入](https://github.com/huggingface/diffusers/blob/f9316b0d4d69555c5e3f625585262e79cd299d34/src/diffusers/models/transformers/transformer_qwenimage21.py#L944-L953)，再按 token 类型选取调制：

```python
# forward：增加条件专用的固定时间输入。
timestep = torch.cat([timestep, timestep.new_zeros(1)], dim=0)
modulation_mask = target_token_mask

# _select_modulation_rows：real/zero 是调制参数，不是原始时间值。
real, zero = params[:-1].unsqueeze(1), params[-1:].unsqueeze(0)
return torch.where(target_token_mask.view(1, -1, 1), real, zero)
```

目标 token 使用当前时间调制，条件使用固定 t=0 调制。变量 `zero` 指由 t=0 计算得到的参数，不代表参数本身全为零。

所以可复用性的完整依据是：**条件输入固定、注意力不读取变化目标、条件调制固定，并保持位置、权重和执行协议一致。**

### 6. 三种 mask 不要混用

| 名称 | 在代码中的作用 | 不代表什么 |
| --- | --- | --- |
| `attention_mask`／`attn_mask` | 控制 query 对 key 的读取，可含块结构或 padding 限制 | 图像编辑选区 |
| `img_mask`／`image_pad_mask` | 标记图像槽与展开后的图像 token 位置，用于联合序列组装和元数据 | True 位置需要被遮蔽、不参与注意力 |
| `target_token_mask` | 区分目标与条件，选择时间调制等 | 目标图中的局部修改区域 |

另外，布尔语义要按接口判断。本节的 mask predicate 及 PyTorch SDPA 布尔 `attn_mask` 使用 **True=允许**；某些其他 API（例如 `MultiheadAttention` 的布尔掩码）采用 True=屏蔽，不能原样照搬。加性 mask 则通常用 0 表示不惩罚、负无穷表示禁止。

### 7. 一个可在 CPU 上运行的依赖关系示例

本站提供 [attention_mask_demo.py](/learning/qwen-image21/attention_mask_demo.py)，只使用 Python 标准库：

```bash
python3 research/qwen-image21/attention_mask_demo.py
```

它打印上述七 token 矩阵，并用两层简化注意力比较“改变目标后，条件输出是否改变”：块因果 mask 下条件差异为零；完全双向注意力下条件会受到目标影响。脚本已执行通过。

这是对依赖关系的教学验证，没有加载 Qwen 权重，没有模拟完整 RoPE、时间调制或推理缓存，也不构成真实模型的缓存一致性或性能基准。


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

## 与 Qwen-Image-Edit-2511 对比：Pipeline 与 Backbone 怎样变化

本节的 **2511 指 `Qwen/Qwen-Image-Edit-2511`**，对应官方 README 的 `QwenImageEditPlusPipeline`，不是另一个名为“Qwen-Image-2511”的纯文生图检查点。对比双方使用同一个 Diffusers 提交 `f9316b0d4d69555c5e3f625585262e79cd299d34`，重点看原生实现的推理依赖，不把第三方加速器能力混入基线。

本轮没有读取到 Hugging Face 的 2511 `model_index.json`／checkpoint config（请求返回 403），因此下列层数、头数与通道数字明确区分为**源码构造默认值和标准 VAE 路径的算例**，不冒充权重配置实测。模型名称与 Pipeline 绑定由官方 GitHub 示例核实。

### 1. Pipeline：参考图两条路径都存在，但组织不同

| 环节 | Edit-2511：`QwenImageEditPlusPipeline` | 2.1：`QwenImage21Pipeline` |
| --- | --- | --- |
| 条件理解 | Qwen2.5-VL，源码文档标明 7B；编码文字与参考图 | Qwen3-VL；官方标明 8B；编码文字与参考图 |
| 参考图的另一条路径 | `AutoencoderKLQwenImage` 编码参考 latent | 专用 `AutoencoderKLQwenImage21` 编码参考 latent |
| 图像 token 准备 | VAE latent 做 2×2 打包；图像流排列为 `[目标, 参考图]` | latent 直接空间展平；输入图像张量为 `[参考图, 目标]` |
| 进入 Backbone | 多模态条件 embedding 作为独立 context 流；目标与参考 latent 作为 image 流 | 展开图像占位，将参考 latent 替换进条件位置，再附加目标，组成单流序列 |
| 去噪循环 | 每步把当前目标与固定参考 latent 再拼接，重新执行双流 Backbone | 首步 `extract`，后续 `cached` 可只计算目标 query 与目标 block 状态 |
| 预测截取 | 取图像输出的开头 `:latents.size(1)` | 取输出末尾 `-latents.size(1):` |
| 调度更新 | scheduler 只更新目标 latent，不更新参考 latent | 同样只更新目标 latent |
| 输出 | 旧版标准 RGB VAE 路径 | 原生 RGBA VAE 路径 |

两版都不能简单理解成“参考图只走 VAE”。2511 的参考图也进入 VLM 以提供语义条件，并单独经过 VAE 提供视觉 latent。差异在于 DiT 如何接收和更新这些表示，而不只是换了一个文字编码器。

2511 的图像预处理还区分 VLM 条件图与 VAE 图的尺寸路径。不能把同一个经过任意缩放的张量同时塞给两个编码器，期待自动得到相同效果。

### 2. 先看循环代码：目标的位置恰好相反

[2511 的去噪循环](https://github.com/huggingface/diffusers/blob/f9316b0d4d69555c5e3f625585262e79cd299d34/src/diffusers/pipelines/qwenimage/pipeline_qwenimage_edit_plus.py#L805-L849)关键代码可缩写为：

```python
# Edit-2511：目标在前，固定参考 latent 在后。
latent_model_input = torch.cat([latents, image_latents], dim=1)
noise_pred = transformer(
    hidden_states=latent_model_input,
    encoder_hidden_states=prompt_embeds,
    # 另传 timestep、img_shapes、padding mask 等
)[0]
noise_pred = noise_pred[:, :latents.size(1)]
latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]
```

[2.1 的循环](https://github.com/huggingface/diffusers/blob/f9316b0d4d69555c5e3f625585262e79cd299d34/src/diffusers/pipelines/qwenimage21/pipeline_qwenimage21.py#L759-L816)则包含显式的逐层条件 KV 缓存协议：

```python
# 2.1：输入图像 latent 中，参考在前、目标在后。
latent_model_input = torch.cat([input_images_latents, latents], dim=1)
kv_mode = "extract" if i == 0 else "cached"  # 此处假定缓存启用
noise_pred = transformer(
    hidden_states=latent_model_input,
    encoder_hidden_states=prompt_embeds,
    img_mask=image_pad_mask,
    kv_cache=cond_cache,
    kv_cache_mode=kv_mode,
    # 另传 timestep、img_shapes、padding mask 等
)[0]
noise_pred = noise_pred[:, -latents.size(1):]
latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]
```

这些片段是省略参数的阅读骨架，不可直接作为完整调用复制运行。2.1 在 Pipeline 层仍会拼接和准备部分输入；真正“只计算目标”的裁剪发生在 Transformer cached 分支中，不能误写成后续步完全不处理任何前缀元数据。

迁移时如果照搬旧版的输出截取位置、`img_shapes`、位置编码或 mask，会把参考与目标弄反。权重文件、latent 和编码器 embedding 也不能因为都叫 Qwen-Image 就直接互换。

### 3. Backbone：双流 joint attention 变成单流 block-causal

| 结构项 | `QwenImageTransformer2DModel`（2511 使用的类） | `QwenImage21Transformer2DModel` |
| --- | --- | --- |
| block 数，源码默认 | 60 | 32 |
| 注意力宽度，源码默认 | 24 头 × 128 = 3072 | 32 头 × 128 = 4096 |
| 数据流 | image / context 分别投影、调制及更新 | 联合 token 序列通过同一套 block |
| 注意力 | 两流各自产生 Q/K/V，拼接做双向 joint attention，再拆回两流 | 联合序列遵循块因果；图像块内部双向 |
| FFN | 图像 `img_mlp` 与文本 `txt_mlp` 分开，GELU 类激活 | 单流 SwiGLU，默认 ratio=3 |
| 调制 | 每层独立 `img_mod`、`txt_mod`；包含 shift/scale/gate | 共享 modulation；按目标与条件选当前 t／固定 0，使用 scale 与门控 |
| 条件跨步独立性 | 不能由固定输入直接推出，联合注意力会重新混入目标 | 块因果隔离 + 固定条件时间调制，使前缀可独立于目标 |

2511 的 [attention processor](https://github.com/huggingface/diffusers/blob/f9316b0d4d69555c5e3f625585262e79cd299d34/src/diffusers/models/transformers/transformer_qwenimage.py#L514-L587)中，有这样的调用：

```python
# 两流分别投影、QK norm 和 RoPE 后：
joint_query = torch.cat([txt_query, img_query], dim=1)
joint_key   = torch.cat([txt_key,   img_key], dim=1)
joint_value = torch.cat([txt_value, img_value], dim=1)

joint_hidden_states = dispatch_attention_fn(
    joint_query, joint_key, joint_value,
    attn_mask=attention_mask,  # 正常路径主要用于屏蔽无效文本 key
    is_causal=False,
    # 省略 dropout、backend、parallel_config
)
# 随后按 seq_txt 拆回文本与图像输出，并分别执行残差和 MLP。
```

“joint”不代表它已经是 2.1 的单流 block：2511 在 attention 处联合交互，投影、调制、输出和 FFN 仍分成两套。它的文本流会被当前图像流影响，参考图的高层状态也会与目标交互。

### 4. 为什么不能把 2.1 的前缀缓存直接搬给 2511

即使参考图文件、VLM embedding 和初始参考 latent 都不变，也不等于参考条件在每个 DiT 层里的 K/V 不变。2511 的双向联合注意力让条件输出依赖当前目标；这些输出进入下一层，就会改变下一层条件 K/V。

源码还存在 `zero_cond_t` 可选分支：启用时参考图可以选 t=0 调制，但这**不等价于 2.1 的 `causal_condition`**。2511 的该分支并未同时建立“条件不能读取目标”的块因果隔离，文本流也仍使用当前时间对应的调制。因此仅启用固定参考时间，不足以得到整套精确条件缓存。

同样，两版都能看到 `cache_context("cond")` 之类的框架上下文，不能仅根据这个名称判断缓存内容。2.1 的证据是具体的 `QwenImage21KVCache`、每层写入／拼接逻辑与 `extract/cached` 执行模式；2511 所比较的原生路径没有这套前缀协议。第三方框架的近似特征缓存、编译缓存或其他加速，是另一项需独立核对的能力。

### 5. VAE 与 token：同为 64 维输入不代表相同编码

旧版标准路径将 16 通道、空间压缩 8× 的 latent 按 2×2 打包：每 token 包含 $16\times4=64$ 个数。2.1 则直接消费 64 通道、空间压缩 16× 的 latent，不再做这层 2×2 打包。

以 1024×1024 目标图像为算例：

```text
2511 标准路径：1024² → 128×128×16 latent → 64×64 个 token，每个 64 维
2.1 路径：    1024² →  64×64×64 latent → 64×64 个 token，每个 64 维

两边目标 token 均为 4096；表示语义、VAE 和输出通道能力不同。
```

因此不能说“VAE 压缩率从 8× 到 16×，目标序列就一定少四倍”。还要计入打包。参考图 token、VLM 展开长度、模型宽度、层数、步数和缓存会共同决定总成本。标准旧版路径的数值来自类默认与代码注释，真实部署仍须检查所加载配置。

### 6. CFG、调度与输出也不能直接照搬

两条 Pipeline 都使用 FlowMatch Euler 类调度，并可按条件启用正／负两次前向的 true CFG；但固定版本的代码还有具体区别：

- 2511 在 CFG 线性组合后，额外按正条件预测的范数进行重缩放；2.1 对应循环没有同一段范数重缩放逻辑。
- 2.1 的正负分支各自持有前缀缓存，不应混用；旧版 `cond/uncond` 上下文名称不是这套缓存机制的证明。
- scheduler shift、时间网格、negative prompt、步数以及 Turbo／Lightning 适配权重，应使用各自匹配的配置，不能为了公平而机械复用一组值。
- RGB 输出与 RGBA 输出需要不同的交付检查；PNG 扩展名不能替代 alpha 有效性验证。

因此迁移评估应对齐任务、尺寸、质量目标、精度和实际前向次数，分别测编码、首步、缓存后续步及 VAE，而不是把 60/32 的层数比当成整体加速倍率。

### 7. 并排可视化：从输入到每一步计算

<figure>
  <div style="overflow-x:auto">
    <a href="/images/qwen-image21/2511-vs-21.svg" target="_blank" rel="noopener"><img src="/images/qwen-image21/2511-vs-21.svg" width="1120" height="1480" loading="lazy" alt="Qwen-Image-Edit-2511 与 Qwen-Image-2.1 并排对比：上半部分比较条件编码、latent 排列、循环和解码，下半部分比较双流联合注意力与单流块因果、缓存首步和后续步；底部解释相同图像尺寸的 token 数。" /></a>
  </div>
  <figcaption>图 3 · Pipeline 与 Backbone 的两级对比。层数等为固定源码默认，图中箭头表示数据依赖，不代表实测速度；手机端可横向滑动，点击查看 SVG 原图。</figcaption>
</figure>

源码入口：[官方 2511 用法](https://github.com/QwenLM/Qwen-Image/blob/6b5e1f5cec987d404be5ac6657db3b9aacb56a89/README.md#qwen-image-edit-2511-for-image-editing-multiple-image-support-and-improved-consistency)、[2511 Pipeline](https://github.com/huggingface/diffusers/blob/f9316b0d4d69555c5e3f625585262e79cd299d34/src/diffusers/pipelines/qwenimage/pipeline_qwenimage_edit_plus.py)、[旧版 Transformer](https://github.com/huggingface/diffusers/blob/f9316b0d4d69555c5e3f625585262e79cd299d34/src/diffusers/models/transformers/transformer_qwenimage.py)、[2.1 Pipeline](https://github.com/huggingface/diffusers/blob/f9316b0d4d69555c5e3f625585262e79cd299d34/src/diffusers/pipelines/qwenimage21/pipeline_qwenimage21.py)、[2.1 Transformer](https://github.com/huggingface/diffusers/blob/f9316b0d4d69555c5e3f625585262e79cd299d34/src/diffusers/models/transformers/transformer_qwenimage21.py)。图的[可复现脚本](/learning/qwen-image21/build_comparison.py)随文章提供。


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

