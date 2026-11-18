# Qwen-Image-2.1 技术与产品研究

文章：[Qwen-Image-2.1：从模型结构到产品落地](https://wenmang98k.github.io/articles/qwen-image-21/)。

## 本轮核对范围

固定版本的官方 README／LICENSE、Diffusers 2.1 Transformer／Pipeline、LightX2V 2.1 配置与 LoRA 校验，共 12 个已读文件。具体版本和 SHA256 见 `sources.lock.json`。没有复制模型材料或下载权重，没有执行 GPU 生图或速度基准。

用户给定的 `lightx2v/Qwen-Image-2.1-Lightning` 模型卡被网络代理阻断。其权重名、步数、sigma、CFG、rank、alpha、许可和质量／收益仍待核实；本文没有拿旧版 Lightning 数据补空。

## 本站参考代码

```bash
python3 research/qwen-image21/reference_inference.py --dry-run
python3 research/qwen-image21/reference_inference.py --variant base --width 2048 --height 2048 --dry-run
```

dry-run 不导入 GPU 框架、不下载模型、不执行推理，只展示所选路径和输入设置。真实推理需要相容的软件、硬件、模型访问与许可。运行前参阅固定版本官方用法；脚本没有未经核实的 Lightning 路径。

`evaluation-template.csv` 是产品测试用例模板，结果字段刻意留空，并非已通过的评测。可以扩充任务后隔离调参与验收样本。

官方 2.1 仓库的 Qwen Research License 明确商用需要单独许可；推理代码或 LoRA 的其他许可不替代基座材料条款。不要将本目录的源码链接记录当作授予模型许可。

## 结构与推理图

`build_diagrams.py` 使用 Python 标准库生成本站原创 SVG；运行 `python3 research/qwen-image21/build_diagrams.py` 可更新 `public/images/qwen-image21/architecture.svg` 与 `inference.svg`。图中展开单层 DiT 算子，并分开表示首步 extract 与后续 cached；源码映射见文章。图不代表已验证 checkpoint 配置或实测吞吐。

## attn_mask 源码补充（2026-11-17）

文章新增块因果掩码、分段切片与 cached 目标 query 的调用链，继续使用 sources.lock.json 中已固定的 Transformer 提交与 SHA256；本次重新下载核对，hash 一致。

`python3 research/qwen-image21/attention_mask_demo.py` 使用标准库打印 token 可见性矩阵，验证两层简化注意力中的条件独立性。已在 CPU 执行；它不是官方模型推理，不验证 GPU、实际 KV 缓存或生成质量。构建时由现有 prepare:legacy 流程复制到 `/learning/qwen-image21/`。

## Edit-2511 与 2.1 对比（2026-11-17）

新增核对官方 Qwen-Image README、旧版 Diffusers EditPlus Pipeline 和双流 Transformer，共 3 个文件；固定提交和 SHA256 已追加到 sources.lock.json。2511 Hugging Face model_index 请求 403，未读取 checkpoint config，因此文章明确区分源码默认与已加载权重。没有模型推理或 GPU 基准。

运行 `python3 research/qwen-image21/build_comparison.py` 生成 `public/images/qwen-image21/2511-vs-21.svg`，对比 Pipeline 与 Backbone，显示两条条件路径、目标位置、双流／单流注意力、缓存依赖及 token 数算例。它复用 build_diagrams.py 的标准库绘图组件，不会修改旧图。

## 条件 KV 完整数据流（2026-11-18）

原文章新增一次编辑请求从 VLM/VAE 输入准备、extract 逐层写缓存、cached 每层读取，到 scheduler 和最终 VAE 解码的图文导读；解释网络层与去噪步两条轴、目标每步从第一层重算、前缀长度不增长、正负 CFG 分支隔离及请求缓存生命周期。沿用已固定 Diffusers 源码，无新增权重或 GPU 实测。

`PYTHONDONTWRITEBYTECODE=1 python3 research/qwen-image21/build_kv_dataflow.py` 可重绘 `public/images/qwen-image21/kv-dataflow.svg`，采用现有标准库绘图组件。正文 token 长度为教学算例，不是吞吐测量。

## 具体尺寸 shape 算例（2026-11-18）

新增输出 W1024×H1024、参考 W1024×H1204（默认缩放到 W960×H1120）、有效非图像token 2048 的完整张量表和 `kv-shapes.svg`。明确原始prompt与模板后有效长度、VLM图像槽与DiT参考token的差别。单分支、B=1、无TP、BF16，结构采用源码默认；processor需匹配并验证槽数，不宣称执行真实模型。

`PYTHONDONTWRITEBYTECODE=1 python3 research/qwen-image21/shape_example.py` 输出shape与缓存大小，验证算术；`PYTHONDONTWRITEBYTECODE=1 python3 research/qwen-image21/build_kv_shapes.py` 重绘插图。脚本不分配模型张量、不下载权重。

## 原数据流图内联 shape（2026-11-18）

将输入、联合序列、首步 QKV、逐层缓存、后续目标 QKV、拼接 KV 与 scheduler/VAE 输出 shape 直接整合到原 `kv-dataflow.svg` 节点中，保留读写、跨步保留和目标回环箭头。原图高更新为2320；脚本复用 shape_example.py 的假设与算术校验，未做GPU推理。
