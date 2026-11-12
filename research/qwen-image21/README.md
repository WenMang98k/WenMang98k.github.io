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
