# WenMang 的个人网站与技术笔记

个人主页：[wenmang98k.github.io](https://wenmang98k.github.io/)

记录 AI Infra、算法优化、模型压缩与推理加速，也分享生活与兴趣。

## 技术文章

- **[LLM 投机解码综述](articles/speculative-decoding/README.md)** · [网页阅读](https://wenmang98k.github.io/articles/speculative-decoding/)  
  经典 SD、EAGLE、MTP、Medusa、Hydra、ReDrafter、DFlash、DSpark 等路线；训练与推理流程、概率校正、工程选型，以及 vLLM / SGLang 官方主分支支持矩阵。研究初稿，附资料核对范围与待复核项，未进行 GPU 性能对比实验。

## 本地阅读与更新

网站是纯静态 HTML/CSS，无需安装应用依赖：

```bash
python3 -m http.server 8000 --bind 127.0.0.1
```

文章 Markdown 是编辑源文件；修改后使用 Pandoc 重新生成网页版：

```bash
bash articles/speculative-decoding/build.sh
python3 articles/speculative-decoding/scripts/verify_probabilities.py
```

公式脚本仅校验有限词表上的接受／残差概率恒等式，不代表 LLM 性能测试。GitHub Pages 使用提交的 HTML，部署时不依赖 Pandoc。
