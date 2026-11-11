# WenMang · 技术与生活

[个人网站](https://wenmang98k.github.io/) 只保留两大入口：

- **[技术](https://wenmang98k.github.io/tech/)**：按 topic 整理 AI Infra、Agent、AIGC、模型压缩与推理加速的文章。
- **[生活](https://wenmang98k.github.io/life/)**：按兴趣整理摄影、运动、骑行、美食等记录，支持图文混排。

职业经历与开源项目移至[技术介绍](https://wenmang98k.github.io/tech/about/)，不再铺在首页。

## 现有文章

归类于「技术 → LLM 推理与投机解码」：

- [LLM 投机解码综述](articles/speculative-decoding/README.md) · [网页](https://wenmang98k.github.io/articles/speculative-decoding/)
- [投机解码技术时间轴](articles/speculative-decoding/timeline.md) · [网页](https://wenmang98k.github.io/articles/speculative-decoding/timeline.html)

研究初稿附有资料核对范围与待复核项，没有进行 GPU 性能对比实验。

## 本地维护

```bash
python3 scripts/build_site.py
python3 -m http.server 8000 --bind 127.0.0.1
```

目录数据统一维护在 `content/catalog.json`。添加文章、主题与图片的方法见[写作说明](docs/WRITING.md)。分类页生成仅需 Python 标准库。

修改综述 Markdown 后，用 Pandoc 重建阅读页：

```bash
bash articles/speculative-decoding/build.sh
python3 articles/speculative-decoding/scripts/verify_probabilities.py
```

GitHub Pages 使用已经提交的静态 HTML，不需要在部署时安装 Pandoc 或运行生成器。
