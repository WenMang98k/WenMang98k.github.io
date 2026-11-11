# 内容结构与写作方式

首页只保留两大内容入口：技术、生活。文章属于一个大类及其中一个主题，不直接堆在首页。

```text
首页 /
├── 技术 /tech/
│   ├── LLM 推理与投机解码 /tech/llm-inference/
│   │   ├── LLM 投机解码综述
│   │   └── 投机解码技术时间轴
│   ├── Agent /tech/agents/
│   ├── AIGC 与多模态 /tech/aigc/
│   ├── 模型压缩与硬件加速 /tech/compression/
│   ├── AI Infra 与服务工程 /tech/infrastructure/
│   └── 关于我的技术经历 /tech/about/ （辅助链接）
└── 生活 /life/
    ├── 摄影 /life/photography/
    ├── 健身 /life/fitness/
    ├── 跑步 /life/running/
    ├── 摩托 /life/motorcycling/
    ├── 羽毛球 /life/badminton/
    ├── 美食 /life/food/
    ├── 麻将 /life/mahjong/
    └── 游戏 /life/gaming/
```

现有文章保留 `/articles/speculative-decoding/` 地址，避免旧链接失效。页面通过面包屑与导航体现所属分类；磁盘路径不必和阅读分类完全相同。

## 添加主题与文章

1. 在 `content/catalog.json` 的 `tech.topics` 或 `life.topics` 添加主题；不要手工编辑生成的主题列表页。
2. 在 `articles/` 中新建文章目录，保存 Markdown 和图片；先生成对应 `index.html`。
3. 在 `posts` 添加一项：`title`、`section`（tech 或 life）、`topic`（对应 slug）、`url`（站点绝对路径）、`date`、`kind`、`summary`。列表按 catalog 顺序展示，可将新文章加在前面。
4. 执行 `python3 scripts/build_site.py`，刷新首页、分类页及主题页。该命令会检查文章地址对应的本地文件是否存在。

没有文章的主题显示真实空状态，不生成占位文章或虚构日期。

## 生成新的普通文章页

新文章可使用 `templates/article.html`。例如在创建并写好 `articles/my-photo-note/README.md` 后，从仓库根目录运行：

```bash
pandoc articles/my-photo-note/README.md -f markdown -t html5 -s --mathml \
  --template templates/article.html \
  -M title='文章标题' -M description='文章简介' -M date='实际发布日期' \
  -M category='生活' -M topic='摄影' \
  -V site-root='../../' -V category-url='../../life/' \
  -V topic-url='../../life/photography/' \
  -o articles/my-photo-note/index.html
```

技术文章使用对应的 category、topic 与链接。新增文章后仍需在 catalog 登记，模板不会自动扫描文章目录。

现有投机解码综述沿用专门的阅读版式：

```bash
bash articles/speculative-decoding/build.sh
```

## 摄影与图文文章

图片建议放在文章自己的 `images/` 下，用相对路径引用，避免依赖聊天附件或会失效的外链。以下只是写作示例，不代表已经发布作品：

```markdown
<figure>
  <img src="images/autumn.jpg" alt="银杏树叶间的高塔" loading="lazy" width="1280" height="1920">
  <figcaption>作品说明；地点与日期按实际情况填写。</figcaption>
</figure>
```

文章模板支持自适应图片、图注和长代码／表格；相片可与文字混排。导出适合网页的图片尺寸，并自行确认是否保留拍摄地点等 EXIF 信息。当前没有接入摄影原图。

## 检查与发布

```bash
python3 scripts/build_site.py
python3 -m http.server 8000 --bind 127.0.0.1
```

检查首页 → 大类 → 主题 → 文章 → 返回主题的路径，以及手机布局。提交生成后的 HTML；GitHub Pages 无需运行 Python 或 Pandoc。
