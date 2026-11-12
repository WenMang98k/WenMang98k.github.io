# 文氓 · WenMang 的个人网站

Astro + Markdown 构建，GitHub Pages 托管。项目介绍、文章归档与个人主页统一在同一站点。

## 开发与验证

需要 Node.js 22.12+（也支持 24）。在本仓库目录运行：

```bash
export ASTRO_TELEMETRY_DISABLED=1 # 云环境中避免写入用户配置目录
npm ci --cache /workspace/.npm-cache
npm run dev
npm run check
npm run build
npm run preview
```

`dev` 默认使用 4321 端口。`build` 的产物位于 `dist/`。依赖缓存路径可按本机需要调整。

## 写文章

```bash
npm run new:post -- my-topic "我的文章标题"
```

编辑 `src/content/posts/my-topic.md`，填写摘要、日期、标签与正文，将 `draft: true` 改成 `false` 后公开。文章列表、标题/摘要/标签搜索和 RSS 自动更新。草稿不会生成网页，但公开仓库中的 Markdown 仍可被看到。文件名决定文章 URL，公开后尽量保持不变。`speculative-decoding` 是保留的旧文章路径。

支持 Markdown 标题、代码块、表格和链接；图片可放到 `public/images/`，正文使用 `/images/文件名`。新增数学公式支持需要单独配置；旧综述仍使用原有 MathML 页面。

## 更新项目与介绍

- `src/data/site.ts`：项目名称、描述、技术栈与仓库地址；旧文章的列表元数据。
- `src/pages/index.astro`：技术首页。
- `src/styles/global.css`：新版网站样式。
- `tech/about/index.html` 与 `style.css`：原有个人介绍，构建时也保留为 `/about/`。
- `articles/speculative-decoding/`：原有综述及原有地址，构建时复制保留。

旧综述继续用 Pandoc 更新，但请仅在主动修改文章时运行其构建脚本：

```bash
bash articles/speculative-decoding/build.sh
python3 articles/speculative-decoding/scripts/verify_probabilities.py
```

仅预览或构建整站不需要 Pandoc。`public/about/`、`public/articles/` 和对应旧样式由 `scripts/prepare-legacy.mjs` 生成，不要手动修改生成副本。

## 发布

1. 在 GitHub 仓库 Settings → Pages，将 Source 设为 **GitHub Actions**。
2. 将审阅后的修改推送到 `main`，工作流会执行安装、类型检查、构建及 Pages 部署。
3. Pull Request 只验证构建，不部署。也可在 Actions 页面手动运行部署工作流。

正式地址：https://wenmang98k.github.io 。修改站点域名时同步调整 `astro.config.mjs` 与 `src/pages/rss.xml.ts`。此代码不需要服务端密钥或数据库。

本地验证不等于已经发布；GitHub Pages 设置与远程 Actions 运行需在推送后确认。

## 保留的分类站点

远程最新的 `/tech/`、`/life/` 与主题页，以及投机解码时间轴均保留。旧分类目录维护在 `content/catalog.json`，说明见 [docs/WRITING.md](docs/WRITING.md)。`python3 scripts/build_site.py` 会重写根目录的旧首页和分类页；Astro 正式首页由 `src/pages/index.astro` 生成。新 Markdown 文章通过新版 `/articles/` 自动收录，旧目录仍使用 catalog 维护。

## 投机解码学习系列

- `/articles/speculative-decoding/` 现由 Astro 渲染，保留原地址，提供导读、公式、分组目录及 35 篇分篇链接。
- `src/content/posts/speculative-decoding-overview.md` 是新版综述源文件。
- `src/content/posts/sd-*.md` 是方法文章与源码索引；`src/data/speculative-series.json` 维护分组和顺序。
- `research/speculative-decoding/` 保存本站原创有限词表参考代码与已读源码来源记录，不复制上游整仓。
- `articles/speculative-decoding/README.md` 保留早期长版与文献记录；其 Pandoc HTML 不再作为正式综述入口。`prepare:legacy` 会移除该公开副本，避免覆盖 Astro 入口；旧时间轴地址保留。
- 数学公式使用本地 KaTeX 资源。文章目录、复制代码和阅读进度在普通阅读器中统一提供。

更新内容后执行 `python3 scripts/build_site.py` 刷新技术 Topic 归档，再运行 `npm run check && npm run build`。新增分篇需同步系列 JSON 与 catalog；构建脚本会校验文章目标是否存在。

## 主流模型技术观察

- [Qwen-Image-2.1：从模型结构到产品落地](src/content/posts/qwen-image-21.md) · [网页阅读](https://wenmang98k.github.io/articles/qwen-image-21/)。归档在 AIGC 与多模态主题，覆盖技术结构、Lightning／Turbo 区别、产品经理验收和许可边界。
- [研究来源、参考脚本与产品验收模板](research/qwen-image21/README.md)。源码／文档检查不是 GPU 性能实测；Lightning 模型卡仍待补核。
