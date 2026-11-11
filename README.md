# WenMang 的个人技术网站

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
- 根目录 `index.html` 与 `style.css`：原有个人介绍，构建时保留为 `/about/`。
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
