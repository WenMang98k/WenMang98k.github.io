import { mkdir, cp, readFile, writeFile } from 'node:fs/promises';
await mkdir('public/about', { recursive: true });
await cp('articles', 'public/articles', { recursive: true });
for (const file of ['style.css', 'jemdoc.css']) await cp(file, `public/${file}`);
let html = await readFile('index.html', 'utf8');
html = html.replace('href="style.css', 'href="/style.css').replaceAll('href="articles/', 'href="/articles/');
html = html.replace('<nav aria-label="主导航">', '<nav aria-label="主导航"><a href="/">技术首页</a>');
await writeFile('public/about/index.html', html);
