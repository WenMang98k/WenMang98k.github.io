import { mkdir, cp, readFile, writeFile, rm } from 'node:fs/promises';
await mkdir('public/about', { recursive: true });
await cp('articles', 'public/articles', { recursive: true });
// The overview is rendered by Astro; keep its old source and timeline URL.
await rm('public/articles/speculative-decoding/index.html', { force: true });
for (const file of ['style.css', 'jemdoc.css', 'site.css']) await cp(file, `public/${file}`);
for (const dir of ['tech', 'life']) await cp(dir, `public/${dir}`, { recursive: true });
let html = await readFile('tech/about/index.html', 'utf8');
html = html.replace('href="style.css', 'href="/style.css').replaceAll('href="articles/', 'href="/articles/');
await writeFile('public/about/index.html', html);

await mkdir('public/learning/speculative-decoding', { recursive: true });
await cp('research/speculative-decoding', 'public/learning/speculative-decoding', { recursive: true });
