import { writeFile } from 'node:fs/promises';
const [slug, title = slug] = process.argv.slice(2);
if (!slug || !/^[a-z0-9]+(?:-[a-z0-9]+)*$/.test(slug) || slug === 'speculative-decoding') {
  console.error('用法：npm run new:post -- my-post "文章标题"（slug 使用小写英文、数字和连字符；不能占用已有文章地址）'); process.exit(1);
}
const date = new Intl.DateTimeFormat('en-CA', { timeZone: 'Asia/Shanghai', year: 'numeric', month: '2-digit', day: '2-digit' }).format(new Date());
await writeFile(`src/content/posts/${slug}.md`, `---\ntitle: ${JSON.stringify(title)}\ndescription: "填写文章摘要"\ndate: ${date}\ntags: ["工程实践"]\ndraft: true\n---\n\n## 问题与背景\n\n## 实现思路\n\n## 验证与结论\n`, { flag: 'wx' });
console.log(`已创建草稿 src/content/posts/${slug}.md，发布前将 draft 改为 false。`);
