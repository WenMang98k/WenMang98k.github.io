import rss from '@astrojs/rss';
import { articles } from '../data/site';
export async function GET() {
  return rss({ title: '文氓 · WenMang｜技术与实践', description: 'AI Infra、模型压缩与推理加速的实践记录', site: 'https://wenmang98k.github.io', items: (await articles()).map(p=>({title:p.title, description:p.description, pubDate:p.date, link:p.href})), customData: '<language>zh-cn</language>' });
}
