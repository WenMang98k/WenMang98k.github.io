"""Generate static category pages from content/catalog.json; no third-party packages."""
import json
from html import escape as e
from pathlib import Path
from urllib.parse import urlsplit
ROOT = Path(__file__).resolve().parents[1]
DATA = json.loads((ROOT / 'content/catalog.json').read_text())

def shell(title, description, body, prefix='', active=''):
    nav=''.join(f'<a href="{prefix}{k}/"'+(' aria-current="page"' if k==active else '')+f'>{DATA[k]["label"]}</a>' for k in ['tech','life'])
    return f'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta name="description" content="{e(description,quote=True)}"><meta name="theme-color" content="#f6f5ef"><title>{e(title)} · 文氓 WenMang</title><link rel="stylesheet" href="{prefix}site.css?v=20261111"></head>
<body class="{active or 'home'}"><a class="skip-link" href="#main">跳转到正文</a><header class="site-header container"><a class="wordmark" href="{prefix or './'}">文氓<span> · WenMang</span></a><nav aria-label="主导航">{nav}</nav></header><main id="main" class="container">{body}</main><footer class="site-footer container"><span>© 文氓 · WenMang · 保持好奇，记录日常</span><a href="mailto:WenMang98k@gmail.com">打个招呼 ↗</a></footer></body></html>'''

def write(path, text):
    p=ROOT/path;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(text)

def posts_for(section,topic=None):
    return [p for p in DATA['posts'] if p['section']==section and (topic is None or p['topic']==topic)]

def post_list(posts,prefix):
    if not posts:
        return '<div class="empty-state"><span aria-hidden="true">○</span><h2>这里的记录，慢慢写。</h2><p>这个主题还没有发布文章。新内容会整理在这里。</p></div>'
    out='<div class="post-list">'
    for p in posts:
        out+=f'<a class="post-row" href="{prefix}{e(p["url"].lstrip("/"),quote=True)}"><div><p class="post-meta">{e(p["date"])} <span>／</span> {e(p["kind"])}</p><h3>{e(p["title"])}</h3><p>{e(p["summary"])}</p></div><span class="row-arrow" aria-hidden="true">↗</span></a>'
    return out+'</div>'

# Validate the catalog before generating pages.
for post in DATA['posts']:
    assert post['section'] in ['tech','life']
    assert any(t['slug']==post['topic'] for t in DATA[post['section']]['topics'])
    assert post['url'].startswith('/') and not urlsplit(post['url']).netloc
    file=ROOT/post['url'].lstrip('/')
    assert file.is_file() or (file/'index.html').is_file(),post['url']

home='''<section class="landing-intro"><p class="eyebrow">WENMANG / PERSONAL JOURNAL</p><h1>技术，和生活<span>。</span></h1><p>一边探索模型与系统，一边记录日常与热爱。<br>从这里，走进我的两个世界。</p></section><div class="portal-grid" aria-label="两大内容分类">'''
for key,n,desc,words in [('tech','01','模型、算法与系统的思考，<br>按 topic 整理的技术笔记。','AI Infra / Agent / 模型与推理'),('life','02','运动、镜头与旅途，<br>按兴趣收藏的生活记录。','摄影 / 运动 / 骑行 / 美食')]:
    home+=f'<a class="portal portal-{key}" href="{key}/"><div class="portal-top"><span>{n} / {DATA[key]["english"]}</span><span aria-hidden="true">↗</span></div><div class="portal-art" aria-hidden="true"><span class="art-ring"></span><span class="art-inner">'+('⌘' if key=='tech' else '✳')+f'</span></div><div class="portal-copy"><h2>{DATA[key]["label"]}</h2><p>{desc}</p><span class="portal-tags">{words}</span></div><div class="portal-bottom">进入{DATA[key]["label"]}<span aria-hidden="true">→</span></div></a>'
home+='</div>'
write('index.html',shell('技术与生活','文氓（WenMang）的个人网站：技术笔记与生活记录，两个入口，慢慢探索。',home))

for key in ['tech','life']:
    section=DATA[key];posts=posts_for(key)
    intro=f'<p class="breadcrumb"><a href="../">首页</a> / {section["label"]}</p><section class="page-intro"><p class="eyebrow">{section["english"]}</p><h1>{section["label"]}<span>。</span></h1><p>{section["intro"]}</p>'
    if key=='tech':intro+='<a class="inline-link" href="about/">关于我的技术经历与开源项目 ↗</a>'
    else:intro+='<p class="intro-note">健身、跑步、摄影、骑摩托、羽毛球，也爱麻将、游戏和美食。</p>'
    intro+='</section><section aria-labelledby="topics-title"><div class="list-heading"><h2 id="topics-title">'+('按 topic 阅读' if key=='tech' else '按兴趣阅读')+'</h2><span>'+str(len(section['topics']))+' 个主题</span></div><div class="topic-grid">'
    for t in section['topics']:
        count=len(posts_for(key,t['slug']))
        intro+=f'<a class="topic-card" href="{t["slug"]}/"><div class="topic-card-top"><span aria-hidden="true">{t["symbol"]}</span><small>{str(count)+" 篇文章" if count else "待记录"}</small></div><h3>{t["title"]}</h3><p>{t["description"]}</p><span class="topic-arrow" aria-hidden="true">↗</span></a>'
    intro+='</div></section>'
    if posts:intro+='<section class="recent-posts" aria-labelledby="recent-title"><div class="list-heading"><h2 id="recent-title">最新文章</h2></div>'+post_list(posts,'../')+'</section>'
    write(f'{key}/index.html',shell(section['label'],section['intro'],intro,'../',key))
    for topic in section['topics']:
        body=f'<p class="breadcrumb"><a href="../../">首页</a> / <a href="../">{section["label"]}</a> / {topic["title"]}</p><section class="page-intro"><p class="eyebrow">{section["english"]} / TOPIC</p><h1 class="topic-title">{topic["title"]}<span>。</span></h1><p>{topic["description"]}</p></section>'+post_list(posts_for(key,topic['slug']),'../../')+f'<a class="back-link" href="../">← 返回{section["label"]}</a>'
        write(f'{key}/{topic["slug"]}/index.html',shell(topic['title'],topic['description'],body,'../../',key))
print('Generated home, 2 category indexes and',sum(len(DATA[k]['topics']) for k in ['tech','life']),'topic pages.')
