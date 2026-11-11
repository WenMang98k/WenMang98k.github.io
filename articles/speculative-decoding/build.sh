#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
command -v pandoc >/dev/null || { echo 'Install Pandoc before rebuilding the article.' >&2; exit 1; }
pandoc README.md -f markdown -t html5 -s --toc --toc-depth=2 --mathml \
  --metadata title='LLM 投机解码综述｜WenMang' --metadata lang=zh-CN \
  -H reading-style.html -B article-navigation.html -o index.html

pandoc timeline.md -f markdown -t html5 -s --toc --toc-depth=2 --mathml \
  --metadata title='投机解码技术时间轴｜WenMang' --metadata lang=zh-CN \
  -H reading-style.html -B article-navigation.html -o timeline.html
