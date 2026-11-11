# 投机解码源码学习资料

- `exact_sampling.py`：本站原创的有限词表、单位置精确投机采样参考实现，不是 GPU 推理框架。
- `sources.lock.json`：已读取文件的原仓库、路径、上游 commit（如已锁定）与 SHA256；失败路径保留为 unavailable，不作为代码入口。
- 网站入口：[原始仓库与源码阅读索引](https://wenmang98k.github.io/articles/sd-source-map/)。

没有把上游整仓复制进个人网站。学习文章链接原始代码并明确作者仓库、框架移植与第三方复现；本站伪代码和参考实现为独立编写。若后续确需 vendor 上游文件，应先核对对应 revision 的许可证，保留版权、许可证与修改说明，并避免引入模型权重、数据或大型二进制。

执行参考校验：

```bash
python3 research/speculative-decoding/exact_sampling.py
python3 articles/speculative-decoding/scripts/verify_probabilities.py
```

这些检查验证概率计算和简单参考代码，不能代替整套模型实现、分布统计审计或 GPU 性能评测。
