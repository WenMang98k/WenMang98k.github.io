import { getCollection } from 'astro:content';
export const legacyArticle = {
  title: 'LLM 投机解码综述',
  description: '从经典 SD 到 EAGLE、MTP、DFlash 与 DSpark：机制、概率校正与工程选型，关联 35 篇方法文章和固定版本源码。学习初稿，未进行 GPU 性能对比实验。',
  date: new Date('2026-11-11T00:00:00+08:00'), tags: ['LLM', '推理加速'], href: '/articles/speculative-decoding/',
};
export async function articles() {
  const posts = await getCollection('posts', ({ data }) => !data.draft && data.seriesRole !== 'overview');
  return [legacyArticle, {
    title: '投机解码技术时间轴', description: '从早期块预测到 EAGLE、MTP、DFlash 与 DSpark，区分论文提出、代码发布和框架支持。',
    date: new Date('2026-11-11T00:00:00+08:00'), tags: ['LLM', '推理加速'], href: '/articles/speculative-decoding/timeline.html'
  }, ...posts.map(p => ({ ...p.data, href: `/articles/${p.id}/` }))]
    .sort((a,b) => b.date.getTime() - a.date.getTime());
}
export const projects = [
  { name: 'BERT 剪枝与知识蒸馏', repo: 'nni_compress', category: '模型压缩', stack: 'PyTorch / Transformers / NNI', description: '面向 GLUE / MNLI 任务，探索注意力模块剪枝、结构精简与层级知识蒸馏。', mark: 'BERT', status: '实验代码' },
  { name: '面向 FPGA 的 YOLOv3 压缩', repo: 'compact_for_FPGA', category: '软硬件协同', stack: 'PyTorch / YOLOv3 / Pruning', description: '结合块通道剪枝与层剪枝，让网络结构匹配 FPGA 输出并行维度。', mark: 'YOLO', status: '研究实践' },
  { name: 'INT8 卷积硬件加速', repo: 'conv_int8_v2', category: '硬件加速', stack: 'C++ / HLS / INT8 / AXI', description: '以特征图与通道分块组织卷积，探索数据重排、流水线和硬件并行实现。', mark: 'INT8', status: '源码与综合记录' },
  { name: 'ThiNet 模型压缩', repo: 'thinet-pytorch', category: '模型压缩', stack: 'PyTorch / Channel Pruning', description: 'ThiNet 通道剪枝的 PyTorch 实践，包含分割模型与训练相关代码。', mark: 'ThiNet', status: '实验代码' },
  { name: '浮点分块卷积', repo: 'conv_fp_git', category: '硬件加速', stack: 'C++ / HLS / FPGA', description: '通过片上缓冲与多路数据流组织输入、权重和输出，探索数据复用。', mark: 'FP32', status: '研究实践' },
  { name: '中文 CLIP 实践', repo: 'cn_clip', category: '多模态', stack: 'PyTorch / Chinese-CLIP', description: '中文图文表征模型相关代码与运行脚本。', mark: 'CLIP', status: '学习与实验' },
];
export const formatDate = (d: Date) => d.toLocaleDateString('zh-CN', { timeZone: 'Asia/Shanghai', year: 'numeric', month: '2-digit', day: '2-digit' });
