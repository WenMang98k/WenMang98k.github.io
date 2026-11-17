"""Original source-aligned comparison figure, no model/runtime dependencies."""
from build_diagrams import Diagram, GREEN, BLUE, AMBER, INK, MUTED


def main():
    d = Diagram(1480, 'Edit-2511 与 2.1：Pipeline 和 Backbone 对比',
                '两列分别展示旧版双流推理和2.1单流块因果推理；明确首步缓存、后续目标计算及相同分辨率token数量。')
    d.text(36,38,'WENMANG / SOURCE NOTES · Diffusers f9316b0',13,MUTED,600)
    d.text(36,80,'Edit-2511 → 2.1：推理发生了什么变化',29,INK,700)
    d.text(36,113,'同版本源码对照 · 结构与依赖示意 · 未运行 GPU 基准',16,MUTED)
    d.rect(36,137,510,48,'#e8eef8'); d.rect(574,137,510,48,'#e6f0e5')
    d.text(54,168,'Qwen-Image-Edit-2511',23,BLUE,700)
    d.text(592,168,'Qwen-Image-2.1',23,GREEN,700)
    for x in [36,574]: d.text(x,215,'A / Pipeline：编码 → 去噪循环 → 解码',16,INK,700)
    left=[
        ('指令 + 参考图：双路径编码',('Qwen2.5-VL → 多模态 context embedding','旧版 VAE → 参考 latent；单独尺寸预处理')),
        ('每步输入：两条流',('context：VLM embedding','image：[当前目标 xₜ | 固定参考 latent]')),
        ('每步完整运行双流 Backbone',('文本与图像高层状态重新交互','取图像输出开头的目标预测')),
        ('scheduler 更新 → 下一步',('只更新目标 latent，参考 latent 不被加噪更新','最终 unpack → 旧版 VAE → RGB')),
    ]
    right=[
        ('指令 + 参考图：双路径编码',('Qwen3-VL → 联合条件表示','RGBA VAE → 参考 latent')),
        ('构造单流：[交错条件 c | 目标 xₜ]',('参考 latent 替换图像槽，再附加目标','不能照搬旧版的 token 顺序与切片位置')),
        ('首步 extract / 后续 cached',('首步：全序列，写入各层条件 K/V','后续：裁掉前缀 query，只更新目标状态')),
        ('scheduler 更新 → 下一步',('取末尾目标预测；正负 CFG 缓存分开','最终 flatten 还原 → RGBA VAE → RGBA')),
    ]
    for x, items, color in [(36,left,BLUE),(574,right,GREEN)]:
        for i,(title,lines) in enumerate(items):
            y=236+i*117;d.box(x,y,510,96,title,lines,color,size=19)
            if i<3:d.line([(x+255,y+96),(x+255,y+117)],color)
    d.rect(36,711,1048,49,'#fbf0df')
    d.text(54,742,'共同点：参考图参与 VLM 与 VAE；去噪更新的是目标，不是参考。',17,AMBER,600)
    for x in [36,574]: d.text(x,796,'B / Backbone：一个 block 内部',16,INK,700)
    d.box(36,816,510,109,'双流：分别投影与调制',('图像：img Q/K/V + img_mod','文本：context Q/K/V + txt_mod','两套流在 attention 中交互'),BLUE,size=19)
    d.box(574,816,510,109,'单流：联合 token，共享 block',('同一套 Q/K/V；QK norm 与 RoPE','共享 modulation：条件 t=0，目标当前 t','同一套 SwiGLU FFN'),GREEN,size=19)
    d.line([(291,925),(291,951)],BLUE);d.line([(829,925),(829,951)],GREEN)
    d.box(36,951,510,118,'双向 joint attention',('[文本 | 目标 | 参考] 的 Q/K/V 联合计算','padding mask 不隔离条件与目标','再拆回两流，各做残差与 MLP'),BLUE,size=19)
    d.box(574,951,510,118,'块因果 attention',('条件 ✕→ 目标；目标 → 全部有效条件','每张图内部双向；文本保持因果','cached 仅保留目标 query，可省块因果 mask'),GREEN,size=19)
    d.line([(291,1069),(291,1095)],BLUE);d.line([(829,1069),(829,1095)],GREEN)
    d.box(36,1095,510,96,'固定输入 ≠ 固定逐层 K/V',('目标变化会传入条件高层状态','参考 t=0 分支也不足以隔离目标'),BLUE,size=20)
    d.box(574,1095,510,96,'隔离依赖 + 固定条件调制 → 可缓存',('每层保存 prefix K/V；后续与目标 K/V 拼接','目标 FFN、Q/K/V 与输出仍需每步计算'),GREEN,size=18)
    d.rect(36,1217,1048,144,'#edf1f3')
    d.text(54,1248,'C / 1024 × 1024 算例：两边目标 token 都可能是 4096',20,INK,650)
    d.text(54,1280,'2511 标准路径：8× VAE · 16 通道 → 2×2 打包 → 64×64 token × 64 维',17,BLUE)
    d.text(54,1311,'2.1 路径：      16× VAE · 64 通道 → 直接展平 → 64×64 token × 64 维',17,GREEN)
    d.text(54,1341,'相同 token 数 ≠ 相同表示；压缩率、层数或参数量的比值都不能直接当作端到端加速。',15,MUTED)
    d.text(36,1394,'结构默认：旧版 60 层 / 24×128；2.1 为 32 层 / 32×128；未读取 2511 checkpoint config。',15,MUTED)
    d.text(36,1423,'✕→ 表示禁止读取；cached 去掉的是条件 query，不是目标对条件 K/V 的访问。',15,MUTED)
    d.text(36,1452,'来源、代码映射与限制见正文；本图为本站原创，可由 build_comparison.py 重绘。',14,MUTED)
    d.save('2511-vs-21.svg')


if __name__ == '__main__':
    main()
