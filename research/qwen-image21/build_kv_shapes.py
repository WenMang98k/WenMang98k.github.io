"""Generate the concrete shape dataflow, using checked shape arithmetic."""
from build_diagrams import Diagram, GREEN, BLUE, AMBER, INK, MUTED
from shape_example import example


def main():
    s=example()
    assert s['prefix_tokens']==6248 and s['expanded_joint'][1]==10344
    d=Diagram(1700,'Qwen-Image-2.1 KV 数据流：带 shape 的完整实例','参考图宽1024高1204缩放至宽960高1120；有效非图像token2048，条件前缀6248，目标4096，首步联合10344；每层KV及最终输出形状。')
    d.text(36,38,'WENMANG / QWEN-IMAGE-2.1 · TENSOR SHAPES',13,MUTED,600)
    d.text(36,80,'把实际尺寸放进 KV 数据流',31,INK,700)
    d.text(36,113,'B=1 · 无 TP · 32 头 × 128 · 宽度 4096 · 单分支 BF16 · 图像尺寸为 W×H',15,MUTED)
    d.rect(36,135,1048,55,'#fbf0df')
    d.text(54,158,'2048 = 进入 DiT 的有效非图像 token，不含图像槽；非原始 prompt 长度保证。',17,AMBER,650)
    d.text(54,179,'模板 / 特殊 token 若另有 δ，条件与联合序列长度要同步加 δ；processor 必须匹配。',14,MUTED)
    d.text(36,224,'A / 输入准备：参考图会缩放；目标图固定为 1024×1024',18,INK,700)
    d.box(36,245,510,145,'参考图：原始 W1024 × H1204',('默认缩放 → W960 × H1120','VAE 像素 [1,4,1,1120,960]','VAE latent [1,64,1,70,60]','展平参考 [1,4200,64]'),GREEN,size=19)
    d.box(574,245,510,145,'目标：W1024 × H1024',('目标 latent 空间 H64 × W64','展平目标 x [1,4096,64]','目标占位槽 = 4096 / 4 = 1024','img_in 后目标 [1,4096,4096]'),BLUE,size=20)
    d.line([(829,390),(829,414)],BLUE)
    d.box(36,414,510,124,'参考图另一路 → VLM 条件',('2048 非图像 token + 1050 图像槽','prompt_embeds [1,3098,4096]','1050 槽 × 4 → 4200 参考 latent token'),GREEN,size=19)
    d.box(574,414,510,124,'追加目标槽，再展开、替换',('img_mask [1,4122] = 3098 + 1024','图像输入 [1,8296,64] → [1,8296,4096]','参考与目标占位均展开；文本长度不变'),BLUE,size=19)
    d.rect(36,562,1048,88,'#e6f0e5')
    d.text(54,591,'条件 P = 2048 + 4200 = 6248；目标 T = 4096；联合 N = 10344',21,GREEN,650)
    d.text(54,622,'联合隐藏状态 [1,10344,4096] = [条件前缀 | 目标]；图像槽被替换，不重复追加参考。',16,MUTED)
    d.text(36,687,'B / 首步 extract：每一层都做同一形状的计算，缓存数值各不相同',18,INK,700)
    d.box(36,708,691,145,'第 ℓ 层：联合序列全部参与计算',('新算 Q / K / V：各 [1,10344,32,128]','K 在 QK norm + RoPE 后切片；V 不做 RoPE','attention 合头 → [1,10344,4096]','FFN 后 → 下一层；共 L=32 层'),BLUE,size=21)
    d.box(797,708,287,145,'cache[ℓ]',('K [1,6248,32,128]','V [1,6248,32,128]','从前 6248 个 token 切出','不缓存目标、不缓存 Q'),GREEN,'#e6f0e5',20)
    d.line([(727,778),(797,778)],GREEN);d.text(744,763,'写',14,GREEN,650)
    d.box(36,877,1048,80,'输出头：全序列 [1,10344,64] → 取末尾目标 [1,4096,64]',('scheduler 更新目标 x，shape 仍为 [1,4096,64]；逐层条件缓存原样保留。',),BLUE,size=19)
    d.line([(381,853),(381,877)],BLUE)
    d.text(36,995,'C / 每个后续 cached 步：目标重新从第 1 层进入；按层号读取缓存',18,INK,700)
    d.box(36,1016,691,145,'第 ℓ 层：只新算目标，仍读取完整条件',('目标隐藏状态 [1,4096,4096]','新算 Q_x / K_x / V_x：各 [1,4096,32,128]','拼接 K/V：[缓存条件6248 | 本步目标4096]','K_all / V_all：各 [1,10344,32,128]'),BLUE,size=20)
    d.box(797,1016,287,145,'只读 cache[ℓ]',('K [1,6248,32,128]','V [1,6248,32,128]','每个后续步长度不变','目标 K/V 不追加进缓存'),GREEN,'#e6f0e5',19)
    d.line([(797,1086),(727,1086)],GREEN,True);d.text(744,1071,'读',14,GREEN,650)
    d.box(36,1185,1048,118,'Q_x 查询 K_all / V_all → 仅目标输出',('逻辑 score [1,32,4096,10344]；高效内核不必物化完整矩阵','attention 输出 [1,4096,32,128] → 合头 / FFN [1,4096,4096]','各层后 → 目标预测 [1,4096,64] → scheduler；下一步仍只重算目标'),BLUE,size=20)
    d.line([(381,1161),(381,1185)],BLUE)
    d.text(36,1329,'D / 结束去噪后：目标 latent 还原，才做最终 VAE 解码',18,INK,700)
    d.box(36,1350,1048,104,'[1,4096,64] → [1,64,1,64,64] → VAE → [1,4,1,1024,1024]',('去掉单帧轴 → [1,4,1024,1024]；最终 PIL：1024×1024，RGBA','VAE 不读取 DiT 的条件 KV；C=64 是 latent 通道，C=4 是 RGBA 通道。'),GREEN,size=18)
    d.rect(36,1482,1048,142,'#edf1f3')
    d.text(54,1515,'缓存纯数据：每层 K+V 约 97.625 MiB；32 层约 3.051 GiB。',21,INK,650)
    d.text(54,1548,'不含权重、临时拼接、激活和工作区；true CFG 负分支长度要独立计算。',17,MUTED)
    d.text(54,1578,'Q/K/V 布局 [B,tokens,heads,head_dim]；内核可能换轴。无 padding 时 cached mask 可为 None。',15,MUTED)
    d.text(54,1605,'参考槽数需核对实际 image_pad_mask；这是源码协议下的 shape 算例，不是 GPU 实测。',15,MUTED)
    d.text(36,1664,'来源：Diffusers f9316b0 · shape_example.py 复核算术 · build_kv_shapes.py 重绘',14,MUTED)
    d.save('kv-shapes.svg')


if __name__=='__main__':
    main()
