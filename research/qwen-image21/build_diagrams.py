"""Draw original, source-aligned SVG figures; standard library only."""
from pathlib import Path
from html import escape

OUT = Path(__file__).resolve().parents[2] / 'public/images/qwen-image21'
INK, MUTED = '#20352f', '#64766e'
GREEN, BLUE, AMBER = '#19735c', '#366baf', '#a76326'
class Diagram:
    def __init__(self, height, title, desc):
        self.height = height
        self.parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="1120" height="{height}" viewBox="0 0 1120 {height}" role="img" aria-labelledby="title desc">', f'<title id="title">{escape(title)}</title><desc id="desc">{escape(desc)}</desc>', '''<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="context-stroke"/></marker><pattern id="grid" width="24" height="24" patternUnits="userSpaceOnUse"><circle cx="1" cy="1" r=".8" fill="#dae3db"/></pattern></defs>''',f'<rect width="1120" height="{height}" fill="#f7f9f5"/><rect width="1120" height="{height}" fill="url(#grid)"/>']
    def text(self,x,y,s,size=17,color=INK,weight=400,mono=False):
        font = 'ui-monospace, SFMono-Regular, Consolas, monospace' if mono else 'system-ui, -apple-system, Noto Sans CJK SC, Microsoft YaHei, sans-serif'
        self.parts.append(f'<text x="{x}" y="{y}" font-family="{font}" font-size="{size}" font-weight="{weight}" fill="{color}">{escape(s)}</text>')
    def rect(self,x,y,w,h,fill='#ffffff',stroke='#d6e1d8',r=14,dashed=False):
        self.parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" fill="{fill}" stroke="{stroke}" stroke-width="1.4"'+(' stroke-dasharray="6 5"' if dashed else '')+'/>')
    def line(self,pts,color=GREEN,dashed=False,arrow=True):
        p=' '.join(f'{x},{y}' for x,y in pts)
        self.parts.append(f'<polyline points="{p}" fill="none" stroke="{color}" stroke-width="2" stroke-linejoin="round"'+(' stroke-dasharray="6 5"' if dashed else '')+(' marker-end="url(#arrow)"' if arrow else '')+'/>')
    def box(self,x,y,w,h,title,lines=(),color=GREEN,fill='#fff',size=18):
        self.rect(x,y,w,h,fill)
        self.parts.append(f'<rect x="{x}" y="{y+15}" width="3" height="22" rx="1.5" fill="{color}"/>')
        self.text(x+18,y+32,title,size,color,650)
        for i,t in enumerate(lines): self.text(x+18,y+58+i*23,t,15,MUTED)
    def header(self,n,title,sub):
        self.text(36,37,f'WENMANG / MODEL NOTES                                      QWEN-IMAGE-2.1 / {n}',12,MUTED,600,True)
        self.text(36,82,title,31,INK,700)
        self.text(36,114,sub,16,MUTED)
    def save(self,name):
        OUT.mkdir(parents=True,exist_ok=True)
        (OUT/name).write_text('\n'.join(self.parts+['</svg>']),encoding='utf8')

def backbone():
    d=Diagram(1180,'Qwen-Image-2.1 Backbone 与单层 DiT 结构','左侧为联合条件与目标 latent 构成单流序列，右侧展开 32 个 Transformer block 中的一层；蓝色表示目标，绿色表示条件，虚线表示调制或缓存。')
    d.header('01','Backbone：一个联合序列，32 层单流 DiT','从输入投影到单层算子展开 · 对齐 Diffusers QwenImage21Transformer2DModel')
    d.rect(36,143,1048,61,'#edf4ed')
    d.text(55,168,'7B 视觉生成组件',18,GREEN,650)
    d.text(290,168,'32 × Single-Stream Block',18,INK,650)
    d.text(680,168,'width 4096 · heads 32 × 128',17,BLUE,600,True)
    d.text(55,191,'结构参数采用已固定源码的构造默认值；不等同于本轮直接读取了 checkpoint config。',14,MUTED)
    d.text(36,237,'A / 构造联合序列',15,GREEN,700)
    d.text(574,237,'B / 展开一个 Transformer Block',15,GREEN,700)
    d.box(36,255,222,105,'文本 + 参考图',('Qwen3-VL 8B 联合编码','输出 encoder_hidden_states'))
    d.box(282,255,224,105,'参考图 / 目标噪声',('VAE → 条件 latent','xₜ → 目标 latent'),BLUE)
    d.line([(147,360),(147,387)])
    d.line([(394,360),(394,387)],BLUE)
    d.box(36,387,222,103,'txt_in',('RMSNorm → Linear','GELU → Linear'),size=19)
    d.box(282,387,224,103,'img_in',('Linear：64 → 4096','latent patch_size = 1'),BLUE,size=19)
    d.line([(147,490),(147,518)])
    d.line([(394,490),(394,518)],BLUE)
    d.rect(36,518,470,190,'#edf4ed')
    d.text(54,549,'图像占位展开并替换，再附加目标块',18,INK,650)
    for x,w,t,fill,col in [(54,70,'文本','#e0eee4',GREEN),(130,100,'参考图 1','#e0eee4',GREEN),(236,56,'文本','#e0eee4',GREEN),(298,77,'参考图 N','#e0eee4',GREEN),(381,107,'目标 xₜ','#e5edf9',BLUE)]:
        d.rect(x,567,w,39,fill,fill,7); d.text(x+9,592,t,13,col,600)
    d.text(54,634,'固定条件前缀 c',15,GREEN)
    d.text(359,634,'目标 token',15,BLUE)
    d.text(54,664,'文本与图像交错；不同参考图保持独立 block。',15,MUTED)
    d.text(54,689,'RoPE：frame / height / width 三轴位置编码',15,MUTED)
    d.line([(271,708),(271,752)])
    d.box(36,752,470,104,'Single-Stream DiT × 32',('同一套 block 处理联合序列','条件 t = 0；目标 t = 当前去噪时间'),GREEN,'#e5efdf',23)
    d.line([(506,803),(539,803),(539,299),(574,299)],GREEN,True)
    d.text(36,901,'输出头',15,GREEN,700)
    d.box(36,920,470,97,'norm_out → proj_out',('自适应 LayerNorm → Linear：4096 → 64','Pipeline 仅取末尾目标 token 的预测'),BLUE)
    d.line([(271,856),(271,920)],BLUE)
    # expanded block: skip routes stay in a dedicated gutter
    d.rect(574,255,510,762,'#ffffff')
    d.text(598,288,'h：联合序列 / cached 时仅目标',17,INK,650)
    d.line([(825,300),(825,326)])
    d.box(610,326,430,77,'LayerNorm → scale modulation',('scale 来自共享 modulation；按 token 选 t / 0',),AMBER,size=17)
    d.line([(825,403),(825,427)])
    d.box(610,427,430,102,'Block-Causal Attention',('Q / K RMSNorm → 3-axis RoPE → Attention','输出投影；条件前缀 K/V 在此写入或读取'),GREEN,size=19)
    d.line([(825,529),(825,555)])
    d.box(610,555,430,61,'h + tanh(gate₁) × Attention(h)',(),BLUE,size=17)
    d.line([(825,312),(1061,312),(1061,585),(1040,585)],BLUE)
    d.line([(825,616),(825,642)])
    d.box(610,642,430,77,'LayerNorm → scale modulation',('第二组 scale / gate；仍由同一共享投影产生',),AMBER,size=17)
    d.line([(825,719),(825,743)])
    d.box(610,743,430,80,'SwiGLU Feed-Forward',('gate: SiLU(Wg h) ⊙ Wp h → Wo；ratio = 3',),GREEN,size=19)
    d.line([(825,823),(825,847)])
    d.box(610,847,430,61,'h + tanh(gate₂) × SwiGLU(h)',(),BLUE,size=17)
    d.line([(825,629),(590,629),(590,877),(610,877)],BLUE)
    d.line([(825,908),(825,935)])
    d.text(704,963,'送往下一层 / 最终输出头',17,INK,650)
    d.text(598,997,'图中省略 batch、padding 与张量并行切分。',14,MUTED)
    d.rect(36,1041,1048,103,'#f7efdf','#e5d8bc')
    d.text(55,1072,'共享时间调制 ≠ 每层各自计算一套调制参数',18,AMBER,650)
    d.text(55,1100,'time_text_embed(t 与 0) → SiLU + Linear → 两组 scale / gate → 供所有 32 层读取。',16,INK)
    d.text(55,1125,'缓存前缀每层保存 post-RoPE K/V；正在变化的目标 latent 与目标 K/V 仍需逐步计算。',15,MUTED)
    d.save('architecture.svg')

def inference():
    d=Diagram(1250,'Qwen-Image-2.1 推理调用链与条件 KV 生命周期','上方为请求预处理与时间网格；中间对比首步 extract 与后续 cached；下方为 scheduler 更新、循环与 VAE 输出。')
    d.header('02','Inference：首步写缓存，后续只重算目标','按 QwenImage21Pipeline.__call__ 的执行顺序绘制 · 默认展示无 true CFG 的路径')
    d.box(36,148,318,111,'01 / 请求与条件编码',('prompt + 最多 10 张参考图','encode_prompt：联合语义表示','VAE encode：参考图 latent'),GREEN)
    d.box(392,148,300,111,'02 / 目标初始化',('prepare_latents：随机 xₜ','seed / 输出尺寸 / 参考图顺序','图像布局、padding 与位置元数据'),BLUE)
    d.box(730,148,354,111,'03 / 设置时间网格',('显式 sigmas > sample_sigmas','否则按 num_inference_steps 构造','retrieve_timesteps + dynamic shifting'),AMBER)
    d.line([(195,259),(195,299),(555,299)])
    d.line([(542,259),(542,299),(555,299)],BLUE,arrow=False)
    d.line([(907,259),(907,299),(555,299)],AMBER,arrow=False)
    d.line([(555,299),(555,337)])
    d.rect(36,337,1048,561,'#f0f5ee','#b8cebd',20)
    d.text(56,370,'DENOISING LOOP / i, t in timesteps',19,GREEN,700,True)
    d.text(56,397,'同一个 Transformer，按步数切换计算范围；缓存开启且 causal_condition = true。',16,MUTED)
    d.box(64,427,470,83,'首步 i = 0 · extract',('联合序列：条件前缀 c + 目标 xₜ','计算全部 token；使用 block-causal 注意力'),GREEN,'#fff',21)
    d.box(586,427,470,83,'后续 i > 0 · cached',('block 前切掉条件前缀，只保留目标查询','读取前缀 K/V；重算目标 Q/K/V 与 MLP'),BLUE,'#fff',21)
    d.line([(299,510),(299,539)])
    d.line([(821,510),(821,539)],BLUE)
    d.box(64,539,470,89,'32 × Transformer Block',('条件用 t = 0；目标用当前 t','每层写出 prefix K/V（post-RoPE）'),GREEN)
    d.box(586,539,470,89,'32 × Transformer Block',('Attention：[cached K/V | target K/V]','目标读取全部合法前缀 + 自己的图像块'),BLUE)
    d.line([(299,628),(299,659)],GREEN,True)
    d.rect(64,659,470,75,'#dfeee4','#b8cebd')
    d.text(83,689,'每层条件缓存 Kᶜ / Vᶜ',19,GREEN,650)
    d.text(83,713,'固定 prompt、参考图、权重、LoRA 与位置协议',15,MUTED)
    d.line([(534,697),(556,697),(556,584),(586,584)],GREEN,True)
    d.text(573,656,'只读复用',14,GREEN)
    d.line([(821,628),(821,757)],BLUE)
    d.line([(299,628),(299,641),(50,641),(50,757),(299,757)],BLUE)
    d.line([(299,757),(821,757)],BLUE,arrow=False)
    d.line([(560,757),(560,782)],BLUE)
    d.box(230,782,662,87,'取目标预测 → scheduler.step(noise_pred, t, latents)',('截取 noise_pred[:, -latents.size(1):] → 更新 xₜ → x_next','只循环目标 latent；条件前缀保持固定'),BLUE,size=18)
    # loop return in reserved outer margin, avoiding content
    d.line([(892,826),(1103,826),(1103,403),(821,403),(821,427)],BLUE,True)
    d.text(929,796,'下一步',15,BLUE,650)
    d.line([(560,898),(560,934)],BLUE)
    d.box(230,934,662,99,'04 / 完成去噪 → VAE decode → postprocess',('unpack → 恢复 latent mean / std → VAE 解码','RGB / RGBA → alpha、文字与主体质量检查 → 返回结果'),GREEN,size=19)
    d.rect(36,1061,1048,150,'#f7efdf','#e5d8bc')
    d.text(55,1094,'三条执行边界',19,AMBER,700)
    d.text(55,1126,'① Base 常用 40 步；官方 Turbo 为 8 步配方。实际循环次数以时间网格为准。',16,INK)
    d.text(55,1156,'② true CFG 启用时另跑 negative 分支，使用独立 neg_cache，再合并预测。',16,INK)
    d.text(55,1186,'③ 关闭缓存时每步重算联合序列；换条件 / 适配器后必须重新建立相应缓存。',16,INK)
    d.save('inference.svg')

if __name__ == '__main__':
    backbone()
    inference()
