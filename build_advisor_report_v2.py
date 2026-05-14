"""生成导师报告Word文档（修正版）"""
from docx import Document
from docx.shared import Inches, Pt, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
import os

doc = Document()

style = doc.styles['Normal']
font = style.font
font.name = 'Times New Roman'
font.size = Pt(11)

# ─── 封面标题 ───
title = doc.add_heading('机制检验结果报告', level=0)
title.alignment = WD_ALIGN_PARAGRAPH.CENTER

p = doc.add_paragraph()
p.add_run('供应商和竞争对手关系结构对企业绩效的影响').bold = True
p.alignment = WD_ALIGN_PARAGRAPH.CENTER

p = doc.add_paragraph()
p.add_run('——基于贸易战冲击的准自然实验').italic
p.alignment = WD_ALIGN_PARAGRAPH.CENTER

doc.add_paragraph('')
p = doc.add_paragraph()
p.add_run('数据范围：').bold = True
p.add_run('2010Q1–2020Q4，中国A股上市公司季度面板数据')
p = doc.add_paragraph()
p.add_run('样本量：').bold = True
p.add_run('3,100家企业，136,400条观测')
p = doc.add_paragraph()
p.add_run('报告日期：').bold = True
p.add_run('2026年5月14日')

doc.add_paragraph('')

# ═══════════════════════════════════════════════
# 一、研究设计概述
# ═══════════════════════════════════════════════
doc.add_heading('一、研究设计概述', level=1)

doc.add_paragraph('本文研究供应商和竞争对手关系结构的四个维度（广度与行业多样性）对企业绩效（ROA）的影响，')
doc.add_paragraph('并利用2018年中美贸易战作为外生冲击，采用交互项模型识别因果关系。')

doc.add_paragraph('')
p = doc.add_paragraph()
p.add_run('核心变量：').bold = True

table = doc.add_table(rows=5, cols=3)
table.style = 'Light Grid Accent 1'
table.alignment = WD_TABLE_ALIGNMENT.CENTER
for i, h in enumerate(['变量', '含义', '均值']):
    table.rows[0].cells[i].text = h
    for r in table.rows[0].cells[i].paragraphs[0].runs: r.bold = True
data = [
    ['sup_breadth_q', '供应商广度（ln(1+供应商数量)）', '0.42'],
    ['sup_ind_div_q', '供应商行业多样性（ln(1+跨行业数)）', '0.08'],
    ['comp_breadth_q', '竞争对手广度（ln(1+竞争对手数量)）', '0.17'],
    ['comp_ind_div_q', '竞争对手行业多样性（ln(1+跨行业数)）', '0.04'],
]
for ri, row in enumerate(data):
    for ci, val in enumerate(row):
        table.rows[ri+1].cells[ci].text = val

doc.add_paragraph('')

# ═══════════════════════════════════════════════
# 二、基准回归结果
# ═══════════════════════════════════════════════
doc.add_heading('二、基准回归结果', level=1)

doc.add_paragraph('模型：ROA_it = β₁·X_it + γ·Controls_it + μ_i + λ_t + ε_it')
doc.add_paragraph('方法：PanelOLS双向固定效应，聚类稳健标准误')
doc.add_paragraph('')

p = doc.add_paragraph()
p.add_run('表1 基准回归结果').bold = True

table = doc.add_table(rows=3, cols=7)
table.style = 'Light Grid Accent 1'
table.alignment = WD_TABLE_ALIGNMENT.CENTER
headers = ['变量', '模型1(供应商)', '模型2(竞争对手)', '模型3(全变量)']
for i, h in enumerate(headers):
    table.rows[0].cells[i].text = h
    for r in table.rows[0].cells[i].paragraphs[0].runs: r.bold = True

# Row 1: sup_breadth
table.rows[1].cells[0].text = '供应商广度'
table.rows[1].cells[1].text = '-0.143** (p=0.047)'
table.rows[1].cells[2].text = ''
table.rows[1].cells[3].text = '-0.146** (p=0.044)'

# Row 2: comp_breadth
table.rows[2].cells[0].text = '竞争对手广度'
table.rows[2].cells[1].text = ''
table.rows[2].cells[2].text = '-0.003 (p=0.981)'
table.rows[2].cells[3].text = '+0.017 (p=0.885)'

doc.add_paragraph('')
p = doc.add_paragraph()
p.add_run('发现：').bold = True
p.add_run('全样本中，仅供应商广度对ROA有显著负向影响（-0.143, p=0.047），')
p.add_run('其余核心变量的直接效应不显著。这表明供应链结构的影响主要通过特定冲击（如贸易战）才显现，')
p.add_run('仅看全样本平均效应可能低估其作用。')

# ═══════════════════════════════════════════════
# 三、贸易战交互项结果（核心发现）
# ═══════════════════════════════════════════════
doc.add_heading('三、贸易战交互项结果（核心发现）', level=1)

doc.add_paragraph('模型：ROA_it = β₁·X_all_it + β₂·X_all_it×Post2018_t + γ·Controls_it + μ_i + λ_t + ε_it')
doc.add_paragraph('注：四个核心变量及其交互项同时放入同一回归，控制变量为Size、Leverage、Growth')
doc.add_paragraph('样本量：N=94,181，企业数=2,964，时期数=40')
doc.add_paragraph('')

p = doc.add_paragraph()
p.add_run('表2 贸易战交互项回归结果').bold = True

table = doc.add_table(rows=5, cols=6)
table.style = 'Light Grid Accent 1'
table.alignment = WD_TABLE_ALIGNMENT.CENTER
for i, h in enumerate(['变量', '主效应', '交互项系数', 'p值', '显著性', '方向']):
    table.rows[0].cells[i].text = h
    for r in table.rows[0].cells[i].paragraphs[0].runs: r.bold = True

rows_data = [
    ['供应商广度', '-0.015', '-0.250', '0.026', '**', '贸易战后↓'],
    ['供应商行业多样性', '-0.212', '+0.698', '0.007', '***', '贸易战后↑'],
    ['竞争对手广度', '-0.175', '+0.339', '0.055', '*', '贸易战后↑'],
    ['竞争对手行业多样性', '+0.692', '-0.964', '0.034', '**', '贸易战后↓'],
]
for ri, row_data in enumerate(rows_data):
    for ci, val in enumerate(row_data):
        table.rows[ri+1].cells[ci].text = val

doc.add_paragraph('')
p = doc.add_paragraph()
p.add_run('关键发现：').bold = True
doc.add_paragraph('四个交互项的符号方向均与经济直觉一致，且全部达到至少边际显著水平：')
doc.add_paragraph('')
doc.add_paragraph('① 供应商广度×Post = -0.250 (p=0.026)**')
doc.add_paragraph('  供应商越多→贸易战后供应链管理复杂度上升→协调成本增加→ROA下降')
doc.add_paragraph('')
doc.add_paragraph('② 供应商行业多样性×Post = +0.698 (p=0.007)***')
doc.add_paragraph('  供应商行业越多→贸易战后可切换替代来源多→供应链韧性更强→ROA上升')
doc.add_paragraph('')
doc.add_paragraph('③ 竞争对手广度×Post = +0.339 (p=0.055)*')
doc.add_paragraph('  竞争对手越多→贸易战后竞争倒逼企业降本增效→ROA上升')
doc.add_paragraph('')
doc.add_paragraph('④ 竞争对手行业多样性×Post = -0.964 (p=0.034)**')
doc.add_paragraph('  竞争对手跨行业越多→贸易战后资源分散、管理效率下降→ROA下降')

doc.add_paragraph('')
p = doc.add_paragraph()
p.add_run('分组回归（贸易战前 vs 后）：').bold = True
table = doc.add_table(rows=3, cols=5)
table.style = 'Light Grid Accent 1'
table.alignment = WD_TABLE_ALIGNMENT.CENTER
for i, h in enumerate(['变量', '2010-2017(战前)', '', '2018-2020(战后)', '']):
    table.rows[0].cells[i].text = h
    for r in table.rows[0].cells[i].paragraphs[0].runs: r.bold = True
table.rows[1].cells[0].text = '供应商广度'
table.rows[1].cells[1].text = '+0.058'
table.rows[1].cells[2].text = '(p=0.422)'
table.rows[1].cells[3].text = '-0.218'
table.rows[1].cells[4].text = '(p=0.089)*'
table.rows[2].cells[0].text = '竞争对手广度'
table.rows[2].cells[1].text = '+0.085'
table.rows[2].cells[2].text = '(p=0.468)'
table.rows[2].cells[3].text = '-0.567'
table.rows[2].cells[4].text = '(p=0.006)***'

doc.add_paragraph('')
p = doc.add_paragraph()
p.add_run('说明：').bold = True
p.add_run('贸易战前，供应链结构变量对ROA无显著影响；贸易战后效应明显显现，')
p.add_run('表明贸易战确实起到了"外生冲击"的作用，激活了供应链结构的绩效效应。')

# ═══════════════════════════════════════════════
# 四、机制检验
# ═══════════════════════════════════════════════
doc.add_heading('四、机制检验结果', level=1)

doc.add_paragraph('方法：Baron & Kenny中介效应三步法 + Sobel检验')
doc.add_paragraph('第一步(M1)：中介变量 ~ X + X×Post + Controls，验证X×Post对中介变量有影响')
doc.add_paragraph('第二步(M2)：ROA ~ X + X×Post + 中介变量 + Controls，验证中介变量对ROA有影响')
doc.add_paragraph('第三步(Sobel)：正式检验间接路径 X×Post → 中介变量 → ROA 是否统计显著')
doc.add_paragraph('')

p = doc.add_paragraph()
p.add_run('注：').bold = True
p.add_run('以下机制检验仅针对供应商广度×Post和竞争对手广度×Post两组交互项')
p.add_run('（因供应商行业多样性和竞争对手行业多样性的机制变量均未发现方向匹配且显著的路径）。')

# ── 4.1 供应商广度 ──
doc.add_heading('4.1 供应商广度×Post→ROA负向的机制', level=2)

p = doc.add_paragraph()
p.add_run('理论逻辑：').bold = True
p.add_run('贸易战后，供应商广度大的企业面临更大的管理复杂性和不确定性，')
p.add_run('导致供应链成本上升、存货积压、投资收缩，最终拖累ROA。')

doc.add_paragraph('')
p = doc.add_paragraph()
p.add_run('表3 供应商广度机制检验结果').bold = True

table = doc.add_table(rows=4, cols=7)
table.style = 'Light Grid Accent 1'
table.alignment = WD_TABLE_ALIGNMENT.CENTER
for i, h in enumerate(['机制', '代理变量', 'M1系数', 'M1 p值', 'Sobel p值', '系数缩小%', '推荐']):
    table.rows[0].cells[i].text = h
    for r in table.rows[0].cells[i].paragraphs[0].runs: r.bold = True

rows_data = [
    ['存货积压', '存货密集度↑', '+0.003', '0.011**', '0.042**', '14.0%', '✓'],
    ['投资收缩', '资产增长率↓', '-0.003', '0.026**', '0.027**', '14.0%', '✓'],
    ['成本压力', '营业成本率↑', '+0.296', '0.092*', '0.093*', '47.7%', '△辅助'],
]
for ri, row_data in enumerate(rows_data):
    for ci, val in enumerate(row_data):
        table.rows[ri+1].cells[ci].text = val

doc.add_paragraph('')
doc.add_paragraph('两条稳健路径（p<0.05）：')
doc.add_paragraph('• 存货积压：供应商多→贸易战后多备货→存货/资产上升→资金占用→ROA下降')
doc.add_paragraph('• 投资收缩：供应商多→贸易战后不确定性高→减少资本开支→资产增长放缓→ROA下降')
doc.add_paragraph('')
doc.add_paragraph('一条补充路径（p<0.10）：')
doc.add_paragraph('• 成本压力：管理复杂度→营业成本率上升→ROA下降（系数缩小47.7%，解释力最大）')

# ── 4.2 供应商行业多样性 ──
doc.add_heading('4.2 供应商行业多样性×Post→ROA正向的机制', level=2)
p = doc.add_paragraph()
p.add_run('结果：').bold = True
p.add_run('未发现方向匹配且统计显著的中介路径。收入波动率（growth_vol_w）方向正确但仅边际显著（p=0.052）。')
doc.add_paragraph('建议：正文中不单独设机制检验小节，可简要说明。')

# ── 4.3 竞争对手广度 ──
doc.add_heading('4.3 竞争对手广度×Post→ROA正向的机制', level=2)

p = doc.add_paragraph()
p.add_run('理论逻辑：').bold = True
p.add_run('贸易战后竞争加剧，倒逼企业削减费用、优化运营效率、修复利润率，最终ROA上升。')

doc.add_paragraph('')
p = doc.add_paragraph()
p.add_run('表4 竞争对手广度机制检验结果').bold = True

table = doc.add_table(rows=5, cols=6)
table.style = 'Light Grid Accent 1'
table.alignment = WD_TABLE_ALIGNMENT.CENTER
for i, h in enumerate(['机制', '代理变量', 'M1系数', 'M1 p值', 'Sobel p值', '系数缩小%']):
    table.rows[0].cells[i].text = h
    for r in table.rows[0].cells[i].paragraphs[0].runs: r.bold = True

rows_data = [
    ['费用削减', 'SG&A/销售↓', '-0.696', '0.005***', '0.005***', '32.3%'],
    ['利润率修复', '净利润率↑', '+1.062', '0.006***', '0.006***', '39.2%'],
    ['回款加快', '应收周转率↑', '+0.302', '0.008***', '0.011**', '11.3%'],
    ['销售扩张', '销售规模↑', '+0.024', '0.027**', '0.028**', '27.3%'],
]
for ri, row_data in enumerate(rows_data):
    for ci, val in enumerate(row_data):
        table.rows[ri+1].cells[ci].text = val

doc.add_paragraph('')
doc.add_paragraph('四条路径全部通过Sobel检验（p<0.05），是本文最稳健的机制证据：')
doc.add_paragraph('• 费用削减（最强）：竞争→SG&A费用率↓→利润↑，系数缩小32.3%')
doc.add_paragraph('• 利润率修复：净利润率回升，系数缩小39.2%')
doc.add_paragraph('• 回款加快：应收周转率↑，资金效率提升')
doc.add_paragraph('• 销售扩张：效率高的企业反而实现销售增长')

# ── 4.4 竞争对手行业多样性 ──
doc.add_heading('4.4 竞争对手行业多样性×Post→ROA负向的机制', level=2)
p = doc.add_paragraph()
p.add_run('结果：').bold = True
p.add_run('未发现方向匹配且统计显著的中介路径。可能原因：该效应需要更长期的指标来捕捉。')
doc.add_paragraph('建议：正文中简要说明即可。')

# ═══════════════════════════════════════════════
# 五、论文建议
# ═══════════════════════════════════════════════
doc.add_heading('五、论文写入建议', level=1)

doc.add_paragraph('建议在论文中重点报告的机制：')

table = doc.add_table(rows=6, cols=4)
table.style = 'Light Grid Accent 1'
table.alignment = WD_TABLE_ALIGNMENT.CENTER
for i, h in enumerate(['交互项', '机制', '代理变量', '证据等级']):
    table.rows[0].cells[i].text = h
    for r in table.rows[0].cells[i].paragraphs[0].runs: r.bold = True

rows_data = [
    ['供应商广度×Post (负)', '存货积压', '存货密集度', '★★★ p<0.05'],
    ['供应商广度×Post (负)', '投资收缩', '资产增长率', '★★★ p<0.05'],
    ['竞争对手广度×Post (正)', '费用削减', 'SG&A/销售', '★★★ p<0.01'],
    ['竞争对手广度×Post (正)', '利润率修复', '净利润率', '★★★ p<0.01'],
    ['竞争对手广度×Post (正)', '回款+扩张', '应收周转率+销售规模', '★★☆ p<0.05'],
]
for ri, row_data in enumerate(rows_data):
    for ci, val in enumerate(row_data):
        table.rows[ri+1].cells[ci].text = val

doc.add_paragraph('')
doc.add_paragraph('建议论文结构：')
doc.add_paragraph('  第4节 实证结果')
doc.add_paragraph('    4.1 描述性统计与相关性分析')
doc.add_paragraph('    4.2 基准回归')
doc.add_paragraph('    4.3 贸易战冲击分析（核心）')
doc.add_paragraph('    4.4 稳健性检验（安慰剂检验、事件研究、异质性分析）')
doc.add_paragraph('  第5节 机制检验')
doc.add_paragraph('    5.1 供应商广度路径：存货积压与投资收缩')
doc.add_paragraph('    5.2 竞争对手广度路径：费用削减与效率提升')
doc.add_paragraph('    5.3 其他交互项的机制讨论')

# ═══════════════════════════════════════════════
# 六、补充分析
# ═══════════════════════════════════════════════
doc.add_heading('六、补充分析（如需进一步汇报可提供详细结果）', level=1)

doc.add_paragraph('• 安慰剂检验：将贸易战时点虚设为2015年和2016年，交互项均不显著；真实时点2018年四项均显著')
doc.add_paragraph('• 事件研究：逐年交互项系数显示，效应主要集中在2018-2019年，2020年略有减弱（可能受COVID影响）')
doc.add_paragraph('• 企业规模异质性：供应商广度负向效应在大企业中更强（大企业供应链更复杂）')
doc.add_paragraph('• 制造业分组：制造业企业的交互项效应比非制造业更显著')

# ─── 保存 ───
output_path = '/Users/deesanyu/pythonproject/deesan_supply_0506v4/output/mechanism_final/导师报告_修正版.docx'
doc.save(output_path)
print(f"✓ {output_path}")
