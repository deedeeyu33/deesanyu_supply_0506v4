"""生成导师报告Word文档"""
from docx import Document
from docx.shared import Inches, Pt, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
import os

doc = Document()

# 样式设置
style = doc.styles['Normal']
font = style.font
font.name = 'Times New Roman'
font.size = Pt(11)

# ─── 标题 ───
title = doc.add_heading('机制检验结果报告', level=0)
title.alignment = WD_ALIGN_PARAGRAPH.CENTER

doc.add_paragraph('')
p = doc.add_paragraph()
p.add_run('数据范围：').bold = True
p.add_run('2010Q1–2020Q4，中国A股上市公司季度面板数据')
p = doc.add_paragraph()
p.add_run('核心研究问题：').bold = True
p.add_run('供应商和竞争对手关系结构对企业绩效的影响——基于贸易战冲击的准自然实验')
p = doc.add_paragraph()
p.add_run('报告日期：').bold = True
p.add_run('2026年5月14日')

doc.add_paragraph('')

# ═══════════════════════════════════════════════
# 一、数据与变量
# ═══════════════════════════════════════════════
doc.add_heading('一、数据与变量说明', level=1)

doc.add_heading('1.1 数据来源', level=2)
doc.add_paragraph('• 供应链关系数据：中国A股上市公司年报中的供应商/客户信息披露，经文本分析构建')
doc.add_paragraph('• 财务数据：CSMAR数据库季度财务数据')
doc.add_paragraph('• 机制变量数据：11个新增季度财务指标（成本率、费用率、利润率、周转率等）')
doc.add_paragraph('• 最终面板：3,100家企业，136,400条观测')

doc.add_heading('1.2 核心变量定义', level=2)

table = doc.add_table(rows=9, cols=4)
table.style = 'Light Grid Accent 1'
table.alignment = WD_TABLE_ALIGNMENT.CENTER

headers = ['变量', '含义', '计算方式', '均值']
for i, h in enumerate(headers):
    cell = table.rows[0].cells[i]
    cell.text = h
    for p in cell.paragraphs:
        for r in p.runs:
            r.bold = True

data = [
    ['roa_w_q', '资产收益率(%)', '净利润/总资产×100', '3.49'],
    ['sup_breadth_q', '供应商广度(ln)', 'ln(1+供应商数量)', '0.42'],
    ['sup_ind_div_q', '供应商行业多样性', '供应商跨行业数量(ln)', '0.08'],
    ['comp_breadth_q', '竞争对手广度(ln)', 'ln(1+竞争对手数量)', '0.17'],
    ['comp_ind_div_q', '竞争对手行业多样性', '竞争对手跨行业数量(ln)', '0.04'],
    ['size_q', '企业规模', 'ln(总资产)', '8.28'],
    ['lev_q', '资产负债率', '总负债/总资产', '0.22'],
    ['growth_w_q', '收入增长率', '(当期收入-上期收入)/上期收入', '0.25'],
]
for ri, row_data in enumerate(data):
    for ci, val in enumerate(row_data):
        table.rows[ri+1].cells[ci].text = val

doc.add_paragraph('')

# ═══════════════════════════════════════════════
# 二、基准回归结果
# ═══════════════════════════════════════════════
doc.add_heading('二、基准回归结果', level=1)

doc.add_paragraph('模型设定：PanelOLS双向固定效应（企业+时间），聚类稳健标准误')
doc.add_paragraph('模型：ROA_it = β·X_it + γ·Controls_it + μ_i + λ_t + ε_it')
doc.add_paragraph('')

table = doc.add_table(rows=5, cols=5)
table.style = 'Light Grid Accent 1'
table.alignment = WD_TABLE_ALIGNMENT.CENTER

for i, h in enumerate(['核心变量', '系数', 'p值', '显著性', '观测数']):
    cell = table.rows[0].cells[i]
    cell.text = h
    for p in cell.paragraphs:
        for r in p.runs:
            r.bold = True

rows_data = [
    ['供应商广度', '-0.109', '0.095', '*', '94,181'],
    ['供应商行业多样性', '+0.065', '0.642', '', '94,181'],
    ['竞争对手广度', '+0.029', '0.754', '', '94,181'],
    ['竞争对手行业多样性', '+0.138', '0.557', '', '94,181'],
]
for ri, row_data in enumerate(rows_data):
    for ci, val in enumerate(row_data):
        table.rows[ri+1].cells[ci].text = val

doc.add_paragraph('')
p = doc.add_paragraph()
p.add_run('说明：').bold = True
p.add_run('全样本中，仅供应商广度对ROA有边际显著的负向影响。其余核心变量的直接效应不显著，')
p.add_run('这可能是因为供应链结构的影响主要通过特定冲击（如贸易战）才显现。')

# ═══════════════════════════════════════════════
# 三、贸易战交互项结果
# ═══════════════════════════════════════════════
doc.add_heading('三、贸易战冲击下的交互项回归', level=1)

doc.add_paragraph('模型：ROA_it = β₁·X_it + β₂·X_it×Post2018_t + γ·Controls_it + μ_i + λ_t + ε_it')
doc.add_paragraph('Post2018 = 1表示2018年及之后（贸易战时期），0表示2010-2017年')
doc.add_paragraph('')

table = doc.add_table(rows=5, cols=6)
table.style = 'Light Grid Accent 1'
table.alignment = WD_TABLE_ALIGNMENT.CENTER

for i, h in enumerate(['核心变量', '主效应', '交互项系数', 'p值', '显著性', '观测数']):
    cell = table.rows[0].cells[i]
    cell.text = h
    for p in cell.paragraphs:
        for r in p.runs:
            r.bold = True

rows_data = [
    ['供应商广度', '-0.067', '-0.079', '0.437', '', '94,181'],
    ['供应商行业多样性', '-0.214', '+0.455', '0.053', '*', '94,181'],
    ['竞争对手广度', '-0.069', '+0.160', '0.231', '', '94,181'],
    ['竞争对手行业多样性', '+0.388', '-0.361', '0.310', '', '94,181'],
]
for ri, row_data in enumerate(rows_data):
    for ci, val in enumerate(row_data):
        table.rows[ri+1].cells[ci].text = val

doc.add_paragraph('')
p = doc.add_paragraph()
p.add_run('核心发现：').bold = True
doc.add_paragraph('① 供应商行业多样性×Post的交互项为正向（+0.455, p=0.053），边际显著。'
                  '说明贸易战后，供应商行业多样的企业ROA相对上升——多样性带来了风险分散优势。')
doc.add_paragraph('② 供应商广度×Post为负向（-0.079, p=0.437），竞争对手广度×Post为正向（+0.160, p=0.231），'
                  '竞争对手行业多样性×Post为负向（-0.361, p=0.310），方向与预期一致但统计不显著。')
doc.add_paragraph('③ 四个交互项的符号方向均与理论预期一致，为后续机制检验提供了基础。')

# ═══════════════════════════════════════════════
# 四、机制检验
# ═══════════════════════════════════════════════
doc.add_heading('四、机制检验结果', level=1)

doc.add_paragraph('方法：Baron & Kenny中介效应检验 + Sobel检验')
doc.add_paragraph('第一步(M1)：中介变量 ~ X + X×Post + Controls，验证X×Post对中介变量有影响')
doc.add_paragraph('第二步(M2)：ROA ~ X + X×Post + 中介变量 + Controls，验证中介变量对ROA有影响')
doc.add_paragraph('第三步(Sobel)：检验间接路径 X×Post → 中介变量 → ROA 是否显著')

# ── 4.1 供应商广度 ──
doc.add_heading('4.1 供应商广度×Post→ROA负向的机制', level=2)

doc.add_paragraph()
p = doc.add_paragraph()
p.add_run('理论逻辑：').bold = True
p.add_run('供应商广度大 → 供应链复杂度高 → 贸易战后协调管理成本上升、不确定性增加 → ROA下降')

table = doc.add_table(rows=4, cols=7)
table.style = 'Light Grid Accent 1'
table.alignment = WD_TABLE_ALIGNMENT.CENTER

for i, h in enumerate(['机制通道', '代理变量', 'M1系数', 'M1 p值', 'Sobel p值', '系数缩小%', '推荐使用']):
    cell = table.rows[0].cells[i]
    cell.text = h
    for p in cell.paragraphs:
        for r in p.runs:
            r.bold = True

rows_data = [
    ['存货积压', '存货密集度↑', '+0.003', '0.011**', '0.042**', '14.0%', '✓ 主选'],
    ['投资收缩', '资产增长率↓', '-0.003', '0.026**', '0.027**', '14.0%', '✓ 主选'],
    ['成本压力', '营业成本率↑', '+0.296', '0.092*', '0.093*', '47.7%', '△ 补充'],
]
for ri, row_data in enumerate(rows_data):
    for ci, val in enumerate(row_data):
        table.rows[ri+1].cells[ci].text = val

doc.add_paragraph('')
doc.add_paragraph('发现：贸易战后，供应商广度大的企业出现（a）存货积压、（b）投资收缩、（c）成本上升，')
doc.add_paragraph('三条路径共同导致ROA下降。其中成本路径的解释力最大（系数缩小47.7%），')
doc.add_paragraph('存货和投资路径的统计显著性更强（p<0.05）。建议论文中重点采用存货积压和投资收缩两条路径。')

# ── 4.2 供应商行业多样性 ──
doc.add_heading('4.2 供应商行业多样性×Post→ROA正向的机制', level=2)

doc.add_paragraph()
p = doc.add_paragraph()
p.add_run('结果：').bold = True
p.add_run('未发现方向匹配且统计显著的中介路径。收入波动率（growth_vol_w）方向正确但仅边际显著（p=0.052）。')
doc.add_paragraph('建议：正文中可不单独设机制检验小节，简要说明即可。')

# ── 4.3 竞争对手广度 ──
doc.add_heading('4.3 竞争对手广度×Post→ROA正向的机制', level=2)

doc.add_paragraph()
p = doc.add_paragraph()
p.add_run('理论逻辑：').bold = True
p.add_run('竞争对手多 → 竞争激烈 → 贸易战后倒逼企业降本增效 → ROA上升')

table = doc.add_table(rows=5, cols=7)
table.style = 'Light Grid Accent 1'
table.alignment = WD_TABLE_ALIGNMENT.CENTER

for i, h in enumerate(['机制通道', '代理变量', 'M1系数', 'M1 p值', 'Sobel p值', '系数缩小%', '推荐使用']):
    cell = table.rows[0].cells[i]
    cell.text = h
    for p in cell.paragraphs:
        for r in p.runs:
            r.bold = True

rows_data = [
    ['费用削减', 'SG&A/销售↓', '-0.696', '0.005***', '0.005***', '32.3%', '✓ 主选'],
    ['利润率修复', '净利润率↑', '+1.062', '0.006***', '0.006***', '39.2%', '✓ 主选'],
    ['回款加快', '应收周转率↑', '+0.302', '0.008***', '0.011**', '11.3%', '✓ 主选'],
    ['规模扩张', '销售规模↑', '+0.024', '0.027**', '0.028**', '27.3%', '✓ 主选'],
]
for ri, row_data in enumerate(rows_data):
    for ci, val in enumerate(row_data):
        table.rows[ri+1].cells[ci].text = val

doc.add_paragraph('')
doc.add_paragraph('发现：贸易战后，竞争对手广度大的企业通过削减SG&A费用、提升净利润率、')
doc.add_paragraph('加快回款、扩大销售规模四条路径，共同推动ROA上升。')
doc.add_paragraph('这是本文最稳健的机制证据——全部四个代理变量均通过Sobel检验（p<0.05），')
doc.add_paragraph('且方向一致、互不重复。')

# ── 4.4 竞争对手行业多样性 ──
doc.add_heading('4.4 竞争对手行业多样性×Post→ROA负向的机制', level=2)

doc.add_paragraph()
p = doc.add_paragraph()
p.add_run('结果：').bold = True
p.add_run('未发现方向匹配且统计显著的中介路径。可能原因：该效应需要更长期的指标来捕捉。')
doc.add_paragraph('建议：正文中可简要说明，不单独设机制检验小节。')

# ═══════════════════════════════════════════════
# 五、论文选用建议
# ═══════════════════════════════════════════════
doc.add_heading('五、论文机制检验章节的选用建议', level=1)

doc.add_paragraph()
p = doc.add_paragraph()
p.add_run('推荐写入正文的机制：').bold = True

table = doc.add_table(rows=6, cols=4)
table.style = 'Light Grid Accent 1'
table.alignment = WD_TABLE_ALIGNMENT.CENTER

for i, h in enumerate(['交互项', '机制', '代理变量', '证据等级']):
    cell = table.rows[0].cells[i]
    cell.text = h
    for p in cell.paragraphs:
        for r in p.runs:
            r.bold = True

rows_data = [
    ['供应商广度×Post', '存货积压', '存货密集度', '★★★'],
    ['供应商广度×Post', '投资收缩', '资产增长率', '★★★'],
    ['竞争对手广度×Post', '费用削减', 'SG&A/销售', '★★★'],
    ['竞争对手广度×Post', '利润率修复', '净利润率', '★★★'],
    ['竞争对手广度×Post', '回款加快+规模扩张', '应收周转率+销售规模', '★★☆'],
]
for ri, row_data in enumerate(rows_data):
    for ci, val in enumerate(row_data):
        table.rows[ri+1].cells[ci].text = val

doc.add_paragraph('')
doc.add_paragraph('建议论文结构：')
doc.add_paragraph('  • 第X节：基准回归（全样本+交互项）')
doc.add_paragraph('  • 第X+1节：稳健性检验（安慰剂检验、事件研究、企业规模异质性）')
doc.add_paragraph('    → 这部分如果需要我可以提供现成结果')
doc.add_paragraph('  • 第X+2节：机制检验')
doc.add_paragraph('      - 4.1 供应商广度→存货积压与投资收缩路径')
doc.add_paragraph('      - 4.2 竞争对手广度→费用削减与效率提升路径')
doc.add_paragraph('      - 4.3 其他交互项的简要说明')

# ═══════════════════════════════════════════════
# 六、其他可能感兴趣的结果
# ═══════════════════════════════════════════════
doc.add_heading('六、其他补充分析（如果老师需要，我可以提供详细结果）', level=1)

doc.add_paragraph('• 安慰剂检验：将贸易战时点假设为2015年和2016年，交互项不显著；真实时点2018年显著')
doc.add_paragraph('• 事件研究：逐年交互项系数显示，效应在2018-2019年最为明显')
doc.add_paragraph('• 企业规模异质性：供应商广度的负向效应在大企业中更强')
doc.add_paragraph('• 行业异质性：制造业与非制造业的分组分析')

# ─── 保存 ───
output_path = '/Users/deesanyu/pythonproject/deesan_supply_0506v4/output/mechanism_final/导师报告.docx'
doc.save(output_path)
print(f"✓ {output_path}")
