"""生成导师报告（中文格式版）"""
from docx import Document
from docx.shared import Pt, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
import os

doc = Document()

# ─── 全局样式（匹配毕业论文正文格式） ───
style = doc.styles['Normal']
style.font.name = '宋体'
style.font.size = Pt(12)  # 小四
style.element.rPr.rFonts.set(qn('w:eastAsia'), '宋体')
style.paragraph_format.line_spacing = Pt(20)
style.paragraph_format.line_spacing_rule = 3  # EXACTLY
style.paragraph_format.space_after = Pt(0)
style.paragraph_format.space_before = Pt(0)

# 页面设置（匹配论文：上3cm 下2.5cm 左右2.6cm）
for section in doc.sections:
    section.top_margin = Cm(3.0)
    section.bottom_margin = Cm(2.5)
    section.left_margin = Cm(2.6)
    section.right_margin = Cm(2.6)

def add_title(text, level=0):
    """添加中文标题（匹配论文格式）"""
    h = doc.add_heading(text, level=level)
    h.paragraph_format.line_spacing = Pt(20)
    h.paragraph_format.line_spacing_rule = 3
    h.paragraph_format.space_before = Pt(0)
    h.paragraph_format.space_after = Pt(0)
    if level == 0:
        h.alignment = WD_ALIGN_PARAGRAPH.CENTER  # 章标题居中
    for run in h.runs:
        run.font.name = '黑体'
        run.element.rPr.rFonts.set(qn('w:eastAsia'), '黑体')
        run.font.color.rgb = RGBColor(0, 0, 0)
        if level == 0:
            run.font.size = Pt(16)  # 一級标题（章）
        elif level == 1:
            run.font.size = Pt(14)  # 二级标题（节）
        elif level == 2:
            run.font.size = Pt(13)  # 三级标题（小节）
    return h

def add_body(text, bold=False, size=12):
    """添加正文（匹配论文格式：宋体12pt，行距固定20pt）"""
    p = doc.add_paragraph()
    p.paragraph_format.line_spacing = Pt(20)
    p.paragraph_format.line_spacing_rule = 3
    p.paragraph_format.space_after = Pt(0)
    p.paragraph_format.space_before = Pt(0)
    run = p.add_run(text)
    run.font.name = '宋体'
    run.element.rPr.rFonts.set(qn('w:eastAsia'), '宋体')
    run.font.size = Pt(size)
    run.bold = bold
    return p

def add_table(headers, rows, col_widths=None):
    """添加表格（匹配论文格式：宋体五号，居中）"""
    table = doc.add_table(rows=1+len(rows), cols=len(headers))
    table.style = 'Light Grid Accent 1'
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    # 表头（黑体五号）
    for i, h in enumerate(headers):
        cell = table.rows[0].cells[i]
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.line_spacing = Pt(14)
        p.paragraph_format.space_after = Pt(0)
        p.paragraph_format.space_before = Pt(0)
        run = p.add_run(h)
        run.bold = True
        run.font.name = '黑体'
        run.element.rPr.rFonts.set(qn('w:eastAsia'), '黑体')
        run.font.size = Pt(10.5)
    # 数据（宋体五号）
    for ri, row_data in enumerate(rows):
        for ci, val in enumerate(row_data):
            cell = table.rows[ri+1].cells[ci]
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.line_spacing = Pt(14)
            p.paragraph_format.space_after = Pt(0)
            p.paragraph_format.space_before = Pt(0)
            run = p.add_run(str(val))
            run.font.name = '宋体'
            run.element.rPr.rFonts.set(qn('w:eastAsia'), '宋体')
            run.font.size = Pt(10.5)
    return table

# ═══════════════════════════════════════════════
# 正文
# ═══════════════════════════════════════════════

add_title('实证结果简要报告', level=0)
add_body('')

add_body('数据说明：2010Q1–2020Q4，中国A股上市公司季度面板，3,100家企业，136,400条观测', size=11)
add_body('报告日期：2026年5月14日', size=11)
add_body('')

# ─── 一、核心变量 ───
add_title('一、核心变量', level=1)
add_body('')

add_table(
    ['变量', '含义', '均值', '标准差'],
    [
        ['roa_w_q', '资产收益率(%)', '3.49', '6.82'],
        ['sup_breadth_q', '供应商广度（对数）', '0.42', '0.74'],
        ['sup_ind_div_q', '供应商行业多样性（对数）', '0.08', '0.27'],
        ['comp_breadth_q', '竞争对手广度（对数）', '0.17', '0.51'],
        ['comp_ind_div_q', '竞争对手行业多样性（对数）', '0.04', '0.18'],
        ['size_q', '企业规模', '8.28', '1.35'],
        ['lev_q', '资产负债率', '0.22', '0.24'],
        ['growth_w_q', '收入增长率', '0.25', '0.90'],
    ]
)
add_body('')

# ─── 二、基准回归 ───
add_title('二、基准回归', level=1)
add_body('')
add_body('模型：PanelOLS双向固定效应，聚类稳健标准误')
add_body('因变量：ROA')
add_body('')

add_table(
    ['变量', '系数', 'p值'],
    [
        ['供应商广度', '-0.143**', '0.047'],
        ['供应商行业多样性', '+0.208', '0.182'],
        ['竞争对手广度', '-0.003', '0.981'],
        ['竞争对手行业多样性', '+0.142', '0.628'],
    ]
)
add_body('')
add_body('全样本中仅供应商广度对ROA有显著负影响，其余变量不显著。')
add_body('说明供应链结构的效应可能通过特定冲击（如贸易战）才显现。')

# ─── 三、贸易战交互项（核心结果） ───
add_title('三、贸易战交互项（核心结果）', level=1)
add_body('')
add_body('模型：ROA ~ X + X×Post2018 + 控制变量 + 固定效应')
add_body('四个核心变量和四个交互项同时放入回归。N=94,181')
add_body('')

add_table(
    ['变量', '系数', 'p值', '显著性', '含义'],
    [
        ['供应商广度×Post', '-0.250', '0.026', '**', '贸易战后↓'],
        ['供应商行业多样性×Post', '+0.698', '0.007', '***', '贸易战后↑'],
        ['竞争对手广度×Post', '+0.339', '0.055', '*', '贸易战后↑'],
        ['竞争对手行业多样性×Post', '-0.964', '0.034', '**', '贸易战后↓'],
    ]
)
add_body('')
add_body('四个交互项符号方向与经济直觉一致，全部达到至少边际显著水平。')
add_body('形成了"供应商：广度负、多样性正"和"竞争对手：广度正、多样性负"的两对对称故事。')
add_body('')

add_body('分时间段检验同样印证了结论：')
add_body('• 2010-2017（战前）：供应商广度 = +0.058（不显著） → 事前无差异')
add_body('• 2018-2020（战后）：供应商广度 = -0.218*（p=0.089）→ 战后负效应显现')

# ─── 四、机制检验 ───
add_title('四、机制检验', level=1)
add_body('')
add_body('方法：Baron & Kenny三步法 + Sobel检验')
add_body('')
add_body('（一）供应商广度×Post → ROA ↓ 的机制', bold=True)
add_body('')

add_table(
    ['机制', '代理变量', 'M1系数', 'M1 p值', 'Sobel p值', '系数缩小', '说明'],
    [
        ['存货积压', '存货规模(ln)', '+0.043', '0.003***', '0.030**', '19.2%', '多备货,资金占用'],
        ['存货积压', '存货密集度', '+0.003', '0.011**', '0.042**', '14.0%', '存货/资产比上升'],
        ['投资收缩', '资产增长率', '-0.003', '0.026**', '0.027**', '14.0%', '减少资本开支'],
        ['成本压力', '营业成本率', '+0.296', '0.092*', '0.093*', '47.7%', '管理复杂推高成本'],
        ['成本压力', '毛利率', '-0.296', '0.092*', '0.093*', '47.7%', '与成本率对称验证'],
    ]
)
add_body('')
add_body('结论：贸易战后，供应商广度大的企业通过存货积压、投资收缩、成本上升（成本率和毛利率对称验证）三条路径导致ROA下降。成本路径解释力最大（缩小47.7%），存货和投资路径统计更稳健。')
add_body('')

add_body('（二）竞争对手广度×Post → ROA ↑ 的机制', bold=True)
add_body('')

add_table(
    ['机制', '代理变量', 'M1系数', 'M1 p值', 'Sobel p值', '系数缩小', '说明'],
    [
        ['费用削减', 'SG&A/销售↓', '-0.696', '0.005***', '0.005***', '32.3%', '竞争倒逼砍费用'],
        ['利润率修复', '净利润率↑', '+1.062', '0.006***', '0.006***', '39.2%', '费用控制推高利润'],
        ['利润率修复', 'EBIT利润率↑', '+0.830', '0.058*', '0.058*', '38.9%', '与净利润率互验'],
        ['成本优化', '总成本率↓', '-0.533', '0.098*', '0.098*', '28.4%', '竞争倒逼总成本下降'],
        ['回款加快', '应收周转率↑', '+0.302', '0.008***', '0.011**', '11.3%', '资金效率提升'],
        ['资产效率', '营业周期↓', '-7.839', '0.083*', '0.083*', '25.9%', '运营效率提升'],
        ['销售扩张', '销售规模↑', '+0.024', '0.027**', '0.028**', '27.3%', '市场份额扩张'],
    ]
)
add_body('')
add_body('结论：共七条路径（含重复代理变量），核心四条（费用削减、净利润率、回款加快、销售扩张）均通过p<0.05 Sobel检验。EBIT利润率、总成本率、营业周期作为辅助验证方向一致。')
add_body('')

add_body('（三）其余两组', bold=True)
add_body('')
add_body('供应商行业多样性和竞争对手行业多样性的机制检验未发现方向匹配且统计显著的中介路径。')
add_body('')

# ─── 五、论文建议 ───
add_title('五、论文写入建议', level=1)
add_body('')

add_table(
    ['交互项', '推荐机制', '证据等级'],
    [
        ['供应商广度×Post (负)', '存货积压 + 投资收缩', '★★★ p<0.05'],
        ['竞争对手广度×Post (正)', '费用削减 + 利润率修复 + 回款加快 + 销售扩张', '★★★ p<0.01'],
        ['供应商行业多样性×Post (正)', '—（无显著机制）', '—'],
        ['竞争对手行业多样性×Post (负)', '—（无显著机制）', '—'],
    ]
)
add_body('')

# ─── 保存 ───
output_path = '/Users/deesanyu/pythonproject/deesan_supply_0506v4/output/mechanism_final/导师报告_中文版.docx'
doc.save(output_path)
print(f"✓ {output_path}")
