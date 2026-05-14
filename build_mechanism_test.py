"""
机制检验：解释四个交互项为何各自具有特定的方向
数据来源：已有季度面板 + 11个新增财务数据文件

核心逻辑：
  ROA = 利润率(Profit Margin) × 资产周转率(Asset Turnover)
  通过分解ROA，找到每个交互项影响ROA的具体渠道

四组假设：
  ① sup_breadth × post → ROA负向: 交易成本↑ → 成本率↑ → EBIT利润率↓
  ② sup_ind_div × post → ROA正向: 风险分散 → 效率↑ → EBIT利润率↑
  ③ comp_breadth × post → ROA正向: 竞争降本 → 成本率↓ → EBIT利润率↑
  ④ comp_ind_div × post → ROA负向: 资源分散 → 周转率↓ → EBIT利润率↓
"""
import pandas as pd, numpy as np, os, warnings, re, gc
warnings.filterwarnings('ignore')
from linearmodels.panel import PanelOLS
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

DATA_DIR = '/Users/deesanyu/pythonproject/deesan_supply_0506v4/data'
OUT_DIR  = '/Users/deesanyu/pythonproject/deesan_supply_0506v4/output/mechanism_test'
FONT_PATH = '/Users/deesanyu/pythonproject/deesan_supply_0506v4/simhei.ttf'
QP_PATH  = '/Users/deesanyu/pythonproject/deesan_supply_0506v4/output/quarterly_panel/firm_quarterly_panel.parquet'
os.makedirs(OUT_DIR, exist_ok=True)
zh_font = FontProperties(fname=FONT_PATH)
plt.rcParams['axes.unicode_minus'] = False

print("=" * 60)
print("机制检验：渠道分解分析")
print("=" * 60)

# ====================================================================
# 1. 加载已有季度面板
# ====================================================================
print("\n1. 加载已有季度面板...")
qp = pd.read_parquet(QP_PATH)
qp['quarter_dt'] = pd.to_datetime(qp['quarter'])
print(f"  面板: {len(qp):,} 行, {qp['ISIN'].nunique():,} 企业")

# ====================================================================
# 2. 加载11个新增数据文件
# ====================================================================
print("\n2. 加载新增财务数据...")

def load_mechanism_data(filepath, vname):
    """加载季度机制变量数据（列在FE Isin之后）"""
    raw = pd.read_excel(filepath, header=None)
    h = raw.iloc[4]
    # 找FE Isin列
    try:
        isin_pos = next(i for i, v in enumerate(h) if pd.notna(v) and 'Isin' in str(v))
    except StopIteration:
        return None

    # 季度列在 FE Isin 之后
    col_to_q = {}
    for i in range(isin_pos + 1, len(h)):
        v = str(h[i]) if pd.notna(h[i]) else ''
        m = re.search(r'(201[0-9]|2020).*?Q(\d)', v)
        if m:
            qcode = f"{m.group(1)}Q{m.group(2)}"
            col_to_q[v] = qcode

    d = raw.iloc[6:].copy()
    d.columns = list(h)
    d = d[d['Symbol'].notna()]

    value_cols = list(col_to_q.keys())
    if not value_cols:
        return None

    melted = d.melt(id_vars=['FE Isin', 'Symbol', 'Name'],
                    value_vars=value_cols,
                    var_name='qname', value_name=vname)
    melted['qcode'] = melted['qname'].map(col_to_q)
    melted[vname] = pd.to_numeric(melted[vname], errors='coerce')

    def qcode_to_int(q):
        m = re.search(r'(201[0-9]|2020)Q(\d)', str(q))
        if m:
            y, q_ = int(m.group(1)), int(m.group(2))
            return (y - 2010) * 4 + (q_ - 1)
        return -1

    melted = melted[melted['FE Isin'].astype(str) != '@NA']
    melted['qi'] = melted['qcode'].apply(qcode_to_int)
    melted = melted[(melted['qi'] >= 0) & (melted['qi'] <= 43)].dropna(subset=[vname])
    return melted[['FE Isin', 'qi', vname]]


files_to_load = [
    ('china-quarterly-cogs-sales-ratio.xlsx', 'cogs_sales_ratio'),
    ('china-quarterly-sga-sales-ratio.xlsx', 'sga_sales_ratio'),
    ('china-quarterly-sga-expense.xlsx', 'sga_expense'),
    ('china-quarterly-ebit-margin.xlsx', 'ebit_margin'),
    ('china-quarterly-inventories.xlsx', 'inventories'),
    ('china-quarterly-inventory-turnover.xlsx', 'inventory_turnover'),
    ('china-quarterly-asset-turnover.xlsx', 'asset_turnover'),
    ('china-quarterly-payables-turnover.xlsx', 'payables_turnover'),
    ('china-quarterly-days-payables-outstanding.xlsx', 'days_payables'),
    ('china-quarterly-accounts-payable-sales-ratio.xlsx', 'ap_sales_ratio'),
    ('china-quarterly-receivables-turnover.xlsx', 'receivables_turnover'),
]

for fn, vn in files_to_load:
    df = load_mechanism_data(f'{DATA_DIR}/{fn}', vn)
    if df is not None:
        df_dedup = df.groupby(['FE Isin', 'qi'], as_index=False)[vn].mean()
        temp = df_dedup.rename(columns={'FE Isin': 'ISIN'})
        qp = qp.merge(temp, on=['ISIN', 'qi'], how='left')
        n_valid = qp[vn].notna().sum()
        print(f"  {vn:25s}: N={n_valid:>8,}")
    else:
        print(f"  {vn:25s}: FAILED")

# ====================================================================
# 3. 处理变量：缩尾
# ====================================================================
print("\n3. 变量处理...")

def winsor(s, l=0.01, u=0.99):
    lo, hi = s.quantile(l), s.quantile(u)
    return s.clip(lo, hi)

mechanism_vars = ['cogs_sales_ratio', 'sga_sales_ratio', 'sga_expense',
                  'ebit_margin', 'inventories', 'inventory_turnover',
                  'asset_turnover', 'payables_turnover', 'days_payables',
                  'ap_sales_ratio', 'receivables_turnover']

for v in mechanism_vars:
    if v in qp.columns:
        w_col = v + '_w'
        qp[w_col] = winsor(qp[v].dropna())
        n = qp[w_col].notna().sum()
        mean_v = qp[w_col].mean()
        print(f"  {w_col:25s}: N={n:>8,}, 均值={mean_v:.4f}")
    else:
        print(f"  {v:25s}: NOT FOUND")

# ====================================================================
# 4. 变量映射与回归函数
# ====================================================================
print("\n4. 准备回归...")

# 扩展 var_map_q 包含机制变量
var_map_q = {
    'ROA': 'roa_w_q', 'ROE': 'roe_w_q',
    'supplier_count_log': 'sup_breadth_q', 'supplier_industry_count': 'sup_ind_div_q',
    'competitor_count_log': 'comp_breadth_q', 'competitor_industry_count': 'comp_ind_div_q',
    'Size': 'size_q', 'Leverage': 'lev_q', 'Growth': 'growth_w_q',
}
# 添加机制变量 (使用缩尾后的版本)
for v in mechanism_vars:
    w_col = v + '_w'
    if w_col in qp.columns:
        var_map_q[v] = w_col

col_to_disp_q = {v: k for k, v in var_map_q.items()}
controls = ['Size', 'Leverage', 'Growth']
core_vars_disp = ['supplier_count_log', 'supplier_industry_count',
                  'competitor_count_log', 'competitor_industry_count']
core_vars_col = [var_map_q[v] for v in core_vars_disp]

# 设置面板索引 (PanelOLS需要MultiIndex)
qp = qp.sort_values(['ISIN', 'qi']).set_index(['ISIN', 'quarter_dt'])


def run_panelols(y_display, x_displays, data, label=''):
    y_name = var_map_q[y_display]

    def _resolve(v):
        return var_map_q[v] if v in var_map_q else v

    x_names = [_resolve(v) for v in x_displays]
    available = [c for c in x_names if c in data.columns]
    missing = [c for c in x_names if c not in data.columns]
    if missing:
        print(f"  ⚠ {label}: missing columns: {missing}")
        return None
    sub = data[[y_name] + available].dropna().copy()
    n = len(sub)
    n_firms = sub.index.get_level_values(0).nunique()
    n_periods = sub.index.get_level_values(1).nunique()
    if n < 50:
        return None
    formula = f'{y_name} ~ EntityEffects + TimeEffects + ' + ' + '.join(available)
    try:
        mod = PanelOLS.from_formula(formula, data=sub, drop_absorbed=True)
        res = mod.fit(cov_type='clustered', cluster_entity=True)
        rows = []
        for v_col in available:
            v_disp = col_to_disp_q.get(v_col, v_col)
            coef = res.params.get(v_col, np.nan)
            se = res.std_errors.get(v_col, np.nan)
            t = res.tstats.get(v_col, np.nan)
            pv = res.pvalues.get(v_col, np.nan)
            star = ''
            if not np.isnan(pv):
                if pv < 0.01: star = '***'
                elif pv < 0.05: star = '**'
                elif pv < 0.1: star = '*'
            rows.append({
                'model': label, 'variable': v_disp,
                'coef': round(coef, 5), 'se': round(se, 5),
                't': round(t, 3), 'pval': round(pv, 4),
                'sig': star, 'N': n, 'N_firms': int(n_firms),
                'N_periods': int(n_periods),
                'rsq_within': round(res.rsquared_within, 4),
            })
        return pd.DataFrame(rows)
    except Exception as e:
        print(f"  ⚠ {label}: {e}")
        return None


# 创建交互项
p_int = qp.copy()
for v_col in core_vars_col:
    p_int[f'{v_col}_x_post'] = p_int[v_col] * p_int['post2018']

# ====================================================================
# 5. 机制检验
# ====================================================================
print("\n" + "=" * 60)
print("5. 机制检验结果")
print("=" * 60)

all_results = []

# 定义每组交互项对应的机制检验
# 格式：(核心变量display名, 核心变量列名, 中文名, 机制变量列表, 对每个机制变量的预期方向)
mechanism_tests = [
    ('supplier_count_log', 'sup_breadth_q', '供应商广度',
     ['cogs_sales_ratio', 'sga_sales_ratio', 'ebit_margin',
      'inventory_turnover', 'ap_sales_ratio'],
     ['+', '+', '-', '-', '+']),
    ('supplier_industry_count', 'sup_ind_div_q', '供应商行业多样性',
     ['ebit_margin', 'asset_turnover', 'inventory_turnover'],
     ['+', '+', '+']),
    ('competitor_count_log', 'comp_breadth_q', '竞争对手广度',
     ['cogs_sales_ratio', 'ebit_margin', 'sga_sales_ratio'],
     ['-', '+', '-']),
    ('competitor_industry_count', 'comp_ind_div_q', '竞争对手行业多样性',
     ['ebit_margin', 'asset_turnover', 'receivables_turnover', 'inventory_turnover'],
     ['-', '-', '-', '-']),
]

mechanism_summaries = []

for core_disp, core_col, core_zh, med_list, expected in mechanism_tests:
    print(f"\n  === {core_zh} ({core_disp}) ===")
    print(f"  交互项系数: 已从主回归得知 → ROA方向确定")
    row_summary = {'变量': core_zh}

    for med_name, exp_dir in zip(med_list, expected):
        med_col = var_map_q[med_name]

        # Step 1: 检验 X × post → 中介变量
        x_mech = [core_disp, f'{core_col}_x_post'] + controls
        label_mech = f'mech_{core_disp}_{med_name}'
        r_mech = run_panelols(med_name, x_mech, p_int, label_mech)

        if r_mech is not None:
            row_interact = r_mech[r_mech['variable'] == f'{core_col}_x_post']
            if len(row_interact):
                rw = row_interact.iloc[0]
                dir_actual = '+' if rw['coef'] > 0 else '-'
                match = '✓' if dir_actual == exp_dir else '✗'
                sig_info = f"{rw['coef']:+.5f}{rw['sig']} (p={rw['pval']:.4f})"
                print(f"    {med_name:25s} → {dir_actual}(预期{exp_dir}) {match}  coef={sig_info}")
                row_summary[f'{med_name}_coef'] = f"{rw['coef']:+.5f}{rw['sig']}"
                row_summary[f'{med_name}_pval'] = rw['pval']
                row_summary[f'{med_name}_direction'] = dir_actual
                row_summary[f'{med_name}_match'] = match
                all_results.append(r_mech)

                # Step 2: 加入中介变量到ROA回归，看交互项系数变化
                x_med = [core_disp, f'{core_col}_x_post', med_name] + controls
                label_med2 = f'med_{core_disp}_{med_name}_add'
                r_med2 = run_panelols('ROA', x_med, p_int, label_med2)
                if r_med2 is not None:
                    all_results.append(r_med2)
                    row_interact2 = r_med2[r_med2['variable'] == f'{core_col}_x_post']
                    if len(row_interact2):
                        rw2 = row_interact2.iloc[0]
                        print(f"     加入{med_name}后交互项: {rw2['coef']:+.5f}{rw2['sig']}")
            else:
                print(f"    {med_name:25s} → 交互项无结果")
        else:
            print(f"    {med_name:25s} → N不足或模型失败")

    mechanism_summaries.append(row_summary)

# ====================================================================
# 6. 保存结果
# ====================================================================
if all_results:
    rdf = pd.concat(all_results, ignore_index=True)
    rdf.to_excel(f'{OUT_DIR}/mechanism_results.xlsx', index=False)
    print(f"\n  ✓ {OUT_DIR}/mechanism_results.xlsx")

# ====================================================================
# 7. 生成可视化
# ====================================================================
print("\n6. 生成可视化...")

# 提取关键的机制检验系数
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
zh_names = ['供应商广度', '供应商行业多样性', '竞争对手广度', '竞争对手行业多样性']
core_cols = ['sup_breadth_q', 'sup_ind_div_q', 'comp_breadth_q', 'comp_ind_div_q']

all_med_names = ['cogs_sales_ratio', 'sga_sales_ratio', 'ebit_margin',
                 'asset_turnover', 'inventory_turnover', 'ap_sales_ratio',
                 'receivables_turnover']
all_med_labels = ['成本率\n(COGS/Sales)', '管理费用率\n(SG&A/Sales)', 'EBIT\n利润率',
                  '总资产\n周转率', '存货\n周转率', '应付账款\n比率', '应收账款\n周转率']
med_color_map = {'cogs_sales_ratio': 'coral', 'sga_sales_ratio': 'orange',
                 'ebit_margin': 'steelblue', 'asset_turnover': 'green',
                 'inventory_turnover': 'teal', 'ap_sales_ratio': 'purple',
                 'receivables_turnover': 'brown'}

for idx, (core_disp, core_col, zh_name) in enumerate(zip(
        ['supplier_count_log', 'supplier_industry_count', 'competitor_count_log',
         'competitor_industry_count'],
        core_cols, zh_names)):
    ax = axes[idx // 2, idx % 2]
    ax.axvline(x=0, color='gray', lw=1, alpha=0.5)

    bar_data = []
    for med_name in all_med_names:
        model_name = f'mech_{core_disp}_{med_name}'
        sub = rdf[(rdf['model'] == model_name) &
                  (rdf['variable'] == f'{core_col}_x_post')]
        if len(sub):
            rw = sub.iloc[0]
            bar_data.append({
                'label': med_name,
                'coef': rw['coef'],
                'ci': rw['se'] * 1.96,
                'sig': rw['sig'],
                'color': med_color_map.get(med_name, 'gray'),
            })

    if bar_data:
        y_labels = [all_med_labels[all_med_names.index(b['label'])] for b in bar_data]
        coefs = [b['coef'] for b in bar_data]
        cis = [b['ci'] for b in bar_data]
        colors = [b['color'] for b in bar_data]
        sigs = [b['sig'] for b in bar_data]

        y_pos = range(len(bar_data))
        bars = ax.barh(y_pos, coefs, xerr=cis, color=colors, alpha=0.75, capsize=4)
        for i, (c, ci, s) in enumerate(zip(coefs, cis, sigs)):
            offset = ci + 0.02 if c >= 0 else -(ci + 0.15)
            ax.text(c + offset, i, f'{c:+.3f}{s}', va='center',
                    fontsize=8, fontweight='bold')

        ax.set_yticks(list(y_pos))
        ax.set_yticklabels(y_labels, fontsize=9)
        ax.set_xlabel('中介效应系数 (X×post → 中介变量)', fontproperties=zh_font, fontsize=9)

    ax.set_title(f'{zh_name}的渠道分解', fontproperties=zh_font, fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.2, axis='x')

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/mechanism_channel_plot.png', dpi=200, bbox_inches='tight')
plt.close()
print(f"  ✓ {OUT_DIR}/mechanism_channel_plot.png")

# ====================================================================
# 8. 生成分析报告
# ====================================================================
print("\n7. 生成分析报告...")

# 构建机制汇总表
report = [
    "# 机制检验分析报告：渠道分解",
    "",
    "## 一、分析框架",
    "",
    "将ROA拆解为利润率和资产周转率两个渠道，通过面板固定效应模型检验",
    "贸易战后供应链结构变量变化影响ROA的具体传导路径。",
    "",
    "```",
    "ROA = 利润率(Profit Margin) × 资产周转率(Asset Turnover)",
    "",
    "每个交互项的传导路径：",
    "X × post2018 → 中介变量(M) → ROA",
    "```",
    "",
    "## 二、四组机制假设",
    "",
    "### 2.1 供应商广度 × post → ROA 负向 (-0.250**)",
    "- 假设：交易成本渠道",
    "- 供应商越多 → 贸易战后协调管理成本↑ → 成本率↑ → EBIT利润率↓ → ROA↓",
    "- 预测中介变量方向: 成本率(+), 费用率(+), EBIT利润率(-), 存货周转率(-), 应付账款率(+)",
    "",
    "### 2.2 供应商行业多样性 × post → ROA 正向 (+0.698***)",
    "- 假设：风险分散渠道",
    "- 供应商行业多样 → 贸易战中可切换替代来源 → 运营更稳定 → EBIT利润率↑ / 周转率↑ → ROA↑",
    "- 预测中介变量方向: EBIT利润率(+), 资产周转率(+), 存货周转率(+)",
    "",
    "### 2.3 竞争对手广度 × post → ROA 正向边际显著 (+0.339*)",
    "- 假设：竞争降本渠道",
    "- 竞争对手多 → 贸易战倒逼降本增效 → 成本率↓ → EBIT利润率↑ → ROA↑",
    "- 预测中介变量方向: 成本率(-), 费用率(-), EBIT利润率(+)",
    "",
    "### 2.4 竞争对手行业多样性 × post → ROA 负向 (-0.964**)",
    "- 假设：资源分散渠道",
    "- 竞争对手跨太多行业 → 管理资源分散 → 运营效率↓ → 周转率↓ / EBIT利润率↓ → ROA↓",
    "- 预测中介变量方向: EBIT利润率(-), 资产周转率(-), 应收账款周转率(-), 存货周转率(-)",
    "",
    "## 三、检验结果",
    "",
]

# 为每个核心变量生成结果表格
for core_disp, core_zh, core_col, med_list, expected in [
    ('supplier_count_log', '供应商广度', 'sup_breadth_q',
     ['cogs_sales_ratio', 'sga_sales_ratio', 'ebit_margin',
      'inventory_turnover', 'ap_sales_ratio'],
     ['+', '+', '-', '-', '+']),
    ('supplier_industry_count', '供应商行业多样性', 'sup_ind_div_q',
     ['ebit_margin', 'asset_turnover', 'inventory_turnover'],
     ['+', '+', '+']),
    ('competitor_count_log', '竞争对手广度', 'comp_breadth_q',
     ['cogs_sales_ratio', 'ebit_margin', 'sga_sales_ratio'],
     ['-', '+', '-']),
    ('competitor_industry_count', '竞争对手行业多样性', 'comp_ind_div_q',
     ['ebit_margin', 'asset_turnover', 'receivables_turnover', 'inventory_turnover'],
     ['-', '-', '-', '-']),
]:
    report.append(f"### {core_zh}")
    report.append("")
    report.append("| 中介变量 | 预期方向 | 实际系数 | p值 | 匹配 |")
    report.append("|---------|:-------:|:-------:|:---:|:---:|")
    for med_name, exp_dir in zip(med_list, expected):
        model_name = f'mech_{core_disp}_{med_name}'
        sub = rdf[(rdf['model'] == model_name) &
                  (rdf['variable'] == f'{core_col}_x_post')]
        if len(sub):
            rw = sub.iloc[0]
            dir_actual = '+' if rw['coef'] > 0 else '-'
            match = '✓' if dir_actual == exp_dir else '✗'
            report.append(f"| {med_name} | {exp_dir} | {rw['coef']:+.5f}{rw['sig']} | {rw['pval']:.4f} | {match} |")
        else:
            report.append(f"| {med_name} | {exp_dir} | N/A | N/A | - |")
    report.append("")

report += [
    "## 四、结论",
    "",
    "（根据实际运行结果填写）",
    "",
]
with open(f'{OUT_DIR}/mechanism_summary.md', 'w', encoding='utf-8-sig') as f:
    f.write('\n'.join(report))
print(f"  ✓ {OUT_DIR}/mechanism_summary.md")

print("\n" + "=" * 60)
print("机制检验完成!")
print(f"输出目录: {OUT_DIR}")
print("=" * 60)
