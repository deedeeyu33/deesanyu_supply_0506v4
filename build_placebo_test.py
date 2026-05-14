"""
安慰剂检验：虚设贸易战时间点
比较 假2015年、假2016年、真2018年 的交互项结果
"""
import pandas as pd, numpy as np, os, warnings
warnings.filterwarnings('ignore')
from linearmodels.panel import PanelOLS
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

OUT_DIR  = '/Users/deesanyu/pythonproject/deesan_supply_0506v4/output/placebo_test'
FONT_PATH = '/Users/deesanyu/pythonproject/deesan_supply_0506v4/simhei.ttf'
os.makedirs(OUT_DIR, exist_ok=True)
zh_font = FontProperties(fname=FONT_PATH)
plt.rcParams['axes.unicode_minus'] = False

qp = pd.read_parquet('/Users/deesanyu/pythonproject/deesan_supply_0506v4/output/quarterly_panel/firm_quarterly_panel.parquet')
qp['quarter_dt'] = pd.to_datetime(qp['quarter'])
qp = qp.sort_values(['ISIN', 'qi']).set_index(['ISIN', 'quarter_dt'])
print(f"面板: {len(qp):,} 行")

var_map_q = {
    'ROA': 'roa_w_q',
    'supplier_count_log': 'sup_breadth_q', 'supplier_industry_count': 'sup_ind_div_q',
    'competitor_count_log': 'comp_breadth_q', 'competitor_industry_count': 'comp_ind_div_q',
    'Size': 'size_q', 'Leverage': 'lev_q', 'Growth': 'growth_w_q',
}
col_to_disp_q = {v: k for k, v in var_map_q.items()}
controls = ['Size', 'Leverage', 'Growth']
core_vars_disp = ['supplier_count_log', 'supplier_industry_count', 'competitor_count_log', 'competitor_industry_count']
core_vars_col = [var_map_q[v] for v in core_vars_disp]

def run_panelols(y_display, x_displays, data, label=''):
    y_name = var_map_q[y_display]
    def _resolve(v): return var_map_q[v] if v in var_map_q else v
    x_names = [_resolve(v) for v in x_displays]
    sub = data[[y_name] + x_names].dropna().copy()
    n = len(sub)
    if n < 100: return None
    formula = f'{y_name} ~ EntityEffects + TimeEffects + ' + ' + '.join(x_names)
    try:
        mod = PanelOLS.from_formula(formula, data=sub, drop_absorbed=True)
        res = mod.fit(cov_type='clustered', cluster_entity=True)
        rows = []
        for v_col in x_names:
            v_disp = col_to_disp_q.get(v_col, v_col)
            coef = res.params.get(v_col, np.nan)
            se = res.std_errors.get(v_col, np.nan)
            pv = res.pvalues.get(v_col, np.nan)
            star = ''
            if not np.isnan(pv):
                if pv < 0.01: star = '***'
                elif pv < 0.05: star = '**'
                elif pv < 0.1: star = '*'
            rows.append({
                'model': label, 'variable': v_disp,
                'coef': round(coef, 5), 'se': round(se, 5),
                'pval': round(pv, 4), 'sig': star,
                'N': n, 'rsq_within': round(res.rsquared_within, 4),
            })
        return pd.DataFrame(rows)
    except Exception as e:
        print(f"  ⚠ {label}: {e}")
        return None

# ====================================================================
# 三个时间点对比
# ====================================================================
print("\n安慰剂检验：假2015年 vs 假2016年 vs 真2018年")
print("=" * 60)
all_results = []

for fake_year, label in [(2015, 'placebo_2015_ROA'), (2016, 'placebo_2016_ROA'),
                          (2018, 'true_2018_ROA')]:
    p_int = qp.copy()
    # 避免与已存在的 post2018 混淆
    p_int['fake_post'] = (qp.index.get_level_values('quarter_dt').year >= fake_year).astype(int)

    for v_col in core_vars_col:
        p_int[f'{v_col}_x_fake'] = p_int[v_col] * p_int['fake_post']

    x_b = core_vars_disp + [f'{v}_x_fake' for v in core_vars_col] + controls
    r = run_panelols('ROA', x_b, p_int, label)
    if r is not None:
        all_results.append(r)
        print(f"\n  假贸易战={fake_year}:")
        for v_col in core_vars_col:
            row = r[r['variable'] == f'{v_col}_x_fake']
            if len(row):
                rw = row.iloc[0]
                vd = col_to_disp_q.get(v_col, v_col)
                ds = '+' if rw['coef'] > 0 else ''
                print(f"    {vd:35s} × post: {ds}{rw['coef']:.5f}{rw['sig']}  p={rw['pval']:.4f}")

if all_results:
    rdf = pd.concat(all_results, ignore_index=True)
    rdf.to_excel(f'{OUT_DIR}/placebo_test_results.xlsx', index=False)
    print(f"\n  ✓ {OUT_DIR}/placebo_test_results.xlsx")

# ====================================================================
# 可视化
# ====================================================================
print("\n生成对比图...")
zh_names = ['供应商广度', '供应商行业多样性', '竞争对手广度', '竞争对手行业多样性']
scenarios = [
    ('placebo_2015', '假2015年'),
    ('placebo_2016', '假2016年'),
    ('true_2018', '真2018年'),
]
colors = ['lightgray', 'darkgray', 'coral']

fig, axes = plt.subplots(2, 2, figsize=(12, 9))

for vi, (core_col, zh_name) in enumerate(zip(core_vars_col, zh_names)):
    ax = axes[vi // 2, vi % 2]
    for si, (sc_label, sc_name) in enumerate(scenarios):
        model_name = f'{sc_label}_ROA'
        row = rdf[(rdf['model']==model_name) & (rdf['variable']==f'{core_col}_x_fake')]
        if len(row):
            rw = row.iloc[0]
            ci = rw['se'] * 1.96
            ax.barh(sc_name, rw['coef'], xerr=ci, color=colors[si], alpha=0.8, capsize=4)
            ax.text(rw['coef'] + ci + 0.03, sc_name,
                   f'{rw["coef"]:.3f}{rw["sig"]}', va='center', fontsize=9, fontweight='bold')
    ax.axvline(x=0, color='gray', lw=1)
    ax.set_xlabel('交互项系数', fontproperties=zh_font, fontsize=10)
    ax.set_title(f'{zh_name} × 假贸易战年份', fontproperties=zh_font, fontsize=12)
    ax.grid(True, alpha=0.2, axis='x')

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/placebo_comparison.png', dpi=200, bbox_inches='tight')
plt.close()
print(f"  ✓ {OUT_DIR}/placebo_comparison.png")

# ====================================================================
# 报告
# ====================================================================
print("\n生成报告...")
report = [
    "# 安慰剂检验报告",
    "",
    "## 一、检验目的",
    "",
    "通过将贸易战时间点虚设为2015年和2016年，检验交互项效应是否仅在真实贸易战（2018年）时出现。",
    "如果虚设年份不显著而真实年份显著，则支持贸易战确实是触发供应链结构效应变化的外生冲击。",
    "",
    "## 二、结果",
    "",
    "| 变量 | 假2015年(系数/显著性) | 假2016年(系数/显著性) | 真2018年(系数/显著性) |",
    "|------|---------------------|---------------------|---------------------|",
]
for vi, (core_col, zh_name) in enumerate(zip(core_vars_col, zh_names)):
    vals = []
    for sc_label, _ in scenarios:
        row = rdf[(rdf['model']==f'{sc_label}_ROA') & (rdf['variable']==f'{core_col}_x_fake')]
        if len(row):
            rw = row.iloc[0]
            vals.append(f"{rw['coef']:.4f}{rw['sig']}")
        else:
            vals.append("NA")
    report.append(f"| {zh_name} | {vals[0]} | {vals[1]} | {vals[2]} |")

report += [
    "",
    "## 三、结论",
    "",
]
with open(f'{OUT_DIR}/placebo_summary.md', 'w', encoding='utf-8-sig') as f:
    f.write('\n'.join(report))
print(f"  ✓ {OUT_DIR}/placebo_summary.md")
print("\n完成!")
