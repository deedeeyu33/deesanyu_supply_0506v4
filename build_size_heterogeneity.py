"""
企业规模分组异质性分析
按企业规模（总资产）中位数分为大/小两组
"""
import pandas as pd, numpy as np, os, warnings
warnings.filterwarnings('ignore')
from linearmodels.panel import PanelOLS
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

OUT_DIR  = '/Users/deesanyu/pythonproject/deesan_supply_0506v4/output/size_heterogeneity'
FONT_PATH = '/Users/deesanyu/pythonproject/deesan_supply_0506v4/simhei.ttf'
os.makedirs(OUT_DIR, exist_ok=True)
zh_font = FontProperties(fname=FONT_PATH)
plt.rcParams['axes.unicode_minus'] = False

# 用季度面板
print("=" * 60)
print("企业规模分组异质性分析")
print("=" * 60)
qp = pd.read_parquet('/Users/deesanyu/pythonproject/deesan_supply_0506v4/output/quarterly_panel/firm_quarterly_panel.parquet')
qp['quarter_dt'] = pd.to_datetime(qp['quarter'])
qp = qp.sort_values(['ISIN', 'qi']).set_index(['ISIN', 'quarter_dt'])

# 计算每个企业的size均值，按中位数分组
firm_size = qp.groupby(level='ISIN')['size_q'].mean()
size_median = firm_size.median()
print(f"企业规模中位数: {size_median:.4f}")
qp['size_group'] = 'small'
large_firms = firm_size[firm_size > size_median].index
qp.loc[qp.index.get_level_values(0).isin(large_firms), 'size_group'] = 'large'
print(f"大企业: {qp[qp['size_group']=='large'].index.get_level_values(0).nunique():,} 家")
print(f"小企业: {qp[qp['size_group']=='small'].index.get_level_values(0).nunique():,} 家")

var_map_q = {
    'ROA': 'roa_w_q', 'ROE': 'roe_w_q',
    'supplier_count_log': 'sup_breadth_q', 'supplier_industry_count': 'sup_ind_div_q',
    'competitor_count_log': 'comp_breadth_q', 'competitor_industry_count': 'comp_ind_div_q',
    'Size': 'size_q', 'Leverage': 'lev_q', 'Growth': 'growth_w_q',
}
col_to_disp_q = {v: k for k, v in var_map_q.items()}
controls = ['Size', 'Leverage', 'Growth']
core_vars_disp = ['supplier_count_log', 'supplier_industry_count', 'competitor_count_log', 'competitor_industry_count']
core_vars_col = [var_map_q[v] for v in core_vars_disp]
base_x = core_vars_disp + controls

def run_panelols(y_display, x_displays, data, label=''):
    y_name = var_map_q[y_display]
    def _resolve(v): return var_map_q[v] if v in var_map_q else v
    x_names = [_resolve(v) for v in x_displays]
    sub = data[[y_name] + x_names].dropna().copy()
    n = len(sub); n_firms = sub.index.get_level_values(0).nunique()
    n_periods = sub.index.get_level_values(1).nunique()
    if n < 50: return None
    formula = f'{y_name} ~ EntityEffects + TimeEffects + ' + ' + '.join(x_names)
    try:
        mod = PanelOLS.from_formula(formula, data=sub, drop_absorbed=True)
        res = mod.fit(cov_type='clustered', cluster_entity=True)
        rows = []
        for v_col in x_names:
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
                'sig': star, 'N': n, 'N_firms': n_firms,
                'N_periods': n_periods,
                'rsq_within': round(res.rsquared_within, 4),
            })
        return pd.DataFrame(rows)
    except Exception as e:
        print(f"  ⚠ {label}: {e}")
        return None

# ====================================================================
# 描述性统计
# ====================================================================
print("\n描述性统计对比:")
for grp_name in ['large', 'small']:
    sub = qp[qp['size_group'] == grp_name]
    print(f"\n  {grp_name} (N={len(sub):,}):")
    for v in ['sup_breadth_q','sup_ind_div_q','comp_breadth_q','comp_ind_div_q','roa_w_q']:
        print(f"    {v:20s} 均值={sub[v].mean():.4f}  零值={(sub[v]==0).mean()*100:.1f}%")

# ====================================================================
# 1. 分规模基准回归 (M3)
# ====================================================================
print("\n1. 分规模基准回归...")
all_results = []
for grp_name, grp_label in [('全部企业', 'all'), ('大企业', 'large'), ('小企业', 'small')]:
    if grp_name == '全部企业':
        sub = qp
    else:
        sub = qp[qp['size_group'] == grp_label]
    print(f"  {grp_name}: {len(sub):,} 行")
    for y in ['ROA', 'ROE']:
        r = run_panelols(y, base_x, sub, f'M3_{grp_label}_{y}')
        if r is not None:
            all_results.append(r)
            print(f"    {y}: N={r.iloc[0]['N']:,}, R²={r.iloc[0]['rsq_within']:.4f}")
            for _, rw in r[r['variable'].isin(core_vars_disp)].iterrows():
                ds = '+' if rw['coef'] > 0 else ''
                print(f"      {rw['variable']:35s} {ds}{rw['coef']:.5f}{rw['sig']}  p={rw['pval']:.4f}")

# ====================================================================
# 2. 分规模交互项检验
# ====================================================================
print("\n2. 分规模交互项检验...")
for grp_name, grp_label in [('全部企业', 'all'), ('大企业', 'large'), ('小企业', 'small')]:
    if grp_name == '全部企业':
        sub = qp
    else:
        sub = qp[qp['size_group'] == grp_label]
    p_int = sub.copy()
    for v_col in core_vars_col:
        p_int[f'{v_col}_x_post'] = p_int[v_col] * p_int['post2018']
    x_b = core_vars_disp + [f'{v}_x_post' for v in core_vars_col] + controls
    print(f"\n  {grp_name}:")
    for y in ['ROA', 'ROE']:
        r = run_panelols(y, x_b, p_int, f'VB_{grp_label}_{y}')
        if r is not None:
            all_results.append(r)
            print(f"    {y}:")
            for v_col in core_vars_col:
                row = r[r['variable'] == f'{v_col}_x_post']
                if len(row):
                    rw = row.iloc[0]
                    vd = col_to_disp_q.get(v_col, v_col)
                    ds = '+' if rw['coef'] > 0 else ''
                    print(f"      {vd:35s} × post: {ds}{rw['coef']:.5f}{rw['sig']}  p={rw['pval']:.4f}")

if all_results:
    rdf = pd.concat(all_results, ignore_index=True)
    rdf.to_excel(f'{OUT_DIR}/size_heterogeneity_results.xlsx', index=False)
    print(f"\n  ✓ {OUT_DIR}/size_heterogeneity_results.xlsx")

# ====================================================================
# 3. 可视化
# ====================================================================
print("\n3. 生成对比图...")

# 3.1 交互项系数对比
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
zh_names = ['供应商广度', '供应商行业多样性', '竞争对手广度', '竞争对手行业多样性']
for vi, (core_col, zh_name) in enumerate(zip(core_vars_col, zh_names)):
    ax = axes[vi // 2, vi % 2]
    groups = [('all', '全样本', 'gray'), ('large', '大企业', 'steelblue'), ('small', '小企业', 'coral')]
    for gi, (gl, gn, gc) in enumerate(groups):
        r_sub = rdf[(rdf['model']==f'VB_{gl}_ROA') & (rdf['variable']==f'{core_col}_x_post')]
        if len(r_sub):
            rw = r_sub.iloc[0]
            ci = rw['se'] * 1.96
            ax.barh(gn, rw['coef'], xerr=ci, color=gc, alpha=0.7, capsize=4)
            ax.text(rw['coef'] + ci + 0.02, gn,
                   f'{rw["coef"]:.3f}{rw["sig"]}', va='center', fontsize=9)
    ax.axvline(x=0, color='gray', lw=1)
    ax.set_xlabel('交互项系数', fontproperties=zh_font, fontsize=10)
    ax.set_title(f'{zh_name} × post', fontproperties=zh_font, fontsize=12)
    ax.grid(True, alpha=0.2, axis='x')
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/size_interaction_comparison.png', dpi=200, bbox_inches='tight')
plt.close()
print(f"  ✓ {OUT_DIR}/size_interaction_comparison.png")

# 3.2 边际效应变化图
fig, ax = plt.subplots(figsize=(8, 6))
margin_results = {}
for grp_label, grp_color, grp_marker in [('large','steelblue','o'), ('small','coral','s')]:
    sub = qp[qp['size_group'] == grp_label]
    p_int = sub.copy()
    for v_col in core_vars_col:
        p_int[f'{v_col}_x_post'] = p_int[v_col] * p_int['post2018']
    x_margin = ['supplier_count_log', f'sup_breadth_q_x_post'] + controls
    r = run_panelols('ROA', x_margin, p_int, f'margin_{grp_label}')
    if r is not None:
        b_pre = r[r['variable']=='supplier_count_log']['coef'].values[0]
        b_post = b_pre + r[r['variable']=='sup_breadth_q_x_post']['coef'].values[0]
        margin_results[grp_label] = (b_pre, b_post)

if margin_results:
    x_pos = [0, 1]
    labels = ['贸易战前', '贸易战后']
    for grp_label, grp_color in [('large','steelblue'), ('small','coral')]:
        if grp_label in margin_results:
            vals = margin_results[grp_label]
            ax.plot(x_pos, vals, '-', color=grp_color, linewidth=2, marker='o', markersize=8,
                   label=f'{"大企业" if grp_label=="large" else "小企业"}')
    ax.set_xticks(x_pos); ax.set_xticklabels(labels, fontproperties=zh_font, fontsize=11)
    ax.set_ylabel('供应商广度对ROA的边际效应', fontproperties=zh_font, fontsize=11)
    ax.set_title('供应商广度效应: 大企业 vs 小企业', fontproperties=zh_font, fontsize=13)
    ax.legend(prop=zh_font, fontsize=10)
    ax.axhline(y=0, color='gray', ls='--', lw=1)
    ax.grid(True, alpha=0.2)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/size_margin_comparison.png', dpi=200, bbox_inches='tight')
plt.close()
print(f"  ✓ {OUT_DIR}/size_margin_comparison.png")

# ====================================================================
# 4. 报告
# ====================================================================
print("\n4. 生成报告...")
n_large = qp[qp['size_group']=='large'].index.get_level_values(0).nunique()
n_small = qp[qp['size_group']=='small'].index.get_level_values(0).nunique()
report = [
    "# 企业规模分组异质性分析报告",
    "",
    "## 一、分组方法",
    "",
    f"按企业季度面板size_q的均值取中位数({size_median:.4f})分组",
    f"- 大企业: {n_large} 家 (size_q > {size_median:.4f})",
    f"- 小企业: {n_small} 家 (size_q ≤ {size_median:.4f})",
    "",
    "## 二、预期逻辑",
    "",
    "大企业资源多、供应链管理能力强 → 贸易战冲击效应应更小",
    "小企业资源有限、缺乏议价能力 → 贸易战冲击效应应更大",
    "",
]
with open(f'{OUT_DIR}/size_heterogeneity_summary.md', 'w', encoding='utf-8-sig') as f:
    f.write('\n'.join(report))
print(f"  ✓ {OUT_DIR}/size_heterogeneity_summary.md")
print("\n完成!")
