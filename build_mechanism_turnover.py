"""
机制检验：运营效率（资产周转率）路径
数据来源：季度面板
"""
import pandas as pd, numpy as np, os, warnings
warnings.filterwarnings('ignore')
from linearmodels.panel import PanelOLS
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

OUT_DIR  = '/Users/deesanyu/pythonproject/deesan_supply_0506v4/output/mechanism_turnover'
FONT_PATH = '/Users/deesanyu/pythonproject/deesan_supply_0506v4/simhei.ttf'
os.makedirs(OUT_DIR, exist_ok=True)
zh_font = FontProperties(fname=FONT_PATH)
plt.rcParams['axes.unicode_minus'] = False

# ====================================================================
# 加载数据
# ====================================================================
print("=" * 60)
print("机制检验：运营效率路径")
print("=" * 60)
qp = pd.read_parquet('/Users/deesanyu/pythonproject/deesan_supply_0506v4/output/quarterly_panel/firm_quarterly_panel.parquet')
qp['quarter_dt'] = pd.to_datetime(qp['quarter'])
qp = qp.sort_values(['ISIN', 'qi']).set_index(['ISIN', 'quarter_dt'])

# 计算资产周转率 = sales / assets
qp['turnover_q'] = qp['sales_q'] / qp['assets_q'].replace(0, np.nan)
# 缩尾 1%/99%
lo, hi = qp['turnover_q'].quantile([0.01, 0.99]).values
qp['turnover_w_q'] = qp['turnover_q'].clip(lo, hi)
print(f"周转率: N={qp['turnover_w_q'].notna().sum():,}, "
      f"均值={qp['turnover_w_q'].mean():.4f}, 中位数={qp['turnover_w_q'].median():.4f}")

var_map_q = {
    'ROA': 'roa_w_q', 'ROE': 'roe_w_q', 'Turnover': 'turnover_w_q',
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
    n = len(sub); n_f = sub.index.get_level_values(0).nunique()
    n_p = sub.index.get_level_values(1).nunique()
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
            star = ''; sig = ''
            if not np.isnan(pv):
                if pv < 0.01: star = '***'; sig = '***'
                elif pv < 0.05: star = '**'; sig = '**'
                elif pv < 0.1: star = '*'; sig = '*'
            rows.append({
                'model': label, 'variable': v_disp,
                'coef': round(coef, 5), 'se': round(se, 5),
                't': round(t, 3), 'pval': round(pv, 4),
                'sig': star, 'N': n, 'N_firms': n_f,
                'N_periods': n_p,
                'rsq_within': round(res.rsquared_within, 4),
            })
        return pd.DataFrame(rows)
    except Exception as e:
        print(f"  ⚠ {label}: {e}")
        return None

all_results = []

# ====================================================================
# 1. 供应链结构 → 运营效率
# ====================================================================
print("\n1. 供应链结构 → 运营效率")
for x_list, label in [
    (['supplier_count_log','supplier_industry_count']+controls, 'M_sup_turnover'),
    (['competitor_count_log','competitor_industry_count']+controls, 'M_comp_turnover'),
    (core_vars_disp+controls, 'M_full_turnover'),
]:
    r = run_panelols('Turnover', x_list, qp, label)
    if r is not None:
        all_results.append(r)
        print(f"  {label}: N={r.iloc[0]['N']:,}, R²={r.iloc[0]['rsq_within']:.4f}")
        for _, rw in r[r['variable'].isin(core_vars_disp)].iterrows():
            ds = '+' if rw['coef'] > 0 else ''
            print(f"    {rw['variable']:35s} {ds}{rw['coef']:.5f}{rw['sig']}  p={rw['pval']:.4f}")

# ====================================================================
# 2. 贸易战交互项 → 运营效率
# ====================================================================
print("\n2. 贸易战交互项 → 运营效率")
p_int = qp.copy()
for v_col in core_vars_col:
    p_int[f'{v_col}_x_post'] = p_int[v_col] * p_int['post2018']
x_b = core_vars_disp + [f'{v}_x_post' for v in core_vars_col] + controls
r = run_panelols('Turnover', x_b, p_int, 'VB_turnover')
if r is not None:
    all_results.append(r)
    print(f"  N={r.iloc[0]['N']:,}, R²={r.iloc[0]['rsq_within']:.4f}")
    for v_col in core_vars_col:
        row = r[r['variable']==f'{v_col}_x_post']
        if len(row):
            rw = row.iloc[0]
            vd = col_to_disp_q.get(v_col, v_col)
            ds = '+' if rw['coef'] > 0 else ''
            print(f"    {vd:35s} × post: {ds}{rw['coef']:.5f}{rw['sig']}  p={rw['pval']:.4f}")

# ====================================================================
# 3. 基准 + 周转率作为控制变量
# ====================================================================
print("\n3. 基准回归 + 周转率作为控制变量")
base_x2 = core_vars_disp + controls + ['Turnover']
r = run_panelols('ROA', base_x2, qp, 'M3_with_turnover')
if r is not None:
    all_results.append(r)
    print(f"  N={r.iloc[0]['N']:,}, R²={r.iloc[0]['rsq_within']:.4f}")
    for _, rw in r[r['variable'].isin(core_vars_disp + ['Turnover'])].iterrows():
        ds = '+' if rw['coef'] > 0 else ''
        print(f"    {rw['variable']:35s} {ds}{rw['coef']:.5f}{rw['sig']}  p={rw['pval']:.4f}")

# ====================================================================
# 4. 按周转率分组
# ====================================================================
print("\n4. 按运营效率分组交互项")
firm_turnover = qp.groupby(level='ISIN')['turnover_w_q'].mean()
t_median = firm_turnover.median()
print(f"  周转率中位数: {t_median:.4f}")
qp['high_turnover'] = qp.index.get_level_values(0).isin(
    firm_turnover[firm_turnover > t_median].index)

for grp_name, grp_bool in [('高周转率', True), ('低周转率', False)]:
    sub = qp[qp['high_turnover'] == grp_bool]
    p_sub = sub.copy()
    for v_col in core_vars_col:
        p_sub[f'{v_col}_x_post'] = p_sub[v_col] * p_sub['post2018']
    x_b = core_vars_disp + [f'{v}_x_post' for v in core_vars_col] + controls
    r = run_panelols('ROA', x_b, p_sub, f'VB_{grp_name}')
    if r is not None:
        all_results.append(r)
        print(f"\n  {grp_name}: N={r.iloc[0]['N']:,}, R²={r.iloc[0]['rsq_within']:.4f}")
        for v_col in core_vars_col:
            row = r[r['variable']==f'{v_col}_x_post']
            if len(row):
                rw = row.iloc[0]
                vd = col_to_disp_q.get(v_col, v_col)
                ds = '+' if rw['coef'] > 0 else ''
                print(f"    {vd:35s} × post: {ds}{rw['coef']:.5f}{rw['sig']}  p={rw['pval']:.4f}")

# ====================================================================
# 保存
# ====================================================================
if all_results:
    rdf = pd.concat(all_results, ignore_index=True)
    rdf.to_excel(f'{OUT_DIR}/turnover_mechanism_results.xlsx', index=False)
    print(f"\n  ✓ {OUT_DIR}/turnover_mechanism_results.xlsx")

# ====================================================================
# 报告
# ====================================================================
print("\n生成报告...")
report = [
    "# 机制检验报告：运营效率路径",
    "",
    "## 一、检验逻辑",
    "",
    "供应商广度增加 → 供应链管理复杂度上升 → 运营效率(资产周转率)下降 → ROA下降",
    "",
    "## 二、结果",
    "",
    "### 2.1 供应链结构 → 运营效率",
    "",
    "预期：供应商广度越大，周转率越低（管理复杂度）",
    "",
    "### 2.2 贸易战交互项 → 运营效率",
    "",
    "预期：贸易战后，供应商广度对周转率的负向效应增强",
    "",
    "### 2.3 加入周转率控制变量后的基准回归",
    "",
    "预期：加入周转率后，供应商广度的系数应减小（部分被周转率解释）",
    "",
    "### 2.4 按周转率分组的贸易战交互项",
    "",
    "预期：低周转率组（效率低）的贸易战调节效应更大",
    "",
]
with open(f'{OUT_DIR}/turnover_mechanism_summary.md', 'w', encoding='utf-8-sig') as f:
    f.write('\n'.join(report))
print(f"  ✓ {OUT_DIR}/turnover_mechanism_summary.md")
print("\n完成!")
