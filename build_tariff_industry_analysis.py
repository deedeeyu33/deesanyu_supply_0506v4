"""
贸易战补充分析：分行业异质性 + 边际效应图
基于季度面板 (2010Q1-2020Q4)
"""
import pandas as pd, numpy as np, os, warnings, gc
warnings.filterwarnings('ignore')
from linearmodels.panel import PanelOLS
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

DATA_DIR = '/Users/deesanyu/pythonproject/deesan_supply_0506v4/data'
OUT_DIR  = '/Users/deesanyu/pythonproject/deesan_supply_0506v4/output/tariff_industry'
FONT_PATH = '/Users/deesanyu/pythonproject/deesan_supply_0506v4/simhei.ttf'
os.makedirs(OUT_DIR, exist_ok=True)
zh_font = FontProperties(fname=FONT_PATH)
plt.rcParams['axes.unicode_minus'] = False

# ====================================================================
# 加载季度面板
# ====================================================================
print("=" * 60)
print("贸易战补充分析：分行业异质性 + 边际效应图")
print("=" * 60)

print("\n1. 加载季度面板...")
qp = pd.read_parquet(f'{DATA_DIR}/../output/quarterly_panel/firm_quarterly_panel.parquet')
print(f"  季度面板: {len(qp):,} 行, {qp['ISIN'].nunique():,} 企业")

# ====================================================================
# 合并行业分类
# ====================================================================
print("\n2. 合并行业分类...")
ar = pd.read_excel(f'{DATA_DIR}/china-quarterly-assets.xlsx', header=None)
ind_info = ar.iloc[6:, [10, -1]].copy()
ind_info.columns = ['industry_l1', 'ISIN']
ind_info = ind_info.dropna(subset=['ISIN'])
ind_info['ISIN'] = ind_info['ISIN'].astype(str).str.strip()
ind_info = ind_info.drop_duplicates('ISIN')

qp = qp.merge(ind_info, on='ISIN', how='left')
print(f"  合并后: {qp['ISIN'].nunique():,} 企业")

# 制造业分类
manu_list = ['Industrials', 'Non-Energy Materials', 'Technology',
             'Consumer Cyclicals', 'Consumer Non-Cyclicals']
qp['is_manu'] = qp['industry_l1'].isin(manu_list)
print(f"  制造业: {qp[qp['is_manu']]['ISIN'].nunique():,} 企业")
print(f"  非制造业: {qp[~qp['is_manu']]['ISIN'].nunique():,} 企业")
print(f"\n  各行业企业数:")
print(qp.groupby('industry_l1')['ISIN'].nunique().to_string())

# ====================================================================
# 变量映射
# ====================================================================
var_map_q = {
    'ROA': 'roa_w_q', 'ROE': 'roe_w_q',
    'supplier_count_log': 'sup_breadth_q', 'supplier_industry_count': 'sup_ind_div_q',
    'competitor_count_log': 'comp_breadth_q', 'competitor_industry_count': 'comp_ind_div_q',
    'Size': 'size_q', 'Leverage': 'lev_q', 'Growth': 'growth_w_q',
}
col_to_disp_q = {v: k for k, v in var_map_q.items()}
controls = ['Size', 'Leverage', 'Growth']
core_vars_disp = ['supplier_count_log', 'supplier_industry_count',
                  'competitor_count_log', 'competitor_industry_count']
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
# 准备面板索引数据
# ====================================================================
qp['quarter_dt'] = pd.to_datetime(qp['quarter'])
qp = qp.sort_values(['ISIN', 'qi']).set_index(['ISIN', 'quarter_dt'])

print("\n3. 分行业基准回归 (M3)...")
# ====================================================================
# 分行业 M3 基准
# ====================================================================
all_results = []
for grp_name, grp_label in [('全部企业', 'all'), ('制造业', 'manu'), ('非制造业', 'non_manu')]:
    if grp_name == '全部企业':
        sub = qp
    else:
        sub = qp[qp['is_manu'] == (grp_name == '制造业')]

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
# 分行业交互项检验 (Version B)
# ====================================================================
print("\n4. 分行业交互项检验...")
for grp_name, grp_label in [('全部企业', 'all'), ('制造业', 'manu'), ('非制造业', 'non_manu')]:
    if grp_name == '全部企业':
        sub = qp
    else:
        sub = qp[qp['is_manu'] == (grp_name == '制造业')]

    p_int = sub.copy()
    for v_col in core_vars_col:
        p_int[f'{v_col}_x_post'] = p_int[v_col] * p_int['post2018']
    x_b = core_vars_disp + [f'{v}_x_post' for v in core_vars_col] + controls

    print(f"\n  {grp_name} 交互项:")
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
    with pd.ExcelWriter(f'{OUT_DIR}/industry_heterogeneity_results.xlsx') as writer:
        rdf.to_excel(writer, sheet_name='all_results', index=False)
    print(f"\n  ✓ {OUT_DIR}/industry_heterogeneity_results.xlsx")

# ====================================================================
# 5. 边际效应图 (Margin Plot)
# ====================================================================
print("\n5. 生成边际效应图...")

# 准备交互项模型数据
p_int = qp.copy()
for v_col in core_vars_col:
    p_int[f'{v_col}_x_post'] = p_int[v_col] * p_int['post2018']

# 对每个核心变量做交互项模型，提取边际效应
margin_results = []
for core_disp, core_col in zip(core_vars_disp, core_vars_col):
    x_margin = [core_disp, f'{core_col}_x_post'] + [c for c in controls if c != core_disp]
    r = run_panelols('ROA', x_margin, p_int, f'margin_{core_col}')
    if r is not None:
        # 提取系数
        beta_core = r[r['variable'] == core_disp]['coef'].values[0]
        beta_interact = r[r['variable'] == f'{core_col}_x_post']['coef'].values[0]
        # 提取标准误
        se_core = r[r['variable'] == core_disp]['se'].values[0]
        se_interact = r[r['variable'] == f'{core_col}_x_post']['se'].values[0]
        margin_results.append({
            'var_disp': core_disp, 'var_col': core_col,
            'beta_pre': beta_core, 'se_pre': se_core,
            'beta_post': beta_core + beta_interact,
            'se_post': np.sqrt(se_core**2 + se_interact**2),
        })
        print(f"  {core_disp:35s} | pre={beta_core:+.5f} | post={beta_core+beta_interact:+.5f}")

# 生成漂亮的边际效应图
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
var_labels_zh = ['供应商广度', '供应商行业多样性', '竞争对手广度', '竞争对手行业多样性']

for idx, (ax, mr) in enumerate(zip(axes.flat, margin_results)):
    # 取实际数据范围
    d = qp[mr['var_col']].dropna()
    x_vals = np.linspace(d.quantile(0.05), d.quantile(0.95), 100)

    # Pre-2018: 边际效应 = beta_core (常数)
    me_pre = np.full_like(x_vals, mr['beta_pre'])
    ci_pre = mr['se_pre'] * 1.96

    # Post-2018: 边际效应 = beta_core + beta_interact (常数)
    me_post = np.full_like(x_vals, mr['beta_post'])
    ci_post = mr['se_post'] * 1.96

    # 画图
    ax.plot(x_vals, me_pre, 'b-', linewidth=2.5, label='贸易战前 (2010-2017)')
    ax.fill_between(x_vals, me_pre - ci_pre, me_pre + ci_pre, color='blue', alpha=0.1)
    ax.plot(x_vals, me_post, 'r-', linewidth=2.5, label='贸易战后 (2018-2020)')
    ax.fill_between(x_vals, me_post - ci_post, me_post + ci_post, color='red', alpha=0.1)
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.8)

    # 标注文字
    ax.text(0.05, 0.95, f'pre: {mr["beta_pre"]:+.4f}', transform=ax.transAxes,
            fontsize=10, color='blue', verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
    ax.text(0.05, 0.85, f'post: {mr["beta_post"]:+.4f}', transform=ax.transAxes,
            fontsize=10, color='red', verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

    ax.set_xlabel(var_labels_zh[idx], fontproperties=zh_font, fontsize=12)
    ax.set_ylabel('对ROA的边际效应', fontproperties=zh_font, fontsize=11)
    ax.set_title(f'{var_labels_zh[idx]}的边际效应变化', fontproperties=zh_font, fontsize=13)
    ax.legend(prop=zh_font, fontsize=9, loc='lower left')
    ax.grid(True, alpha=0.2)

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/margin_plot.png', dpi=200, bbox_inches='tight')
plt.close()
print(f"  ✓ {OUT_DIR}/margin_plot.png")

# ====================================================================
# 6. 变化幅度图 (更直观的展示)
# ====================================================================
print("\n6. 生成系数变化幅度图...")
fig, ax = plt.subplots(figsize=(10, 6))
if margin_results:
    var_names = var_labels_zh
    pre_v = [mr['beta_pre'] for mr in margin_results]
    post_v = [mr['beta_post'] for mr in margin_results]
    pre_ci = [mr['se_pre'] * 1.96 for mr in margin_results]
    post_ci = [mr['se_post'] * 1.96 for mr in margin_results]

    xp = np.arange(len(var_names))
    w = 0.35
    bars1 = ax.bar(xp - w/2, pre_v, w, yerr=pre_ci, capsize=4,
                   color='steelblue', alpha=0.75, label='贸易战前')
    bars2 = ax.bar(xp + w/2, post_v, w, yerr=post_ci, capsize=4,
                   color='coral', alpha=0.75, label='贸易战后')

    ax.axhline(y=0, color='gray', lw=1)
    ax.set_xticks(xp)
    ax.set_xticklabels(var_names, fontproperties=zh_font, fontsize=11)
    ax.set_ylabel('对ROA的边际效应', fontproperties=zh_font, fontsize=12)
    ax.set_title('贸易战前后核心变量的边际效应变化(含95%CI)', fontproperties=zh_font, fontsize=14)
    ax.legend(prop=zh_font, fontsize=11)
    ax.grid(True, alpha=0.2, axis='y')

    # 标注显著性
    for i, mr in enumerate(margin_results):
        pv = None
        for _, row in pd.concat(all_results, ignore_index=True).iterrows():
            if f'{mr["var_col"]}_x_post' in str(row['variable']):
                pv = row['pval']
                break
        if pv is not None:
            star = ''
            if pv < 0.01: star = '***'
            elif pv < 0.05: star = '**'
            elif pv < 0.1: star = '*'
            ax.annotate(f'p={pv:.3f}{star}', (xp[i], post_v[i] + post_ci[i] + 0.05),
                       ha='center', fontsize=8, color='coral')

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/margin_coef_change.png', dpi=200, bbox_inches='tight')
plt.close()
print(f"  ✓ {OUT_DIR}/margin_coef_change.png")

# ====================================================================
# 7. 分行业交互项可视化
# ====================================================================
print("\n7. 分行业交互项系数对比图...")
fig, ax = plt.subplots(figsize=(12, 7))
groups = [(qp, '全样本', 'gray'), (qp[qp['is_manu']], '制造业', 'steelblue'),
          (qp[~qp['is_manu']], '非制造业', 'coral')]
group_labels = ['全样本', '制造业', '非制造业']
colors = ['gray', 'steelblue', 'coral']

# 对每个核心变量画分组图
for vi, (core_disp, core_col, zh_name) in enumerate(zip(core_vars_disp, core_vars_col, var_labels_zh)):
    ax_sub = fig.add_subplot(2, 2, vi+1)
    for gi, (gdata, glabel, gcolor) in enumerate(groups):
        pi = gdata.copy()
        pi[f'{core_col}_x_post'] = pi[core_col] * pi['post2018']
        x_s = [core_disp, f'{core_col}_x_post'] + [c for c in controls]
        r = run_panelols('ROA', x_s, pi, f'group_{glabel}_{core_col}')
        if r is not None:
            row = r[r['variable'] == f'{core_col}_x_post']
            if len(row):
                rw = row.iloc[0]
                ci = rw['se'] * 1.96
                ax_sub.barh(glabel, rw['coef'], xerr=ci, color=gcolor, alpha=0.7, capsize=4)
                ax_sub.text(rw['coef'] + ci + 0.02, glabel,
                           f'{rw["coef"]:.3f}{rw["sig"]}', va='center', fontsize=9)
    ax_sub.axvline(x=0, color='gray', lw=1)
    ax_sub.set_xlabel('交互项系数', fontproperties=zh_font, fontsize=10)
    ax_sub.set_title(f'{zh_name} × post (按行业)', fontproperties=zh_font, fontsize=12)
    ax_sub.grid(True, alpha=0.2, axis='x')

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/industry_interaction_comparison.png', dpi=200, bbox_inches='tight')
plt.close()
print(f"  ✓ {OUT_DIR}/industry_interaction_comparison.png")

# ====================================================================
# 8. 生成报告
# ====================================================================
print("\n8. 生成分析报告...")
# qp在set_index后没有ISIN列了，用index level
all_firms = qp.index.get_level_values(0).nunique()
manu_firms = qp[qp['is_manu']].index.get_level_values(0).nunique()
non_manu_firms = qp[~qp['is_manu']].index.get_level_values(0).nunique()

report = [
    "# 贸易战补充分析报告：分行业异质性 + 边际效应",
    "",
    "## 一、分析目的",
    "",
    "本分析包含两个部分：",
    "1. **分行业异质性**：检验制造业与非制造业在贸易战调节效应上的差异",
    "2. **边际效应图**：直观展示贸易战前后核心变量对ROA的边际效应变化",
    "",
    "## 二、行业分布",
    "",
    f"季度面板共 {all_firms:,} 家企业，全部具有RBICS L1行业分类。",
    "其中：",
    f"- 制造业: {manu_firms:,} 家",
    f"- 非制造业: {non_manu_firms:,} 家",
    "",
]
with open(f'{OUT_DIR}/analysis_summary.md', 'w', encoding='utf-8-sig') as f:
    f.write('\n'.join(report))

print(f"  ✓ {OUT_DIR}/analysis_summary.md")
print("\n" + "=" * 60)
print("分析完成!")
print(f"输出目录: {OUT_DIR}")
print("=" * 60)
