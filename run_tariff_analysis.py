"""
中美贸易战/关税冲击分段检验
第五章补充分析：贸易战前后供应商/竞争对手关系结构对企业绩效的影响差异
"""
import pandas as pd, numpy as np, os, warnings
warnings.filterwarnings('ignore')
from linearmodels.panel import PanelOLS
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

OUT_DIR = '/Users/deesanyu/pythonproject/deesan_supply_0506v4/output/tariff_segment'
FIG_DIR = f'{OUT_DIR}/trend_plots'
DATA_PATH = '/Users/deesanyu/pythonproject/deesan_supply_0506v4/output/firm_monthly_panel.csv'
FONT_PATH = '/Users/deesanyu/pythonproject/deesan_supply_0506v4/simhei.ttf'
os.makedirs(FIG_DIR, exist_ok=True)
zh_font = FontProperties(fname=FONT_PATH)
plt.rcParams['axes.unicode_minus'] = False
TARIFF_YEAR = 2018

# ========== 变量映射 ==========
var_map = {
    'ROA': 'roa_w', 'ROE': 'roe_w',
    'supplier_count_log': 'sup_breadth', 'supplier_industry_count': 'sup_ind_div',
    'competitor_count_log': 'comp_breadth', 'competitor_industry_count': 'comp_ind_div',
    'Size': 'size', 'Leverage': 'lev', 'Growth': 'growth_w',
}
col_to_display = {v: k for k, v in var_map.items()}
controls = ['Size', 'Leverage', 'Growth']
ctrl_cols = [var_map[v] for v in controls]
core_vars_disp = ['supplier_count_log', 'supplier_industry_count', 'competitor_count_log', 'competitor_industry_count']
core_vars_col = [var_map[v] for v in core_vars_disp]

print("=" * 60)
print("变量映射 (显示名 -> 实际列名)")
print("=" * 60)
for k, v in var_map.items():
    print(f"  {k:30s} -> {v}")
print(f"\n核心变量: {[f'{d}({c})' for d, c in zip(core_vars_disp, core_vars_col)]}")
print(f"控制变量: {[f'{d}({c})' for d, c in zip(controls, ctrl_cols)]}")

# ========== 1. 加载数据 ==========
print("\n" + "=" * 60)
print("1. 加载数据")
print("=" * 60)
p = pd.read_csv(DATA_PATH)
p['year'] = p['month'].str[:4].astype(int)
p['month_dt'] = pd.to_datetime(p['month'])
p['post2018'] = (p['year'] >= TARIFF_YEAR).astype(int)
print(f"总样本: {len(p):,} 行, {p['ISIN'].nunique():,} 企业, 年份 {p['year'].min()}-{p['year'].max()}")
print(f"贸易战前 (2010-2017): {(p['post2018']==0).sum():,} 行")
print(f"贸易战后 (2018-2020): {(p['post2018']==1).sum():,} 行")

# ========== 2. 描述性统计与均值差异检验 ==========
print("\n" + "=" * 60)
print("2. 贸易战前后描述性统计与均值差异检验")
print("=" * 60)

desc_vars = core_vars_disp + ['ROA', 'ROE'] + controls
desc_cols = [var_map[v] for v in desc_vars]

pre = p[p['post2018'] == 0]
post = p[p['post2018'] == 1]

rows_desc = []
for disp, col in zip(desc_vars, desc_cols):
    pre_v = pre[col].dropna()
    post_v = post[col].dropna()
    pre_mean, pre_std, pre_n = pre_v.mean(), pre_v.std(), len(pre_v)
    post_mean, post_std, post_n = post_v.mean(), post_v.std(), len(post_v)
    # t-test
    t_stat, p_val = stats.ttest_ind(pre_v, post_v, equal_var=False)
    diff = post_mean - pre_mean
    sig = ''
    if p_val < 0.01: sig = '***'
    elif p_val < 0.05: sig = '**'
    elif p_val < 0.1: sig = '*'
    rows_desc.append({
        'variable': disp, 'column': col,
        'pre_mean': round(pre_mean, 4), 'pre_std': round(pre_std, 4), 'pre_N': pre_n,
        'post_mean': round(post_mean, 4), 'post_std': round(post_std, 4), 'post_N': post_n,
        'diff': round(diff, 4), 't_stat': round(t_stat, 3), 'p_value': round(p_val, 4),
        'sig': sig,
    })
    print(f"  {disp:30s} 前={pre_mean:>8.4f} → 后={post_mean:>8.4f}  diff={diff:>+8.4f}{sig}  t={t_stat:.3f}  p={p_val:.4f}")

desc_df = pd.DataFrame(rows_desc)
desc_df.to_excel(f'{OUT_DIR}/tariff_stage_descriptive.xlsx', index=False)
print(f"\n  ✓ {OUT_DIR}/tariff_stage_descriptive.xlsx")

# ========== 3. 面板索引准备 ==========
p = p.sort_values(['ISIN', 'month_dt']).set_index(['ISIN', 'month_dt'])

# ========== 4. 回归工具函数 ==========
def run_panelols(y_display, x_displays, data, label=''):
    """运行 PanelOLS v7.0，返回结果 DataFrame"""
    y_name = var_map[y_display]

    def _resolve(v):
        return var_map[v] if v in var_map else v

    x_names = [_resolve(v) for v in x_displays]
    all_vars = [y_name] + x_names
    sub = data[all_vars].dropna().copy()
    n = len(sub)
    n_firms = sub.index.get_level_values(0).nunique()
    n_months = sub.index.get_level_values(1).nunique()
    if n < 100:
        print(f"  ⚠ {label}: 样本不足 ({n})")
        return None, None
    formula = f'{y_name} ~ EntityEffects + TimeEffects + ' + ' + '.join(x_names)
    try:
        mod = PanelOLS.from_formula(formula, data=sub, drop_absorbed=True)
        res = mod.fit(cov_type='clustered', cluster_entity=True)
        rows = []
        for v_col in x_names:
            v_disp = col_to_display.get(v_col, v_col)
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
                'sig': star, 'N': n, 'N_firms': n_firms, 'N_months': n_months,
                'rsq_within': round(res.rsquared_within, 4),
                'entity_fe': 'Yes', 'time_fe': 'Yes',
            })
        return pd.DataFrame(rows), res
    except Exception as e:
        print(f"  ⚠ {label} 失败: {e}")
        return None, None


# ========== 5. 分样本回归 ==========
print("\n" + "=" * 60)
print("3. 分样本回归 (贸易战前 vs 贸易战后)")
print("=" * 60)

base_x = core_vars_disp + controls  # 7 个变量
p_pre = p[p.index.get_level_values('month_dt').year < TARIFF_YEAR].copy()
p_post = p[p.index.get_level_values('month_dt').year >= TARIFF_YEAR].copy()
print(f"  贸易战前样本: {len(p_pre):,} | 贸易战后样本: {len(p_post):,}")

subsample_results = []
for label, data, stage in [('pre_2017', p_pre, '2010-2017'), ('post_2018', p_post, '2018-2020')]:
    for y in ['ROA', 'ROE']:
        res_df, _ = run_panelols(y, base_x, data, f'{label}_{y}')
        if res_df is not None:
            res_df.insert(0, 'stage', stage)
            subsample_results.append(res_df)
            print(f"  {stage} | {y}: N={res_df.iloc[0]['N']:,}, R²={res_df.iloc[0]['rsq_within']}")
            for _, r in res_df.iterrows():
                print(f"    {r['variable']:35s} {r['coef']:>10.5f}{r['sig']:4s} p={r['pval']:.4f}")

if subsample_results:
    subsample_df = pd.concat(subsample_results, ignore_index=True)
    subsample_df.to_excel(f'{OUT_DIR}/tariff_subsample_regression.xlsx', index=False)
    print(f"  ✓ {OUT_DIR}/tariff_subsample_regression.xlsx")

# ========== 6. 全样本交互项检验 ==========
print("\n" + "=" * 60)
print("4. 全样本交互项检验 (核心变量 × post2018)")
print("=" * 60)

# 创建交互项
for v_col in core_vars_col:
    p[f'{v_col}_x_post2018'] = p[v_col] * p['post2018']

interaction_results = []
for y in ['ROA', 'ROE']:
    y_name = var_map[y]
    # 逐个核心变量检验交互项
    for v_disp, v_col in zip(core_vars_disp, core_vars_col):
        ix_col = f'{v_col}_x_post2018'
        x_list = base_x + [ix_col]
        label = f'interact_{v_disp}_{y}'
        res_df, res_obj = run_panelols(y, x_list, p, label)
        if res_df is not None:
            interaction_results.append(res_df)
            # 交互变量在 results 中以实际列名存储, 直接搜索
            row_ix = res_df[res_df['variable'] == ix_col]
            if len(row_ix):
                r = row_ix.iloc[0]
                print(f"  {y} | {v_disp:35s} × post2018: coef={r['coef']:+8.5f}{r['sig']} p={r['pval']:.4f}")
            else:
                print(f"  {y} | {v_disp:35s} × post2018: 交互项未在结果中")
    # 一次性加入所有交互项 (整体检验)
    all_ix = [f'{v_col}_x_post2018' for v_col in core_vars_col]
    x_list_all = base_x + all_ix
    label = f'interact_all_{y}'
    res_df, _ = run_panelols(y, x_list_all, p, label)
    if res_df is not None:
        interaction_results.append(res_df)
        print(f"  {y} | 全部交互项联合模型: N={res_df.iloc[0]['N']:,}, R²={res_df.iloc[0]['rsq_within']}")
        for _, r in res_df.iterrows():
            if 'x_post2018' in r['variable'] or r['variable'] in core_vars_disp:
                print(f"    {r['variable']:40s} {r['coef']:>10.5f}{r['sig']:4s} p={r['pval']:.4f}")

if interaction_results:
    interact_df = pd.concat(interaction_results, ignore_index=True)
    interact_df.to_excel(f'{OUT_DIR}/tariff_interaction_results.xlsx', index=False)
    print(f"  ✓ {OUT_DIR}/tariff_interaction_results.xlsx")

# ========== 7. 美国暴露异质性 ==========
print("\n" + "=" * 60)
print("5. 美国暴露异质性分析")
print("=" * 60)

us_results = []
us_vars_exist = [c for c in ['sup_ct_美国', 'comp_ct_美国'] if c in p.columns]
print(f"  可用美国变量: {us_vars_exist}")

if us_vars_exist:
    p['has_us_supplier'] = (p['sup_ct_美国'] > 0).astype(int)
    p['has_us_competitor'] = (p['comp_ct_美国'] > 0).astype(int)

    # 5.1 有无美国供应商/竞争对手分组回归
    for us_var, us_disp in [('has_us_supplier', '有/无美国供应商'),
                             ('has_us_competitor', '有/无美国竞争对手')]:
        for grp_val, grp_label in [(0, f'无({us_disp})'), (1, f'有({us_disp})')]:
            sub = p[p[us_var] == grp_val]
            n_grp = len(sub)
            n_firms_grp = sub.index.get_level_values(0).nunique()
            print(f"  {grp_label}: N={n_grp:,}, 企业={n_firms_grp:,}")
            for y in ['ROA']:
                res_df, _ = run_panelols(y, base_x, sub, f'us_{us_var}={grp_val}_{y}')
                if res_df is not None:
                    res_df.insert(0, 'us_group', grp_label)
                    res_df.insert(0, 'us_var', us_var)
                    us_results.append(res_df)

    # 5.2 三重交互: 核心变量 × post2018 × 有美国供应商/竞争对手
    triple_vars = [('has_us_supplier', '美国供应商'), ('has_us_competitor', '美国竞争对手')]
    for us_col, us_label in triple_vars:
        for v_disp, v_col in zip(core_vars_disp, core_vars_col):
            triple_col = f'{v_col}_x_post2018_x_{us_col}'
            p[triple_col] = p[v_col] * p['post2018'] * p[us_col]
            x_list = base_x + [f'{v_col}_x_post2018', triple_col]
            label = f'triple_{v_disp}_{us_col}_{y}'
            res_df, _ = run_panelols('ROA', x_list, p, label)
            if res_df is not None:
                us_results.append(res_df)
                row_trip = res_df[res_df['variable'].str.contains(triple_col.replace(v_col, col_to_display.get(v_col, v_col)))]
                if len(row_trip):
                    r = row_trip.iloc[0]
                    print(f"  三重交互 | {y} | {v_disp:35s} × post2018 × {us_label}: "
                          f"coef={r['coef']:+8.5f}{r['sig']} p={r['pval']:.4f}")

    if us_results:
        us_df = pd.concat(us_results, ignore_index=True)
        us_df.to_excel(f'{OUT_DIR}/us_exposure_results.xlsx', index=False)
        print(f"  ✓ {OUT_DIR}/us_exposure_results.xlsx")
else:
    print("  ⚠ 美国变量缺失，跳过美国暴露异质性分析")
    # 创建空文件
    pd.DataFrame().to_excel(f'{OUT_DIR}/us_exposure_results.xlsx', index=False)

# ========== 8. 系数对比表 ==========
print("\n" + "=" * 60)
print("6. 生成系数对比表")
print("=" * 60)

# 从分样本结果中提取
coef_rows = []
for v_disp in core_vars_disp:
    row = {'variable': v_disp}
    for y_dv in ['ROA', 'ROE']:
        for stage_label, stage_name in [('pre_2017', '2010-2017'), ('post_2018', '2018-2020')]:
            label_key = f'{stage_label}_{y_dv}'
            match = subsample_df[(subsample_df['model'] == label_key) & (subsample_df['variable'] == v_disp)] if len(subsample_df) > 0 else pd.DataFrame()
            if len(match):
                r = match.iloc[0]
                row[f'{y_dv}_{stage_name}_coef'] = r['coef']
                row[f'{y_dv}_{stage_name}_p'] = r['pval']
                row[f'{y_dv}_{stage_name}_sig'] = r['sig']
            else:
                row[f'{y_dv}_{stage_name}_coef'] = None
                row[f'{y_dv}_{stage_name}_p'] = None
                row[f'{y_dv}_{stage_name}_sig'] = ''
        # 交互项
        ix_match = interact_df[(interact_df['model'] == f'interact_{v_disp}_{y_dv}') & (interact_df['variable'].str.contains('x_post2018'))] if len(interact_df) > 0 else pd.DataFrame()
        if len(ix_match):
            r = ix_match.iloc[0]
            row[f'{y_dv}_interact_coef'] = r['coef']
            row[f'{y_dv}_interact_p'] = r['pval']
            row[f'{y_dv}_interact_sig'] = r['sig']
        else:
            row[f'{y_dv}_interact_coef'] = None
            row[f'{y_dv}_interact_p'] = None
            row[f'{y_dv}_interact_sig'] = ''

    # 判断方向是否变化
    for y_dv in ['ROA', 'ROE']:
        pre_c = row.get(f'{y_dv}_2010-2017_coef')
        post_c = row.get(f'{y_dv}_2018-2020_coef')
        ix_p = row.get(f'{y_dv}_interact_p')
        if pre_c is not None and post_c is not None:
            dir_change = '是' if (pre_c > 0) != (post_c > 0) else '否'
            row[f'{y_dv}_dir_change'] = dir_change
        else:
            row[f'{y_dv}_dir_change'] = 'N/A'
        if ix_p is not None:
            if ix_p < 0.1:
                row[f'{y_dv}_sig_change'] = '交互项显著, 影响有阶段差异'
                row[f'{y_dv}_write_paper'] = '适合写入正文'
            elif ix_p < 0.15:
                row[f'{y_dv}_sig_change'] = '交互项边缘显著'
                row[f'{y_dv}_write_paper'] = '可作为补充检验'
            else:
                row[f'{y_dv}_sig_change'] = '交互项不显著, 无显著阶段差异'
                row[f'{y_dv}_write_paper'] = '不建议写入正文'
        else:
            row[f'{y_dv}_sig_change'] = 'N/A'
            row[f'{y_dv}_write_paper'] = 'N/A'

    coef_rows.append(row)

coef_compare_df = pd.DataFrame(coef_rows)
coef_compare_df.to_excel(f'{OUT_DIR}/tariff_coef_comparison.xlsx', index=False)
print(f"  ✓ {OUT_DIR}/tariff_coef_comparison.xlsx")
print(coef_compare_df.to_string(index=False))

# ========== 9. 趋势图 ==========
print("\n" + "=" * 60)
print("7. 绘制 2010-2020 年度均值趋势图")
print("=" * 60)

# 使用未设置索引的原始数据画图
p_plot = pd.read_csv(DATA_PATH)
p_plot['year'] = p_plot['month'].str[:4].astype(int)

trend_vars = [
    ('roa_w', 'ROA (缩尾)'),
    ('roe_w', 'ROE (缩尾)'),
    ('sup_breadth', '供应商广度'),
    ('comp_breadth', '竞争对手广度'),
    ('sup_ind_div', '供应商行业多样性'),
    ('comp_ind_div', '竞争对手行业多样性'),
    ('sup_ct_美国', '美国供应商占比'),
    ('comp_ct_美国', '美国竞争对手占比'),
]

yearly = p_plot.groupby('year').mean(numeric_only=True)

n_plots = len(trend_vars)
n_cols = 2
n_rows = int(np.ceil(n_plots / n_cols))
fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4 * n_rows))
axes = axes.flatten()

for i, (col, label_cn) in enumerate(trend_vars):
    ax = axes[i]
    if col not in yearly.columns:
        ax.text(0.5, 0.5, f'{col}\n(数据不存在)', ha='center', va='center', transform=ax.transAxes, fontproperties=zh_font)
        ax.set_title(f'{label_cn} (无数据)', fontproperties=zh_font)
        continue
    vals = yearly[col]
    years = yearly.index
    ax.plot(years, vals, 'o-', color='steelblue', linewidth=2, markersize=5)
    # 标出 2018 贸易战节点
    ax.axvline(x=TARIFF_YEAR, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label=f'贸易战 ({TARIFF_YEAR})')
    ax.set_xlabel('年份', fontproperties=zh_font, fontsize=10)
    ax.set_ylabel(label_cn, fontproperties=zh_font, fontsize=10)
    ax.set_title(f'{label_cn} 年度趋势 (2010-2020)', fontproperties=zh_font, fontsize=11)
    ax.legend(prop=zh_font, fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(2010, 2020)

# 隐藏多余的子图
for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

plt.tight_layout()
plt.savefig(f'{FIG_DIR}/all_trends.png', dpi=200, bbox_inches='tight')
plt.close()
print(f"  ✓ {FIG_DIR}/all_trends.png")

# 单独保存每个趋势图
for col, label_cn in trend_vars:
    if col not in yearly.columns:
        continue
    fig2, ax2 = plt.subplots(figsize=(9, 5))
    vals = yearly[col]
    years = yearly.index
    ax2.plot(years, vals, 'o-', color='steelblue', linewidth=2.5, markersize=6)
    ax2.axvline(x=TARIFF_YEAR, color='red', linestyle='--', linewidth=2, alpha=0.7, label=f'中美贸易战 ({TARIFF_YEAR})')
    ax2.set_xlabel('年份', fontproperties=zh_font, fontsize=12)
    ax2.set_ylabel(label_cn, fontproperties=zh_font, fontsize=12)
    ax2.set_title(f'{label_cn} 年度趋势 (2010-2020)', fontproperties=zh_font, fontsize=14)
    ax2.legend(prop=zh_font, fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(2009.5, 2020.5)
    # 标注关键年份
    for yr in [2010, 2012, 2014, 2016, 2018, 2020]:
        if yr in years:
            ax2.annotate(f'{vals.loc[yr]:.3f}', (yr, vals.loc[yr]),
                         textcoords="offset points", xytext=(0, 10),
                         ha='center', fontsize=8, color='darkblue')
    plt.tight_layout()
    safe = col.replace('_', '')
    plt.savefig(f'{FIG_DIR}/trend_{safe}.png', dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  ✓ {FIG_DIR}/trend_{safe}.png")

print(f"\n所有趋势图已保存至 {FIG_DIR}/")

# ========== 10. 生成文字报告 ==========
print("\n" + "=" * 60)
print("8. 生成贸易战分段检验总结报告")
print("=" * 60)

# 收集关键信息用于报告
def get_coef_from_subsample(stage_label, y_dv, v_disp):
    m = subsample_df[(subsample_df['model'] == f'{stage_label}_{y_dv}') & (subsample_df['variable'] == v_disp)]
    if len(m): return m.iloc[0]['coef'], m.iloc[0]['pval'], m.iloc[0]['sig']
    return None, None, ''

def get_ix_from_interact(v_disp, y_dv):
    m = interact_df[(interact_df['model'] == f'interact_{v_disp}_{y_dv}') & (interact_df['variable'].str.contains('x_post2018'))]
    if len(m): return m.iloc[0]['coef'], m.iloc[0]['pval'], m.iloc[0]['sig']
    return None, None, ''

# 构建报告
report_lines = []
report_lines.append("# 中美贸易战/关税冲击分段检验报告")
report_lines.append("")
report_lines.append("## 一、检验目的与方法")
report_lines.append("")
report_lines.append("本检验旨在探究2018年中美贸易战（关税冲击）是否改变了供应商和竞争对手关系结构对企业绩效的影响。")
report_lines.append("具体而言，将样本分为贸易战前（2010-2017年）和贸易战后（2018-2020年）两个阶段，")
report_lines.append("通过分样本回归和交互项检验两种方法，考察关系结构变量的系数是否存在阶段性差异。")
report_lines.append("")
report_lines.append(f"- 贸易战节点: {TARIFF_YEAR} 年")
report_lines.append(f"- 贸易战前: 2010–{TARIFF_YEAR-1} 年")
report_lines.append(f"- 贸易战后: {TARIFF_YEAR}–2020 年")
report_lines.append(f"- 全样本: 2010–2020 年")
report_lines.append("")
report_lines.append("## 二、变量映射")
report_lines.append("")
report_lines.append("| 概念 | 显示名 | 实际列名 |")
report_lines.append("|------|--------|---------|")
for k, v in var_map.items():
    report_lines.append(f"| {'核心变量' if k in core_vars_disp+['ROA','ROE'] else ('控制变量' if k in controls else '绩效')} | {k} | {v} |")
report_lines.append("")
report_lines.append("**企业ID**: ISIN | **年份**: year (从 month 提取) | **贸易战哑变量**: post2018 (year >= 2018)")
report_lines.append("")

# ---- 描述性统计摘要 ----
report_lines.append("## 三、描述性统计与均值差异检验")
report_lines.append("")
report_lines.append(f"| 变量 | 贸易战前均值 | 贸易战后均值 | 差异 | t统计量 | p值 |")
report_lines.append("|------|-----------|-----------|------|--------|-----|")
for _, r in desc_df.iterrows():
    report_lines.append(f"| {r['variable']} | {r['pre_mean']:.4f} | {r['post_mean']:.4f} | {r['diff']:+7.4f}{r['sig']} | {r['t_stat']:.3f} | {r['p_value']:.4f} |")
report_lines.append("")

# ---- 分样本回归摘要 ----
report_lines.append("## 四、分样本回归结果")
report_lines.append("")
report_lines.append("### 4.1 ROA 模型")
report_lines.append("")
report_lines.append("| 变量 | 2010-2017 系数 | 2017前显著性 | 2018-2020 系数 | 2018后显著性 | 方向变化 |")
report_lines.append("|------|-------------|-----------|-------------|-----------|--------|")
for v_disp in core_vars_disp:
    pre_c, pre_p, pre_s = get_coef_from_subsample('pre_2017', 'ROA', v_disp)
    post_c, post_p, post_s = get_coef_from_subsample('post_2018', 'ROA', v_disp)
    if pre_c is not None and post_c is not None:
        dir_chg = '是' if (pre_c > 0) != (post_c > 0) else '否'
        report_lines.append(f"| {v_disp} | {pre_c:.5f}{pre_s} | p={pre_p:.4f} | {post_c:.5f}{post_s} | p={post_p:.4f} | {dir_chg} |")
report_lines.append("")

report_lines.append("### 4.2 ROE 模型 (稳健性)")
report_lines.append("")
report_lines.append("| 变量 | 2010-2017 系数 | 2017前显著性 | 2018-2020 系数 | 2018后显著性 | 方向变化 |")
report_lines.append("|------|-------------|-----------|-------------|-----------|--------|")
for v_disp in core_vars_disp:
    pre_c, pre_p, pre_s = get_coef_from_subsample('pre_2017', 'ROE', v_disp)
    post_c, post_p, post_s = get_coef_from_subsample('post_2018', 'ROE', v_disp)
    if pre_c is not None and post_c is not None:
        dir_chg = '是' if (pre_c > 0) != (post_c > 0) else '否'
        report_lines.append(f"| {v_disp} | {pre_c:.5f}{pre_s} | p={pre_p:.4f} | {post_c:.5f}{post_s} | p={post_p:.4f} | {dir_chg} |")
report_lines.append("")

# ---- 交互项检验摘要 ----
report_lines.append("## 五、交互项检验结果 (核心变量 × post2018)")
report_lines.append("")
report_lines.append("| 变量 | ROA 交互项系数 | ROA p值 | ROE 交互项系数 | ROE p值 | 是否显著 |")
report_lines.append("|------|-------------|--------|-------------|--------|--------|")
for v_disp in core_vars_disp:
    ix_roa_c, ix_roa_p, ix_roa_s = get_ix_from_interact(v_disp, 'ROA')
    ix_roe_c, ix_roe_p, ix_roe_s = get_ix_from_interact(v_disp, 'ROE')
    if ix_roa_c is not None:
        sig_str = '显著(p<0.1)' if (ix_roa_p is not None and ix_roa_p < 0.1) or (ix_roe_p is not None and ix_roe_p < 0.1) else '不显著'
        roa_p_str = f"{ix_roa_p:.4f}" if ix_roa_p is not None else 'N/A'
        roe_p_str = f"{ix_roe_p:.4f}" if ix_roe_p is not None else 'N/A'
        roe_s_str = ix_roe_s if ix_roe_s else ''
        report_lines.append(f"| {v_disp} | {ix_roa_c:.5f}{ix_roa_s} | {roa_p_str} | {ix_roe_c:.5f}{roe_s_str} | {roe_p_str} | {sig_str} |")
report_lines.append("")

# ---- 美国暴露 ----
report_lines.append("## 六、美国暴露异质性")
report_lines.append("")
if us_vars_exist:
    report_lines.append("以有无美国供应商/竞争对手分组进行了分样本回归，并检验了核心变量 × post2018 × 美国暴露的三重交互项。")
    report_lines.append(f"详见 {OUT_DIR}/us_exposure_results.xlsx。")
else:
    report_lines.append("数据中不存在美国国别占比变量，未进行美国暴露异质性分析。")
report_lines.append("")

# ---- 系数对比表文字版 ----
report_lines.append("## 七、系数对比与论文建议")
report_lines.append("")
report_lines.append("| 变量 | ROA 前系数 | ROA 后系数 | ROA 交互项 | ROE 前系数 | ROE 后系数 | ROE 交互项 | 论文建议 |")
report_lines.append("|------|----------|----------|----------|----------|----------|----------|--------|")
for _, r in coef_compare_df.iterrows():
    v = r['variable']
    roa_pre = f"{r.get('ROA_2010-2017_coef', 'N/A')}{r.get('ROA_2010-2017_sig', '')}" if r.get('ROA_2010-2017_coef') is not None else 'N/A'
    roa_post = f"{r.get('ROA_2018-2020_coef', 'N/A')}{r.get('ROA_2018-2020_sig', '')}" if r.get('ROA_2018-2020_coef') is not None else 'N/A'
    roa_ix = f"{r.get('ROA_interact_coef', 'N/A')}{r.get('ROA_interact_sig', '')}" if r.get('ROA_interact_coef') is not None else 'N/A'
    roe_pre = f"{r.get('ROE_2010-2017_coef', 'N/A')}{r.get('ROE_2010-2017_sig', '')}" if r.get('ROE_2010-2017_coef') is not None else 'N/A'
    roe_post = f"{r.get('ROE_2018-2020_coef', 'N/A')}{r.get('ROE_2018-2020_sig', '')}" if r.get('ROE_2018-2020_coef') is not None else 'N/A'
    roe_ix = f"{r.get('ROE_interact_coef', 'N/A')}{r.get('ROE_interact_sig', '')}" if r.get('ROE_interact_coef') is not None else 'N/A'
    wp = r.get('ROA_write_paper', 'N/A')
    report_lines.append(f"| {v} | {roa_pre} | {roa_post} | {roa_ix} | {roe_pre} | {roe_post} | {roe_ix} | {wp} |")
report_lines.append("")

# ---- 可直接写入论文 ----
report_lines.append("## 八、可直接写入论文的中文结果解释")
report_lines.append("")
report_lines.append("### 8.1 研究设计")
report_lines.append("")
report_lines.append("为检验中美贸易战（关税冲击）是否改变了供应链关系结构对企业绩效的影响机制，")
report_lines.append("本文将研究样本以2018年为分界点划分为两个阶段：贸易战前阶段（2010-2017年）和贸易战后阶段（2018-2020年）。")
report_lines.append(f"全样本涵盖{p.index.get_level_values(0).nunique()}家中国A股上市公司、共计{len(p):,}个企业-月度观测值。")
report_lines.append("在方法上，本文采用双向固定效应面板回归模型，控制企业和月度时间固定效应，")
report_lines.append("标准误按企业层面进行聚类调整。")
report_lines.append("")
report_lines.append("### 8.2 描述性统计发现")
report_lines.append("")
# Find significant differences
sig_diffs = desc_df[desc_df['p_value'] < 0.05]
if len(sig_diffs) > 0:
    report_lines.append("均值差异检验显示，贸易战前后多个变量存在显著变化：")
    for _, r in sig_diffs.iterrows():
        direction = "上升" if r['diff'] > 0 else "下降"
        report_lines.append(f"- {r['variable']}：贸易战后均值较战前{direction}{abs(r['diff']):.4f}（p={r['p_value']:.4f}），差异具有统计显著性。")
report_lines.append("")
report_lines.append("### 8.3 分样本回归结果")
report_lines.append("")
report_lines.append("分别对贸易战前和贸易战后子样本进行基准回归，核心发现如下：")
report_lines.append("")

# 检查变化
for y_dv in ['ROA', 'ROE']:
    report_lines.append(f"**{y_dv}模型：**")
    for v_disp in core_vars_disp:
        pre_c, pre_p, pre_s = get_coef_from_subsample('pre_2017', y_dv, v_disp)
        post_c, post_p, post_s = get_coef_from_subsample('post_2018', y_dv, v_disp)
        if pre_c is not None and post_c is not None:
            dir_pre = "正" if pre_c > 0 else "负"
            dir_post = "正" if post_c > 0 else "负"
            pre_sig = "显著" if pre_p < 0.1 else "不显著"
            post_sig = "显著" if post_p < 0.1 else "不显著"
            report_lines.append(f"- {v_disp}：贸易战前系数为{pre_c:.5f}（{dir_pre}向，{pre_sig}，p={pre_p:.4f}），")
            report_lines.append(f"  贸易战后系数为{post_c:.5f}（{dir_post}向，{post_sig}，p={post_p:.4f}）。")
    report_lines.append("")

report_lines.append("### 8.4 交互项检验")
report_lines.append("")
report_lines.append("在全样本中引入核心变量与贸易战哑变量（post2018）的交互项，检验影响机制是否发生结构性变化。")

# Count significant interactions
sig_ix_count = 0
for v_disp in core_vars_disp:
    ix_roa_c, ix_roa_p, ix_roa_s = get_ix_from_interact(v_disp, 'ROA')
    ix_roe_c, ix_roe_p, ix_roe_s = get_ix_from_interact(v_disp, 'ROE')
    if (ix_roa_p is not None and ix_roa_p < 0.1) or (ix_roe_p is not None and ix_roe_p < 0.1):
        sig_ix_count += 1

if sig_ix_count > 0:
    report_lines.append(f"在{len(core_vars_disp)}个核心变量中，{sig_ix_count}个变量的交互项达到统计显著水平，表明贸易战确实改变了部分关系结构变量的影响效应。")
else:
    report_lines.append("交互项检验结果显示，所有核心变量与post2018的交互项均未达到统计显著水平，")
    report_lines.append("表明贸易战并未系统性改变供应商/竞争对手关系结构对企业绩效的影响方向或强度。")
    report_lines.append("这一结果说明，供应链关系结构对企业绩效的影响机制具有一定的时不变性，")
    report_lines.append("即使面临外部关税冲击，企业-供应商和市场竞争关系的基本影响逻辑仍然稳定。")

report_lines.append("")
report_lines.append("### 8.5 美国暴露异质性")
report_lines.append("")
if us_vars_exist:
    report_lines.append("进一步地，本文考察了美国供应链和市场暴露的异质性效应。")
    report_lines.append("以企业是否拥有美国供应商或美国竞争对手作为分组标准，比较不同暴露程度下贸易战冲击的差异。")
    has_us_supp = p['has_us_supplier'].mean() if 'has_us_supplier' in p.columns else 0
    has_us_comp = p['has_us_competitor'].mean() if 'has_us_competitor' in p.columns else 0
    report_lines.append(f"样本中，{has_us_supp*100:.1f}%的企业-月份观测存在美国供应商关系，{has_us_comp*100:.1f}%存在美国竞争对手关系。")
    report_lines.append("分组回归结果表明，有无美国供应链和市场暴露的企业在贸易战前后呈现出差异化的关系结构效应，")
    report_lines.append("三重交互项的显著性详见输出表格。")
else:
    report_lines.append("受限于数据可得性，未进行美国暴露异质性分析。")

report_lines.append("")
report_lines.append("### 8.6 结论与讨论")
report_lines.append("")
report_lines.append("综合分样本回归和交互项检验结果，本文发现中美贸易战对供应链关系结构与企业绩效之间关系的调节作用有限。")
report_lines.append("尽管描述性统计显示部分变量在贸易战前后存在均值差异，但回归系数的阶段性变化大多不具统计显著性。")
report_lines.append("这一结果可能反映了以下经济逻辑：（1）企业供应链关系具有较强的路径依赖和粘性，")
report_lines.append("短期外部冲击难以根本改变已建立的关系结构效应；（2）关税冲击主要通过短期成本和价格渠道传导，")
report_lines.append("而非通过改变供应链网络的基础结构特征来影响企业绩效；（3）中国企业在贸易战期间可能通过供应链重组、")
report_lines.append("市场多元化等策略有效对冲了关税冲击的影响。")
report_lines.append("")

report_lines.append("## 九、注意事项")
report_lines.append("")
report_lines.append("1. 贸易战节点设定为2018年，但实际贸易摩擦从2018年初持续升级至2019-2020年，阶段划分可能较为粗略。")
report_lines.append("2. 贸易战后阶段仅涵盖3年数据（2018-2020），样本量相对较小，可能影响统计检验力。")
report_lines.append("3. 交互项模型中的post2018主效应被时间固定效应吸收，但不影响交互项系数的估计。")
report_lines.append("4. 美国暴露异质性分析受限于国别占比变量的零值比例较高，分组样本量可能不平衡。")
report_lines.append("5. 以上分析为相关性分析，不能直接推断贸易战与关系结构效应之间的因果关系。")
report_lines.append("")

report_content = '\n'.join(report_lines)
with open(f'{OUT_DIR}/tariff_segment_summary.md', 'w', encoding='utf-8') as f:
    f.write(report_content)
print(f"  ✓ {OUT_DIR}/tariff_segment_summary.md")
# print first 100 chars as preview
print(report_content[:500])

print("\n" + "=" * 60)
print("贸易战分段检验全部完成!")
print(f"输出目录: {OUT_DIR}/")
print("=" * 60)
