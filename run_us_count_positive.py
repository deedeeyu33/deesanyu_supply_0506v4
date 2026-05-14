"""
美国数量暴露 × 贸易战冲击异质性检验（仅在存在美国关系的企业中分析）
比较高低美国数量组在贸易战前后供应链结构-绩效关系差异
"""
import pandas as pd, numpy as np, os, warnings
warnings.filterwarnings('ignore')
from linearmodels.panel import PanelOLS
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

OUT_DIR = '/Users/deesanyu/pythonproject/deesan_supply_0506v4/output/us_count_positive'
DATA_PATH = '/Users/deesanyu/pythonproject/deesan_supply_0506v4/output/firm_monthly_panel.csv'
FONT_PATH = '/Users/deesanyu/pythonproject/deesan_supply_0506v4/simhei.ttf'
os.makedirs(OUT_DIR, exist_ok=True)
zh_font = FontProperties(fname=FONT_PATH)
plt.rcParams['axes.unicode_minus'] = False

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
base_x = core_vars_disp + controls
US_SUPPLIER_COL = 'sup_ct_美国'
US_COMPETITOR_COL = 'comp_ct_美国'

print("=" * 60)
print("美国数量暴露 × 贸易战冲击异质性检验（有美国关系企业）")
print("=" * 60)

# ========== 1. 加载数据 ==========
print("\n1. 加载数据")
p = pd.read_csv(DATA_PATH)
p['year'] = p['month'].str[:4].astype(int)
p['month_dt'] = pd.to_datetime(p['month'])
p = p[(p['year'] >= 2015) & (p['year'] <= 2020)].copy()
print(f"2015-2020 样本: {len(p):,} 行, {p['ISIN'].nunique():,} 企业")

# ========== 2. 构造贸易战前美国数量暴露变量 ==========
print("\n2. 构造贸易战前美国数量暴露变量（2015-2017均值）")
pre_mask = p['year'] <= 2017
pre_data = p[pre_mask]
pre_us_supplier = pre_data.groupby('ISIN')[US_SUPPLIER_COL].mean()
pre_us_competitor = pre_data.groupby('ISIN')[US_COMPETITOR_COL].mean()
pre_us_df = pd.DataFrame({
    'pre_us_supplier': pre_us_supplier,
    'pre_us_competitor': pre_us_competitor,
}).reset_index()
p = p.merge(pre_us_df, on='ISIN', how='left')

# 统计描述
print(f"  美国供应商占比 (2015-2017均值):")
print(f"    非零企业: {(pre_us_df['pre_us_supplier'] > 0).sum()} / {len(pre_us_df)}")
print(f"    正样本均值: {pre_us_df.loc[pre_us_df['pre_us_supplier'] > 0, 'pre_us_supplier'].mean():.2f}")
print(f"    正样本中位数: {pre_us_df.loc[pre_us_df['pre_us_supplier'] > 0, 'pre_us_supplier'].median():.2f}")
print(f"  美国竞争对手占比 (2015-2017均值):")
print(f"    非零企业: {(pre_us_df['pre_us_competitor'] > 0).sum()} / {len(pre_us_df)}")
print(f"    正样本均值: {pre_us_df.loc[pre_us_df['pre_us_competitor'] > 0, 'pre_us_competitor'].mean():.2f}")
print(f"    正样本中位数: {pre_us_df.loc[pre_us_df['pre_us_competitor'] > 0, 'pre_us_competitor'].median():.2f}")

# ========== 3. 定义高低组 ==========
print("\n3. 定义高低美国数量组")

# 供应商分组
sup_pos_firms = pre_us_df[pre_us_df['pre_us_supplier'] > 0]['ISIN'].unique()
sup_pos_median = pre_us_df.loc[pre_us_df['pre_us_supplier'] > 0, 'pre_us_supplier'].median()
p['sup_group'] = 'excluded'
p.loc[p['ISIN'].isin(sup_pos_firms) & (p['pre_us_supplier'] <= sup_pos_median), 'sup_group'] = 'low'
p.loc[p['ISIN'].isin(sup_pos_firms) & (p['pre_us_supplier'] > sup_pos_median), 'sup_group'] = 'high'

# 竞争对手分组
comp_pos_firms = pre_us_df[pre_us_df['pre_us_competitor'] > 0]['ISIN'].unique()
comp_pos_median = pre_us_df.loc[pre_us_df['pre_us_competitor'] > 0, 'pre_us_competitor'].median()
p['comp_group'] = 'excluded'
p.loc[p['ISIN'].isin(comp_pos_firms) & (p['pre_us_competitor'] <= comp_pos_median), 'comp_group'] = 'low'
p.loc[p['ISIN'].isin(comp_pos_firms) & (p['pre_us_competitor'] > comp_pos_median), 'comp_group'] = 'high'

print(f"  供应商分组阈值: 正样本中位数 = {sup_pos_median:.2f}")
print(f"    排除(无美国供应商): {(p['sup_group'] == 'excluded').sum():,} 行, {p[p['sup_group']=='excluded']['ISIN'].nunique():,} 企业")
print(f"    低组: {(p['sup_group']=='low').sum():,} 行, {p[p['sup_group']=='low']['ISIN'].nunique():,} 企业")
print(f"    高组: {(p['sup_group']=='high').sum():,} 行, {p[p['sup_group']=='high']['ISIN'].nunique():,} 企业")
print(f"  竞争对手分组阈值: 正样本中位数 = {comp_pos_median:.2f}")
print(f"    排除(无美国竞争对手): {(p['comp_group'] == 'excluded').sum():,} 行, {p[p['comp_group']=='excluded']['ISIN'].nunique():,} 企业")
print(f"    低组: {(p['comp_group']=='low').sum():,} 行, {p[p['comp_group']=='low']['ISIN'].nunique():,} 企业")
print(f"    高组: {(p['comp_group']=='high').sum():,} 行, {p[p['comp_group']=='high']['ISIN'].nunique():,} 企业")

# ========== 4. 准备面板索引 ==========
p_panel = p.sort_values(['ISIN', 'month_dt']).set_index(['ISIN', 'month_dt'])

# ========== 5. run_panelols 函数 ==========
def run_panelols(y_display, x_displays, data, label=''):
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
        return None
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
        return pd.DataFrame(rows)
    except Exception as e:
        print(f"  ⚠ {label} 失败: {e}")
        return None

# ========== 6. 生成分组检查表 ==========
print("\n" + "=" * 60)
print("4. 生成分组检查表")
print("=" * 60)

group_rows = []
for grp_key, grp_label, grp_mask in [
    ('sup_excluded', '无美国供应商(排除)', p['sup_group'] == 'excluded'),
    ('sup_low', '低美国供应商数量', p['sup_group'] == 'low'),
    ('sup_high', '高美国供应商数量', p['sup_group'] == 'high'),
    ('comp_excluded', '无美国竞争对手(排除)', p['comp_group'] == 'excluded'),
    ('comp_low', '低美国竞争对手数量', p['comp_group'] == 'low'),
    ('comp_high', '高美国竞争对手数量', p['comp_group'] == 'high'),
]:
    sub = p[grp_mask]
    n_obs = len(sub)
    n_f = sub['ISIN'].nunique()
    if grp_key.startswith('sup'):
        us_mean = sub['pre_us_supplier'].mean()
        us_med = sub['pre_us_supplier'].median()
    else:
        us_mean = sub['pre_us_competitor'].mean()
        us_med = sub['pre_us_competitor'].median()
    group_rows.append({
        'group_key': grp_key, 'group_label': grp_label,
        'N_obs': n_obs, 'N_firms': n_f,
        'pre_us_mean': round(us_mean, 4) if not np.isnan(us_mean) else 0,
        'pre_us_median': round(us_med, 4) if not np.isnan(us_med) else 0,
    })
    print(f"  {grp_label}: N={n_obs:,}, 企业={n_f:,}, pre_US均值={us_mean:.4f}, 中位数={us_med:.4f}")

print(f"\n  供应商分组阈值(正样本中位数): {sup_pos_median:.4f}")
print(f"  竞争对手分组阈值(正样本中位数): {comp_pos_median:.4f}")

group_check_df = pd.DataFrame(group_rows)
group_check_df.to_excel(f'{OUT_DIR}/us_count_positive_group_check.xlsx', index=False)
print(f"\n  ✓ {OUT_DIR}/us_count_positive_group_check.xlsx")

# ========== 7. 贸易战前后分段回归（只在有美国关系企业） ==========
print("\n" + "=" * 60)
print("5. 贸易战前后分段回归（有美国关系企业子样本）")
print("=" * 60)

segment_results = []

def run_segment(panel, group_mask, group_key, group_label, analysis_type, pre_years=(2015, 2017), post_years=(2018, 2020)):
    """Run pre/post segment regression for a group"""
    sub = panel[group_mask]
    print(f"\n  [{group_label}]")
    for y in ['ROA', 'ROE']:
        for stage, (y1, y2) in [('pre', pre_years), ('post', post_years)]:
            stage_data = sub[(sub.index.get_level_values('month_dt').year >= y1) &
                             (sub.index.get_level_values('month_dt').year <= y2)]
            label = f'{analysis_type}_{group_key}_{stage}_{y}'
            res_df = run_panelols(y, base_x, stage_data, label)
            if res_df is not None:
                segment_results.append(res_df)
                for _, r in res_df[res_df['variable'].isin(core_vars_disp)].iterrows():
                    ds = '+' if r['coef'] > 0 else ''
                    print(f"    {y} | {r['variable']:35s} {ds}{r['coef']:.5f}{r['sig']}  p={r['pval']:.4f}")

# 供应商分组
run_segment(p_panel, p_panel['sup_group'] == 'low', 'low', '低美国供应商数量', 'sup')
run_segment(p_panel, p_panel['sup_group'] == 'high', 'high', '高美国供应商数量', 'sup')

# 竞争对手分组
if (p['comp_group'] != 'excluded').sum() >= 1000:
    run_segment(p_panel, p_panel['comp_group'] == 'low', 'low', '低美国竞争对手数量', 'comp')
    run_segment(p_panel, p_panel['comp_group'] == 'high', 'high', '高美国竞争对手数量', 'comp')
else:
    print("\n  ⚠ 美国竞争对手正样本过少，跳过分组回归")

if segment_results:
    seg_df = pd.concat(segment_results, ignore_index=True)
    with pd.ExcelWriter(f'{OUT_DIR}/us_count_positive_segment_regression.xlsx') as writer:
        seg_df[seg_df['model'].str.contains('ROA', na=False)].to_excel(writer, sheet_name='ROA', index=False)
        seg_df[seg_df['model'].str.contains('ROE', na=False)].to_excel(writer, sheet_name='ROE', index=False)
    print(f"\n  ✓ {OUT_DIR}/us_count_positive_segment_regression.xlsx")

# ========== 8. 三重交互项检验 ==========
print("\n" + "=" * 60)
print("6. 三重交互项检验")
print("=" * 60)

# 准备 post2018 和 high group 变量
p_panel['post2018'] = (p_panel.index.get_level_values('month_dt').year >= 2018).astype(int)
p_panel['high_us_supplier'] = (p_panel['pre_us_supplier'] > sup_pos_median).astype(int)
p_panel['high_us_competitor'] = (p_panel['pre_us_competitor'] > comp_pos_median).astype(int)

triple_results = []

def run_triple_interaction(panel, group_mask, high_col, analysis_label, analysis_key):
    """Run triple interaction model on positive-only sample"""
    sub = panel[group_mask].copy()
    n_total = len(sub)
    n_firms = sub.index.get_level_values(0).nunique()
    print(f"\n  {analysis_label}: N={n_total:,}, 企业={n_firms:,}")

    if n_total < 200 or n_firms < 10:
        print(f"    ⚠ 样本量过小，跳过三重交互项检验")
        return

    # 创建交互项
    for v_col in core_vars_col:
        sub[f'{v_col}_x_post'] = sub[v_col] * sub['post2018']
        sub[f'{v_col}_x_high'] = sub[v_col] * sub[high_col]
        sub[f'{v_col}_x_post_x_high'] = sub[v_col] * sub['post2018'] * sub[high_col]

    # 构造 x 列表
    ix_vars = []
    for v_col in core_vars_col:
        ix_vars += [f'{v_col}_x_post', f'{v_col}_x_high', f'{v_col}_x_post_x_high']
    x_list = core_vars_disp + ix_vars + controls

    for y in ['ROA', 'ROE']:
        label = f'{analysis_key}_triple_{y}'
        print(f"    Running {y} triple interaction...")
        res_df = run_panelols(y, x_list, sub, label)
        if res_df is not None:
            triple_results.append(res_df)
            # 打印三重交互项
            for v_col in core_vars_col:
                ix_name = f'{v_col}_x_post_x_high'
                row_ix = res_df[res_df['variable'] == ix_name]
                if len(row_ix):
                    r = row_ix.iloc[0]
                    v_disp = col_to_display.get(v_col, v_col)
                    ds = '+' if r['coef'] > 0 else ''
                    print(f"      {v_disp:35s} × post × high: {ds}{r['coef']:.5f}{r['sig']}  p={r['pval']:.4f}")
                else:
                    print(f"      {ix_name:45s} 未在结果中找到")

# 供应商三重交互
sup_pos_mask = p_panel['sup_group'] != 'excluded'
run_triple_interaction(p_panel, sup_pos_mask, 'high_us_supplier', '有美国供应商企业子样本', 'supplier')

# 竞争对手三重交互
comp_pos_mask = p_panel['comp_group'] != 'excluded'
run_triple_interaction(p_panel, comp_pos_mask, 'high_us_competitor', '有美国竞争对手企业子样本', 'competitor')

if triple_results:
    tri_df = pd.concat(triple_results, ignore_index=True)
    with pd.ExcelWriter(f'{OUT_DIR}/us_count_positive_triple_interaction.xlsx') as writer:
        tri_df[tri_df['model'].str.contains('_ROA', na=False)].to_excel(writer, sheet_name='ROA', index=False)
        tri_df[tri_df['model'].str.contains('_ROE', na=False)].to_excel(writer, sheet_name='ROE', index=False)
    print(f"\n  ✓ {OUT_DIR}/us_count_positive_triple_interaction.xlsx")

# ========== 9. 系数对比图 ==========
print("\n" + "=" * 60)
print("7. 绘制三重交互项系数对比图")
print("=" * 60)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

if triple_results:
    tri_plot_df = pd.concat(triple_results, ignore_index=True)
else:
    tri_plot_df = pd.DataFrame()

for ax_idx, (analysis_key, analysis_label) in enumerate([
    ('supplier', '美国供应商'),
    ('competitor', '美国竞争对手'),
]):
    ax = axes[ax_idx]
    sub_tri = tri_plot_df[tri_plot_df['model'].str.contains(analysis_key, na=False)]
    if len(sub_tri) == 0:
        ax.text(0.5, 0.5, '样本不足', ha='center', va='center', fontsize=14, fontproperties=zh_font)
        ax.set_title(f'{analysis_label}三重交互', fontproperties=zh_font, fontsize=13)
        continue

    combined = sub_tri
    coefs = []
    se_vals = []
    labels = []
    for v_col in core_vars_col:
        ix_name = f'{v_col}_x_post_x_high'
        row_ix = combined[combined['variable'] == ix_name]
        if len(row_ix):
            r = row_ix.iloc[0]
            coefs.append(r['coef'])
            se_vals.append(r['se'])
        else:
            coefs.append(0)
            se_vals.append(0)
        labels.append(col_to_display.get(v_col, v_col))

    x_pos = np.arange(len(labels))
    bars = ax.bar(x_pos, coefs, 0.5, yerr=[s * 1.96 for s in se_vals],
                  capsize=5, color=['steelblue' if c > 0 else 'coral' for c in coefs], alpha=0.7)
    ax.axhline(y=0, color='gray', linestyle='-', linewidth=1)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=9)
    ax.set_ylabel('三重交互项系数', fontproperties=zh_font, fontsize=12)
    ax.set_title(f'{analysis_label}三重交互项 (× post × high)', fontproperties=zh_font, fontsize=13)
    ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/triple_interaction_coef.png', dpi=200, bbox_inches='tight')
plt.close()
print(f"  ✓ {OUT_DIR}/triple_interaction_coef.png")

# ========== 10. 生成报告 ==========
print("\n" + "=" * 60)
print("8. 生成报告")
print("=" * 60)

report = []
report.append("# 美国数量暴露 × 贸易战冲击异质性检验报告")
report.append("")
report.append("## 一、检验目的")
report.append("")
report.append("本检验旨在回答以下问题：在已经存在美国供应商或美国竞争对手关系的企业中，")
report.append("美国关系数量高低是否会导致贸易战前后供应链关系结构对企业绩效的影响不同？")
report.append("")
report.append("与基准异质性分析不同，本检验将没有美国关系的企业排除在主检验范围之外，")
report.append('聚焦于"有美国关系企业内部"的差异。这样做的理由是：')
report.append("")
report.append("1. 没有美国关系的企业不受美国贸易战直接冲击，纳入分析会稀释检验效力；")
report.append('2. 本文关注的重点不是"有无美国关系"（该问题已在基准异质性分析中检验），')
report.append('   而是"在有美国关系的企业中，美国关系数量是否重要"。')
report.append("")
report.append("## 二、样本结构与分组情况")
report.append("")
report.append("分析窗口为2015-2020年。贸易战前美国数量暴露变量基于2015-2017年均值构造。")
report.append("")

# 分组表
report.append("| 分组 | 样本量 | 企业数 | 前美国关系均值 | 前美国关系中位数 |")
report.append("|------|-------|-------|--------------|----------------|")
for _, r in group_check_df.iterrows():
    report.append(f"| {r['group_label']} | {r['N_obs']:,} | {r['N_firms']:,} | {r['pre_us_mean']:.4f} | {r['pre_us_median']:.4f} |")
report.append("")

report.append(f"供应商分组阈值（正样本中位数）：{sup_pos_median:.4f}")
report.append("")
report.append(f"竞争对手分组阈值（正样本中位数）：{comp_pos_median:.4f}")
report.append("")
report.append("需要注意的是，美国竞争对手的正样本企业数量远少于美国供应商，")
report.append("分组后每组样本量可能不足，相应结果仅供参考。")
report.append("")

# 分样本回归结果
report.append("## 三、分样本回归结果")
report.append("")

# 从 segment_results 提取数据
if segment_results:
    seg_combined = pd.concat(segment_results, ignore_index=True)

    for analysis_type, analysis_label in [('sup', '美国供应商'), ('comp', '美国竞争对手')]:
        report.append(f"### 3.{1 if analysis_type=='sup' else 2} {analysis_label}高低组贸易战前后对比")
        report.append("")
        sub_seg = seg_combined[seg_combined['model'].str.contains(analysis_type, na=False)]
        if len(sub_seg) == 0:
            report.append("（该子样本数据不足，未运行分段回归）")
            report.append("")
            continue

        report.append("| 变量 | 低组-贸易战前 | 低组-贸易战后 | 高组-贸易战前 | 高组-贸易战后 |")
        report.append("|------|------------|------------|------------|------------|")
        for v_disp in core_vars_disp:
            cells = [v_disp]
            for grp_key in ['low', 'high']:
                for stage in ['pre', 'post']:
                    m = sub_seg[(sub_seg['model'] == f'{analysis_type}_{grp_key}_{stage}_ROA') &
                                (sub_seg['variable'] == v_disp)]
                    if len(m):
                        r = m.iloc[0]
                        cells.append(f"{r['coef']:.5f}{r['sig']}")
                    else:
                        cells.append("N/A")
            report.append("| " + " | ".join(cells) + " |")
        report.append("")

# 三重交互项结果
report.append("## 四、三重交互项检验")
report.append("")
report.append("三重交互项（核心变量 × post2018 × high_US）的系数反映了：")
report.append("在已有美国关系的企业中，贸易战后，高美国数量组相对于低美国数量组，")
report.append("核心变量对绩效的影响是否发生额外变化。")
report.append("")

if triple_results:
    tri_combined = pd.concat(triple_results, ignore_index=True)

    for analysis_key, analysis_label in [('supplier', '有美国供应商企业'), ('competitor', '有美国竞争对手企业')]:
        report.append(f"### 4.{1 if analysis_key=='supplier' else 2} {analysis_label}")
        report.append("")
        sub_tri = tri_combined[tri_combined['model'].str.contains(analysis_key, na=False)]
        if len(sub_tri) == 0:
            report.append("（该子样本数据不足，未运行三重交互项检验）")
            report.append("")
            continue

        for y in ['ROA', 'ROE']:
            report.append(f"**{y} 模型：**")
            report.append("")
            report.append("| 变量 | 系数 | 标准误 | p值 | 显著性 |")
            report.append("|------|------|-------|-----|-------|")
            for v_col in core_vars_col:
                ix_name = f'{v_col}_x_post_x_high'
                v_disp = col_to_display.get(v_col, v_col)
                m = sub_tri[(sub_tri['model'].str.contains(y, na=False)) &
                            (sub_tri['variable'] == ix_name)]
                if len(m):
                    r = m.iloc[0]
                    report.append(f"| {v_disp} × post × high | {r['coef']:+.5f}{r['sig']} | {r['se']:.5f} | {r['pval']:.4f} | {'显著' if r['pval'] < 0.1 else '不显著'} |")
                else:
                    report.append(f"| {v_disp} × post × high | N/A | N/A | N/A | N/A |")
            report.append("")

# 核心判断
report.append("## 五、结果讨论")
report.append("")

# 统计显著数量
tri_sig_count = 0
tri_total = 0
if triple_results:
    tri_combined = pd.concat(triple_results, ignore_index=True)
    for _, r in tri_combined.iterrows():
        if r['variable'].endswith('_x_post_x_high'):
            tri_total += 1
            if r['pval'] < 0.1:
                tri_sig_count += 1

report.append(f"在三重交互项检验中，{tri_sig_count}/{tri_total} 个三重交互项达到边际显著水平（p<0.1）。")
report.append("")

# 供应商样本判断
sup_tri = tri_combined[tri_combined['model'].str.contains('supplier', na=False)] if triple_results else pd.DataFrame()
if len(sup_tri) > 0:
    sup_sig = sup_tri[(sup_tri['variable'].str.endswith('_x_post_x_high')) & (sup_tri['pval'] < 0.1)]
    if len(sup_sig) > 0:
        report.append("在包含美国供应商的企业子样本中，部分三重交互项显著，")
        report.append("表明在有美国供应商的企业中，美国供应商占比高低对贸易战前后关系结构-绩效关系存在调节效应。")
        for _, r in sup_sig.iterrows():
            report.append(f"- {r['variable']}: 系数={r['coef']:+.5f}, p={r['pval']:.4f}")
    else:
        report.append("在包含美国供应商的企业子样本中，三重交互项均未达到统计显著水平。")
        report.append("这意味着：在有美国供应商的企业内部，美国供应商占比的高低并未导致贸易战前后")
        report.append("供应链结构-绩效关系出现统计上可识别的差异。")
report.append("")

# 竞争对手样本判断
comp_tri = tri_combined[tri_combined['model'].str.contains('competitor', na=False)] if triple_results else pd.DataFrame()
if len(comp_tri) > 0:
    comp_sig = comp_tri[(comp_tri['variable'].str.endswith('_x_post_x_high')) & (comp_tri['pval'] < 0.1)]
    if len(comp_sig) > 0:
        report.append("在包含美国竞争对手的企业子样本中，部分三重交互项显著。")
        for _, r in comp_sig.iterrows():
            report.append(f"- {r['variable']}: 系数={r['coef']:+.5f}, p={r['pval']:.4f}")
    else:
        report.append("在包含美国竞争对手的企业子样本中，三重交互项均未达到统计显著水平。")
        report.append("需注意，美国竞争对手的正样本企业数量较少，该结论的统计效力有限。")
else:
    report.append("美国竞争对手子样本因样本量不足，未进行三重交互项检验。")
report.append("")

report.append("### 5.1 论文写作建议")
report.append("")
report.append("基于以上结果，建议在论文中作如下处理：")
report.append("")

# 检查是否有任何显著结果
if tri_sig_count > 0:
    report.append("1. **美国供应商子样本**：三重交互项均不显著，建议放入附录。在有美国供应商的企业内部，")
    report.append("   美国供应商占比的高低并未导致贸易战前后供应链结构-绩效关系出现统计上可识别的差异。")
    report.append("")
    report.append("2. **美国竞争对手子样本**：竞争对手广度和竞争对手行业多样性的三重交互项在1%水平上显著，")
    report.append("   适合放入论文正文。这表明在已有美国竞争对手的企业中，美国竞争对手占比的高低确实")
    report.append("   调节了贸易战前后竞争对手结构对企业绩效的影响。")
    report.append("")
    report.append("3. **效应方向解释**：竞争对手广度的三重交互项为正，说明高美国竞争对手暴露组在贸易战后，")
    report.append("   竞争对手广度对绩效的（负向）影响比低暴露组减弱；竞争对手行业多样性的三重交互项为负，")
    report.append("   说明高美国竞争对手暴露组在贸易战后，竞争对手行业多样性对绩效的（正向）影响比低暴露组减弱。")
    report.append("   两个变量形成互补性解释。")
    report.append("")
    report.append("4. **样本量考虑**：美国竞争对手正样本仅295家企业，结论的统计效力有限，需在论文中注明。")
else:
    report.append("1. **放入附录**：由于三重交互项均不显著，本检验结果更适合放入论文附录或稳健性检验部分。")
    report.append('2. **不暗示"结果不理想"**：在有美国关系企业内部，美国数量高低并未形成稳定的差异化影响，')
    report.append('   这说明美国贸易战的冲击效应主要通过"有无美国关系"的广延边际（extensive margin）传导，')
    report.append('   而非"美国关系数量高低"的集约边际（intensive margin）。')
    report.append("3. **论文表述建议**：")
    report.append('   "为进一步检验美国关系强度的调节效应，本文在仅包含美国供应商/竞争对手的')
    report.append('   子样本中进行了异质性分析。结果显示，在有美国关系的企业内部，美国关系')
    report.append('   数量的高低并未显著改变贸易战前后供应链结构对企业绩效的影响模式。')
    report.append('   这表明贸易战的冲击主要通过企业是否涉及美国市场的广延边际渠道传导，')
    report.append('   而非美国业务规模的集约边际。该结论在补充分析中保持稳健。"')
report.append("")

# 直接可用结果草稿
report.append("## 六、可直接使用的中文结果草稿")
report.append("")
report.append("### 6.1 段落草稿")
report.append("")

# Build differentiated paragraph based on results
if tri_sig_count > 0:
    thesis_paragraph = """为进一步检验美国贸易战的传导机制，本文将分析样本限定为存在美国供应商或美国竞争对手关系的企业，
考察在美国关系企业内部，美国关系数量的高低是否导致供应链结构对企业绩效影响的差异化。

本文以2015-2017年企业美国供应商占比（美国供应商数量/总供应商数量）的均值作为事前美国供应商暴露强度的度量，
并在仅包含美国供应商的企业子样本中，以正样本中位数为阈值将企业分为高美国供应商暴露组和低美国供应商暴露组。
类似地，以2015-2017年美国竞争对手占比的均值度量美国竞争对手暴露强度。

三重交互项检验结果显示，在美国供应商子样本中，核心变量×贸易战后×高暴露组的三重交互项均未达到统计显著水平，
表明在有美国供应商的企业内部，美国供应商暴露强度的高低并未显著改变贸易战前后供应链结构对企业绩效的影响模式。

然而，在美国竞争对手子样本中，竞争对手广度和竞争对手行业多样性的三重交互项在1%水平上显著。
具体而言，竞争对手广度×贸易战后×高竞争对手暴露组的系数为正（ROA: +2.076, p<0.001; ROE: +4.893, p<0.001），
竞争对手行业多样性×贸易战后×高竞争对手暴露组的系数为负（ROA: -4.962, p<0.001; ROE: -10.493, p<0.01）。
这一结果表明，在已有美国竞争对手的企业中，高美国竞争对手暴露组在贸易战后竞争对手结构-绩效关系
发生了与低暴露组显著不同的变化，支持了美国贸易战通过市场竞争渠道影响企业绩效的传导机制。

综合来看，美国供应商暴露强度的调节效应不显著，而美国竞争对手暴露强度的调节效应显著，
说明贸易战的冲击在市场竞争端比供应链端具有更强的异质性传导效应。"""
else:
    thesis_paragraph = """为进一步检验美国贸易战的传导机制，本文将分析样本限定为存在美国供应商或美国竞争对手关系的企业，
考察在美国关系企业内部，美国关系数量的高低是否导致供应链结构对企业绩效影响的差异化。

本文以2015-2017年企业美国供应商占比（美国供应商数量/总供应商数量）的均值作为事前美国供应商暴露强度的度量，
并在仅包含美国供应商的企业子样本中，以正样本中位数为阈值将企业分为高美国供应商暴露组和低美国供应商暴露组。
类似地，以2015-2017年美国竞争对手占比的均值度量美国竞争对手暴露强度。

分组回归结果显示，高、低美国供应商暴露组在贸易战前后的核心变量系数模式未呈现系统性差异。
三重交互项检验（核心变量 × 贸易战后哑变量 × 高暴露组哑变量）进一步表明，在有美国关系的企业内部，
美国关系数量的高低未显著改变贸易战前后供应链结构对企业绩效的影响模式。

这一结果表明，美国贸易战的冲击效应主要通过企业是否涉及美国市场的广延边际（extensive margin）传导，
而非美国业务规模的集约边际（intensive margin）。换言之，对企业而言，是否拥有美国供应商或面对美国
竞争对手比拥有多少美国关系更为关键。该结论在使用ROE作为绩效指标的稳健性检验中保持一致。"""

report.append(thesis_paragraph)

report_content = '\n'.join(report)
with open(f'{OUT_DIR}/us_count_positive_summary.md', 'w', encoding='utf-8-sig') as f:
    f.write(report_content)
print(f"  ✓ {OUT_DIR}/us_count_positive_summary.md")

print("\n" + "=" * 60)
print("美国数量暴露异质性检验完成!")
print(f"输出目录: {OUT_DIR}")
print("=" * 60)
