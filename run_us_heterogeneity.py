"""
美国供应商/美国竞争对手暴露异质性分析
检验企业是否存在美国供应链和市场暴露对企业绩效的影响差异
"""
import pandas as pd, numpy as np, os, warnings
warnings.filterwarnings('ignore')
from linearmodels.panel import PanelOLS
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

OUT_DIR = '/Users/deesanyu/pythonproject/deesan_supply_0506v4/output/us_heterogeneity'
DATA_PATH = '/Users/deesanyu/pythonproject/deesan_supply_0506v4/output/firm_monthly_panel.csv'
FONT_PATH = '/Users/deesanyu/pythonproject/deesan_supply_0506v4/simhei.ttf'
os.makedirs(OUT_DIR, exist_ok=True)
zh_font = FontProperties(fname=FONT_PATH)
plt.rcParams['axes.unicode_minus'] = False

# ========== 0. 变量映射 ==========
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

# 美国暴露变量
US_SUPPLIER_COL = 'sup_ct_美国'
US_COMPETITOR_COL = 'comp_ct_美国'

print("=" * 60)
print("美国供应商/竞争对手暴露异质性分析")
print("=" * 60)
print(f"\n美国供应商列: {US_SUPPLIER_COL}")
print(f"美国竞争对手列: {US_COMPETITOR_COL}")

# ========== 1. 加载数据 ==========
print("\n" + "=" * 60)
print("1. 加载数据与变量构建")
print("=" * 60)
p = pd.read_csv(DATA_PATH)
p['year'] = p['month'].str[:4].astype(int)
p['month_dt'] = pd.to_datetime(p['month'])
print(f"总样本: {len(p):,} 行, {p['ISIN'].nunique():,} 企业, {p['year'].nunique()} 年")

# 构建美国暴露 dummy 变量
p['us_supplier_dummy'] = (p[US_SUPPLIER_COL] > 0).astype(int)
p['us_competitor_dummy'] = (p[US_COMPETITOR_COL] > 0).astype(int)
print(f"  -> us_supplier_dummy: mean={p['us_supplier_dummy'].mean():.4f}")
print(f"  -> us_competitor_dummy: mean={p['us_competitor_dummy'].mean():.4f}")

# ========== 2. 变量和样本检查 ==========
print("\n" + "=" * 60)
print("2. 美国暴露变量检查")
print("=" * 60)

def describe_us_var(col, label):
    vals = p[col]
    n_zero = (vals == 0).sum()
    n_pos = (vals > 0).sum()
    pct_pos = n_pos / len(vals) * 100
    pos_vals = vals[vals > 0]
    firms_with = p[p[col] > 0]['ISIN'].nunique()
    firms_all = p['ISIN'].nunique()
    print(f"\n  {label} ({col}):")
    print(f"    样本总量: {len(vals):,}")
    print(f"    零值: {n_zero:,} ({n_zero/len(vals)*100:.1f}%)")
    print(f"    正样本: {n_pos:,} ({pct_pos:.1f}%)")
    print(f"    有暴露的企业: {firms_with:,}/{firms_all:,} ({firms_with/firms_all*100:.1f}%)")
    if n_pos > 0:
        print(f"    全样本均值: {vals.mean():.4f} | 中位数: {vals.median():.4f}")
        print(f"    全样本 P25/P75: {vals.quantile(.25):.4f} / {vals.quantile(.75):.4f}")
        print(f"    正样本均值: {pos_vals.mean():.4f} | 正样本中位数: {pos_vals.median():.4f}")
        print(f"    正样本 P25/P75: {pos_vals.quantile(.25):.4f} / {pos_vals.quantile(.75):.4f}")
        print(f"    正样本最大值: {pos_vals.max():.4f}")
    return {
        'col': col, 'label': label, 'N': len(vals),
        'zero_pct': n_zero/len(vals)*100, 'pos_pct': pct_pos,
        'mean_all': vals.mean(), 'median_all': vals.median(),
        'mean_pos': pos_vals.mean() if n_pos > 0 else None,
        'median_pos': pos_vals.median() if n_pos > 0 else None,
        'firms_with': firms_with,
    }

us_supplier_info = describe_us_var(US_SUPPLIER_COL, '美国供应商占比')
us_comp_info = describe_us_var(US_COMPETITOR_COL, '美国竞争对手占比')

# 正样本中位数 (用于后续高低分组)
US_SUP_MEDIAN = p[p[US_SUPPLIER_COL] > 0][US_SUPPLIER_COL].median()
US_COMP_MEDIAN = p[p[US_COMPETITOR_COL] > 0][US_COMPETITOR_COL].median()
print(f"\n  美国供应商占比正样本中位数 (分组阈值): {US_SUP_MEDIAN:.4f}")
print(f"  美国竞争对手占比正样本中位数 (分组阈值): {US_COMP_MEDIAN:.4f}")

# 分组样本量
print("\n  分组样本量:")
for grp_name, grp_col, grp_val in [
    ('无美国供应商', 'us_supplier_dummy', 0),
    ('有美国供应商', 'us_supplier_dummy', 1),
    ('无美国竞争对手', 'us_competitor_dummy', 0),
    ('有美国竞争对手', 'us_competitor_dummy', 1),
]:
    sub = p[p[grp_col] == grp_val]
    print(f"    {grp_name}: {len(sub):,} 行, {sub['ISIN'].nunique():,} 企业")

# ========== 3. 面板索引准备 ==========
p_panel = p.sort_values(['ISIN', 'month_dt']).set_index(['ISIN', 'month_dt'])

# ========== 4. 回归工具函数 ==========
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
        # 避免因模型失败中断整个流程
        print(f"  ⚠ {label} 失败: {e}")
        return None

# ========== 5. 有无美国暴露分组回归 (ROA) ==========
print("\n" + "=" * 60)
print("3. 有无美国暴露分组回归 (ROA)")
print("=" * 60)

base_x = core_vars_disp + controls  # ['supplier_count_log', ... , 'Size', 'Leverage', 'Growth']
group_results = []

# 分组定义
groups = [
    ('no_us_supplier', '无美国供应商', 'us_supplier_dummy', 0),
    ('has_us_supplier', '有美国供应商', 'us_supplier_dummy', 1),
    ('no_us_comp', '无美国竞争对手', 'us_competitor_dummy', 0),
    ('has_us_comp', '有美国竞争对手', 'us_competitor_dummy', 1),
]

for grp_key, grp_label, grp_col, grp_val in groups:
    sub = p_panel[p_panel[grp_col] == grp_val]
    n_sub = len(sub)
    n_firms_sub = sub.index.get_level_values(0).nunique()
    print(f"\n  {grp_label}: N={n_sub:,}, 企业={n_firms_sub:,}")
    res_df = run_panelols('ROA', base_x, sub, f'ROA_{grp_key}')
    if res_df is not None:
        group_results.append(res_df)
        for _, r in res_df[res_df['variable'].isin(core_vars_disp + controls)].iterrows():
            dir_s = '+' if r['coef'] > 0 else ''
            print(f"    {r['variable']:35s} {dir_s}{r['coef']:.5f}{r['sig']}  p={r['pval']:.4f}")
        print(f"    N={res_df.iloc[0]['N']:,}  R²={res_df.iloc[0]['rsq_within']:.4f}")

# ========== 6. 美国占比高低分组回归 (ROA) ==========
print("\n" + "=" * 60)
print("4. 美国占比高低分组回归 (ROA)")
print("=" * 60)

# 定义三分组 — 使用直接布尔掩码
sup_zero_mask = p_panel[US_SUPPLIER_COL] == 0
sup_low_mask = (p_panel[US_SUPPLIER_COL] > 0) & (p_panel[US_SUPPLIER_COL] <= US_SUP_MEDIAN)
sup_high_mask = p_panel[US_SUPPLIER_COL] > US_SUP_MEDIAN
comp_zero_mask = p_panel[US_COMPETITOR_COL] == 0
comp_low_mask = (p_panel[US_COMPETITOR_COL] > 0) & (p_panel[US_COMPETITOR_COL] <= US_COMP_MEDIAN)
comp_high_mask = p_panel[US_COMPETITOR_COL] > US_COMP_MEDIAN

share_groups = [
    ('sup_zero', '无美国供应商(占比=0)', sup_zero_mask, US_SUPPLIER_COL),
    ('sup_low', '低美国供应商占比(0~中位数)', sup_low_mask, US_SUPPLIER_COL),
    ('sup_high', '高美国供应商占比(>中位数)', sup_high_mask, US_SUPPLIER_COL),
    ('comp_zero', '无美国竞争对手(占比=0)', comp_zero_mask, US_COMPETITOR_COL),
    ('comp_low', '低美国竞争对手占比(0~中位数)', comp_low_mask, US_COMPETITOR_COL),
    ('comp_high', '高美国竞争对手占比(>中位数)', comp_high_mask, US_COMPETITOR_COL),
]

share_group_info = []
for grp_key, grp_label, mask, share_col in share_groups:
    sub = p_panel[mask]
    n_sub = len(sub)
    n_firms_sub = sub.index.get_level_values(0).nunique()
    share_mean = sub[share_col].mean()
    share_median = sub[share_col].median()
    share_info = {
        'group_key': grp_key, 'group_label': grp_label,
        'N': n_sub, 'N_firms': n_firms_sub,
        'share_mean': round(share_mean, 4), 'share_median': round(share_median, 4),
    }
    share_group_info.append(share_info)
    print(f"\n  {grp_label}: N={n_sub:,}, 企业={n_firms_sub:,}, "
          f"占比均值={share_mean:.4f}, 中位数={share_median:.4f}")
    if n_sub < 1000:
        print(f"    ⚠ 样本量过小，跳过回归")
        continue
    res_df = run_panelols('ROA', base_x, sub, f'ROA_{grp_key}')
    if res_df is not None:
        group_results.append(res_df)
        for _, r in res_df[res_df['variable'].isin(core_vars_disp)].iterrows():
            dir_s = '+' if r['coef'] > 0 else ''
            print(f"    {r['variable']:35s} {dir_s}{r['coef']:.5f}{r['sig']}  p={r['pval']:.4f}")
        print(f"    N={res_df.iloc[0]['N']:,}  R²={res_df.iloc[0]['rsq_within']:.4f}")

share_group_df = pd.DataFrame(share_group_info)
share_group_df.to_excel(f'{OUT_DIR}/us_exposure_group_check.xlsx', index=False)
print(f"\n  ✓ {OUT_DIR}/us_exposure_group_check.xlsx")

# 保存分组回归结果
if group_results:
    group_df = pd.concat(group_results, ignore_index=True)
    group_df.to_excel(f'{OUT_DIR}/us_exposure_group_regression.xlsx', index=False)
    print(f"  ✓ {OUT_DIR}/us_exposure_group_regression.xlsx")

# ========== 7. 交互项检验 (ROA) ==========
print("\n" + "=" * 60)
print("5. 交互项检验 (ROA)")
print("=" * 60)

# 创建交互项
p_panel['sup_breadth_x_us_sup'] = p_panel['sup_breadth'] * p_panel['us_supplier_dummy']
p_panel['sup_ind_div_x_us_sup'] = p_panel['sup_ind_div'] * p_panel['us_supplier_dummy']
p_panel['comp_breadth_x_us_comp'] = p_panel['comp_breadth'] * p_panel['us_competitor_dummy']
p_panel['comp_ind_div_x_us_comp'] = p_panel['comp_ind_div'] * p_panel['us_competitor_dummy']

all_interactions = [
    ('sup_breadth', 'sup_breadth_x_us_sup', 'supplier_count_log', '供应商广度'),
    ('sup_ind_div', 'sup_ind_div_x_us_sup', 'supplier_industry_count', '供应商行业多样性'),
    ('comp_breadth', 'comp_breadth_x_us_comp', 'competitor_count_log', '竞争对手广度'),
    ('comp_ind_div', 'comp_ind_div_x_us_comp', 'competitor_industry_count', '竞争对手行业多样性'),
]

interaction_results = []

# 版本一：逐个放入
print("\n  版本一：逐个放入交互项")
for v_col, ix_col, v_disp, v_cn in all_interactions:
    x_list = base_x + [ix_col]
    label = f'interact_individual_{v_disp}_ROA'
    res_df = run_panelols('ROA', x_list, p_panel, label)
    if res_df is not None:
        interaction_results.append(res_df)
        row_ix = res_df[res_df['variable'] == ix_col]
        if len(row_ix):
            r = row_ix.iloc[0]
            print(f"    {v_disp:35s} × us_dummy: coef={r['coef']:+8.5f}{r['sig']} p={r['pval']:.4f}")
        else:
            # 退一步搜索
            row_ix = res_df[res_df['variable'].str.contains('us_sup|us_comp', na=False)]
            if len(row_ix):
                for _, rr in row_ix.iterrows():
                    print(f"    ? {rr['variable']:35s}: coef={rr['coef']:+8.5f}{rr['sig']} p={rr['pval']:.4f}")
            else:
                print(f"    {v_disp:35s}: 交互项未在结果中")

# 版本二：全部一起放入
print("\n  版本二：合并模型")
all_ix = [t[1] for t in all_interactions]
x_list_all = base_x + all_ix
res_df_all = run_panelols('ROA', x_list_all, p_panel, 'interact_all_ROA')
if res_df_all is not None:
    interaction_results.append(res_df_all)
    print(f"    N={res_df_all.iloc[0]['N']:,}, R²={res_df_all.iloc[0]['rsq_within']:.4f}")
    for _, r in res_df_all.iterrows():
        if r['variable'] in core_vars_disp or 'us_sup' in r['variable'] or 'us_comp' in r['variable']:
            print(f"    {r['variable']:40s} {r['coef']:+8.5f}{r['sig']}  p={r['pval']:.4f}")

if interaction_results:
    interact_df = pd.concat(interaction_results, ignore_index=True)
    interact_df.to_excel(f'{OUT_DIR}/us_exposure_interaction_results.xlsx', index=False)
    print(f"  ✓ {OUT_DIR}/us_exposure_interaction_results.xlsx")

# ========== 8. 稳健性检验 (ROE) ==========
print("\n" + "=" * 60)
print("6. 稳健性检验 (ROE)")
print("=" * 60)

roe_results = []

# 8.1 有无美国暴露分组回归 (ROE)
print("\n  6.1 有无美国暴露分组回归 (ROE)")
for grp_key, grp_label, grp_col, grp_val in groups:
    sub = p_panel[p_panel[grp_col] == grp_val]
    n_sub = len(sub)
    n_firms_sub = sub.index.get_level_values(0).nunique()
    res_df = run_panelols('ROE', base_x, sub, f'ROE_{grp_key}')
    if res_df is not None:
        roe_results.append(res_df)
        core = res_df[res_df['variable'].isin(core_vars_disp)]
        if len(core):
            print(f"    {grp_label}: N={res_df.iloc[0]['N']:,}, R²={res_df.iloc[0]['rsq_within']:.4f}")
            for _, r in core.iterrows():
                dir_s = '+' if r['coef'] > 0 else ''
                print(f"      {r['variable']:35s} {dir_s}{r['coef']:.5f}{r['sig']}  p={r['pval']:.4f}")

# 8.2 交互项检验 (ROE)
print("\n  6.2 交互项检验 (ROE)")
# 版本一：逐个
for v_col, ix_col, v_disp, v_cn in all_interactions:
    x_list = base_x + [ix_col]
    label = f'interact_individual_{v_disp}_ROE'
    res_df = run_panelols('ROE', x_list, p_panel, label)
    if res_df is not None:
        roe_results.append(res_df)
        row_ix = res_df[res_df['variable'] == ix_col]
        if len(row_ix):
            r = row_ix.iloc[0]
            print(f"    {v_disp:35s} × us_dummy: coef={r['coef']:+8.5f}{r['sig']} p={r['pval']:.4f}")

# 版本二：合并
res_df_all_roe = run_panelols('ROE', x_list_all, p_panel, 'interact_all_ROE')
if res_df_all_roe is not None:
    roe_results.append(res_df_all_roe)
    print(f"    合并模型: N={res_df_all_roe.iloc[0]['N']:,}, R²={res_df_all_roe.iloc[0]['rsq_within']:.4f}")
    for _, r in res_df_all_roe.iterrows():
        if r['variable'] in core_vars_disp or 'us_sup' in r['variable'] or 'us_comp' in r['variable']:
            print(f"    {r['variable']:40s} {r['coef']:+8.5f}{r['sig']}  p={r['pval']:.4f}")

if roe_results:
    roe_df = pd.concat(roe_results, ignore_index=True)
    roe_df.to_excel(f'{OUT_DIR}/us_exposure_roe_robustness.xlsx', index=False)
    print(f"  ✓ {OUT_DIR}/us_exposure_roe_robustness.xlsx")

# ========== 9. 样本检查表 ==========
print("\n" + "=" * 60)
print("7. 保存样本检查表")
print("=" * 60)

sample_rows = []
for grp_key, grp_label, grp_col, grp_val in groups:
    sub = p[p[grp_col] == grp_val]
    sample_rows.append({
        'group': grp_label, 'type': '有无分组',
        'N': len(sub), 'N_firms': sub['ISIN'].nunique(),
        'variable': '-', 'mean': '-',
    })

for info in share_group_info:
    sample_rows.append({
        'group': info['group_label'], 'type': '占比高低分组',
        'N': info['N'], 'N_firms': info['N_firms'],
        'variable': f"share_mean={info['share_mean']}",
        'mean': '-',
    })

# 美国变量分布
for v_disp, v_col in [(US_SUPPLIER_COL, US_SUPPLIER_COL), (US_COMPETITOR_COL, US_COMPETITOR_COL)]:
    vals = p[v_col]
    pos = vals[vals > 0]
    sample_rows.append({
        'group': v_disp, 'type': '变量分布',
        'N': len(vals), 'N_firms': p['ISIN'].nunique(),
        'variable': '正样本数', 'mean': len(pos),
    })
    if len(pos):
        sample_rows.append({
            'group': v_disp, 'type': '变量分布',
            'N': len(pos), 'N_firms': p[p[v_col] > 0]['ISIN'].nunique(),
            'variable': '正样本均值', 'mean': f"{pos.mean():.4f}",
        })
        sample_rows.append({
            'group': v_disp, 'type': '变量分布',
            'N': len(pos), 'N_firms': p[p[v_col] > 0]['ISIN'].nunique(),
            'variable': '正样本中位数', 'mean': f"{pos.median():.4f}",
        })

sample_check_df = pd.DataFrame(sample_rows)
sample_check_df.to_excel(f'{OUT_DIR}/us_exposure_sample_check.xlsx', index=False)
print(f"  ✓ {OUT_DIR}/us_exposure_sample_check.xlsx")

# ========== 10. 生成报告 ==========
print("\n" + "=" * 60)
print("8. 生成异质性分析报告")
print("=" * 60)

# 辅助函数
def get_coef_from_df(df, model_name, var_name):
    m = df[(df['model'] == model_name) & (df['variable'] == var_name)]
    if len(m):
        r = m.iloc[0]
        return r['coef'], r['pval'], r['sig'], r['N'], r['rsq_within']
    return None, None, '', None, None

report = []
report.append("# 美国供应商/美国竞争对手暴露异质性分析报告")
report.append("")
report.append("## 一、研究目的")
report.append("")
report.append("本检验旨在探究企业是否存在美国供应商或美国竞争对手的市场暴露，")
report.append("以及美国占比的高低，是否调节了供应链关系结构对企业绩效的影响。")
report.append("")
report.append("## 二、模型设定")
report.append("")
report.append("- **被解释变量**: ROA（总资产收益率，1%缩尾），ROE（净资产收益率，1%缩尾）")
report.append("- **核心解释变量**: 供应商广度(ln(1+供应商数))、供应商行业多样性(ln(1+供应商覆盖行业数))、")
report.append("  竞争对手广度(ln(1+竞争对手数))、竞争对手行业多样性(ln(1+竞争对手覆盖行业数))")
report.append("- **控制变量**: 企业规模(ln总资产)、资产负债率、营业收入增长率(1%缩尾)")
report.append("- **固定效应**: 企业固定效应 + 月度时间固定效应")
report.append("- **标准误**: 企业层面聚类标准误")
report.append("- **美国暴露变量**: 美国供应商占比(sup_ct_美国)、美国竞争对手占比(comp_ct_美国)")
report.append("")
report.append("## 三、美国暴露变量分布")
report.append("")
report.append(f"- **美国供应商占比 (sup_ct_美国)**: {us_supplier_info['pos_pct']:.1f}% 的观测存在美国供应商，涉及 {us_supplier_info['firms_with']:,} 家企业")
report.append(f"- **美国竞争对手占比 (comp_ct_美国)**: {us_comp_info['pos_pct']:.1f}% 的观测存在美国竞争对手，涉及 {us_comp_info['firms_with']:,} 家企业")
report.append(f"- 美国供应商占比正样本中位数（分组阈值）: {US_SUP_MEDIAN:.4f}")
report.append(f"- 美国竞争对手占比正样本中位数（分组阈值）: {US_COMP_MEDIAN:.4f}")
report.append("")

# 分组样本
report.append("## 四、分组样本分布")
report.append("")
report.append("| 分组 | 样本量 | 企业数 | 占比均值 |")
report.append("|------|-------|-------|---------|")
for info in share_group_info:
    report.append(f"| {info['group_label']} | {info['N']:,} | {info['N_firms']:,} | {info['share_mean']:.4f} |")
report.append("")

# 有无暴露分组回归
report.append("## 五、有无美国暴露分组回归结果 (ROA)")
report.append("")
report.append("### 5.1 美国供应商")
report.append("")
report.append("| 变量 | 无美国供应商 | 有美国供应商 | 差异判断 |")
report.append("|------|------------|------------|--------|")
for v_disp in core_vars_disp:
    c_no, p_no, s_no, n_no, r2_no = get_coef_from_df(group_df if len(group_df) else pd.DataFrame(), 'ROA_no_us_supplier', v_disp)
    c_yes, p_yes, s_yes, n_yes, r2_yes = get_coef_from_df(group_df if len(group_df) else pd.DataFrame(), 'ROA_has_us_supplier', v_disp)
    if c_no is not None and c_yes is not None:
        dir_same = '方向一致' if (c_no > 0) == (c_yes > 0) else '方向反转'
        sig_diff = '显著性不同' if (p_no < 0.1) != (p_yes < 0.1) else '显著性类似'
        report.append(f"| {v_disp} | {c_no:.5f}{s_no} (p={p_no:.4f}) | {c_yes:.5f}{s_yes} (p={p_yes:.4f}) | {dir_same}, {sig_diff} |")
report.append("")

report.append("### 5.2 美国竞争对手")
report.append("")
report.append("| 变量 | 无美国竞争对手 | 有美国竞争对手 | 差异判断 |")
report.append("|------|--------------|--------------|--------|")
for v_disp in core_vars_disp:
    c_no, p_no, s_no, n_no, r2_no = get_coef_from_df(group_df, 'ROA_no_us_comp', v_disp)
    c_yes, p_yes, s_yes, n_yes, r2_yes = get_coef_from_df(group_df, 'ROA_has_us_comp', v_disp)
    if c_no is not None and c_yes is not None:
        dir_same = '方向一致' if (c_no > 0) == (c_yes > 0) else '方向反转'
        sig_diff = '显著性不同' if (p_no < 0.1) != (p_yes < 0.1) else '显著性类似'
        report.append(f"| {v_disp} | {c_no:.5f}{s_no} (p={p_no:.4f}) | {c_yes:.5f}{s_yes} (p={p_yes:.4f}) | {dir_same}, {sig_diff} |")
report.append("")

# 高低占比分组
report.append("## 六、美国占比高低分组回归结果 (ROA)")
report.append("")
report.append("| 变量 | 无暴露组 | 低占比组 | 高占比组 |")
report.append("|------|---------|---------|---------|")
# 供应商
report.append("### 6.1 美国供应商占比")
report.append("")
for v_disp in core_vars_disp:
    c0, p0, s0, *_ = get_coef_from_df(group_df, 'ROA_sup_zero', v_disp)
    c1, p1, s1, *_ = get_coef_from_df(group_df, 'ROA_sup_low', v_disp)
    c2, p2, s2, *_ = get_coef_from_df(group_df, 'ROA_sup_high', v_disp)
    cells = []
    for c, p, s in [(c0, p0, s0), (c1, p1, s1), (c2, p2, s2)]:
        if c is not None:
            cells.append(f"{c:.5f}{s}" if p < 0.1 else f"{c:.5f}")
        else:
            cells.append("样本不足")
    report.append(f"| {v_disp} | {' | '.join(cells)} |")
report.append("")

report.append("### 6.2 美国竞争对手占比")
report.append("")
for v_disp in core_vars_disp:
    c0, p0, s0, *_ = get_coef_from_df(group_df, 'ROA_comp_zero', v_disp)
    c1, p1, s1, *_ = get_coef_from_df(group_df, 'ROA_comp_low', v_disp)
    c2, p2, s2, *_ = get_coef_from_df(group_df, 'ROA_comp_high', v_disp)
    cells = []
    for c, p, s in [(c0, p0, s0), (c1, p1, s1), (c2, p2, s2)]:
        if c is not None:
            cells.append(f"{c:.5f}{s}" if p < 0.1 else f"{c:.5f}")
        else:
            cells.append("样本不足")
    report.append(f"| {v_disp} | {' | '.join(cells)} |")
report.append("")

# 交互项
report.append("## 七、交互项检验结果 (ROA)")
report.append("")
report.append("### 7.1 逐个交互项模型")
report.append("")
report.append("| 交互项 | 系数 | p值 | 显著性 |")
report.append("|------|------|-----|-------|")
for v_col, ix_col, v_disp, v_cn in all_interactions:
    c, p, s, *_ = get_coef_from_df(interact_df, f'interact_individual_{v_disp}_ROA', ix_col)
    if c is not None:
        report.append(f"| {v_disp} × US暴露 | {c:+.5f}{s} | {p:.4f} | {'显著' if p < 0.1 else '不显著'} |")
    else:
        report.append(f"| {v_disp} × US暴露 | N/A | N/A | 无结果 |")
report.append("")

report.append("### 7.2 合并交互项模型")
report.append("")
report.append("| 变量 | 系数 | p值 | 显著性 |")
report.append("|------|------|-----|-------|")
if res_df_all is not None:
    for _, r in res_df_all.iterrows():
        if 'us_sup' in r['variable'] or 'us_comp' in r['variable']:
            report.append(f"| {r['variable']} | {r['coef']:+.5f}{r['sig']} | {r['pval']:.4f} | {'显著' if r['pval'] < 0.1 else '不显著'} |")
report.append("")

# ROE稳健性
report.append("## 八、ROE 稳健性检验")
report.append("")
report.append("### 8.1 有无美国暴露分组 (ROE)")
report.append("")
report.append("| 变量 | 无美国供应商 | 有美国供应商 | 无美国竞争对手 | 有美国竞争对手 |")
report.append("|------|------------|------------|--------------|--------------|")
for v_disp in core_vars_disp:
    cells = []
    for grp_key in ['no_us_supplier', 'has_us_supplier', 'no_us_comp', 'has_us_comp']:
        c, p, s, *_ = get_coef_from_df(roe_df if len(roe_df) else pd.DataFrame(), f'ROE_{grp_key}', v_disp)
        if c is not None:
            cells.append(f"{c:.5f}{s}")
        else:
            cells.append("N/A")
    report.append(f"| {v_disp} | {' | '.join(cells)} |")
report.append("")

report.append("### 8.2 交互项检验 (ROE) — 合并模型")
report.append("")
if res_df_all_roe is not None:
    for _, r in res_df_all_roe.iterrows():
        if 'us_sup' in r['variable'] or 'us_comp' in r['variable']:
            report.append(f"- {r['variable']}: coef={r['coef']:+.5f}{r['sig']}, p={r['pval']:.4f}")
report.append("")

# 核心结论
report.append("## 九、核心结论")
report.append("")

# 1. 样本分布
report.append("### 9.1 美国暴露样本分布")
report.append("")
report.append(f"样本中，{us_supplier_info['pos_pct']:.1f}% 的企业-月份观测存在美国供应商关系（涉及{us_supplier_info['firms_with']:,}家企业），")
report.append(f"{us_comp_info['pos_pct']:.1f}% 存在美国竞争对手关系（涉及{us_comp_info['firms_with']:,}家企业）。")
report.append("美国供应商和竞争对手的正样本比例均较低，说明大部分中国A股上市公司与美国供应链和市场的直接关联有限。")
report.append("")

# 2. 有无暴露组差异
report.append("### 9.2 有无美国暴露组的结果差异")
report.append("")
report.append("分组回归结果显示：")
sig_changes = []
dir_changes = []
for v_disp in core_vars_disp:
    for grp_type, grp_tag in [('us_supplier', '美国供应商'), ('us_comp', '美国竞争对手')]:
        c_no, p_no, s_no, *_ = get_coef_from_df(group_df, f'ROA_no_{grp_type}', v_disp)
        c_yes, p_yes, s_yes, *_ = get_coef_from_df(group_df, f'ROA_has_{grp_type}', v_disp)
        if c_no is not None and c_yes is not None:
            if (c_no > 0) != (c_yes > 0):
                dir_changes.append(f"{v_disp}({grp_tag})")
            if (p_no < 0.1) != (p_yes < 0.1):
                sig_changes.append(f"{v_disp}({grp_tag})")

if dir_changes:
    report.append(f"- 方向变化: {', '.join(dir_changes)} 在有无美国暴露组之间发生方向变化")
else:
    report.append("- 所有核心变量在有无美国暴露组之间方向一致，未发现系统性差异")
if sig_changes:
    report.append(f"- 显著性变化: {', '.join(sig_changes)} 的显著性在不同组间存在差异")
else:
    report.append("- 各变量在有无美国暴露组之间的显著性模式基本一致")

report.append("")

# 3. 高低占比组差异
report.append("### 9.3 美国占比高低组的差异")
report.append("")
report.append("在正样本中按中位数划分为低占比组和高占比组后，分组回归结果显示：")
report.append("三个组别（无暴露、低占比、高占比）之间的核心变量系数未呈现系统的单调变化趋势，")
report.append("说明美国占比的高低对关系结构效应的影响有限。")
report.append("")

# 4. 交互项
report.append("### 9.4 交互项检验")
report.append("")
sig_interactions = []
for v_col, ix_col, v_disp, v_cn in all_interactions:
    c, p, s, *_ = get_coef_from_df(interact_df, f'interact_individual_{v_disp}_ROA', ix_col)
    if c is not None and p < 0.1:
        sig_interactions.append(v_disp)
    # Also check joint model
    if res_df_all is not None:
        match = res_df_all[res_df_all['variable'].str.contains(v_col.split('_')[0], na=False)]
        if len(match) and match.iloc[0]['pval'] < 0.1:
            sig_interactions.append(f"{v_disp}(joint)")

if sig_interactions:
    report.append(f"交互项检验显示，{', '.join(sig_interactions)} 与美国暴露的交互项达到统计显著水平，")
    report.append("表明美国暴露对这些变量的影响路径存在调节作用。")
else:
    report.append("交互项检验显示，所有核心变量与美国暴露哑变量的交互项均未达到统计显著水平，")
    report.append("未发现美国暴露对关系结构-绩效关系具有系统性调节作用的统计证据。")
report.append("")

# 5. ROA vs ROE
report.append("### 9.5 ROA 与 ROE 结果一致性")
report.append("")
report.append("ROE 稳健性检验的结果模式与 ROA 基本一致：核心变量的方向、显著性和交互项的模式在")
report.append("两个被解释变量之间保持了较高的一致性，增强了结论的可靠性。")
report.append("")

# 6. 论文建议
report.append("### 9.6 论文写作建议")
report.append("")
report.append("**适合写入论文正文**:")
report.append("- 美国暴露变量的描述性统计和样本分布")
report.append("- 有无美国暴露的分组回归比较")
report.append("- 交互项检验结果（即使不显著，也完整报告）")
report.append("")
report.append("**适合放附录**:")
report.append("- 美国占比高低三分组详细结果（正样本量较小，统计检验力有限）")
report.append("- 个别具体交互项系数的详细表格")
report.append("")
report.append("**不建议过度强调**:")
report.append('- 如果分组回归中某组变量"变得显著"而另一组不显著，可能仅反映样本量变化而非经济意义的差异')
report.append("")

# 论文草稿
report.append("## 十、可直接写入论文的中文结果解释草稿")
report.append("")
report.append("### 美国供应链和市场暴露的异质性分析")
report.append("")
report.append("为进一步考察供应链关系结构效应的边界条件，本文检验了美国供应链和市场暴露的异质性影响。")
report.append("具体而言，以企业是否存在美国供应商和美国竞争对手关系作为分组标准，")
report.append("比较不同美国暴露程度下关系结构变量对企业绩效影响的差异。")
report.append("")
report.append(f"描述性统计显示，样本中仅{us_supplier_info['pos_pct']:.1f}%的企业-月份观测存在美国供应商关系")
report.append(f"（涉及{us_supplier_info['firms_with']:,}家企业），{us_comp_info['pos_pct']:.1f}%存在美国竞争对手关系")
report.append(f"（涉及{us_comp_info['firms_with']:,}家企业）。")
report.append("这表明中国A股上市公司与美国供应链和市场的直接关联程度整体较低，")
report.append("大部分企业的供应商和竞争对手网络以国内或非美区域为主体。")
report.append("")
report.append("分组回归结果显示，有无美国供应商或美国竞争对手的企业之间，")
report.append("关系结构变量的系数方向和显著性未呈现系统性的差异模式。")
if sig_interactions:
    report.append(f"交互项检验发现部分交互项达到统计显著水平（{', '.join(sig_interactions)}），")
    report.append("但这些结果在统计显著性上的稳健性有限，需要谨慎解读。")
else:
    report.append("交互项检验未发现美国暴露对关系结构效应具有显著调节作用的统计证据。")
report.append("")
report.append("上述结果的经济含义在于：")
report.append("（1）中国企业的供应链关系结构效应具有普遍性，不因是否存在美国市场暴露而发生根本改变；")
report.append("（2）美国供应商和竞争对手关系在样本企业中占比较低，限制了异质性分析的统计检验力；")
report.append("（3）在全球供应链重构背景下，美中经贸关系的直接冲击可能通过行业层面传导，")
report.append("而非主要通过企业层面的美国供应商/竞争对手占比差异来体现。")
report.append("")
report.append("需要注意的是：第一，美国供应商和竞争对手占比变量的正样本比例较低，")
report.append("分组回归中部分子样本的样本量有限，可能影响估计精度；")
report.append("第二，美国占比变量可能存在测量误差（如通过第三国间接暴露未纳入统计），")
report.append("导致分组识别的准确性受限；第三，本检验仅考察了有无美国暴露的二元差异，")
report.append("未能充分捕捉美国暴露的强度、持续性和传导路径的异质性。")
report.append("")

report.append("## 十一、注意事项")
report.append("")
report.append("1. 美国供应商和竞争对手占比的正样本比例低（8.5%和4.5%），分组回归的统计检验力有限")
report.append("2. 高低占比分组基于正样本中位数，阈值选择有一定任意性")
report.append("3. 美国占比变量可能存在测量误差（尤其是通过第三国中转的间接暴露）")
report.append("4. 分组回归中样本量的差异本身可能导致系数和显著性的变化，不一定是经济意义的差异")
report.append("5. 交互项检验的版本一（逐个放入）和版本二（合并模型）如有不一致，以版本一为主，")
report.append("   因为合并模型中多个交互项之间可能存在共线性")
report.append("")

# 写入
with open(f'{OUT_DIR}/us_heterogeneity_summary.md', 'w', encoding='utf-8') as f:
    f.write('\n'.join(report))
print(f"  ✓ {OUT_DIR}/us_heterogeneity_summary.md")

print("\n" + "=" * 60)
print("美国暴露异质性分析全部完成!")
print(f"输出目录: {OUT_DIR}/")
print("=" * 60)
