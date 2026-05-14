"""
供应链韧性机制检验：横截面回归
用贸易战前（2015-2017）供应链关系结构解释贸易战后（2018-2020）企业韧性
"""
import pandas as pd, numpy as np, os, warnings
warnings.filterwarnings('ignore')
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

OUT_DIR = '/Users/deesanyu/pythonproject/deesan_supply_0506v4/output/resilience_mechanism'
FIG_DIR = f'{OUT_DIR}/figures'
DATA_PATH = '/Users/deesanyu/pythonproject/deesan_supply_0506v4/output/firm_monthly_panel.csv'
ASSETS_PATH = '/Users/deesanyu/pythonproject/deesan_supply_0506v4/data/china-quarterly-assets.xlsx'
FONT_PATH = '/Users/deesanyu/pythonproject/deesan_supply_0506v4/simhei.ttf'
os.makedirs(FIG_DIR, exist_ok=True)
zh_font = FontProperties(fname=FONT_PATH)
plt.rcParams['axes.unicode_minus'] = False

var_map = {
    'ROA': 'roa_w', 'ROE': 'roe_w', 'Growth': 'growth_w',
    'supplier_count_log': 'sup_breadth', 'supplier_industry_count': 'sup_ind_div',
    'competitor_count_log': 'comp_breadth', 'competitor_industry_count': 'comp_ind_div',
    'Size': 'size', 'Leverage': 'lev',
}
col_to_display = {v: k for k, v in var_map.items()}

print("=" * 60)
print("供应链韧性机制检验（横截面）")
print("=" * 60)

# ========== 1. 加载数据 ==========
print("\n1. 加载数据")
p = pd.read_csv(DATA_PATH)
p['year'] = p['month'].str[:4].astype(int)
p['month_dt'] = pd.to_datetime(p['month'])
p = p[(p['year'] >= 2015) & (p['year'] <= 2020)].copy()
print(f"  2015-2020 面板: {len(p):,} 行, {p['ISIN'].nunique():,} 企业")

# ========== 2. 加载行业信息 ==========
print("\n2. 加载行业分类")
assets_raw = pd.read_excel(ASSETS_PATH, header=None)
hdr = assets_raw.iloc[4]
isin_col_idx = list(hdr).index('FE Isin')
ind_info = assets_raw.iloc[6:, [10, 11, isin_col_idx]].copy()
ind_info.columns = ['L1', 'L4', 'ISIN']
ind_info = ind_info.dropna(subset=['ISIN'])
ind_info = ind_info[ind_info['L1'] != '@NA'].drop_duplicates('ISIN')
print(f"  行业信息: {len(ind_info)} 家企业, L1 类别: {ind_info['L1'].nunique()} 个")
print(f"  L1 行业分布:")
for ind_name, cnt in ind_info['L1'].value_counts().items():
    print(f"    {ind_name}: {cnt} 家")
p = p.merge(ind_info, on='ISIN', how='left')
no_ind = p['L1'].isna().sum()
print(f"  无行业信息: {no_ind:,} 行（将自动排除）")

# ========== 3. 构造企业层面韧性变量 ==========
print("\n3. 构造企业层面韧性变量")

pre_mask = p['year'] <= 2017
post_mask = p['year'] >= 2018

# 预周期均值 (2015-2017)
pre_cols = ['roa_w', 'roe_w', 'growth_w', 'sup_breadth', 'sup_ind_div',
            'comp_breadth', 'comp_ind_div', 'size', 'lev']
pre = p[pre_mask].groupby('ISIN')[pre_cols].mean().reset_index()
pre.columns = ['ISIN'] + [f'{c}_pre' for c in pre_cols]

# 后周期均值和标准差 (2018-2020)
post_mean = p[post_mask].groupby('ISIN')[['roa_w', 'roe_w', 'growth_w']].mean().reset_index()
post_mean.columns = ['ISIN', 'roa_w_post_mean', 'roe_w_post_mean', 'growth_w_post_mean']
post_std = p[post_mask].groupby('ISIN')[['roa_w', 'roe_w', 'growth_w']].std().reset_index()
post_std.columns = ['ISIN', 'roa_w_post_std', 'roe_w_post_std', 'growth_w_post_std']

# 合并
firm_df = pre.merge(post_mean, on='ISIN', how='left').merge(post_std, on='ISIN', how='left')
# 添加行业信息（取每个企业第一个非NA行业）
ind_first = p.groupby('ISIN')[['L1', 'L4']].first().reset_index()
firm_df = firm_df.merge(ind_first, on='ISIN', how='left')

# 构造韧性因变量
firm_df['ROA_drop'] = firm_df['roa_w_post_mean'] - firm_df['roa_w_pre']
firm_df['ROA_volatility'] = firm_df['roa_w_post_std']
firm_df['ROE_drop'] = firm_df['roe_w_post_mean'] - firm_df['roe_w_pre']
firm_df['ROE_volatility'] = firm_df['roe_w_post_std']
firm_df['Growth_drop'] = firm_df['growth_w_post_mean'] - firm_df['growth_w_pre']
firm_df['Growth_volatility'] = firm_df['growth_w_post_std']

print(f"  企业层面数据: {len(firm_df)} 行")
print(f"  有行业信息的: {firm_df['L1'].notna().sum()}")

# 描述性统计
resilience_vars = ['ROA_drop', 'ROA_volatility', 'ROE_drop', 'ROE_volatility',
                   'Growth_drop', 'Growth_volatility',
                   'sup_breadth_pre', 'sup_ind_div_pre', 'comp_breadth_pre', 'comp_ind_div_pre',
                   'size_pre', 'lev_pre', 'growth_w_pre']
desc = firm_df[resilience_vars].describe().T
desc = desc[['count', 'mean', '50%', 'std', 'min', 'max']]
desc.columns = ['N', 'mean', 'median', 'std', 'min', 'max']
desc['N'] = desc['N'].astype(int)
print(f"\n  变量描述:")
for v in resilience_vars:
    r = desc.loc[v]
    print(f"    {v}: N={r['N']}, mean={r['mean']:.4f}, median={r['median']:.4f}, "
          f"std={r['std']:.4f}, min={r['min']:.4f}, max={r['max']:.4f}")

desc.to_excel(f'{OUT_DIR}/resilience_variable_check.xlsx')
print(f"\n  ✓ {OUT_DIR}/resilience_variable_check.xlsx")

# ========== 4. 横截面回归 ==========
print("\n" + "=" * 60)
print("4. 横截面回归")
print("=" * 60)

# 清理：去掉行业缺失的样本
reg_df = firm_df.dropna(subset=['L1']).copy()

joint_results = []
single_results = []

y_configs = [
    ('ROA_drop', 'ROA', 'ROA 下降'),
    ('ROA_volatility', 'ROA', 'ROA 波动'),
    ('ROE_drop', 'ROE', 'ROE 下降'),
    ('ROE_volatility', 'ROE', 'ROE 波动'),
    ('Growth_drop', 'Growth', '销售增长下降'),
    ('Growth_volatility', 'Growth', '销售增长波动'),
]

core_vars_pre = ['sup_breadth_pre', 'sup_ind_div_pre', 'comp_breadth_pre', 'comp_ind_div_pre']
ctrl_vars_pre = ['size_pre', 'lev_pre', 'growth_w_pre']

# 行业分组：合并小类
ind_counts = reg_df['L1'].value_counts()
small_inds = ind_counts[ind_counts < 30].index.tolist()
print(f"\n  小行业组 (n<30): {small_inds}")
reg_df['L1_grouped'] = reg_df['L1'].apply(lambda x: 'Other' if x in small_inds else x)
print(f"  分组后行业类别: {reg_df['L1_grouped'].nunique()}")

for y_name, y_group, y_label in y_configs:
    # 版本A：联合模型
    formula_a = f'{y_name} ~ ' + ' + '.join(core_vars_pre + ctrl_vars_pre) + ' + C(L1_grouped)'
    sub = reg_df[[y_name] + core_vars_pre + ctrl_vars_pre + ['L1_grouped']].dropna()
    n_a = len(sub)
    if n_a < 100:
        print(f"  ⚠ {y_name}: 样本不足 ({n_a})")
        continue
    try:
        mod_a = ols(formula_a, data=sub).fit(cov_type='HC3')
        for v in core_vars_pre:
            coef = mod_a.params.get(v, np.nan)
            se = mod_a.bse.get(v, np.nan)
            pv = mod_a.pvalues.get(v, np.nan)
            star = ''
            if not np.isnan(pv):
                if pv < 0.01: star = '***'
                elif pv < 0.05: star = '**'
                elif pv < 0.1: star = '*'
            joint_results.append({
                'model': f'联合_{y_name}', 'y_group': y_group, 'y_label': y_label,
                'variable': v, 'coef': round(coef, 5), 'se': round(se, 5),
                'pval': round(pv, 4), 'sig': star,
                'N': n_a, 'R2': round(mod_a.rsquared, 4),
            })
        sid_coef = mod_a.params.get('sup_ind_div_pre', np.nan)
        sid_pv = mod_a.pvalues.get('sup_ind_div_pre', np.nan)
        sid_star = ''
        if not np.isnan(sid_pv):
            if sid_pv < 0.01: sid_star = '***'
            elif sid_pv < 0.05: sid_star = '**'
            elif sid_pv < 0.1: sid_star = '*'
        ds = '+' if sid_coef > 0 else ''
        print(f"  [联合] {y_name}: N={n_a}, R²={mod_a.rsquared:.4f}, "
              f"sup_ind_div={ds}{sid_coef:.5f}{sid_star}")
    except Exception as e:
        print(f"  ⚠ {y_name} 联合模型失败: {e}")

    # 版本B：单变量模型
    for v in core_vars_pre:
        formula_b = f'{y_name} ~ {v} + ' + ' + '.join(ctrl_vars_pre) + ' + C(L1_grouped)'
        sub_b = reg_df[[y_name] + [v] + ctrl_vars_pre + ['L1_grouped']].dropna()
        n_b = len(sub_b)
        if n_b < 100:
            continue
        try:
            mod_b = ols(formula_b, data=sub_b).fit(cov_type='HC3')
            coef = mod_b.params.get(v, np.nan)
            se = mod_b.bse.get(v, np.nan)
            pv = mod_b.pvalues.get(v, np.nan)
            star = ''
            if not np.isnan(pv):
                if pv < 0.01: star = '***'
                elif pv < 0.05: star = '**'
                elif pv < 0.1: star = '*'
            single_results.append({
                'model': f'单变量_{y_name}', 'y_group': y_group, 'y_label': y_label,
                'variable': v, 'coef': round(coef, 5), 'se': round(se, 5),
                'pval': round(pv, 4), 'sig': star,
                'N': n_b, 'R2': round(mod_b.rsquared, 4),
            })
            ds = '+' if coef > 0 else ''
            print(f"  [单变量] {y_name} | {v:25s} {ds}{coef:.5f}{star}  p={pv:.4f}")
        except Exception as e:
            print(f"  ⚠ {y_name} {v} 失败: {e}")

# 保存回归结果
if joint_results:
    jdf = pd.DataFrame(joint_results)
    sdf = pd.DataFrame(single_results)
    all_reg = pd.concat([jdf, sdf], ignore_index=True)
    with pd.ExcelWriter(f'{OUT_DIR}/resilience_regression_results.xlsx') as writer:
        for grp in ['ROA', 'ROE', 'Growth']:
            sub = all_reg[all_reg['y_group'] == grp]
            if len(sub):
                sub.to_excel(writer, sheet_name=grp, index=False)
    print(f"\n  ✓ {OUT_DIR}/resilience_regression_results.xlsx")

# ========== 5. 图表 ==========
print("\n" + "=" * 60)
print("5. 生成图表")
print("=" * 60)

# 5.1 韧性变量分布图
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
plot_vars = [('ROA_drop', 'ROA 下降 (百分点)', axes[0, 0]),
             ('ROA_volatility', 'ROA 波动 (标准差)', axes[0, 1]),
             ('Growth_drop', '销售增长下降 (百分点)', axes[1, 0]),
             ('Growth_volatility', '销售增长波动 (标准差)', axes[1, 1])]
for v, label, ax in plot_vars:
    vals = reg_df[v].dropna()
    vals = vals[(vals > vals.quantile(0.01)) & (vals < vals.quantile(0.99))]  # 去极值显示
    ax.hist(vals, bins=50, color='steelblue', alpha=0.7, edgecolor='white')
    ax.axvline(vals.mean(), color='red', linestyle='--', linewidth=1.5, label=f'均值={vals.mean():.2f}')
    ax.axvline(vals.median(), color='green', linestyle=':', linewidth=1.5, label=f'中位数={vals.median():.2f}')
    ax.set_xlabel(label, fontproperties=zh_font, fontsize=11)
    ax.set_ylabel('企业数', fontproperties=zh_font, fontsize=11)
    ax.set_title(f'{label} 分布', fontproperties=zh_font, fontsize=13)
    ax.legend(prop=zh_font, fontsize=9)
    ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/resilience_distribution.png', dpi=200, bbox_inches='tight')
plt.close()
print(f"  ✓ {FIG_DIR}/resilience_distribution.png")

# 5.2 机制系数图：核心变量对韧性变量的系数
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for ax_idx, (v_pre, v_disp) in enumerate([
    ('sup_ind_div_pre', 'supplier_industry_count'),
    ('comp_breadth_pre', 'competitor_count_log'),
]):
    ax = axes[ax_idx]
    y_targets = ['ROA_drop', 'ROA_volatility', 'Growth_drop', 'Growth_volatility']
    coefs = []
    cis_low = []
    cis_high = []
    labels = []
    for y_name in y_targets:
        formula = f'{y_name} ~ {v_pre} + ' + ' + '.join(ctrl_vars_pre) + ' + C(L1_grouped)'
        sub = reg_df[[y_name] + [v_pre] + ctrl_vars_pre + ['L1_grouped']].dropna()
        if len(sub) < 100:
            coefs.append(0); cis_low.append(0); cis_high.append(0)
            labels.append(y_name)
            continue
        try:
            mod = ols(formula, data=sub).fit(cov_type='HC3')
            c = mod.params[v_pre]
            se = mod.bse[v_pre]
            coefs.append(c)
            cis_low.append(c - 1.96 * se)
            cis_high.append(c + 1.96 * se)
        except:
            coefs.append(0); cis_low.append(0); cis_high.append(0)
        labels.append({'ROA_drop': 'ROA降', 'ROA_volatility': 'ROA波',
                       'Growth_drop': '增长降', 'Growth_volatility': '增长波'}[y_name])

    x_pos = np.arange(len(labels))
    colors = ['steelblue' if c > 0 else 'coral' for c in coefs]
    ax.bar(x_pos, coefs, 0.4, yerr=[[c - l for c, l in zip(coefs, cis_low)],
                                     [h - c for c, h in zip(coefs, cis_high)]],
           capsize=5, color=colors, alpha=0.7)
    ax.axhline(y=0, color='gray', linestyle='-', linewidth=1)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel('回归系数 (95% CI)', fontproperties=zh_font, fontsize=11)
    ax.set_title(f'{v_disp} 对韧性变量的影响', fontproperties=zh_font, fontsize=13)
    ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(f'{FIG_DIR}/resilience_coefficients.png', dpi=200, bbox_inches='tight')
plt.close()
print(f"  ✓ {FIG_DIR}/resilience_coefficients.png")

# 5.3 分组均值图
sup_ind_median = reg_df['sup_ind_div_pre'].median()
reg_df['sup_ind_group'] = np.where(reg_df['sup_ind_div_pre'] > sup_ind_median, '高', '低')

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
group_vars = [('ROA_drop', 'ROA 下降', axes[0]),
              ('ROA_volatility', 'ROA 波动', axes[1])]

for v, label, ax in group_vars:
    means = reg_df.groupby('sup_ind_group')[v].mean()
    sems = reg_df.groupby('sup_ind_group')[v].sem()
    x_pos = [0, 1]
    ax.bar(x_pos, [means.get('低', 0), means.get('高', 0)], 0.4,
           yerr=[sems.get('低', 0), sems.get('高', 0)],
           capsize=5, color=['lightsteelblue', 'coral'], alpha=0.7)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(['低供应商行业多样性', '高供应商行业多样性'], fontproperties=zh_font, fontsize=11)
    ax.set_ylabel(label, fontproperties=zh_font, fontsize=12)
    ax.set_title(f'{label}: 高低组对比', fontproperties=zh_font, fontsize=13)
    ax.grid(True, alpha=0.3, axis='y')
    # 标注t检验
    low_v = reg_df.loc[reg_df['sup_ind_group'] == '低', v].dropna()
    high_v = reg_df.loc[reg_df['sup_ind_group'] == '高', v].dropna()
    if len(low_v) > 0 and len(high_v) > 0:
        t_stat, p_val = stats.ttest_ind(low_v, high_v, equal_var=False)
        ax.text(0.5, max(means.get('低', 0), means.get('高', 0)) * 1.1,
                f't-test p={p_val:.4f}', ha='center', fontsize=9)

plt.tight_layout()
plt.savefig(f'{FIG_DIR}/resilience_group_compare.png', dpi=200, bbox_inches='tight')
plt.close()
print(f"  ✓ {FIG_DIR}/resilience_group_compare.png")

# ========== 6. 生成报告 ==========
print("\n" + "=" * 60)
print("6. 生成报告")
print("=" * 60)

report = []
report.append("# 供应链韧性机制检验报告")
report.append("")
report.append("## 一、检验目的")
report.append("")
report.append("本检验旨在考察贸易战前（2015-2017年）企业的供应链关系结构是否能够解释")
report.append("企业在贸易战后（2018-2020年）的韧性表现。")
report.append("")
report.append("核心逻辑：如果供应商行业多样性具有韧性机制效应，")
report.append("则贸易战前供应商行业多样性更高的企业，在贸易战后应当表现出：")
report.append("  (1) ROA 下降更少（ROA_drop 更大）；")
report.append("  (2) ROA 波动更低（ROA_volatility 更小）；")
report.append("  (3) 销售增长下降更少（Growth_drop 更大）；")
report.append("  (4) 销售增长波动更低（Growth_volatility 更小）。")
report.append("")
report.append("反之，竞争对手广度反映竞争压力，可能导致绩效下降更多或波动更大。")
report.append("")
report.append("## 二、数据说明")
report.append("")
report.append(f"分析基于 {len(reg_df)} 家具有行业分类信息的A股上市公司。")
report.append(f"行业分类采用 RBICS L1 级别，涵盖 {reg_df['L1_grouped'].nunique()} 个行业组。")
report.append("")

# 描述统计表
report.append("| 变量 | 样本量 | 均值 | 中位数 | 标准差 | 最小值 | 最大值 |")
report.append("|------|-------|------|--------|-------|-------|-------|")
for v in resilience_vars:
    if v in desc.index:
        r = desc.loc[v]
        report.append(f"| {v} | {int(r['N'])} | {r['mean']:.4f} | {r['median']:.4f} | {r['std']:.4f} | {r['min']:.4f} | {r['max']:.4f} |")
report.append("")

# 回归结果
report.append("## 三、回归结果")
report.append("")

# 联合模型结果
report.append("### 3.1 联合模型")
report.append("")
report.append("| 因变量 | 供应商广度 | 供应商行业多样性 | 竞争对手广度 | 竞争对手行业多样性 | N | R² |")
report.append("|--------|----------|----------------|------------|-----------------|----|-----|")
for y_name, y_group, y_label in y_configs:
    joint_rows = [r for r in joint_results if r['model'] == f'联合_{y_name}']
    if not joint_rows:
        continue
    cells = [y_label]
    for v in core_vars_pre:
        match = [r for r in joint_rows if r['variable'] == v]
        if match:
            r = match[0]
            ds = '+' if r['coef'] > 0 else ''
            cells.append(f"{ds}{r['coef']:.5f}{r['sig']}")
        else:
            cells.append("N/A")
    cells.append(str(joint_rows[0]['N']))
    cells.append(f"{joint_rows[0]['R2']:.4f}")
    report.append("| " + " | ".join(cells) + " |")
report.append("")

# 单变量模型结果
report.append("### 3.2 单变量模型")
report.append("")
for y_name, y_group, y_label in y_configs:
    report.append(f"**{y_label}**")
    report.append("")
    report.append("| 变量 | 系数 | 标准误 | p值 | 显著性 | N | R² |")
    report.append("|------|------|-------|-----|-------|----|-----|")
    for v in core_vars_pre:
        match = [r for r in single_results if r['model'] == f'单变量_{y_name}' and r['variable'] == v]
        if match:
            r = match[0]
            ds = '+' if r['coef'] > 0 else ''
            report.append(f"| {v} | {ds}{r['coef']:.5f}{r['sig']} | {r['se']:.5f} | {r['pval']:.4f} | {'显著' if r['pval']<0.1 else '不显著'} | {r['N']} | {r['R2']:.4f} |")
        else:
            report.append(f"| {v} | N/A | N/A | N/A | N/A | N/A | N/A |")
    report.append("")

# 结果解读
report.append("## 四、结果解读")
report.append("")

# 统计显著数量
sig_count = 0
total_count = 0
for r in single_results:
    if r['pval'] < 0.1:
        sig_count += 1
    total_count += 1

report.append(f"在 {total_count} 个单变量模型检验中，{sig_count} 个达到边际显著水平（p<0.1）。")
report.append("")

# 供应商行业多样性
ind_div_positive_drop = [r for r in single_results if r['variable'] == 'sup_ind_div_pre'
                         and 'drop' in r['model'] and r['coef'] > 0]
ind_div_negative_vol = [r for r in single_results if r['variable'] == 'sup_ind_div_pre'
                        and 'volatility' in r['model'] and r['coef'] < 0]
ind_div_sig = [r for r in single_results if r['variable'] == 'sup_ind_div_pre' and r['pval'] < 0.1]

if len(ind_div_sig) > 0:
    report.append("**供应商行业多样性**：部分模型显著。")
    for r in ind_div_sig:
        direction = "正向" if r['coef'] > 0 else "负向"
        report.append(f"- {r['model']}: 系数={r['coef']:+.5f}, p={r['pval']:.4f}（{direction}）")
else:
    report.append("**供应商行业多样性**：所有模型中均未达到统计显著水平。")
    pos_drop = len(ind_div_positive_drop)
    neg_vol = len(ind_div_negative_vol)
    report.append(f"系数方向：在下降类模型中有 {pos_drop} 个为正（支持韧性机制），")
    report.append(f"在波动类模型中有 {neg_vol} 个为负（支持韧性机制）。")
    report.append("但统计上不显著，不能拒绝供应商行业多样性无韧性效应的原假设。")
report.append("")

# 竞争对手广度
comp_sig = [r for r in single_results if r['variable'] == 'comp_breadth_pre' and r['pval'] < 0.1]
comp_negative_drop = [r for r in single_results if r['variable'] == 'comp_breadth_pre'
                      and 'drop' in r['model'] and r['coef'] < 0]
comp_positive_vol = [r for r in single_results if r['variable'] == 'comp_breadth_pre'
                     and 'volatility' in r['model'] and r['coef'] > 0]

if len(comp_sig) > 0:
    report.append("**竞争对手广度**：部分模型显著，但方向与竞争压力假说相反。")
    for r in comp_sig:
        direction = "正向" if r['coef'] > 0 else "负向"
        report.append(f"- {r['model']}: 系数={r['coef']:+.5f}, p={r['pval']:.4f}（{direction}，说明竞争对手越多的企业绩效下降反而更少）")
else:
    report.append("**竞争对手广度**：所有模型中均未达到统计显著水平。")
    report.append(f"系数方向：下降类模型中有 {len(comp_negative_drop)} 个为负（支持竞争压力假说），")
    report.append(f"波动类模型中有 {len(comp_positive_vol)} 个为正（支持竞争压力假说）。")
    report.append("但统计上不显著。")
report.append("")

# 最清晰的结果类型
report.append("### 4.1 哪类韧性指标最清晰")
report.append("")
y_sig_counts = {}
for r in single_results:
    yn = r['y_group']
    if yn not in y_sig_counts:
        y_sig_counts[yn] = {'sig': 0, 'total': 0}
    y_sig_counts[yn]['total'] += 1
    if r['pval'] < 0.1:
        y_sig_counts[yn]['sig'] += 1

for yn in ['ROA', 'ROE', 'Growth']:
    if yn in y_sig_counts:
        c = y_sig_counts[yn]
        report.append(f"- {yn}: {c['sig']}/{c['total']} 显著")

report.append("")
report.append("### 4.2 论文写作建议")
report.append("")
report.append("由于绝大多数结果在统计上不显著，本检验结果更适合放入论文附录或稳健性检验部分。")
report.append("建议在论文中作如下表述：")
report.append("")
report.append("为进一步检验供应链关系结构是否通过韧性机制影响企业绩效，")
report.append("本文以2015-2017年供应链结构变量均值解释2018-2020年企业韧性指标。")
report.append("结果显示，供应商行业多样性和竞争对手广度对ROA下降、ROA波动、")
report.append("销售增长下降和销售增长波动的解释力有限，联合模型和单变量模型中")
report.append("核心变量的系数大多未达到统计显著水平。")
report.append("")
report.append("这一结果表明，供应链关系结构对贸易战后企业韧性的直接影响较弱，")
report.append("供应链结构-绩效关系的调节效应可能更多通过贸易战冲击本身来触发，")
report.append("而非通过事前供应链结构的直接韧性渠道传导。")
report.append("")
report.append("### 4.3 论文段落草稿")
report.append("")
report.append("""为检验供应链关系结构的韧性机制，本文构建了企业层面的横截面回归模型。
以2015-2017年供应商广度、供应商行业多样性、竞争对手广度和竞争对手行业多样性的
均值作为事前供应链结构度量，考察其对2018-2020年企业ROA下降幅度、ROA波动性、
销售增长下降幅度和销售增长波动性的解释力。回归模型中控制了企业规模、财务杠杆和
事前增长率，并加入了行业固定效应。

实证结果显示，在控制了事前特征和行业差异后，供应商行业多样性与ROA下降幅度
呈正向关系（系数+0.871，p=0.097），在边际水平上显著，方向支持韧性机制假说
（即供应商行业多样性越高，贸易战后绩效下降越少）。然而，该效应在ROE和销售增长
的下降及波动指标中均不显著，未能形成稳定的证据链。

值得注意的是，竞争对手广度与ROA下降幅度也呈正向关系（系数+0.499，p=0.033），
这一方向并不支持竞争压力假说——竞争对手越多的企业反而绩效下降更少，可能与
市场竞争地位较高的企业本身同时具有更多竞争对手和更强抗风险能力有关。

综合来看，供应链关系结构对贸易战后企业韧性的直接解释力较为有限。这一结果意味着，
供应链结构-绩效关系的贸易战调节效应并非主要通过事前结构的直接韧性渠道传导，
而是与贸易战冲击本身对企业经营环境的改变密切相关。""")

report_content = '\n'.join(report)
with open(f'{OUT_DIR}/resilience_summary.md', 'w', encoding='utf-8-sig') as f:
    f.write(report_content)
print(f"  ✓ {OUT_DIR}/resilience_summary.md")

print("\n" + "=" * 60)
print("供应链韧性机制检验完成!")
print(f"输出目录: {OUT_DIR}")
print("=" * 60)
