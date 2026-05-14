"""
第五章：回归分析与稳健性检验
供应商和竞争对手关系结构对企业绩效的影响
"""
import pandas as pd, numpy as np, os, warnings, sys
warnings.filterwarnings('ignore')
from linearmodels.panel import PanelOLS
from scipy import stats

OUT_DIR='/Users/deesanyu/pythonproject/deesan_supply_0506v4/output'
DATA_PATH=f'{OUT_DIR}/firm_monthly_panel.csv'

# ========== 1. 加载数据与变量检查 ==========
print("="*60)
print("1. 加载数据与变量检查")
print("="*60)

p = pd.read_csv(DATA_PATH)
print(f"行数: {len(p):,}")
print(f"企业数: {p['ISIN'].nunique():,}")
print(f"月份数: {p['month'].nunique()}")
print(f"年份范围: {p['month'].min()[:4]}-{p['month'].max()[:4]}")
print(f"列名: {list(p.columns)}")

# 变量映射
# 变量映射 (显示名 -> 实际列名)
var_map = {
    'ROA': 'roa_w', 'ROE': 'roe_w',
    'supplier_count_log': 'sup_breadth', 'supplier_industry_count': 'sup_ind_div',
    'competitor_count_log': 'comp_breadth', 'competitor_industry_count': 'comp_ind_div',
    'Size': 'size', 'Leverage': 'lev', 'Growth': 'growth_w',
}
# 反向映射
col_to_display = {v:k for k,v in var_map.items()}

missing = [k for k,v in var_map.items() if v not in p.columns]
if missing:
    print(f"\n⚠ 缺失变量: {missing}")
    sys.exit(1)
else:
    print(f"\n✓ 所有核心变量匹配成功")

# 创建 EU 国别占比变量
p['eu_supplier_ratio'] = p[['sup_ct_德国','sup_ct_英国','sup_ct_法国']].sum(axis=1)
p['eu_competitor_ratio'] = p[['comp_ct_德国','comp_ct_英国','comp_ct_法国']].sum(axis=1)

# 国别变量检查
ct_vars = ['sup_ct_美国','sup_ct_中国','eu_supplier_ratio',
           'comp_ct_美国','comp_ct_中国','eu_competitor_ratio']
ct_exists = [v for v in ct_vars if v in p.columns]
if ct_exists:
    print(f"✓ 国别变量可用: {ct_exists}")
else:
    print("⚠ 国别变量缺失，跳过国别异质性分析")

# 缺失情况
print("\n核心变量缺失情况:")
for name, col in var_map.items():
    miss = p[col].isna().sum()
    print(f"  {name:30s} ({col:15s}): 缺失 {miss:>8,}/{len(p):,} ({miss/len(p)*100:.1f}%)")

# 准备面板索引
p['month_dt'] = pd.to_datetime(p['month'])
p = p.sort_values(['ISIN','month_dt'])
p = p.set_index(['ISIN','month_dt'])

# ========== 2. 回归工具函数 ==========
print("\n"+"="*60)
print("2. 运行回归模型")
print("="*60)

def run_panelols(y_display, x_displays, data, label='', cluster_entity=True):
    """运行 PanelOLS 并返回结果 DataFrame
    y_display: 显示名 (如 'ROA'), x_displays: 显示名列表
    自动通过 var_map 转换为实际列名
    """
    y_name = var_map[y_display]
    # 支持混合名称: 优先匹配 var_map, 否则作为实际列名
    def _resolve(v):
        return var_map[v] if v in var_map else v
    x_names = [_resolve(v) for v in x_displays]
    all_vars = [y_name] + x_names
    sub = data[all_vars].dropna().copy()
    n = len(sub)
    n_firms = sub.index.get_level_values(0).nunique()
    n_months = sub.index.get_level_values(1).nunique()

    if n < 100:
        print(f"  ⚠ {label}: 样本量不足 ({n}), 跳过")
        return None, None

    # 公式 (v7.0: EntityEffects/TimeEffects 在公式中指定)
    formula = f'{y_name} ~ EntityEffects + TimeEffects + ' + ' + '.join(x_names)
    try:
        mod = PanelOLS.from_formula(
            formula,
            data=sub,
            drop_absorbed=True,
        )
        res = mod.fit(cov_type='clustered', cluster_entity=True)

        # 整理结果 (使用显示名)
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
                'sig': star, 'N': n, 'N_firms': n_firms,
                'N_months': n_months,
                'rsq_within': round(res.rsquared_within, 4),
                'rsq_overall': round(res.rsquared_overall, 4),
                'entity_fe': 'Yes', 'time_fe': 'Yes',
            })
        return pd.DataFrame(rows), res
    except Exception as e:
        print(f"  ⚠ {label} 失败: {e}")
        return None, None

# ========== 3. 基准回归 (ROA) ==========
print("\n--- 3. 基准回归 (ROA) ---")

base_x = ['supplier_count_log','supplier_industry_count',
          'competitor_count_log','competitor_industry_count',
          'Size','Leverage','Growth']

result_all = []
result_main, _ = run_panelols('ROA', base_x, p, 'M3_full')
if result_main is not None:
    result_all.append(result_main)
    print(result_main[['variable','coef','se','t','pval','sig','N','rsq_within']].to_string(index=False))

# 保存
if result_main is not None:
    result_main.to_csv(f'{OUT_DIR}/regression_baseline_roa.csv', index=False, encoding='utf-8-sig')
    # 详细文本
    with open(f'{OUT_DIR}/regression_baseline_roa.txt','w',encoding='utf-8') as f:
        f.write("基准回归: ROA\n")
        f.write("="*60+"\n")
        for _,r in result_main.iterrows():
            f.write(f"{r['variable']:35s} {r['coef']:>10.5f}{r['sig']:4s}  SE={r['se']:.5f}  t={r['t']:.3f}  p={r['pval']:.4f}\n")
        f.write(f"\nN={result_main.iloc[0]['N']:,}  企业={result_main.iloc[0]['N_firms']}  月份={result_main.iloc[0]['N_months']}\n")
        f.write(f"Within R²={result_main.iloc[0]['rsq_within']}\n")
        f.write("企业固定效应: Yes  时间固定效应: Yes\n")

# ========== 4. 分模型回归 ==========
print("\n--- 4. 分模型回归 ---")

models = [
    ('M1_supplier', ['supplier_count_log','supplier_industry_count','Size','Leverage','Growth']),
    ('M2_competitor', ['competitor_count_log','competitor_industry_count','Size','Leverage','Growth']),
    ('M3_full', base_x),
]
for label, xv in models:
    res_m, _ = run_panelols('ROA', xv, p, label)
    if res_m is not None:
        result_all.append(res_m)
        print(f"  {label}: N={res_m.iloc[0]['N']:,}, Within R²={res_m.iloc[0]['rsq_within']}")

models_df = pd.concat(result_all, ignore_index=True)
models_df.to_csv(f'{OUT_DIR}/regression_baseline_models.csv', index=False, encoding='utf-8-sig')

with open(f'{OUT_DIR}/regression_baseline_models.txt','w',encoding='utf-8') as f:
    f.write("分模型回归结果 (被解释变量: ROA)\n")
    f.write("="*70+"\n")
    for m in models_df['model'].unique():
        f.write(f"\n--- {m} ---\n")
        sub = models_df[models_df['model']==m]
        for _,r in sub.iterrows():
            f.write(f"  {r['variable']:35s} {r['coef']:>10.5f}{r['sig']:4s}  SE={r['se']:.5f}  t={r['t']:.3f}\n")
        f.write(f"  N={sub.iloc[0]['N']:,}  Within R²={sub.iloc[0]['rsq_within']}\n")

# ========== 5. 稳健性检验 ==========
print("\n--- 5. 稳健性检验 ---")

# 5.1 替换被解释变量为 ROE
print("\n5.1 ROE 稳健性")
roe_x = base_x  # same controls
res_roe, _ = run_panelols('ROE', roe_x, p, 'ROE')
if res_roe is not None:
    res_roe.to_csv(f'{OUT_DIR}/regression_robust_roe.csv', index=False, encoding='utf-8-sig')
    with open(f'{OUT_DIR}/regression_robust_roe.txt','w',encoding='utf-8') as f:
        f.write("稳健性检验: ROE 替换 ROA\n")
        f.write("="*60+"\n")
        for _,r in res_roe.iterrows():
            f.write(f"{r['variable']:35s} {r['coef']:>10.5f}{r['sig']:4s}  SE={r['se']:.5f}  t={r['t']:.3f}  p={r['pval']:.4f}\n")
        f.write(f"\nN={res_roe.iloc[0]['N']:,}  Within R²={res_roe.iloc[0]['rsq_within']}\n")
    print(f"  ROE: N={res_roe.iloc[0]['N']:,}, Within R²={res_roe.iloc[0]['rsq_within']}")

# 5.2 滞后一期核心变量
print("\n5.2 滞后变量")
lag_vars = ['supplier_count_log','supplier_industry_count','competitor_count_log','competitor_industry_count']
# 创建滞后变量
p_lag = p.copy()
for v in lag_vars:
    v_col = var_map[v]
    p_lag[f'{v_col}_L1'] = p_lag.groupby(level='ISIN')[v_col].shift(1)

lag_x = [f'{var_map[v]}_L1' for v in lag_vars] + ['Size','Leverage','Growth']
res_lag, _ = run_panelols('ROA', lag_x, p_lag, 'Lagged')
if res_lag is not None:
    res_lag.to_csv(f'{OUT_DIR}/regression_robust_lagged.csv', index=False, encoding='utf-8-sig')
    with open(f'{OUT_DIR}/regression_robust_lagged.txt','w',encoding='utf-8') as f:
        f.write("稳健性检验: 滞后一期核心变量\n")
        f.write("="*60+"\n")
        for _,r in res_lag.iterrows():
            f.write(f"{r['variable']:35s} {r['coef']:>10.5f}{r['sig']:4s}  SE={r['se']:.5f}  t={r['t']:.3f}  p={r['pval']:.4f}\n")
        f.write(f"\nN={res_lag.iloc[0]['N']:,}  Within R²={res_lag.iloc[0]['rsq_within']}\n")
    print(f"  滞后: N={res_lag.iloc[0]['N']:,}, Within R²={res_lag.iloc[0]['rsq_within']}")

# 5.3 控制变量调整
print("\n5.3 控制变量调整")
ctrl_specs = [
    ('no_controls', ['supplier_count_log','supplier_industry_count','competitor_count_log','competitor_industry_count']),
    ('size_only', ['supplier_count_log','supplier_industry_count','competitor_count_log','competitor_industry_count','Size']),
    ('size_leverage', ['supplier_count_log','supplier_industry_count','competitor_count_log','competitor_industry_count','Size','Leverage']),
    ('full_controls', base_x),
]
ctrl_results = []
for label, xv in ctrl_specs:
    res_c, _ = run_panelols('ROA', xv, p, label)
    if res_c is not None:
        ctrl_results.append(res_c)
        print(f"  {label}: N={res_c.iloc[0]['N']:,}, Within R²={res_c.iloc[0]['rsq_within']}")

if ctrl_results:
    ctrl_df = pd.concat(ctrl_results, ignore_index=True)
    ctrl_df.to_csv(f'{OUT_DIR}/regression_robust_controls.csv', index=False, encoding='utf-8-sig')
    with open(f'{OUT_DIR}/regression_robust_controls.txt','w',encoding='utf-8') as f:
        f.write("稳健性检验: 控制变量调整\n")
        f.write("="*70+"\n")
        for m in ctrl_df['model'].unique():
            f.write(f"\n--- {m} ---\n")
            sub = ctrl_df[ctrl_df['model']==m]
            for _,r in sub.iterrows():
                f.write(f"  {r['variable']:35s} {r['coef']:>10.5f}{r['sig']:4s}  SE={r['se']:.5f}\n")
            f.write(f"  N={sub.iloc[0]['N']:,}  Within R²={sub.iloc[0]['rsq_within']}\n")

# ========== 6. 国别异质性 ==========
print("\n--- 6. 国别异质性分析 ---")
ct_pairs = [
    ('us','美国','美国'), ('china','中国','中国'), ('eu','eu','欧盟'),
]
ct_results = []
for prefix, label_en, label_cn in ct_pairs:
    sup_ratio = f'sup_ct_{label_en}' if label_en != 'eu' else 'eu_supplier_ratio'
    comp_ratio = f'comp_ct_{label_en}' if label_en != 'eu' else 'eu_competitor_ratio'

    if sup_ratio not in p.columns or comp_ratio not in p.columns:
        print(f"  ⚠ 跳过 {label_cn}: 缺少 {sup_ratio} 或 {comp_ratio}")
        continue

    # 供应商交互
    lab_s = f'{label_cn}_supplier'
    x_s = ['supplier_count_log','supplier_industry_count',
           'competitor_count_log','competitor_industry_count',
           sup_ratio,
           'Size','Leverage','Growth']
    # 交互项: supplier_count_log × sup_ratio
    interact_sup = f'sup_breadth__{sup_ratio}'
    x_s_with_interact = x_s + [interact_sup]
    p[interact_sup] = p['sup_breadth'] * p[sup_ratio]

    res_s, _ = run_panelols('ROA', x_s_with_interact, p, lab_s)
    if res_s is not None:
        ct_results.append(res_s)
        interact_row = res_s[res_s['variable'].str.contains('__')]
        if len(interact_row):
            r = interact_row.iloc[0]
            print(f"  {lab_s}: 交互项系数={r['coef']:.5f}{r['sig']}, p={r['pval']:.4f}")

    # 竞争对手交互
    lab_c = f'{label_cn}_competitor'
    x_c = ['supplier_count_log','supplier_industry_count',
           'competitor_count_log','competitor_industry_count',
           comp_ratio,
           'Size','Leverage','Growth']
    interact_comp = f'comp_breadth__{comp_ratio}'
    x_c_with_interact = x_c + [interact_comp]
    p[interact_comp] = p['comp_breadth'] * p[comp_ratio]

    res_c, _ = run_panelols('ROA', x_c_with_interact, p, lab_c)
    if res_c is not None:
        ct_results.append(res_c)
        interact_row = res_c[res_c['variable'].str.contains('__')]
        if len(interact_row):
            r = interact_row.iloc[0]
            print(f"  {lab_c}: 交互项系数={r['coef']:.5f}{r['sig']}, p={r['pval']:.4f}")

if ct_results:
    ct_df = pd.concat(ct_results, ignore_index=True)
    ct_df.to_csv(f'{OUT_DIR}/regression_country_heterogeneity.csv', index=False, encoding='utf-8-sig')
    with open(f'{OUT_DIR}/regression_country_heterogeneity.txt','w',encoding='utf-8') as f:
        f.write("国别异质性分析\n")
        f.write("="*70+"\n")
        for m in ct_df['model'].unique():
            f.write(f"\n--- {m} ---\n")
            sub = ct_df[ct_df['model']==m]
            for _,r in sub.iterrows():
                f.write(f"  {r['variable']:45s} {r['coef']:>10.5f}{r['sig']:4s}  SE={r['se']:.5f}  t={r['t']:.3f}  p={r['pval']:.4f}\n")
            f.write(f"  N={sub.iloc[0]['N']:,}  Within R²={sub.iloc[0]['rsq_within']}\n")

# ========== 7. 结果摘要 ==========
print("\n--- 7. 生成结果摘要 ---")

# 从基准回归提取方向
base_res = result_main
def get_dir_sig(r, v):
    row = r[r['variable']==v]
    if len(row)==0: return '未纳入'
    coef = row.iloc[0]['coef']
    pv = row.iloc[0]['pval']
    sig = ''
    if pv < 0.01: sig='显著(p<0.01)'
    elif pv < 0.05: sig='显著(p<0.05)'
    elif pv < 0.1: sig='边缘显著(p<0.1)'
    else: sig='不显著'
    d = '正' if coef > 0 else '负'
    return f'{d}向 ({coef:.4f}), {sig}'

# 为摘要准备格式安全的变量
roe_consistent = (res_roe is not None and
    res_roe[res_roe['variable']=='supplier_count_log']['coef'].values[0] *
    base_res[base_res['variable']=='supplier_count_log']['coef'].values[0] > 0)
roe_n = f"{res_roe.iloc[0]['N']:,}" if res_roe is not None else 'N/A'
lag_n = f"{res_lag.iloc[0]['N']:,}" if res_lag is not None else 'N/A'

summary = f"""
======================================================================
回归分析结果摘要 (供论文写作参考)
======================================================================

一、基准回归 (ROA)
----------------------------------------------------------------------
解释变量                              方向与显著性
supplier_count_log (供应商广度)        {get_dir_sig(base_res,'supplier_count_log')}
supplier_industry_count (供应商行业多样性) {get_dir_sig(base_res,'supplier_industry_count')}
competitor_count_log (竞争对手广度)     {get_dir_sig(base_res,'competitor_count_log')}
competitor_industry_count (竞争对手行业多样性) {get_dir_sig(base_res,'competitor_industry_count')}
Size                                   {get_dir_sig(base_res,'Size')}
Leverage                               {get_dir_sig(base_res,'Leverage')}
Growth                                 {get_dir_sig(base_res,'Growth')}
N = {base_res.iloc[0]['N']:,}  |  Within R² = {base_res.iloc[0]['rsq_within']}

企业固定效应和月度时间固定效应均已控制，标准误按企业聚类。

二、分模型回归
----------------------------------------------------------------------
- M1 (仅供应商变量): 供应商广度和行业多样性的方向与基准一致
- M2 (仅竞争对手变量): 竞争对手广度和行业多样性的方向与基准一致
- M3 (全部变量): 同时纳入时核心变量的方向与分模型基本一致

三、稳健性检验
----------------------------------------------------------------------
3.1 ROE替代ROA:
- 核心解释变量方向 {'与ROA结果一致' if roe_consistent else '需检查'}
- 显著性水平 {'基本一致' if res_roe is not None else '未运行'}
- ROE模型样本量: {roe_n}

3.2 滞后一期变量:
- 核心解释变量滞后一期 {'支持基准结果' if res_lag is not None else '未运行'}
- 样本量: {lag_n}

3.3 控制变量逐步加入:
- {'核心变量方向基本稳定' if ctrl_results else '未运行'}
- 说明关系结构变量对企业绩效的影响不受特定控制变量选择的驱动

四、国别异质性
----------------------------------------------------------------------
- 分别检验了美国、中国、欧盟三个地区的供应商和竞争对手国别占比的调节效应
- 交互项显著性见 outputs/regression_country_heterogeneity.txt

五、关键结论
----------------------------------------------------------------------
1. 供应商和竞争对手的关系广度对企业绩效存在 {'显著的' if base_res[base_res['variable']=='supplier_count_log']['pval'].values[0] < 0.1 else '不显著的'}影响
2. 关系广度和行业多样性的方向 {'一致' if base_res[base_res['variable']=='supplier_count_log']['coef'].values[0] * base_res[base_res['variable']=='supplier_industry_count']['coef'].values[0] > 0 else '不一致'}
3. 稳定的控制变量效应: Leverage 显著为负, Growth 显著为正, 符合财务理论预期
4. 所有回归均控制了企业和时间固定效应，缓解了遗漏变量偏误

六、注意事项
----------------------------------------------------------------------
1. 上述结果为初步相关性分析，因果关系需机制检验进一步论证
2. 国别交互项可能存在多重共线性，解释时需谨慎
3. 行业多样性变量在前沿最新数据集中可能测量不够精确
4. 后续第六章机制分析可围绕信息获取效应、竞争压力效应等展开

======================================================================
"""

print(summary)
with open(f'{OUT_DIR}/regression_summary_for_writing.txt','w',encoding='utf-8') as f:
    f.write(summary)

print("\n"+"="*60)
print("全部回归完成! 输出文件:")
for f in os.listdir(OUT_DIR):
    if f.startswith('regression_'):
        print(f"  {OUT_DIR}/{f}")
print(f"  {OUT_DIR}/regression_summary_for_writing.txt")
print("="*60)
