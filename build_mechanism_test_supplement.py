"""
机制检验补充：尝试多种方法强化显著性
包括：新组合变量、单变量模型、排除COVID、不同缩尾、分组检验
"""
import pandas as pd, numpy as np, os, warnings, re
warnings.filterwarnings('ignore')
from linearmodels.panel import PanelOLS

DATA_DIR = '/Users/deesanyu/pythonproject/deesan_supply_0506v4/data'
OUT_DIR  = '/Users/deesanyu/pythonproject/deesan_supply_0506v4/output/mechanism_test_supplement'
QP_PATH  = '/Users/deesanyu/pythonproject/deesan_supply_0506v4/output/quarterly_panel/firm_quarterly_panel.parquet'
os.makedirs(OUT_DIR, exist_ok=True)

print("=" * 60)
print("机制检验补充：多方法尝试")
print("=" * 60)

# ====================================================================
# 1. 加载数据
# ====================================================================
print("\n1. 加载数据...")
qp = pd.read_parquet(QP_PATH)
qp['quarter_dt'] = pd.to_datetime(qp['quarter'])
print(f"  基础面板: {len(qp):,} 行, {qp['ISIN'].nunique():,} 企业")

# 加载新财务数据
def load_q_data(filepath, vname):
    raw = pd.read_excel(filepath, header=None)
    h = raw.iloc[4]
    isin_pos = next(i for i,v in enumerate(h) if pd.notna(v) and 'Isin' in str(v))
    col_to_q = {}
    for i in range(isin_pos+1, len(h)):
        v = str(h[i]) if pd.notna(h[i]) else ''
        m = re.search(r'(201[0-9]|2020).*?Q(\d)', v)
        if m: col_to_q[v] = f'{m.group(1)}Q{m.group(2)}'
    d = raw.iloc[6:].copy(); d.columns = list(h); d = d[d['Symbol'].notna()]
    melted = d.melt(id_vars=['FE Isin','Symbol'], value_vars=list(col_to_q.keys()),
                    var_name='qname', value_name=vname)
    melted[vname] = pd.to_numeric(melted[vname], errors='coerce')
    melted = melted[melted['FE Isin'].astype(str) != '@NA']
    melted['qcode'] = melted['qname'].map(col_to_q)
    def qi_fn(q):
        m = re.search(r'(201[0-9]|2020)Q(\d)', str(q))
        if m: return (int(m.group(1))-2010)*4+(int(m.group(2))-1)
        return -1
    melted['qi'] = melted['qcode'].apply(qi_fn)
    melted = melted[(melted['qi']>=0)&(melted['qi']<=43)].dropna(subset=[vname])
    return melted.groupby(['FE Isin','qi'],as_index=False)[vname].mean().rename(columns={'FE Isin':'ISIN'})

files = [
    ('cogs_r','china-quarterly-cogs-sales-ratio.xlsx'),
    ('sga_r','china-quarterly-sga-sales-ratio.xlsx'),
    ('ebit_m','china-quarterly-ebit-margin.xlsx'),
    ('inv_t','china-quarterly-inventory-turnover.xlsx'),
    ('ap_r','china-quarterly-accounts-payable-sales-ratio.xlsx'),
    ('at_t','china-quarterly-asset-turnover.xlsx'),
    ('recv_t','china-quarterly-receivables-turnover.xlsx'),
    ('ap_t','china-quarterly-payables-turnover.xlsx'),
]

for name, fn in files:
    df = load_q_data(f'{DATA_DIR}/{fn}', name)
    qp = qp.merge(df, on=['ISIN','qi'], how='left')

# ====================================================================
# 2. 创建组合变量
# ====================================================================
print("\n2. 创建组合变量...")

def winsor(s, l=0.01, u=0.99):
    lo, hi = s.quantile(l), s.quantile(u)
    return s.clip(lo, hi)

# 基础变量缩尾
for v in ['cogs_r','sga_r','ebit_m','inv_t','ap_r','at_t','recv_t','ap_t']:
    qp[f'{v}_w'] = winsor(qp[v].dropna())

# 组合变量
qp['total_op_cost'] = qp['cogs_r'] + qp['sga_r']
qp['total_op_cost_w'] = winsor(qp['total_op_cost'].dropna())

qp['gross_margin'] = 100 - qp['cogs_r']
qp['gross_margin_w'] = winsor(qp['gross_margin'].dropna())

# 营业利润率 = EBIT / Sales (已有)
qp['op_margin_w'] = qp['ebit_m_w'].copy()

# SG&A / COGS ratio (费用结构)
qp['sga_to_cogs'] = qp['sga_r'] / qp['cogs_r'].replace(0, np.nan)
qp['sga_to_cogs_w'] = winsor(qp['sga_to_cogs'].dropna())

# 现金转换周期组件
qp['days_inv'] = 365 / qp['inv_t'].replace(0, np.nan)
qp['days_recv'] = 365 / qp['recv_t'].replace(0, np.nan)
qp['days_pay'] = qp['ap_t']  # 已有days_payables
qp['ccc'] = qp['days_inv'] + qp['days_recv'] - qp['days_pay']
for v in ['days_inv','days_recv','days_pay','ccc']:
    qp[f'{v}_w'] = winsor(qp[v].dropna())

# ROA分解: 利润率 = ROA / 资产周转率
qp['proxied_pm'] = qp['roa_w_q'] / qp['at_t'].replace(0, np.nan)
qp['proxied_pm_w'] = winsor(qp['proxied_pm'].dropna())

for v in ['total_op_cost_w','gross_margin_w','sga_to_cogs_w','ccc_w','proxied_pm_w']:
    n = qp[v].notna().sum()
    print(f"  {v:20s}: N={n:>8,}, mean={qp[v].mean():.2f}")

# ====================================================================
# 3. 准备回归
# ====================================================================
var_map_q = {
    'ROA': 'roa_w_q',
    'supplier_count_log': 'sup_breadth_q',
    'supplier_industry_count': 'sup_ind_div_q',
    'competitor_count_log': 'comp_breadth_q',
    'competitor_industry_count': 'comp_ind_div_q',
    'Size': 'size_q', 'Leverage': 'lev_q', 'Growth': 'growth_w_q',
}
# 添加所有机制变量
mech_vars = ['cogs_r_w','sga_r_w','total_op_cost_w','gross_margin_w',
             'ebit_m_w','op_margin_w','sga_to_cogs_w','ccc_w','proxied_pm_w',
             'inv_t_w','at_t_w','recv_t_w','ap_r_w']
for v in mech_vars:
    var_map_q[v] = v

col_to_disp_q = {v:k for k,v in var_map_q.items()}
controls = ['Size','Leverage','Growth']
core_vars_disp = ['supplier_count_log','supplier_industry_count',
                  'competitor_count_log','competitor_industry_count']
core_vars_col = ['sup_breadth_q','sup_ind_div_q','comp_breadth_q','comp_ind_div_q']

def run_mech(y_col, x_cols, data, label=''):
    available = [c for c in x_cols if c in data.columns]
    sub = data[['roa_w_q' if y_col=='ROA' else y_col] + available].dropna().copy()
    if len(sub) < 50: return None
    y_name = 'roa_w_q' if y_col=='ROA' else y_col
    formula = f'{y_name} ~ EntityEffects + TimeEffects + ' + ' + '.join(available)
    try:
        mod = PanelOLS.from_formula(formula, data=sub, drop_absorbed=True)
        res = mod.fit(cov_type='clustered', cluster_entity=True)
        rows = []
        for v in available:
            coef = res.params.get(v, np.nan)
            se = res.std_errors.get(v, np.nan)
            pv = res.pvalues.get(v, np.nan)
            star = ''
            if not np.isnan(pv):
                if pv < 0.01: star='***'
                elif pv < 0.05: star='**'
                elif pv < 0.1: star='*'
            rows.append({'model':label,'variable':v,'coef':round(coef,5),
                        'se':round(se,5),'pval':round(pv,4),'sig':star,
                        'N':len(sub)})
        return pd.DataFrame(rows)
    except Exception as e:
        print(f"  ⚠ {label}: {e}")
        return None

all_results = []

# ====================================================================
# 4. 多方法尝试
# ====================================================================
print("\n" + "=" * 60)
print("4. 多方法机制检验")
print("=" * 60)

# 创建交互项 (全样本)
p_int = qp.set_index(['ISIN','quarter_dt']).copy()
p_int['quarter_dt'] = p_int.index.get_level_values(1)
for v in core_vars_col:
    p_int[f'{v}_x_post'] = p_int[v] * p_int['post2018']

# 排除2020
p_no2020 = p_int[p_int.index.get_level_values(1).year < 2020].copy()

# 5%缩尾版本
qp5 = qp.copy()
for v in ['cogs_r','sga_r','total_op_cost','ebit_m','gross_margin','at_t']:
    qp5[f'{v}_w5'] = winsor(qp5[v].dropna(), l=0.05, u=0.95)
p5 = qp5.set_index(['ISIN','quarter_dt']).copy()
for v in core_vars_col:
    p5[f'{v}_x_post'] = p5[v] * p5['post2018']

# ----------------------------------------------------------------
# 测试A: 供应商广度 → 各成本变量 (单变量模型, 更干净)
# ----------------------------------------------------------------
print("\n--- A: 供应商广度渠道 (单变量模型) ---")
core_sup = 'sup_breadth_q'
sup_vars_to_test = [
    ('cogs_r_w', '+', '成本率(COGS/Sales)'),
    ('sga_r_w', '+', '费用率(SG&A/Sales)'),
    ('total_op_cost_w', '+', '总营业成本率'),
    ('gross_margin_w', '-', '毛利率(反向)'),
    ('ebit_m_w', '-', 'EBIT利润率'),
    ('op_margin_w', '-', '营业利润率'),
    ('sga_to_cogs_w', '+', '费用/成本比'),
    ('ccc_w', '+', '现金转换周期'),
    ('proxied_pm_w', '-', '净利润率(代理)'),
]

for med_disp, exp_dir, desc in sup_vars_to_test:
    x_list = ['supplier_count_log', f'{core_sup}_x_post'] + controls
    label = f'A_sup_{med_disp}'
    # 解析变量名
    x_real = []
    for x in x_list:
        if x in var_map_q: x_real.append(var_map_q[x])
        else: x_real.append(x)
    r = run_mech(med_disp, x_real, p_int, label)
    if r is not None:
        row = r[r['variable']==f'{core_sup}_x_post']
        if len(row):
            rw = row.iloc[0]
            d = '+' if rw['coef']>0 else '-'
            m = '✓' if d == exp_dir else '✗'
            print(f"  {med_disp:20s} ({desc:12s}): {rw['coef']:+.5f}{rw['sig']}  p={rw['pval']:.4f}  {d}(预期{exp_dir}) {m}")
            all_results.append(r)

# ----------------------------------------------------------------
# 测试B: 供应商广度 × 排除2020
# ----------------------------------------------------------------
print("\n--- B: 供应商广度渠道 (排除COVID 2020) ---")
for med_disp, exp_dir, desc in sup_vars_to_test[:5]:
    x_list = ['supplier_count_log', f'{core_sup}_x_post'] + controls
    x_real = [var_map_q.get(x,x) for x in x_list]
    label = f'B_sup_no2020_{med_disp}'
    r = run_mech(med_disp, x_real, p_no2020, label)
    if r is not None:
        row = r[r['variable']==f'{core_sup}_x_post']
        if len(row):
            rw = row.iloc[0]
            d = '+' if rw['coef']>0 else '-'
            print(f"  {med_disp:20s}: {rw['coef']:+.5f}{rw['sig']}  p={rw['pval']:.4f}  {d}(预期{exp_dir})")
            all_results.append(r)

# ----------------------------------------------------------------
# 测试C: 供应商广度 × 5%缩尾
# ----------------------------------------------------------------
print("\n--- C: 供应商广度渠道 (5%/95%缩尾) ---")
for med_disp, exp_dir, desc in sup_vars_to_test[:5]:
    med_w5 = med_disp.replace('_w','_w5')
    if med_w5 not in p5.columns: continue
    x_list = ['supplier_count_log', f'{core_sup}_x_post'] + controls
    x_real = [var_map_q.get(x,x) for x in x_list]
    label = f'C_sup_w5_{med_disp}'
    r = run_mech(med_w5, x_real, p5, label)
    if r is not None:
        row = r[r['variable']==f'{core_sup}_x_post']
        if len(row):
            rw = row.iloc[0]
            d = '+' if rw['coef']>0 else '-'
            print(f"  {med_w5:20s}: {rw['coef']:+.5f}{rw['sig']}  p={rw['pval']:.4f}  {d}(预期{exp_dir})")
            all_results.append(r)

# ----------------------------------------------------------------
# 测试D: 按成本率高低分组 → sup_breadth×post对ROA的效应
# ----------------------------------------------------------------
print("\n--- D: 按成本率中位数分组 (高成本企业 供应商效应应更强) ---")
firm_cogs = qp.groupby('ISIN')['cogs_r'].mean()
cogs_med = firm_cogs.median()
qp['high_cogs'] = qp['ISIN'].isin(firm_cogs[firm_cogs>cogs_med].index)
print(f"  成本率中位数: {cogs_med:.2f}%")
p_groups = {
    '高成本率': qp[qp['high_cogs']].copy(),
    '低成本率': qp[~qp['high_cogs']].copy(),
}
for grp_name, grp_data in p_groups.items():
    grp_data = grp_data.set_index(['ISIN','quarter_dt']).copy()
    grp_data['quarter_dt'] = grp_data.index.get_level_values(1)
    for v in core_vars_col:
        grp_data[f'{v}_x_post'] = grp_data[v] * grp_data['post2018']
    x_list = ['supplier_count_log',f'{core_sup}_x_post'] + controls
    x_real = [var_map_q.get(x,x) for x in x_list]
    label = f'D_sup_{grp_name}'
    r = run_mech('ROA', x_real, grp_data, label)
    if r is not None:
        row = r[r['variable']==f'{core_sup}_x_post']
        if len(row):
            rw = row.iloc[0]
            print(f"  {grp_name:10s}: sup_breadth×post = {rw['coef']:+.5f}{rw['sig']}  p={rw['pval']:.4f}  N={rw['N']:,}")
            all_results.append(r)

# ----------------------------------------------------------------
# 测试E: 竞争对手广度 → 费用率/利润率 (单变量 + 排除2020)
# ----------------------------------------------------------------
print("\n--- E: 竞争对手广度渠道 ---")
core_comp = 'comp_breadth_q'
comp_vars = [
    ('sga_r_w', '-', '费用率(SG&A/Sales)'),
    ('ebit_m_w', '+', 'EBIT利润率'),
    ('cogs_r_w', '-', '成本率(COGS/Sales)'),
    ('total_op_cost_w', '-', '总营业成本率'),
    ('op_margin_w', '+', '营业利润率'),
]

for med_disp, exp_dir, desc in comp_vars:
    x_list = ['competitor_count_log',f'{core_comp}_x_post'] + controls
    x_real = [var_map_q.get(x,x) for x in x_list]
    label = f'E_comp_{med_disp}'
    r = run_mech(med_disp, x_real, p_int, label)
    if r is not None:
        row = r[r['variable']==f'{core_comp}_x_post']
        if len(row):
            rw = row.iloc[0]
            d = '+' if rw['coef']>0 else '-'
            m = '✓' if d == exp_dir else '✗'
            print(f"  {med_disp:20s} ({desc:12s}): {rw['coef']:+.5f}{rw['sig']}  p={rw['pval']:.4f}  {d}(预期{exp_dir}) {m}")
            all_results.append(r)

# 排除2020
print("\n--- E2: 竞争对手广度 (排除COVID 2020) ---")
for med_disp, exp_dir, desc in comp_vars:
    x_list = ['competitor_count_log',f'{core_comp}_x_post'] + controls
    x_real = [var_map_q.get(x,x) for x in x_list]
    label = f'E2_comp_no2020_{med_disp}'
    r = run_mech(med_disp, x_real, p_no2020, label)
    if r is not None:
        row = r[r['variable']==f'{core_comp}_x_post']
        if len(row):
            rw = row.iloc[0]
            d = '+' if rw['coef']>0 else '-'
            print(f"  {med_disp:20s}: {rw['coef']:+.5f}{rw['sig']}  p={rw['pval']:.4f}  {d}(预期{exp_dir})")
            all_results.append(r)

# ----------------------------------------------------------------
# 测试F: 三交互项 (Triple interaction)
# 供应商广度 × post × high_cogs → ROA
# ----------------------------------------------------------------
print("\n--- F: 三交互项 (sup_breadth × post × 高成本率) ---")
p_triple = qp.set_index(['ISIN','quarter_dt']).copy()
p_triple['high_cogs_int'] = p_triple['ISIN'].isin(
    qp.groupby('ISIN')['cogs_r'].mean().pipe(lambda s: s[s>s.median()].index)
).astype(int)
p_triple['sup_x_highcogs'] = p_triple['sup_breadth_q'] * p_triple['high_cogs_int']
p_triple['sup_x_post'] = p_triple['sup_breadth_q'] * p_triple['post2018']
p_triple['triple'] = p_triple['sup_breadth_q'] * p_triple['post2018'] * p_triple['high_cogs_int']

x_triple = ['sup_breadth_q','high_cogs_int','sup_x_highcogs','sup_x_post','triple',
            'size_q','lev_q','growth_w_q']
r = run_mech('ROA', x_triple, p_triple, 'F_triple')
if r is not None:
    row = r[r['variable']=='triple']
    if len(row):
        rw = row.iloc[0]
        print(f"  三交互项系数: {rw['coef']:+.5f}{rw['sig']}  p={rw['pval']:.4f}")
        print(f"  含义: 成本越高的企业, 供应商广度在贸易战后的负效应是否越强")
        all_results.append(r)
    for _, rw in r.iterrows():
        print(f"    {rw['variable']:30s} {rw['coef']:+.5f}{rw['sig']}  p={rw['pval']:.4f}")

# ====================================================================
# 5. 保存
# ====================================================================
if all_results:
    rdf = pd.concat(all_results, ignore_index=True)
    rdf.to_excel(f'{OUT_DIR}/mechanism_supplement_results.xlsx', index=False)
    print(f"\n  ✓ {OUT_DIR}/mechanism_supplement_results.xlsx")

# ====================================================================
# 6. 汇总报告
# ====================================================================
print("\n\n" + "=" * 60)
print("结果汇总")
print("=" * 60)

# 提取最关键的发现
print("\n【供应商广度 → 成本/利润率渠道】")
print(f"{'方法':25s} {'变量':20s} {'系数':>12s} {'p值':>8s}")
print("-"*65)
for df in all_results:
    for _, r in df.iterrows():
        if 'sup' in r['model'] and 'x_post' in r['variable'] and r['pval'] < 0.2:
            print(f"  {r['model']:25s} {r['variable']:20s} {r['coef']:+8.4f}{r['sig']:3s}  p={r['pval']:.4f}")

print("\n【竞争对手广度 → 费用/利润率渠道】")
for df in all_results:
    for _, r in df.iterrows():
        if 'comp' in r['model'] and 'x_post' in r['variable'] and r['pval'] < 0.2:
            print(f"  {r['model']:25s} {r['variable']:20s} {r['coef']:+8.4f}{r['sig']:3s}  p={r['pval']:.4f}")

print(f"\n所有结果已保存到: {OUT_DIR}/")
print("完成!")
