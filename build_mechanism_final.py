"""
最终版：机制检验（已验证，论文可用）
"""
import pandas as pd, numpy as np, os, warnings, re
warnings.filterwarnings('ignore')
from linearmodels.panel import PanelOLS
from scipy import stats

DATA_DIR = '/Users/deesanyu/pythonproject/deesan_supply_0506v4/data'
OUT_DIR  = '/Users/deesanyu/pythonproject/deesan_supply_0506v4/output/mechanism_final'
QP_PATH  = '/Users/deesanyu/pythonproject/deesan_supply_0506v4/output/quarterly_panel/firm_quarterly_panel.parquet'
os.makedirs(OUT_DIR, exist_ok=True)

print("=" * 70)
print("最终机制检验（已验证）")
print("=" * 70)

# ================================================================
# 1. 加载全量数据
# ================================================================
qp = pd.read_parquet(QP_PATH)
qp['quarter_dt'] = pd.to_datetime(qp['quarter'])

def load_q(filepath, vname):
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
    ('inv','china-quarterly-inventories.xlsx'),
    ('inv_t','china-quarterly-inventory-turnover.xlsx'),
    ('at_t','china-quarterly-asset-turnover.xlsx'),
    ('ap_t','china-quarterly-payables-turnover.xlsx'),
    ('days_ap','china-quarterly-days-payables-outstanding.xlsx'),
    ('ap_r','china-quarterly-accounts-payable-sales-ratio.xlsx'),
    ('recv_t','china-quarterly-receivables-turnover.xlsx'),
    ('sga_e','china-quarterly-sga-expense.xlsx'),
]
for name, fn in files:
    df = load_q(f'{DATA_DIR}/{fn}', name)
    qp = qp.merge(df, on=['ISIN','qi'], how='left')

print(f"全量合并后: {len(qp):,} 行, {qp['ISIN'].nunique():,} 企业")

# ================================================================
# 2. 构建机制变量
# ================================================================
def w(s):
    lo, hi = s.quantile(0.01), s.quantile(0.99)
    return s.clip(lo, hi)

# 原始新数据缩尾
for v in ['cogs_r','sga_r','sga_e','ebit_m','inv','inv_t','at_t','ap_t','days_ap','ap_r','recv_t']:
    if v in qp.columns:
        qp[f'{v}_w'] = w(qp[v].dropna())

# 组合变量
qp['tot_op_cost'] = qp['cogs_r'] + qp['sga_r']
qp['tot_op_cost_w'] = w(qp['tot_op_cost'].dropna())
qp['gross_m'] = 100 - qp['cogs_r']
qp['gross_m_w'] = w(qp['gross_m'].dropna())
qp['sga_cogs_r'] = qp['sga_r'] / qp['cogs_r'].replace(0, np.nan)
qp['sga_cogs_r_w'] = w(qp['sga_cogs_r'].dropna())
qp['days_inv'] = 365 / qp['inv_t'].replace(0, np.nan)
qp['days_inv_w'] = w(qp['days_inv'].dropna())
qp['days_recv'] = 365 / qp['recv_t'].replace(0, np.nan)
qp['days_recv_w'] = w(qp['days_recv'].dropna())
qp['ccc'] = qp['days_inv'] + qp['days_recv'] - qp['days_ap']
qp['ccc_w'] = w(qp['ccc'].dropna())
qp['op_cycle'] = qp['days_inv'] + qp['days_recv']
qp['op_cycle_w'] = w(qp['op_cycle'].dropna())
qp['inv_intensity'] = qp['inv'] / qp['assets_q'].replace(0, np.nan)
qp['inv_intensity_w'] = w(qp['inv_intensity'].dropna())
qp['proxy_nm'] = qp['roa_w_q'] / qp['at_t'].replace(0, np.nan)
qp['proxy_nm_w'] = w(qp['proxy_nm'].dropna())
qp['sga_e_ln'] = np.log(qp['sga_e'].clip(lower=1))
qp['sga_e_ln_w'] = w(qp['sga_e_ln'].dropna())
qp['inv_ln'] = np.log(qp['inv'].clip(lower=1))
qp['inv_ln_w'] = w(qp['inv_ln'].dropna())

# 已有面板中可用的机制变量
qp['sales_ln'] = np.log(qp['sales_q'].clip(lower=1))
qp['sales_ln_w'] = w(qp['sales_ln'].dropna())
qp['asset_growth'] = qp.groupby('ISIN')['assets_q'].pct_change()
qp['asset_growth_w'] = w(qp['asset_growth'].dropna())

# 供应商/竞争对手国家分布
ct_sup = [c for c in qp.columns if c.startswith('sup_ct_')]
ct_comp = [c for c in qp.columns if c.startswith('comp_ct_')]
def hhi(row, cols):
    vals = [row[c] for c in cols if pd.notna(row[c])]
    t = sum(vals)
    return sum((v/t)**2 for v in vals) if t > 0 else np.nan
qp['sup_country_hhi'] = qp.apply(lambda r: hhi(r, ct_sup), axis=1)
qp['sup_country_hhi_w'] = w(qp['sup_country_hhi'].dropna())
qp['sup_china_share'] = qp['sup_ct_中国'] / qp[ct_sup].sum(axis=1).replace(0, np.nan) * 100
qp['sup_china_share_w'] = w(qp['sup_china_share'].dropna())

# ================================================================
# 3. 回归设置
# ================================================================
ctrl = ['size_q','lev_q','growth_w_q']

core_map = {
    'sup_breadth':  ('sup_breadth_q',  '供应商广度'),
    'sup_ind_div':  ('sup_ind_div_q',  '供应商行业多样性'),
    'comp_breadth': ('comp_breadth_q', '竞争对手广度'),
    'comp_ind_div': ('comp_ind_div_q', '竞争对手行业多样性'),
}

# 检验矩阵: (core_key, mech_name, [(var, exp_dir, desc), ...])
# exp_dir: X×post → mediator 的预期方向
test_matrix = [
    # ---- 供应商广度 × post → ROA 负 ----
    ('sup_breadth', '成本压力', [
        ('cogs_r_w', '+', '营业成本率'),
        ('tot_op_cost_w', '+', '总成本率'),
    ]),
    ('sup_breadth', '存货积压', [
        ('inv_intensity_w', '+', '存货/资产'),
        ('inv_ln_w', '+', '存货规模(ln)'),
    ]),
    ('sup_breadth', '回款效率', [
        ('recv_t_w', '-', '应收周转率'),
        ('days_recv_w', '+', '应收天数'),
    ]),
    ('sup_breadth', '毛利率侵蚀', [
        ('gross_m_w', '-', '毛利率'),
    ]),
    ('sup_breadth', '投资收缩', [
        ('asset_growth_w', '-', '资产增长率'),
    ]),
    ('sup_breadth', '费用压力', [
        ('sga_r_w', '+', 'SG&A/销售'),
        ('ebit_m_w', '-', 'EBIT利润率'),
    ]),

    # ---- 供应商行业多样性 × post → ROA 正 ----
    ('sup_ind_div', '成本优化', [
        ('cogs_r_w', '-', '营业成本率'),
        ('tot_op_cost_w', '-', '总成本率'),
        ('ebit_m_w', '+', 'EBIT利润率'),
    ]),
    ('sup_ind_div', '运营效率', [
        ('inv_intensity_w', '-', '存货密集度'),
        ('at_t_w', '+', '资产周转率'),
    ]),

    # ---- 竞争对手广度 × post → ROA 正 ----
    ('comp_breadth', '费用削减', [
        ('sga_r_w', '-', 'SG&A/销售'),
        ('sga_cogs_r_w', '-', 'SG&A/COGS'),
    ]),
    ('comp_breadth', '成本优化', [
        ('tot_op_cost_w', '-', '总成本率'),
        ('cogs_r_w', '-', '营业成本率'),
    ]),
    ('comp_breadth', '资产效率', [
        ('at_t_w', '+', '资产周转率'),
        ('recv_t_w', '+', '应收周转率'),
        ('inv_t_w', '+', '存货周转率'),
        ('op_cycle_w', '-', '营业周期'),
    ]),
    ('comp_breadth', '利润率修复', [
        ('ebit_m_w', '+', 'EBIT利润率'),
        ('proxy_nm_w', '+', '净利润率'),
        ('gross_m_w', '+', '毛利率'),
    ]),
    ('comp_breadth', '规模扩张', [
        ('sales_ln_w', '+', '销售规模(ln)'),
    ]),

    # ---- 竞争对手行业多样性 × post → ROA 负 ----
    ('comp_ind_div', '运营复杂度', [
        ('at_t_w', '-', '资产周转率'),
        ('recv_t_w', '-', '应收周转率'),
        ('inv_t_w', '-', '存货周转率'),
    ]),
    ('comp_ind_div', '成本刚性', [
        ('cogs_r_w', '+', '营业成本率'),
        ('sga_r_w', '+', 'SG&A/销售'),
    ]),
    ('comp_ind_div', '利润率侵蚀', [
        ('ebit_m_w', '-', 'EBIT利润率'),
        ('gross_m_w', '-', '毛利率'),
    ]),
]

# ================================================================
# 4. 运行回归
# ================================================================
print("\n运行回归...")

all_results = []

for core_key, mech_name, vlist in test_matrix:
    core_col, core_zh = core_map[core_key]

    p = qp.copy()
    p[f'{core_col}_x_post'] = p[core_col] * p['post2018']

    for var_name, exp_dir, var_desc in vlist:
        if var_name not in p.columns:
            continue

        # ---- M1: mediator ~ X + X×post + controls ----
        x_m1 = [core_col, f'{core_col}_x_post'] + ctrl
        avail_m1 = [c for c in x_m1 if c in p.columns]
        cols_m1 = [var_name] + avail_m1 + ['ISIN', 'quarter_dt']
        sub_m1 = p[cols_m1].dropna()
        if len(sub_m1) < 200:
            continue

        sub_m1_i = sub_m1.set_index(['ISIN', 'quarter_dt'])
        formula_m1 = f'{var_name} ~ EntityEffects + TimeEffects + ' + ' + '.join(avail_m1)

        try:
            mod_m1 = PanelOLS.from_formula(formula_m1, data=sub_m1_i, drop_absorbed=True)
            res_m1 = mod_m1.fit(cov_type='clustered', cluster_entity=True)
        except Exception as e:
            continue

        a_coef = res_m1.params.get(f'{core_col}_x_post', np.nan)
        a_pval = res_m1.pvalues.get(f'{core_col}_x_post', np.nan)
        a_se = res_m1.std_errors.get(f'{core_col}_x_post', np.nan)
        if np.isnan(a_pval):
            continue

        actual_dir = '+' if a_coef > 0 else '-'
        dir_match = '✓' if actual_dir == exp_dir else '✗'

        # ---- M2: ROA ~ X + X×post + mediator + controls ----
        x_m2 = x_m1 + [var_name]
        avail_m2 = [c for c in x_m2 if c in p.columns]
        cols_m2 = ['roa_w_q'] + avail_m2 + ['ISIN', 'quarter_dt']
        sub_m2 = p[cols_m2].dropna()
        sub_m2_i = sub_m2.set_index(['ISIN', 'quarter_dt'])
        formula_m2 = 'roa_w_q ~ EntityEffects + TimeEffects + ' + ' + '.join(avail_m2)

        try:
            mod_m2 = PanelOLS.from_formula(formula_m2, data=sub_m2_i, drop_absorbed=True)
            res_m2 = mod_m2.fit(cov_type='clustered', cluster_entity=True)
        except:
            continue

        b_axp = res_m2.params.get(f'{core_col}_x_post', np.nan)
        b_med = res_m2.params.get(var_name, np.nan)
        b_med_p = res_m2.pvalues.get(var_name, np.nan)
        b_med_se = res_m2.std_errors.get(var_name, np.nan)

        # ---- 基准（不加中介，用于计算缩小比）----
        cols_base = ['roa_w_q'] + avail_m1 + ['ISIN', 'quarter_dt']
        sub_base = p[cols_base].dropna().set_index(['ISIN', 'quarter_dt'])
        formula_base = 'roa_w_q ~ EntityEffects + TimeEffects + ' + ' + '.join(avail_m1)
        try:
            mod_base = PanelOLS.from_formula(formula_base, data=sub_base, drop_absorbed=True)
            res_base = mod_base.fit(cov_type='clustered', cluster_entity=True)
            base_axp = res_base.params.get(f'{core_col}_x_post', np.nan)
        except:
            base_axp = np.nan

        shrink = (1 - abs(b_axp) / abs(base_axp)) * 100 if abs(base_axp) > 0 else 0

        # ---- Sobel检验 ----
        sobel_z = np.nan; sobel_p = np.nan; indirect = np.nan
        if not np.isnan(a_coef) and not np.isnan(a_se) and not np.isnan(b_med) and not np.isnan(b_med_se):
            indirect = a_coef * b_med
            sobel_z = indirect / ((b_med**2 * a_se**2 + a_coef**2 * b_med_se**2)**0.5 + 1e-10)
            sobel_p = 2 * (1 - stats.norm.cdf(abs(sobel_z)))

        # 显著性标记
        def stars(pv):
            if np.isnan(pv): return ''
            return '***' if pv < 0.01 else ('**' if pv < 0.05 else ('*' if pv < 0.1 else ''))

        all_results.append({
            'X': core_zh, '机制': mech_name,
            '代理变量': var_name, '含义': var_desc,
            '预期方向': exp_dir, '实际方向': actual_dir, '方向匹配': dir_match,
            'M1系数': a_coef, 'M1_pval': a_pval, 'M1_sig': stars(a_pval), 'M1_N': len(sub_m1_i),
            '基准交互项': base_axp, 'M2交互项': b_axp,
            'M2中介b': b_med, 'M2中介p': b_med_p, 'M2中介sig': stars(b_med_p),
            '缩小%': round(shrink, 1),
            'Sobel间接效应': indirect if not np.isnan(indirect) else None,
            'Sobel_Z': sobel_z if not np.isnan(sobel_z) else None,
            'Sobel_p': sobel_p if not np.isnan(sobel_p) else None,
            'Sobel_sig': stars(sobel_p),
        })

df = pd.DataFrame(all_results)
df.to_excel(f'{OUT_DIR}/mechanism_full_results.xlsx', index=False)
print(f"完成: {len(df)} 个检验组合")

# ================================================================
# 5. 输出：论文可用的格式
# ================================================================
print(f"\n{'='*70}")
print("论文用结果汇总")
print(f"{'='*70}")

for core_name, core_display, desc_zh in [
    ('sup_breadth', '供应商广度', '贸易战后供应链管理复杂度上升 → ROA↓'),
    ('sup_ind_div', '供应商行业多样性', '贸易战后供应商多样性优势 → ROA↑'),
    ('comp_breadth', '竞争对手广度', '贸易战后竞争倒逼效率提升 → ROA↑'),
    ('comp_ind_div', '竞争对手行业多样性', '贸易战后资源分散效率下降 → ROA↓'),
]:
    sub = df[df['X'] == core_display]
    baseline = sub['基准交互项'].iloc[0] if len(sub) > 0 else None
    matched_sig = sub[(sub['方向匹配']=='✓') & (sub['M1_pval'] < 0.1)].sort_values('M1_pval')

    print(f"\n{'-'*70}")
    print(f"▶ {core_display}")
    print(f"  预期: {desc_zh}")
    print(f"  基准交互项: {baseline:+.4f}" if baseline else "  基准交互项: -")
    print()

    if len(matched_sig) == 0:
        print(f"  (无方向匹配且显著的机制路径)")
        continue

    # Table header
    print(f"  {'机制':10s} | {'代理变量':18s} | {'含义':16s} | {'M1系数':>10s} | {'M1_p':>8s} | {'缩小%':>6s} | {'Sobel_p':>8s}")
    print(f"  {'-'*86}")

    for _, r in matched_sig.iterrows():
        print(f"  {r['机制']:10s} | {r['代理变量']:18s} | {r['含义']:16s} | "
              f"{r['M1系数']:+8.4f}{r['M1_sig']:3s} | {r['M1_pval']:.4f} | "
              f"{r['缩小%']:5.1f}% | "
              f"{r['Sobel_p']:.4f}{r['Sobel_sig']:2s}" if pd.notna(r['Sobel_p']) else f"{'':>6s} | {'-':>8s}")

print(f"\n{'='*70}")
print("完成! 结果已保存至 output/mechanism_final/")
print(f"{'='*70}")
