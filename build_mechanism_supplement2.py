"""
机制检验补充：加入被忽略的已有面板变量 + 新数据的新视角
"""
import pandas as pd, numpy as np, os, warnings, re
warnings.filterwarnings('ignore')
from linearmodels.panel import PanelOLS
from scipy import stats

DATA_DIR = '/Users/deesanyu/pythonproject/deesan_supply_0506v4/data'
OUT_DIR  = '/Users/deesanyu/pythonproject/deesan_supply_0506v4/output/mechanism_supplement2'
QP_PATH  = '/Users/deesanyu/pythonproject/deesan_supply_0506v4/output/quarterly_panel/firm_quarterly_panel.parquet'
os.makedirs(OUT_DIR, exist_ok=True)

print("=" * 70)
print("机制检验补充：全部已有变量 + 新数据新视角")
print("=" * 70)

# ====================================================================
# 1. 加载数据
# ====================================================================
qp = pd.read_parquet(QP_PATH)
qp['quarter_dt'] = pd.to_datetime(qp['quarter'])
print(f"基础面板: {len(qp):,} 行, {qp['ISIN'].nunique():,} 企业")

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
    ('sga_e','china-quarterly-sga-expense.xlsx'),
    ('ebit_m','china-quarterly-ebit-margin.xlsx'),
    ('inv','china-quarterly-inventories.xlsx'),
    ('inv_t','china-quarterly-inventory-turnover.xlsx'),
    ('at_t','china-quarterly-asset-turnover.xlsx'),
    ('ap_t','china-quarterly-payables-turnover.xlsx'),
    ('days_ap','china-quarterly-days-payables-outstanding.xlsx'),
    ('ap_r','china-quarterly-accounts-payable-sales-ratio.xlsx'),
    ('recv_t','china-quarterly-receivables-turnover.xlsx'),
]
for name, fn in files:
    df = load_q_data(f'{DATA_DIR}/{fn}', name)
    qp = qp.merge(df, on=['ISIN','qi'], how='left')

# ====================================================================
# 2. 构建所有机制变量（已有面板 + 新数据）
# ====================================================================
def winsor(s, l=0.01, u=0.99):
    lo, hi = s.quantile(l), s.quantile(u)
    return s.clip(lo, hi)

# --- A. 所有新数据变量缩尾 ---
raw_new = ['cogs_r','sga_r','sga_e','ebit_m','inv','inv_t','at_t','ap_t','days_ap','ap_r','recv_t']
for v in raw_new:
    if v in qp.columns:
        qp[f'{v}_w'] = winsor(qp[v].dropna())

# 组合变量
qp['tot_op_cost'] = qp['cogs_r'] + qp['sga_r']
qp['tot_op_cost_w'] = winsor(qp['tot_op_cost'].dropna())
qp['gross_m'] = 100 - qp['cogs_r']
qp['gross_m_w'] = winsor(qp['gross_m'].dropna())
qp['sga_cogs_r'] = qp['sga_r'] / qp['cogs_r'].replace(0, np.nan)
qp['sga_cogs_r_w'] = winsor(qp['sga_cogs_r'].dropna())
qp['days_inv'] = 365 / qp['inv_t'].replace(0, np.nan)
qp['days_inv_w'] = winsor(qp['days_inv'].dropna())
qp['days_recv'] = 365 / qp['recv_t'].replace(0, np.nan)
qp['days_recv_w'] = winsor(qp['days_recv'].dropna())
qp['ccc'] = qp['days_inv'] + qp['days_recv'] - qp['days_ap']
qp['ccc_w'] = winsor(qp['ccc'].dropna())
qp['op_cycle'] = qp['days_inv'] + qp['days_recv']
qp['op_cycle_w'] = winsor(qp['op_cycle'].dropna())
qp['inv_intensity'] = qp['inv'] / qp['assets_q'].replace(0, np.nan)
qp['inv_intensity_w'] = winsor(qp['inv_intensity'].dropna())
qp['proxy_nm'] = qp['roa_w_q'] / qp['at_t'].replace(0, np.nan)
qp['proxy_nm_w'] = winsor(qp['proxy_nm'].dropna())
qp['ap_cogs_r'] = qp['ap_r'] / (qp['cogs_r']/100).replace(0, np.nan)
qp['ap_cogs_r_w'] = winsor(qp['ap_cogs_r'].dropna())

# --- B. 已有面板变量中被忽略的 ---
# 销售规模（取ln）
qp['sales_ln'] = np.log(qp['sales_q'].clip(lower=1))
qp['sales_ln_w'] = winsor(qp['sales_ln'].dropna())

# 资产规模变化率（投资扩张）
qp['asset_growth'] = qp.groupby('ISIN')['assets_q'].pct_change()
qp['asset_growth_w'] = winsor(qp['asset_growth'].dropna())

# 负债率（债务融资依赖）
qp['debt_ratio'] = qp['debt_q'] / qp['assets_q'].replace(0, np.nan) * 100
qp['debt_ratio_w'] = winsor(qp['debt_ratio'].dropna())

# ROE作为替代绩效
qp['roe_w_q'] = winsor(qp['roe_q'].dropna())

# --- B. 供应商/竞争对手国家分布变量 ---
ct_sup = [c for c in qp.columns if c.startswith('sup_ct_')]
ct_comp = [c for c in qp.columns if c.startswith('comp_ct_')]

# 国家HHI集中度
def hhi(row, cols):
    vals = [row[c] for c in cols if pd.notna(row[c])]
    t = sum(vals)
    return sum((v/t)**2 for v in vals) if t > 0 else np.nan

qp['sup_country_hhi'] = qp.apply(lambda r: hhi(r, ct_sup), axis=1)
qp['sup_country_hhi_w'] = winsor(qp['sup_country_hhi'].dropna())
qp['comp_country_hhi'] = qp.apply(lambda r: hhi(r, ct_comp), axis=1)
qp['comp_country_hhi_w'] = winsor(qp['comp_country_hhi'].dropna())

# 中国供应商/竞争对手占比
for prefix, cols in [('sup', ct_sup), ('comp', ct_comp)]:
    china_col = f'{prefix}_ct_中国'
    if china_col in cols:
        total = qp[cols].sum(axis=1)
        qp[f'{prefix}_china_share'] = qp[china_col] / total.replace(0, np.nan) * 100
        qp[f'{prefix}_china_share_w'] = winsor(qp[f'{prefix}_china_share'].dropna())

# 美国供应商/竞争对手占比
for prefix, cols in [('sup', ct_sup), ('comp', ct_comp)]:
    us_col = f'{prefix}_ct_美国'
    if us_col in cols:
        total = qp[cols].sum(axis=1)
        qp[f'{prefix}_us_share'] = qp[us_col] / total.replace(0, np.nan) * 100
        qp[f'{prefix}_us_share_w'] = winsor(qp[f'{prefix}_us_share'].dropna())

# 西方（美+其他）占比
for prefix, cols in [('sup', ct_sup), ('comp', ct_comp)]:
    west_cols = [c for c in cols if c.endswith(('_美国','_其他'))]
    total = qp[cols].sum(axis=1)
    qp[f'{prefix}_west_share'] = qp[west_cols].sum(axis=1) / total.replace(0, np.nan) * 100
    qp[f'{prefix}_west_share_w'] = winsor(qp[f'{prefix}_west_share'].dropna())

# 供应商国家数量（多样化程度）
qp['sup_country_count'] = qp[ct_sup].apply(lambda r: (r > 0).sum(), axis=1)
qp['sup_country_count_w'] = winsor(qp['sup_country_count'].dropna())
qp['comp_country_count'] = qp[ct_comp].apply(lambda r: (r > 0).sum(), axis=1)
qp['comp_country_count_w'] = winsor(qp['comp_country_count'].dropna())

# --- C. 新数据的新视角 ---
# receivables turnover → 回款速度/下游议价能力
#   low = 客户付款慢 = 下游弱势 or 信用销售多
# 与 days_recv 是同一经济含义的不同表达
qp['recv_t_w'] = winsor(qp['recv_t'].dropna())
qp['days_recv_w'] = winsor((365 / qp['recv_t'].replace(0, np.nan)).dropna())

# payables turnover → 上游议价能力
#   low = 拖欠供应商 = 自身融资压力大 or 对供应商有议价权
qp['ap_t_w'] = winsor(qp['ap_t'].dropna())

# SG&A绝对额（取ln） → 费用规模（非比率）
qp['sga_e_ln'] = np.log(qp['sga_e'].clip(lower=1))
qp['sga_e_ln_w'] = winsor(qp['sga_e_ln'].dropna())

# 存货绝对额（取ln）→ 备货规模
qp['inv_ln'] = np.log(qp['inv'].clip(lower=1))
qp['inv_ln_w'] = winsor(qp['inv_ln'].dropna())

# 各种周转率从"效率"角度统一对待
# ap_t, recv_t, inv_t, at_t 都是周转率概念

# ====================================================================
# 3. 定义检验矩阵
# ====================================================================

# (core_key, mechanism_name, [(var_name, expected_sign, description), ...])
# expected_sign: '+' means X×post → mediator ↑ → ROA ↓ or mediator ↓ → ROA ↑
#               For clarity, we use: '+' = positive mediation (mediator goes UP and that lowers ROA, or mediator goes DOWN and that raises ROA...)
# Actually simpler: just say expected sign of X×post → mediator.
# We'll check: if mediator goes in the expected direction AND mediator in M2 is significant

test_defs = [
    # ========== 供应商广度 × post → ROA 负 ==========
    ('sup', 'A:成本压力', [
        ('cogs_r_w', '+', '营业成本率↑'),
        ('tot_op_cost_w', '+', '总成本率↑'),
    ]),
    ('sup', 'B:存货积压', [
        ('inv_intensity_w', '+', '存货密集度↑'),
        ('inv_ln_w', '+', '存货规模↑'),
    ]),
    ('sup', 'C:融资压力(应付)', [
        ('ap_r_w', '+', '应付/销售↑'),
        ('ap_t_w', '-', '应付周转率↓=拖欠供应商'),
        ('days_ap_w', '+', '应付天数↑'),
    ]),
    ('sup', 'D:回款困难(应收)', [
        ('recv_t_w', '-', '应收周转率↓=收款变慢'),
        ('days_recv_w', '+', '应收天数↑'),
    ]),
    ('sup', 'E:运营效率', [
        ('at_t_w', '-', '资产周转率↓'),
        ('ccc_w', '+', '现金转换周期↑'),
        ('op_cycle_w', '+', '营业周期↑'),
    ]),
    ('sup', 'F:利润率', [
        ('ebit_m_w', '-', 'EBIT利润率↓'),
        ('gross_m_w', '-', '毛利率↓'),
    ]),
    ('sup', 'G:资产扩张', [
        ('asset_growth_w', '-', '资产增长率↓=投资收缩'),
    ]),
    ('sup', 'H:供应商国家集中', [
        ('sup_country_hhi_w', '+', '国家集中度↑=收缩到中国'),
        ('sup_china_share_w', '+', '中国供应商占比↑'),
        ('sup_west_share_w', '-', '西方供应商占比↓'),
    ]),

    # ========== 供应商行业多样性 × post → ROA 正 ==========
    ('sup_ind', 'A:经营稳定', [
        ('growth_vol_w', '-', '收入波动率↓'),
    ]),
    ('sup_ind', 'B:成本优化', [
        ('cogs_r_w', '-', '成本率↓'),
        ('tot_op_cost_w', '-', '总成本率↓'),
        ('ebit_m_w', '+', 'EBIT利润率↑'),
    ]),
    ('sup_ind', 'C:运营效率', [
        ('inv_intensity_w', '-', '存货密集度↓'),
        ('at_t_w', '+', '资产周转率↑'),
        ('inv_t_w', '+', '存货周转率↑'),
        ('recv_t_w', '+', '应收周转率↑'),
    ]),
    ('sup_ind', 'D:供应链地理分散', [
        ('sup_country_hhi_w', '-', '供应商国家集中度↓'),
        ('sup_country_count_w', '+', '供应商国家数↑'),
    ]),

    # ========== 竞争对手广度 × post → ROA 正 ==========
    ('comp', 'A:费用削减', [
        ('sga_r_w', '-', 'SG&A/销售↓'),
        ('sga_e_ln_w', '-', 'SG&A绝对规模↓'),
        ('sga_cogs_r_w', '-', 'SG&A/COGS↓'),
    ]),
    ('comp', 'B:成本优化', [
        ('cogs_r_w', '-', '成本率↓'),
        ('tot_op_cost_w', '-', '总成本率↓'),
    ]),
    ('comp', 'C:资产效率', [
        ('at_t_w', '+', '资产周转率↑'),
        ('inv_t_w', '+', '存货周转率↑'),
        ('recv_t_w', '+', '应收周转率↑=回款加快'),
        ('op_cycle_w', '-', '营业周期↓'),
        ('ccc_w', '-', '现金转换周期↓'),
    ]),
    ('comp', 'D:利润率修复', [
        ('ebit_m_w', '+', 'EBIT利润率↑'),
        ('gross_m_w', '+', '毛利率↑'),
        ('proxy_nm_w', '+', '净利润率↑'),
    ]),
    ('comp', 'E:投资扩张', [
        ('asset_growth_w', '+', '资产增长率↑=扩张'),
        ('sales_ln_w', '+', '销售规模↑'),
    ]),

    # ========== 竞争对手行业多样性 × post → ROA 负 ==========
    ('comp_ind', 'A:运营复杂度', [
        ('at_t_w', '-', '资产周转率↓'),
        ('inv_t_w', '-', '存货周转率↓'),
        ('recv_t_w', '-', '应收周转率↓'),
        ('ccc_w', '+', '现金转换周期↑'),
    ]),
    ('comp_ind', 'B:成本刚性', [
        ('cogs_r_w', '+', '成本率↑'),
        ('sga_r_w', '+', 'SG&A/销售↑'),
        ('sga_e_ln_w', '+', 'SG&A规模↑(难削减)'),
    ]),
    ('comp_ind', 'C:利润率侵蚀', [
        ('ebit_m_w', '-', 'EBIT利润率↓'),
        ('gross_m_w', '-', '毛利率↓'),
        ('proxy_nm_w', '-', '净利润率↓'),
    ]),
]

# ====================================================================
# 4. 运行检验
# ====================================================================
print("\n4. 运行系统化检验...")

ctrl = ['size_q','lev_q','growth_w_q']
var_map = {'ROA':'roa_w_q','Size':'size_q','Lev':'lev_q','Growth':'growth_w_q'}
core_map = {
    'sup': 'sup_breadth_q', 'sup_ind': 'sup_ind_div_q',
    'comp': 'comp_breadth_q', 'comp_ind': 'comp_ind_div_q',
}
core_zh = {
    'sup': '供应商广度', 'sup_ind': '供应商行业多样性',
    'comp': '竞争对手广度', 'comp_ind': '竞争对手行业多样性',
}

all_summary = []
all_regs = []

def run_m1(med_name, x_list, data, label):
    y_name = var_map.get(med_name, med_name)
    available = [c for c in x_list if c in data.columns]
    if y_name not in data.columns or y_name in available:
        available = [c for c in available if c != y_name]
    cols_check = [y_name] + available + ['ISIN','quarter_dt']
    cols_check = [c for c in cols_check if c in data.columns]
    sub = data[cols_check].dropna()
    if len(sub) < 200: return None, None
    sub_i = sub.set_index(['ISIN','quarter_dt'])
    formula = f'{y_name} ~ EntityEffects + TimeEffects + ' + ' + '.join(available)
    try:
        mod = PanelOLS.from_formula(formula, data=sub_i, drop_absorbed=True)
        res = mod.fit(cov_type='clustered', cluster_entity=True)
        return res, len(sub_i)
    except Exception as e:
        return None, None

def run_m2(x_list, data, label):
    available = [c for c in x_list if c in data.columns]
    cols_check = ['roa_w_q'] + available + ['ISIN','quarter_dt']
    sub = data[cols_check].dropna()
    if len(sub) < 200: return None
    sub_i = sub.set_index(['ISIN','quarter_dt'])
    formula = 'roa_w_q ~ EntityEffects + TimeEffects + ' + ' + '.join(available)
    try:
        mod = PanelOLS.from_formula(formula, data=sub_i, drop_absorbed=True)
        res = mod.fit(cov_type='clustered', cluster_entity=True)
        return res
    except:
        return None

total = 0
sig_count = 0

for core_key, mech_name, var_list in test_defs:
    core_col = core_map[core_key]
    core_zn = core_zh[core_key]

    p = qp.copy()
    if 'quarter_dt' not in p.columns:
        p['quarter_dt'] = pd.to_datetime(p['quarter'])
    p[f'{core_col}_x_post'] = p[core_col] * p['post2018']

    for med_name, exp_dir, desc in var_list:
        if med_name not in p.columns:
            continue
        total += 1

        x_list = [core_col, f'{core_col}_x_post'] + ctrl

        # M1: mediator ~ X×post
        res_m1, n_obs = run_m1(med_name, x_list, p, f'M1_{core_key}_{med_name}')
        if res_m1 is None:
            continue

        axp_coef = res_m1.params.get(f'{core_col}_x_post', np.nan)
        axp_pv = res_m1.pvalues.get(f'{core_col}_x_post', np.nan)
        axp_se = res_m1.std_errors.get(f'{core_col}_x_post', np.nan)

        stars = ''
        if not np.isnan(axp_pv):
            if axp_pv < 0.01: stars = '***'
            elif axp_pv < 0.05: stars = '**'
            elif axp_pv < 0.1: stars = '*'

        actual_dir = '+' if axp_coef > 0 else '-'
        dir_match = '✓' if actual_dir == exp_dir else '✗'

        # M2: ROA ~ X + X×post + mediator + controls
        x_list2 = x_list + [med_name]
        res_m2 = run_m2(x_list2, p, f'M2_{core_key}_{med_name}')

        m2_axp = None; m2_med = None; m2_med_pv = None
        if res_m2 is not None:
            m2_axp = res_m2.params.get(f'{core_col}_x_post', np.nan)
            m2_med = res_m2.params.get(med_name, np.nan)
            m2_med_pv = res_m2.pvalues.get(med_name, np.nan)

            # Baseline interaction (without mediator) for shrinkage comparison
            res_base = run_m2(x_list, p, f'BASE_{core_key}_{med_name}')
            if res_base is not None:
                base_axp = res_base.params.get(f'{core_col}_x_post', np.nan)
                shrink = (1 - abs(m2_axp)/abs(base_axp))*100 if abs(base_axp) > 0 else 0
            else:
                base_axp = np.nan; shrink = 0
        else:
            base_axp = np.nan; shrink = 0

        # Sobel test
        sobel_z = np.nan; sobel_p = np.nan; indirect = np.nan
        if not np.isnan(axp_coef) and not np.isnan(axp_se) and not np.isnan(m2_med) and not np.isnan(m2_med_pv) and m2_med_pv < 0.1:
            b_coef = m2_med
            # Need b SE from M2
            b_se = res_m2.std_errors.get(med_name, np.nan) if res_m2 is not None else np.nan
            if not np.isnan(b_se):
                indirect = axp_coef * b_coef
                sobel_z = indirect / ((b_coef**2 * axp_se**2 + axp_coef**2 * b_se**2)**0.5 + 1e-10)
                sobel_p = 2*(1-stats.norm.cdf(abs(sobel_z)))

        if not np.isnan(axp_pv) and axp_pv < 0.1:
            sig_count += 1

        row = {
            'X': core_zn, '机制': mech_name, '代理变量': med_name,
            '代理变量含义': desc, '预期方向': exp_dir, '实际方向': actual_dir,
            '方向匹配': dir_match, 'M1系数': axp_coef, 'M1_pval': axp_pv,
            'M1_sig': stars, 'M1_N': n_obs,
            '基础交互项系数': base_axp if not np.isnan(base_axp) else None,
            'M2交互项系数': m2_axp, 'M2中介系数': m2_med, 'M2中介_pval': m2_med_pv,
            '系数缩小%': round(shrink, 1),
            'Sobel间接效应': indirect if not np.isnan(indirect) else None,
            'Sobel_Z': sobel_z if not np.isnan(sobel_z) else None,
            'Sobel_p': sobel_p if not np.isnan(sobel_p) else None,
        }
        all_summary.append(row)

        # Save regression details
        if res_m1 is not None:
            for v in x_list:
                if v in res_m1.params:
                    r = {'model': f'M1_{core_key}_{med_name}', 'var': v,
                         'coef': res_m1.params.get(v, np.nan),
                         'se': res_m1.std_errors.get(v, np.nan),
                         'pval': res_m1.pvalues.get(v, np.nan),
                         'N': n_obs, 'rsq': res_m1.rsquared_within}
                    all_regs.append(r)
        if res_m2 is not None:
            for v in x_list2:
                if v in res_m2.params:
                    r = {'model': f'M2_{core_key}_{med_name}', 'var': v,
                         'coef': res_m2.params.get(v, np.nan),
                         'se': res_m2.std_errors.get(v, np.nan),
                         'pval': res_m2.pvalues.get(v, np.nan),
                         'N': n_obs, 'rsq': res_m2.rsquared_within}
                    all_regs.append(r)

# ====================================================================
# 5. 输出
# ====================================================================
print(f"\n  总计检验组合: {total}")
print(f"  M1至少边际显著: {sig_count}")

# 只显示方向匹配 + 显著的
df = pd.DataFrame(all_summary)
sig = df[(df['方向匹配']=='✓') & (df['M1_pval'] < 0.1)]

print(f"\n{'='*70}")
print("结果：方向匹配 + M1显著")
print(f"{'='*70}")

for core_name in ['供应商广度','供应商行业多样性','竞争对手广度','竞争对手行业多样性']:
    sub = sig[sig['X']==core_name]
    if len(sub) == 0:
        print(f"\n  {core_name}: 无方向匹配的显著机制")
        continue
    print(f"\n  {core_name}:")
    for _, r in sub.iterrows():
        sob = f" Sobel={r['Sobel_p']:.4f}" if pd.notna(r['Sobel_p']) else ''
        m2s = f" 缩小{r['系数缩小%']:.0f}%" if pd.notna(r['系数缩小%']) and r['系数缩小%'] > 0 else ''
        print(f"    {r['机制']:16s} | {r['代理变量']:20s} | {r['代理变量含义']:20s} | "
              f"M1={r['M1系数']:+.4f}{r['M1_sig']} p={r['M1_pval']:.4f}{sob}{m2s}")

# 保存
if all_summary:
    pd.DataFrame(all_summary).to_excel(f'{OUT_DIR}/supplement2_summary.xlsx', index=False)
if all_regs:
    pd.DataFrame(all_regs).to_excel(f'{OUT_DIR}/supplement2_regressions.xlsx', index=False)

print(f"\n✓ {OUT_DIR}/")
print("完成!")
