"""
探索替代绩效指标的基准回归结果
独立脚本，不影响任何现有文件
"""
import pandas as pd, numpy as np, os, warnings, re
warnings.filterwarnings('ignore')
from linearmodels.panel import PanelOLS

DATA_DIR = '/Users/deesanyu/pythonproject/deesan_supply_0506v4/data'
QP_PATH  = '/Users/deesanyu/pythonproject/deesan_supply_0506v4/output/quarterly_panel/firm_quarterly_panel.parquet'

# ================================================================
# 1. 加载数据（复用build_mechanism_final.py的加载逻辑）
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

# ================================================================
# 2. 构建替代Y变量 + 缩尾
# ================================================================
def w(s):
    lo, hi = s.quantile(0.01), s.quantile(0.99)
    return s.clip(lo, hi)

# 从新数据构建的绩效指标
qp['gross_m'] = 100 - qp['cogs_r']  # 毛利率（%）
qp['gross_m_w'] = w(qp['gross_m'].dropna())

qp['ebit_m_w'] = w(qp['ebit_m'].dropna())  # EBIT利润率

qp['proxy_nm'] = qp['roa_w_q'] / qp['at_t'].replace(0, np.nan)  # 净利润率
qp['proxy_nm_w'] = w(qp['proxy_nm'].dropna())

qp['at_t_w'] = w(qp['at_t'].dropna())  # 资产周转率

qp['cogs_r_w'] = w(qp['cogs_r'].dropna())  # 营业成本率

qp['sga_r_w'] = w(qp['sga_r'].dropna())  # SG&A/销售

# 从面板原生构建的指标
qp['sales_ln'] = np.log(qp['sales_q'].clip(lower=1))
qp['sales_ln_w'] = w(qp['sales_ln'].dropna())

qp['asset_growth'] = qp.groupby('ISIN')['assets_q'].pct_change()
qp['asset_growth_w'] = w(qp['asset_growth'].dropna())

# 经营现金流/总资产（近似指标）
# 我们没有直接的CFO数据，但可以用面板中的构建
# 季度面板没有CFO，跳过

print("数据准备完成")
print(f"观测数: {len(qp):,}")

# ================================================================
# 3. 运行基准回归（四种Y）
# ================================================================
core_vars = ['sup_breadth_q', 'sup_ind_div_q', 'comp_breadth_q', 'comp_ind_div_q']
controls = ['size_q', 'lev_q', 'growth_w_q']

# 定义要测试的Y变量
y_candidates = {
    'ROA (基准)': 'roa_w_q',
    'ROE': 'roe_w_q',
    '毛利率 (Gross Margin)': 'gross_m_w',
    'EBIT利润率': 'ebit_m_w',
    '净利润率 (Proxy)': 'proxy_nm_w',
    '资产周转率': 'at_t_w',
    '营业成本率': 'cogs_r_w',
    'SG&A/销售': 'sga_r_w',
    '销售规模 (ln)': 'sales_ln_w',
    '资产增长率': 'asset_growth_w',
}

results = []
for y_name, y_col in y_candidates.items():
    cols = [y_col] + core_vars + controls + ['ISIN', 'quarter_dt']
    sub = qp[cols].dropna()
    if len(sub) < 1000:
        print(f"  {y_name:25s}: 样本量不足 ({len(sub)})")
        continue

    sub_i = sub.set_index(['ISIN', 'quarter_dt'])
    formula = f'{y_col} ~ EntityEffects + TimeEffects + ' + ' + '.join(core_vars + controls)

    try:
        mod = PanelOLS.from_formula(formula, data=sub_i, drop_absorbed=True)
        res = mod.fit(cov_type='clustered', cluster_entity=True)
    except Exception as e:
        print(f"  {y_name:25s}: 回归失败 - {e}")
        continue

    n_firms = sub_i.index.get_level_values(0).nunique()
    n_obs = len(sub_i)

    row = {'Y变量': y_name, 'N': n_obs, '企业数': n_firms}
    for x in core_vars:
        coef = res.params.get(x, np.nan)
        pval = res.pvalues.get(x, np.nan)
        sig = '' if np.isnan(pval) else (
            '***' if pval < 0.01 else ('**' if pval < 0.05 else ('*' if pval < 0.1 else ''))
        )
        row[f'{x}_coef'] = coef
        row[f'{x}_pval'] = pval
        row[f'{x}_sig'] = f'{coef:+.4f}{sig}' if not np.isnan(coef) else '-'

    results.append(row)
    print(f"  {y_name:25s}: N={n_obs:>6,} ", end="")
    for x in core_vars:
        print(f" {x[:12]}={row[f'{x}_sig']:>12s}", end="")
    print()

# ================================================================
# 4. 输出汇总表
# ================================================================
print(f"\n\n{'='*90}")
print(f"{'替代绩效指标 — 基准回归汇总'}")
print(f"{'='*90}")
print(f"{'Y变量':25s} {'N':>8s} {'供应商广度':>16s} {'供应商多样性':>16s} {'竞争对手广度':>16s} {'竞争对手多样性':>16s}")
print(f"{'-'*90}")
for r in results:
    print(f"{r['Y变量']:25s} {r['N']:>8,} {r['sup_breadth_q_sig']:>16s} {r['sup_ind_div_q_sig']:>16s} {r['comp_breadth_q_sig']:>16s} {r['comp_ind_div_q_sig']:>16s}")

# 保存
out_df = pd.DataFrame(results)
out_path = '/Users/deesanyu/pythonproject/deesan_supply_0506v4/output/mechanism_final/alternative_y_baseline.xlsx'
os.makedirs(os.path.dirname(out_path), exist_ok=True)
out_df.to_excel(out_path, index=False)
print(f"\n结果已保存: {out_path}")
