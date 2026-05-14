"""
系统化机制检验：四组交互项 × 多机制 × 多代理变量 × 多检验方法

数据结构：
  X（4个核心变量）→ 机制（2-4个）→ 代理变量（2-4个）→ 检验方法（2-4种）

输出：一张汇总表，标记每个组合的显著性
"""
import pandas as pd, numpy as np, os, warnings, re
warnings.filterwarnings('ignore')
from linearmodels.panel import PanelOLS
from scipy import stats

DATA_DIR = '/Users/deesanyu/pythonproject/deesan_supply_0506v4/data'
OUT_DIR  = '/Users/deesanyu/pythonproject/deesan_supply_0506v4/output/mechanism_comprehensive'
QP_PATH  = '/Users/deesanyu/pythonproject/deesan_supply_0506v4/output/quarterly_panel/firm_quarterly_panel.parquet'
os.makedirs(OUT_DIR, exist_ok=True)

print("=" * 70)
print("系统化机制检验")
print("=" * 70)

# ====================================================================
# 1. 加载数据
# ====================================================================
print("\n1. 加载数据...")
qp = pd.read_parquet(QP_PATH)
qp['quarter_dt'] = pd.to_datetime(qp['quarter'])
print(f"  基础面板: {len(qp):,} 行, {qp['ISIN'].nunique():,} 企业")

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

# 加载全部11个文件
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
# 2. 计算所有代理变量（原始 + 组合 + 缩尾）
# ====================================================================
print("\n2. 计算代理变量...")

def winsor(s, l=0.01, u=0.99):
    lo, hi = s.quantile(l), s.quantile(u)
    return s.clip(lo, hi)

# --- 原始变量缩尾 ---
raw_vars = ['cogs_r','sga_r','sga_e','ebit_m','inv','inv_t','at_t','ap_t','days_ap','ap_r','recv_t']
for v in raw_vars:
    if v in qp.columns:
        qp[f'{v}_w'] = winsor(qp[v].dropna())

# --- 组合变量 ---
# 总营业成本率 = (COGS + SG&A) / Sales
qp['tot_op_cost'] = qp['cogs_r'] + qp['sga_r']
qp['tot_op_cost_w'] = winsor(qp['tot_op_cost'].dropna())

# 毛利率 = 1 - 成本率
qp['gross_m'] = 100 - qp['cogs_r']
qp['gross_m_w'] = winsor(qp['gross_m'].dropna())

# SG&A / COGS 比值（费用结构）
qp['sga_cogs_r'] = qp['sga_r'] / qp['cogs_r'].replace(0, np.nan)
qp['sga_cogs_r_w'] = winsor(qp['sga_cogs_r'].dropna())

# 天数转换
qp['days_inv'] = 365 / qp['inv_t'].replace(0, np.nan)
qp['days_recv'] = 365 / qp['recv_t'].replace(0, np.nan)
for v in ['days_inv','days_recv']:
    qp[f'{v}_w'] = winsor(qp[v].dropna())

# 现金转换周期
qp['ccc'] = qp['days_inv'] + qp['days_recv'] - qp['days_ap']
qp['ccc_w'] = winsor(qp['ccc'].dropna())

# 营业周期
qp['op_cycle'] = qp['days_inv'] + qp['days_recv']
qp['op_cycle_w'] = winsor(qp['op_cycle'].dropna())

# DuPont 分解净利润率
qp['proxy_nm'] = qp['roa_w_q'] / qp['at_t'].replace(0, np.nan)
qp['proxy_nm_w'] = winsor(qp['proxy_nm'].dropna())

# 收入波动率 (滚动4期标准差)
qp['growth_vol'] = qp.groupby('ISIN')['growth_w_q'].transform(lambda x: x.rolling(4).std())
qp['growth_vol_w'] = winsor(qp['growth_vol'].dropna())

# 存货密集度 = inventories / assets
qp['inv_intensity'] = qp['inv'] / qp['assets_q'].replace(0, np.nan)
qp['inv_intensity_w'] = winsor(qp['inv_intensity'].dropna())

# AP / COGS (应付账款相对成本的比例)
qp['ap_cogs_r'] = qp['ap_r'] / (qp['cogs_r']/100).replace(0, np.nan)
qp['ap_cogs_r_w'] = winsor(qp['ap_cogs_r'].dropna())

# 所有机制变量列表（缩尾后）
all_mech_vars = [c for c in qp.columns if c.endswith('_w') and c not in
                 ['roa_w_q','roe_w_q','growth_w_q','size_q','lev_q','sup_breadth_q',
                  'comp_breadth_q','sup_ind_div_q','comp_ind_div_q']]
print(f"  共计 {len(all_mech_vars)} 个机制代理变量")
for v in all_mech_vars:
    n = qp[v].notna().sum()
    print(f"    {v:20s}: N={n:>8,}, mean={qp[v].mean():.2f}")

# ====================================================================
# 3. 回归框架
# ====================================================================
var_map = {
    'ROA':'roa_w_q',
    'sup':'sup_breadth_q', 'sup_ind':'sup_ind_div_q',
    'comp':'comp_breadth_q', 'comp_ind':'comp_ind_div_q',
    'Size':'size_q','Lev':'lev_q','Growth':'growth_w_q',
}
# 所有机制变量都加入var_map（自映射）
for v in all_mech_vars:
    var_map[v] = v

controls = ['Size','Lev','Growth']
core_map = {
    'sup': ('sup_breadth_q', 'supplier_count_log'),
    'sup_ind': ('sup_ind_div_q', 'supplier_industry_count'),
    'comp': ('comp_breadth_q', 'competitor_count_log'),
    'comp_ind': ('comp_ind_div_q', 'competitor_industry_count'),
}

def run_ols(y_name, x_list, data, label=''):
    y_name = var_map.get(y_name, y_name)  # resolve ROA → roa_w_q
    available = [c for c in x_list if c in data.columns]
    sub = data[[y_name]+available].dropna()
    if len(sub) < 30: return None
    formula = f'{y_name} ~ EntityEffects + TimeEffects + ' + ' + '.join(available)
    try:
        mod = PanelOLS.from_formula(formula, data=sub, drop_absorbed=True)
        res = mod.fit(cov_type='clustered', cluster_entity=True)
        rows = []
        for v in available:
            coef = res.params.get(v, np.nan); se = res.std_errors.get(v, np.nan)
            pv = res.pvalues.get(v, np.nan); t = res.tstats.get(v, np.nan)
            star = '' if np.isnan(pv) else ('***' if pv<0.01 else ('**' if pv<0.05 else ('*' if pv<0.1 else '')))
            rows.append({'model':label,'var':v,'coef':round(coef,5),'se':round(se,5),
                        't':round(t,3),'pval':round(pv,4),'sig':star,'N':len(sub)})
        return pd.DataFrame(rows)
    except Exception as e:
        return None

# 面板索引
panel = qp.sort_values(['ISIN','qi']).set_index(['ISIN','quarter_dt']).copy()
# 创建交互项
for core_col, _ in core_map.values():
    panel[f'{core_col}_x_post'] = panel[core_col] * panel['post2018']

all_results = []

# ====================================================================
# 4. 系统化检验
# ====================================================================
print("\n" + "=" * 70)
print("4. 系统化机制检验")
print("=" * 70)

# ---- 定义完整的检验矩阵 ----
# 格式: (核心变量key, 核心变量中文, 机制名, 代理变量列表, 预期方向列表, 机制描述)
test_matrix = [
    # ============================================================
    # ① 供应商广度 × post → ROA 负向
    # ============================================================
    ('sup', '供应商广度', 'A:成本压力',
     ['cogs_r_w','tot_op_cost_w','sga_cogs_r_w','sga_r_w'],
     ['+','+','+','+'],
     '供应商多 → 管理复杂 → 直接/间接成本上升'),

    ('sup', '供应商广度', 'B:库存积压',
     ['inv_t_w','inv_intensity_w','days_inv_w'],
     ['-','+','+'],
     '供应商多 → 贸易战不确定性 → 过量备货 → 成本上升'),

    ('sup', '供应商广度', 'C:融资压力',
     ['ap_r_w','days_ap_w','ap_cogs_r_w'],
     ['+','-','+'],
     '贸易战后供应商收紧账期 → 应付账款压力上升'),

    ('sup', '供应商广度', 'D:回款困难',
     ['recv_t_w','days_recv_w'],
     ['-','+'],
     '贸易战下游客户付款延迟 → 资金效率下降'),

    ('sup', '供应商广度', 'E:运营效率',
     ['at_t_w','ccc_w','op_cycle_w'],
     ['-','+','+'],
     '供应链冲击 → 资产周转效率全面下降'),

    ('sup', '供应商广度', 'F:利润率侵蚀',
     ['ebit_m_w','gross_m_w','proxy_nm_w'],
     ['-','-','-'],
     '上述所有渠道 → 利润率下降 → ROA下降'),

    # ============================================================
    # ② 供应商行业多样性 × post → ROA 正向
    # ============================================================
    ('sup_ind', '供应商行业多样性', 'A:收入稳定',
     ['growth_vol_w'],
     ['-'],
     '供应商行业多样 → 可替代来源 → 收入稳定性上升'),

    ('sup_ind', '供应商行业多样性', 'B:成本优化',
     ['cogs_r_w','tot_op_cost_w','ebit_m_w'],
     ['-','-','+'],
     '跨行业供应商 → 议价能力 → 成本优化/利润上升'),

    ('sup_ind', '供应商行业多样性', 'C:运营效率',
     ['at_t_w','inv_t_w','ccc_w'],
     ['+','+','-'],
     '供应选择多样 → 资产配置优化 → 效率上升'),

    # ============================================================
    # ③ 竞争对手广度 × post → ROA 正向
    # ============================================================
    ('comp', '竞争对手广度', 'A:费用削减',
     ['sga_r_w','sga_cogs_r_w','sga_e_w'],
     ['-','-','-'],
     '竞争激烈 → 贸易战后被迫削减管理费用'),

    ('comp', '竞争对手广度', 'B:成本优化',
     ['cogs_r_w','tot_op_cost_w'],
     ['-','-'],
     '竞争压力 → 优化直接成本 → 成本率下降'),

    ('comp', '竞争对手广度', 'C:资产效率',
     ['at_t_w','inv_t_w','op_cycle_w'],
     ['+','+','-'],
     '竞争 → 盘活资产 → 周转效率提升'),

    ('comp', '竞争对手广度', 'D:利润率修复',
     ['ebit_m_w','gross_m_w','proxy_nm_w'],
     ['+','+','+'],
     '降本增效 → 利润率回升'),

    # ============================================================
    # ④ 竞争对手行业多样性 × post → ROA 负向
    # ============================================================
    ('comp_ind', '竞争对手行业多样性', 'A:运营复杂度',
     ['at_t_w','inv_t_w','ccc_w'],
     ['-','-','+'],
     '跨行业竞争 → 管理力分散 → 运营效率下降'),

    ('comp_ind', '竞争对手行业多样性', 'B:成本刚性',
     ['cogs_r_w','sga_r_w','tot_op_cost_w'],
     ['+','+','+'],
     '业务面广 → 成本结构调整慢 → 成本率上升'),

    ('comp_ind', '竞争对手行业多样性', 'C:利润率侵蚀',
     ['ebit_m_w','gross_m_w','proxy_nm_w'],
     ['-','-','-'],
     '效率下降+成本上升 → 利润率下降'),
]

# ---- 执行检验 ----
summary_rows = []
total_tests = 0
significant_tests = 0

for core_key, core_zh, mech_name, proxy_list, expected, desc in test_matrix:
    core_col, core_disp = core_map[core_key]

    for med_name, exp_dir in zip(proxy_list, expected):
        if med_name not in panel.columns:
            continue
        total_tests += 1

        # ===== 方法1: 中介效应检验 =====
        # Step A: M ~ X + X×post + controls
        x_list = [core_col, f'{core_col}_x_post'] + controls
        x_real = [var_map.get(x,x) for x in x_list]
        r_a = run_ols(med_name, x_real, panel, f'M1_{core_key}_{med_name}')

        result = {
            'X': core_zh, '机制': mech_name, '代理变量': med_name,
            '预期方向': exp_dir, '描述': desc,
        }

        if r_a is not None:
            row_a = r_a[r_a['var']==f'{core_col}_x_post']
            if len(row_a):
                rw = row_a.iloc[0]
                d = '+' if rw['coef']>0 else '-'
                match = '✓' if d==exp_dir else '✗'
                result['M1方向'] = d
                result['M1系数'] = rw['coef']
                result['M1_pval'] = rw['pval']
                result['M1_sig'] = rw['sig']
                result['M1匹配'] = match
                result['M1_N'] = rw['N']
                all_results.append(r_a)

                if rw['pval'] < 0.1:
                    significant_tests += 1

                # ===== 方法2: 加入中介变量后交互项变化 =====
                x_list2 = [core_disp, f'{core_col}_x_post', med_name] + controls
                x_real2 = [var_map.get(x,x) for x in x_list2]
                r_b = run_ols('ROA', x_real2, panel, f'M2_{core_key}_{med_name}')
                if r_b is not None:
                    row_b_main = r_b[r_b['var']==f'{core_col}_x_post']
                    row_b_med = r_b[r_b['var']==med_name]
                    if len(row_b_main):
                        result['M2_交互项系数'] = row_b_main.iloc[0]['coef']
                        result['M2_交互项pval'] = row_b_main.iloc[0]['pval']
                    if len(row_b_med):
                        result['M2_中介系数'] = row_b_med.iloc[0]['coef']
                        result['M2_中介pval'] = row_b_med.iloc[0]['pval']
                    all_results.append(r_b)

        # ===== 方法3: Sobel检验 =====
        if r_a is not None and len(r_a[r_a['var']==f'{core_col}_x_post']) > 0:
            a_coef = r_a[r_a['var']==f'{core_col}_x_post'].iloc[0]['coef']
            a_se = r_a[r_a['var']==f'{core_col}_x_post'].iloc[0]['se']

            # Sobel也需要b-path，从方法2中取
            if 'M2_中介系数' in result and result['M2_中介系数'] is not None:
                b_coef = result['M2_中介系数']
                for df_r in all_results:
                    if isinstance(df_r, pd.DataFrame):
                        if f'M2_{core_key}_{med_name}' in df_r['model'].values:
                            b_row = df_r[(df_r['model']==f'M2_{core_key}_{med_name}') & (df_r['var']==med_name)]
                            if len(b_row):
                                b_se = b_row.iloc[0]['se']
                                indirect = a_coef * b_coef
                                sobel_z = indirect / ((b_coef**2 * a_se**2 + a_coef**2 * b_se**2)**0.5 + 1e-10)
                                sobel_p = 2*(1-stats.norm.cdf(abs(sobel_z)))
                                result['Sobel间接效应'] = indirect
                                result['Sobel_Z'] = sobel_z
                                result['Sobel_p'] = sobel_p
                                if sobel_p < 0.1:
                                    significant_tests += 1

        summary_rows.append(result)

# ====================================================================
# 5. 汇总表
# ====================================================================
print("\n\n" + "=" * 70)
print("5. 结果汇总")
print("=" * 70)

df_summary = pd.DataFrame(summary_rows)

# 按核心变量分组输出
for core_name in ['供应商广度','供应商行业多样性','竞争对手广度','竞争对手行业多样性']:
    sub = df_summary[df_summary['X']==core_name]
    print(f"\n{'='*60}")
    print(f"  {core_name} × post")
    print(f"{'='*60}")
    print(f"  {'机制':20s} {'代理变量':20s} {'预期':4s} {'实际':4s} {'M1_pval':>8s} {'M2交互项':>10s} {'Sobel_p':>8s}")
    print(f"  {'-'*74}")
    for _, r in sub.iterrows():
        m1_s = f"{r.get('M1_pval','-'):.4f}{r.get('M1_sig','')}" if pd.notna(r.get('M1_pval',None)) else '-'
        m1_d = r.get('M1方向','-')
        exp_d = r.get('预期方向','-')
        m2_c = f"{r.get('M2_交互項系数',0):+.3f}" if pd.notna(r.get('M2_交互项系数',None)) else '-'
        sob_p = f"{r.get('Sobel_p',1):.4f}" if pd.notna(r.get('Sobel_p',None)) and r.get('Sobel_p',1)<1 else '-'
        print(f"  {r['机制']:20s} {r['代理变量']:20s} {exp_d:4s} {m1_d:4s} {m1_s:>8s} {m2_c:>10s} {sob_p:>8s}")

# 保存全量结果
if all_results:
    pd.concat(all_results, ignore_index=True).to_excel(f'{OUT_DIR}/all_regression_results.xlsx', index=False)
df_summary.to_excel(f'{OUT_DIR}/mechanism_summary.xlsx', index=False)
print(f"\n  ✓ {OUT_DIR}/")
print(f"\n  总计检验组合: {total_tests}")
print(f"  其中M1方法至少边际显著的: {significant_tests}")
print(f"\n完成!")
