"""
分样本基准回归
独立脚本
"""
import pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')
from linearmodels.panel import PanelOLS

QP_PATH = '/Users/deesanyu/pythonproject/deesan_supply_0506v4/output/quarterly_panel/firm_quarterly_panel.parquet'
qp = pd.read_parquet(QP_PATH)
qp['quarter_dt'] = pd.to_datetime(qp['quarter'])

core_vars = ['sup_breadth_q', 'sup_ind_div_q', 'comp_breadth_q', 'comp_ind_div_q']
controls = ['size_q', 'lev_q', 'growth_w_q']

# ── 分组定义 ──
splits = {
    '全样本': pd.Series(True, index=qp.index),
    # 按规模：大企业供应链更复杂
    '大企业': qp['size_q'] > qp['size_q'].median(),
    '小企业': qp['size_q'] <= qp['size_q'].median(),
    # 按杠杆：高杠杆企业财务约束更强
    '高杠杆': qp['lev_q'] > qp['lev_q'].median(),
    '低杠杆': qp['lev_q'] <= qp['lev_q'].median(),
    # 按成长性：高增长企业更需要供应链弹性
    '高增长': qp['growth_w_q'] > qp['growth_w_q'].median(),
    '低增长': qp['growth_w_q'] <= qp['growth_w_q'].median(),
}

# 额外：按照是否有海外供应商/竞争对手分
sup_foreign = [c for c in qp.columns if c.startswith('sup_ct_') and c != 'sup_ct_中国']
comp_foreign = [c for c in qp.columns if c.startswith('comp_ct_') and c != 'comp_ct_中国']
if sup_foreign:
    qp['has_foreign_sup'] = qp[sup_foreign].sum(axis=1) > 0
    splits['有海外供应商'] = qp['has_foreign_sup'] == True
    splits['无海外供应商'] = qp['has_foreign_sup'] == False
if comp_foreign:
    qp['has_foreign_comp'] = qp[comp_foreign].sum(axis=1) > 0
    splits['有海外竞争对手'] = qp['has_foreign_comp'] == True
    splits['无海外竞争对手'] = qp['has_foreign_comp'] == False

# ── 运行分样本回归 ──
print(f"{'分组':20s} {'N':>8s} {'供应商广度':>16s} {'供应商多样性':>16s} {'竞争对手广度':>16s} {'竞争对手多样性':>16s}")
print("=" * 80)

results = []
for name in ['全样本', '大企业', '小企业', '高杠杆', '低杠杆', '高增长', '低增长',
             '有海外供应商', '无海外供应商', '有海外竞争对手', '无海外竞争对手']:
    if name not in splits:
        continue
    mask = splits[name]
    sub = qp[mask].copy()

    cols = ['roa_w_q'] + core_vars + controls + ['ISIN', 'quarter_dt']
    sub = sub[cols].dropna()
    if len(sub) < 1000:
        print(f"{name:20s} {'N<1000':>8s}")
        continue

    sub_i = sub.set_index(['ISIN', 'quarter_dt'])
    formula = 'roa_w_q ~ EntityEffects + TimeEffects + ' + ' + '.join(core_vars + controls)

    try:
        mod = PanelOLS.from_formula(formula, data=sub_i, drop_absorbed=True)
        res = mod.fit(cov_type='clustered', cluster_entity=True)
    except Exception as e:
        print(f"{name:20s}: 失败 - {e}")
        continue

    n_f = sub_i.index.get_level_values(0).nunique()
    def stars(p):
        return '' if np.isnan(p) else ('***' if p<0.01 else ('**' if p<0.05 else ('*' if p<0.1 else '')))

    line = f"{name:20s} N={len(sub_i):>6,}"
    for v in core_vars:
        c = res.params.get(v, np.nan)
        p = res.pvalues.get(v, np.nan)
        if not np.isnan(c):
            line += f" {c:+8.4f}{stars(p):4s}"
        else:
            line += f" {'':>12s}"

    print(line)
    results.append({
        '分组': name, 'N': len(sub_i), '企业数': n_f,
        **{v: f"{res.params.get(v, np.nan):+.4f}{stars(res.pvalues.get(v, np.nan))}"
           if not np.isnan(res.params.get(v, np.nan)) else '-'
           for v in core_vars}
    })

# 保存
pd.DataFrame(results).to_excel(
    '/Users/deesanyu/pythonproject/deesan_supply_0506v4/output/mechanism_final/split_sample_baseline.xlsx',
    index=False
)
print(f"\n结果已保存")
