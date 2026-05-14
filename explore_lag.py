"""
探索滞后项：基准回归使用滞后X
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

# 按企业分组，构建滞后项
qp = qp.sort_values(['ISIN', 'qi'])
for v in core_vars + controls:
    qp[f'{v}_lag1'] = qp.groupby('ISIN')[v].shift(1)   # 滞后1季度
    qp[f'{v}_lag2'] = qp.groupby('ISIN')[v].shift(2)   # 滞后2季度
    qp[f'{v}_lag4'] = qp.groupby('ISIN')[v].shift(4)   # 滞后4季度（1年）

print("=" * 70)
print(f"{'模型':20s} {'供应商广度':>16s} {'供应商多样性':>16s} {'竞争对手广度':>16s} {'竞争对手多样性':>16s}")
print("=" * 70)

models = [
    ('当期 (基准)', core_vars + controls, core_vars),
    ('滞后1季 (X_t-1)', [f'{v}_lag1' for v in core_vars + controls], [f'{v}_lag1' for v in core_vars]),
    ('滞后2季 (X_t-2)', [f'{v}_lag2' for v in core_vars + controls], [f'{v}_lag2' for v in core_vars]),
    ('滞后4季 (X_t-4)', [f'{v}_lag4' for v in core_vars + controls], [f'{v}_lag4' for v in core_vars]),
]

for name, x_list, display_vars in models:
    cols = ['roa_w_q'] + x_list + ['ISIN', 'quarter_dt']
    sub = qp[cols].dropna()
    if len(sub) < 1000:
        print(f"{name:20s}: 样本量不足 ({len(sub)})")
        continue
    sub_i = sub.set_index(['ISIN', 'quarter_dt'])

    # 只将X核心变量和controls传入公式
    formula = f'roa_w_q ~ EntityEffects + TimeEffects + ' + ' + '.join(x_list)

    try:
        mod = PanelOLS.from_formula(formula, data=sub_i, drop_absorbed=True)
        res = mod.fit(cov_type='clustered', cluster_entity=True)

        def stars(p):
            return '' if np.isnan(p) else ('***' if p<0.01 else ('**' if p<0.05 else ('*' if p<0.1 else '')))

        line = f"{name:20s} N={len(sub_i):>6,}"
        for dv in display_vars:
            coef = res.params.get(dv, np.nan)
            pval = res.pvalues.get(dv, np.nan)
            if not np.isnan(coef):
                line += f" {coef:+8.4f}{stars(pval):4s}"
            else:
                line += f" {'':>12s}"
        print(line)

    except Exception as e:
        print(f"{name:20s}: 失败 - {e}")

# 额外：滞后1季的控制变量+当期X的混合模型
print()
print("额外尝试：混合模型")
print("-" * 70)

mix_specs = [
    ('滞后1季X + 当期控制', core_vars + [f'{v}_lag1' for v in controls], core_vars),
    ('当期X + 滞后1季控制', [f'{v}_lag1' for v in core_vars] + controls, [f'{v}_lag1' for v in core_vars]),
]

for name, x_list, display_vars in mix_specs:
    cols = ['roa_w_q'] + x_list + ['ISIN', 'quarter_dt']
    sub = qp[cols].dropna()
    sub_i = sub.set_index(['ISIN', 'quarter_dt'])
    formula = f'roa_w_q ~ EntityEffects + TimeEffects + ' + ' + '.join(x_list)
    try:
        mod = PanelOLS.from_formula(formula, data=sub_i, drop_absorbed=True)
        res = mod.fit(cov_type='clustered', cluster_entity=True)
        def stars(p):
            return '' if np.isnan(p) else ('***' if p<0.01 else ('**' if p<0.05 else ('*' if p<0.1 else '')))
        line = f"{name:25s} N={len(sub_i):>6,}"
        for dv in display_vars:
            coef = res.params.get(dv, np.nan)
            pval = res.pvalues.get(dv, np.nan)
            if not np.isnan(coef):
                line += f" {coef:+8.4f}{stars(pval):4s}"
            else:
                line += f" {'':>12s}"
        print(line)
    except Exception as e:
        print(f"{name:25s}: 失败 - {e}")
