"""
探索非线性项：基准回归加入平方项
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

# 构建平方项（去均值后平方，减少共线性）
for v in core_vars:
    qp[f'{v}_sq'] = qp[v] ** 2  # 直接平方，不用去均值（方便解释）

# ── 方式1：逐个加平方项 ──
print("=" * 60)
print("方式1：逐个加入平方项")
print("每个模型 = 线性项 + 该变量的平方项 + 其他三个线性项 + 控制变量")
print("=" * 60)

for v in core_vars:
    cols = ['roa_w_q'] + core_vars + [f'{v}_sq'] + controls + ['ISIN', 'quarter_dt']
    sub = qp[cols].dropna()
    sub_i = sub.set_index(['ISIN', 'quarter_dt'])

    x_vars = core_vars + [f'{v}_sq'] + controls
    formula = 'roa_w_q ~ EntityEffects + TimeEffects + ' + ' + '.join(x_vars)

    try:
        mod = PanelOLS.from_formula(formula, data=sub_i, drop_absorbed=True)
        res = mod.fit(cov_type='clustered', cluster_entity=True)

        coef_lin = res.params.get(v, np.nan)
        p_lin = res.pvalues.get(v, np.nan)
        coef_sq = res.params.get(f'{v}_sq', np.nan)
        p_sq = res.pvalues.get(f'{v}_sq', np.nan)

        def stars(p):
            return '' if np.isnan(p) else ('***' if p<0.01 else ('**' if p<0.05 else ('*' if p<0.1 else '')))

        # 判断形状
        if not np.isnan(coef_lin) and not np.isnan(coef_sq):
            if coef_lin > 0 and coef_sq < 0:
                shape = '倒U型 (先正后负)'
            elif coef_lin < 0 and coef_sq > 0:
                shape = 'U型 (先负后正)'
            else:
                shape = '单调'
        else:
            shape = '-'

        print(f"\n{v}:")
        print(f"  线性项: {coef_lin:+.4f}{stars(p_lin)} (p={p_lin:.4f})")
        print(f"  平方项: {coef_sq:+.6f}{stars(p_sq)} (p={p_sq:.4f})")
        print(f"  形状: {shape}")
        print(f"  N={len(sub_i):,}")
    except Exception as e:
        print(f"\n{v}: 失败 - {e}")

# ── 方式2：全部平方项一起放 ──
print(f"\n\n{'='*60}")
print("方式2：四个平方项同时放入")
print("=" * 60)

sq_vars = [f'{v}_sq' for v in core_vars]
cols = ['roa_w_q'] + core_vars + sq_vars + controls + ['ISIN', 'quarter_dt']
sub = qp[cols].dropna()
sub_i = sub.set_index(['ISIN', 'quarter_dt'])
x_vars = core_vars + sq_vars + controls
formula = 'roa_w_q ~ EntityEffects + TimeEffects + ' + ' + '.join(x_vars)

try:
    mod = PanelOLS.from_formula(formula, data=sub_i, drop_absorbed=True)
    res = mod.fit(cov_type='clustered', cluster_entity=True)
    print(res)
    print(f"\nN={len(sub_i):,}")
except Exception as e:
    print(f"失败: {e}")
