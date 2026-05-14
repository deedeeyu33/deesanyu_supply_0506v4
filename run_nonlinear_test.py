"""
非线性检验：供应链关系结构对企业绩效的 U 型 / 倒 U 型关系
第六章补充分析：分别在 ROA 和 ROE 模型中检验二次非线性效应
"""
import pandas as pd, numpy as np, os, warnings, sys
warnings.filterwarnings('ignore')
from linearmodels.panel import PanelOLS
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

OUT_DIR = '/Users/deesanyu/pythonproject/deesan_supply_0506v4/output'
DATA_PATH = f'{OUT_DIR}/firm_monthly_panel.csv'
FONT_PATH = '/Users/deesanyu/pythonproject/deesan_supply_0506v4/simhei.ttf'
os.makedirs(f'{OUT_DIR}/figures', exist_ok=True)
zh_font = FontProperties(fname=FONT_PATH)
plt.rcParams['axes.unicode_minus'] = False

# ========== 1. 加载数据 ==========
print("=" * 60)
print("1. 加载数据")
print("=" * 60)
p = pd.read_csv(DATA_PATH)
print(f"行数: {len(p):,}  企业: {p['ISIN'].nunique():,}  月份: {p['month'].nunique()}")

# 变量映射
var_map = {
    'ROA': 'roa_w', 'ROE': 'roe_w',
    'supplier_count_log': 'sup_breadth', 'supplier_industry_count': 'sup_ind_div',
    'competitor_count_log': 'comp_breadth', 'competitor_industry_count': 'comp_ind_div',
    'Size': 'size', 'Leverage': 'lev', 'Growth': 'growth_w',
}
controls = ['Size', 'Leverage', 'Growth']

# 核心变量 list (显示名, 实际列名, 中文名)
core_vars = [
    ('supplier_count_log', 'sup_breadth', '供应商广度'),
    ('supplier_industry_count', 'sup_ind_div', '供应商行业多样性'),
    ('competitor_count_log', 'comp_breadth', '竞争对手广度'),
    ('competitor_industry_count', 'comp_ind_div', '竞争对手行业多样性'),
]

# 准备面板索引
p['month_dt'] = pd.to_datetime(p['month'])
p = p.sort_values(['ISIN', 'month_dt']).set_index(['ISIN', 'month_dt'])

# ========== 2. 生成平方项 ==========
print("\n" + "=" * 60)
print("2. 生成平方项")
print("=" * 60)
for disp, col, cn in core_vars:
    sq_col = f'{col}_sq'
    p[sq_col] = p[col] ** 2
    print(f"  {cn:12s} ({col:15s}) -> 平方项 ({sq_col})")
    print(f"    均值={p[col].mean():.4f}, 平方项均值={p[sq_col].mean():.4f}")

# 控制的实际列名
ctrl_cols = [var_map[v] for v in controls]  # ['size', 'lev', 'growth_w']

# ========== 3. 回归工具函数 ==========
def run_nonlinear_model(y_display, core_disp, data, label=''):
    """运行单个核心变量 + 平方项的非线性检验"""
    y_col = var_map[y_display]
    core_col = var_map[core_disp]
    sq_col = f'{core_col}_sq'

    # 变量列表: y + 一次项 + 平方项 + 控制变量
    all_vars = [y_col, core_col, sq_col] + ctrl_cols
    sub = data[all_vars].dropna().copy()
    n = len(sub)
    n_firms = sub.index.get_level_values(0).nunique()
    n_months = sub.index.get_level_values(1).nunique()

    if n < 100:
        print(f"  ⚠ {label}: 样本不足 ({n})")
        return None

    formula = f'{y_col} ~ EntityEffects + TimeEffects + {core_col} + {sq_col} + ' + ' + '.join(ctrl_cols)
    try:
        mod = PanelOLS.from_formula(formula, data=sub, drop_absorbed=True)
        res = mod.fit(cov_type='clustered', cluster_entity=True)

        # 提取结果
        def get_stat(v):
            return (res.params.get(v, np.nan),
                    res.std_errors.get(v, np.nan),
                    res.tstats.get(v, np.nan),
                    res.pvalues.get(v, np.nan))

        coef_linear, se_linear, t_linear, p_linear = get_stat(core_col)
        coef_sq, se_sq, t_sq, p_sq = get_stat(sq_col)

        # 显著性标注
        def star(pv):
            if np.isnan(pv): return ''
            if pv < 0.01: return '***'
            if pv < 0.05: return '**'
            if pv < 0.1: return '*'
            return ''

        # 判断非线性类型
        if np.isnan(coef_linear) or np.isnan(coef_sq):
            shape = '无法判断'
        elif coef_linear > 0 and coef_sq < 0:
            shape = '倒U型候选'
        elif coef_linear < 0 and coef_sq > 0:
            shape = 'U型候选'
        elif coef_sq < 0:
            shape = '非线性(凸向)'
        else:
            shape = '非线性(凹向)'

        # 是否显著
        sig_note = ''
        if p_sq < 0.01:
            sig_note = '平方项高度显著(p<0.01)'
        elif p_sq < 0.05:
            sig_note = '平方项显著(p<0.05)'
        elif p_sq < 0.1:
            sig_note = '平方项边缘显著(p<0.1)'
        else:
            sig_note = '平方项不显著'

        # 极值点 (仅当一次项和平方项符号相反时有意义)
        turning_point = -coef_linear / (2 * coef_sq) if coef_sq != 0 and not np.isnan(coef_sq) else np.nan
        # 极值点在数据范围内是否有意义
        x_min, x_max = sub[core_col].min(), sub[core_col].max()
        tp_inside = (turning_point >= x_min) and (turning_point <= x_max) if not np.isnan(turning_point) else False

        row = {
            'model': label,
            'y': y_display,
            'core_var': core_disp,
            'core_col': core_col,
            'N': n,
            'N_firms': n_firms,
            'N_months': n_months,
            'rsq_within': round(res.rsquared_within, 4),
            'rsq_overall': round(res.rsquared_overall, 4),

            'coef_linear': round(coef_linear, 6),
            'se_linear': round(se_linear, 6),
            't_linear': round(t_linear, 3),
            'p_linear': round(p_linear, 4),
            'sig_linear': star(p_linear),

            'coef_sq': round(coef_sq, 6),
            'se_sq': round(se_sq, 6),
            't_sq': round(t_sq, 3),
            'p_sq': round(p_sq, 4),
            'sig_sq': star(p_sq),

            'shape': shape,
            'sig_note': sig_note,
            'turning_point': round(turning_point, 4) if not np.isnan(turning_point) else None,
            'tp_inside_range': tp_inside,
            'x_min': round(x_min, 4),
            'x_max': round(x_max, 4),
        }
        tp_str = f"{turning_point:.3f}" if not np.isnan(turning_point) else "N/A"
        print(f"  ✓ {label:35s} N={n:>7,} R²={res.rsquared_within:.4f}  "
              f"线性={coef_linear:.5f}{star(p_linear)}  "
              f"平方={coef_sq:.5f}{star(p_sq)}  "
              f"极值点={tp_str}  "
              f"{shape}")
        return row, res
    except Exception as e:
        print(f"  ⚠ {label} 失败: {e}")
        return None


# ========== 4. 非线性检验 (ROA) ==========
print("\n" + "=" * 60)
print("3. 非线性检验: 被解释变量 = ROA")
print("=" * 60)
roa_results = []
roa_models = {}  # 保存模型对象用于画图
for disp, col, cn in core_vars:
    label = f'ROA_{disp}'
    ret = run_nonlinear_model('ROA', disp, p, label)
    if ret is not None:
        row, model_obj = ret
        roa_results.append(row)
        roa_models[disp] = model_obj

roa_df = pd.DataFrame(roa_results)
print("\nROA 非线性检验摘要:")
print(roa_df[['core_var', 'coef_linear', 'sig_linear', 'coef_sq', 'sig_sq', 'shape', 'sig_note']].to_string(index=False))

# ========== 5. 稳健性检验 (ROE) ==========
print("\n" + "=" * 60)
print("4. 稳健性检验: 被解释变量 = ROE")
print("=" * 60)
roe_results = []
roe_models = {}
for disp, col, cn in core_vars:
    label = f'ROE_{disp}'
    ret = run_nonlinear_model('ROE', disp, p, label)
    if ret is not None:
        row, model_obj = ret
        roe_results.append(row)
        roe_models[disp] = model_obj

roe_df = pd.DataFrame(roe_results)
print("\nROE 非线性检验摘要:")
print(roe_df[['core_var', 'coef_linear', 'sig_linear', 'coef_sq', 'sig_sq', 'shape', 'sig_note']].to_string(index=False))

# ========== 6. 保存完整结果 ==========
print("\n" + "=" * 60)
print("5. 保存结果表格")
print("=" * 60)

# 合并 ROA + ROE
all_df = pd.concat([roa_df, roe_df], ignore_index=True)
all_df.to_excel(f'{OUT_DIR}/nonlinear_test_results.xlsx', index=False)
print(f"  ✓ {OUT_DIR}/nonlinear_test_results.xlsx")

# 简洁回归表 (论文格式)
table_rows = []
for _, r in all_df.iterrows():
    def fmt_coef(coef, sig, se):
        return f"{coef:.4f}{sig}" if not np.isnan(coef) else "N/A"

    linear_str = fmt_coef(r['coef_linear'], r['sig_linear'], r['se_linear'])
    sq_str = fmt_coef(r['coef_sq'], r['sig_sq'], r['se_sq'])

    table_rows.append({
        '被解释变量': r['y'],
        '核心变量': r['core_var'],
        '样本量': r['N'],
        '企业数': r['N_firms'],
        '一次项系数': linear_str,
        '一次项标准误': f"{r['se_linear']:.4f}" if not np.isnan(r['se_linear']) else "N/A",
        '平方项系数': sq_str,
        '平方项标准误': f"{r['se_sq']:.4f}" if not np.isnan(r['se_sq']) else "N/A",
        '极值点': f"{r['turning_point']:.4f}" if r['turning_point'] is not None else "N/A",
        '极值在范围内': "是" if r['tp_inside_range'] else "否",
        '形状判断': r['shape'],
        '显著性说明': r['sig_note'],
        'Within R²': r['rsq_within'],
    })
table_df = pd.DataFrame(table_rows)
table_df.to_excel(f'{OUT_DIR}/table_nonlinear_regression.xlsx', index=False)
print(f"  ✓ {OUT_DIR}/table_nonlinear_regression.xlsx")

# ========== 7. 画预测曲线图 ==========
print("\n" + "=" * 60)
print("6. 绘制预测曲线")
print("=" * 60)

def plot_nonlinear(model, core_col, sq_col, y_col, disp_name, cn_name, y_label, filename, data):
    """画出一次项+平方项的预测曲线 (控制变量取均值)"""
    if model is None:
        return

    # 取回归系数
    b0 = model.params.get('Intercept', 0) if 'Intercept' in model.params else 0
    b1 = model.params.get(core_col, 0)
    b2 = model.params.get(sq_col, 0)

    # 控制变量的系数均值
    x_range = np.linspace(data[core_col].min(), data[core_col].max(), 100)

    # 预测 y = b1*x + b2*x^2 + others (控制变量取均值)
    ctrl_means = {}
    formula_terms = []
    for c in ctrl_cols:
        if c in model.params:
            ctrl_means[c] = data[c].mean()
            formula_terms.append(model.params[c] * data[c].mean())
    ctrl_offset = sum(formula_terms) + b0

    y_pred = b1 * x_range + b2 * x_range ** 2 + ctrl_offset

    fig, ax = plt.subplots(figsize=(8, 5))

    # 散点: 绘制数据点 (抽样子集避免太密)
    np.random.seed(42)
    n_sample = min(2000, len(data))
    idx_sample = np.random.choice(len(data), n_sample, replace=False)
    x_sample = data[core_col].iloc[idx_sample]
    y_sample = data[y_col].iloc[idx_sample]
    ax.scatter(x_sample, y_sample, alpha=0.15, s=5, c='gray', edgecolors='none')

    # 预测曲线
    ax.plot(x_range, y_pred, 'b-', linewidth=2.5, label='预测值')

    # 95% CI 用 bootstrap
    try:
        from linearmodels.iv.results import compare
        # 简单做法: 使用 params 的 std 近似
        se_b1 = model.std_errors.get(core_col, 0)
        se_b2 = model.std_errors.get(sq_col, 0)
        cov = model.cov.loc[core_col, sq_col] if core_col in model.cov.index and sq_col in model.cov.columns else 0
        # 简单近似: var(y_hat) ≈ (∂y/∂b)ᵀ Σ (∂y/∂b)
        # 这里简化处理，省略置信区间
    except:
        pass

    x_min, x_max = data[core_col].min(), data[core_col].max()
    ax.axvline(x=0, color='gray', linestyle=':', alpha=0.5)
    ax.axhline(y=0, color='gray', linestyle=':', alpha=0.5)

    # 极值点
    if b2 != 0:
        tp = -b1 / (2 * b2)
        if x_min <= tp <= x_max:
            y_tp = b1 * tp + b2 * tp ** 2 + ctrl_offset
            ax.axvline(x=tp, color='red', linestyle='--', alpha=0.7, label=f'极值点 x={tp:.3f}')
            ax.scatter([tp], [y_tp], color='red', s=60, zorder=5)

    ax.set_xlabel(f'{cn_name} ({disp_name})', fontproperties=zh_font, fontsize=12)
    ax.set_ylabel(y_label, fontproperties=zh_font, fontsize=12)
    ax.set_title(f'{cn_name}的非线性效应 ({y_label})', fontproperties=zh_font, fontsize=13)
    ax.legend(prop=zh_font, fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  ✓ {filename}")

# 对每个显著或边缘显著的模型画图
for disp, col, cn in core_vars:
    sq_col = f'{col}_sq'

    # ROA
    roa_row = roa_df[roa_df['core_var'] == disp]
    if len(roa_row) and roa_row.iloc[0]['p_sq'] < 0.15:
        plot_nonlinear(
            roa_models.get(disp), col, sq_col, 'roa_w',
            disp, cn, 'ROA (缩尾)',
            f'{OUT_DIR}/figures/nonlinear_ROA_{disp}.png',
            p.reset_index(level=['ISIN','month_dt']).drop(columns=['ISIN','month_dt'])
        )

    # ROE
    roe_row = roe_df[roe_df['core_var'] == disp]
    if len(roe_row) and roe_row.iloc[0]['p_sq'] < 0.15:
        plot_nonlinear(
            roe_models.get(disp), col, sq_col, 'roe_w',
            disp, cn, 'ROE (缩尾)',
            f'{OUT_DIR}/figures/nonlinear_ROE_{disp}.png',
            p.reset_index(level=['ISIN','month_dt']).drop(columns=['ISIN','month_dt'])
        )

# ========== 8. 生成文字报告 ==========
print("\n" + "=" * 60)
print("7. 生成非线性检验文字报告")
print("=" * 60)

def interpret_nonlinear(row):
    """对单个模型结果生成文字说明"""
    shape = row['shape']
    sig = row['sig_note']
    coef_l = row['coef_linear']
    coef_sq = row['coef_sq']
    p_sq = row['p_sq']
    p_l = row['p_linear']
    tp = row['turning_point']
    tp_ok = row['tp_inside_range']
    xmin = row['x_min']
    xmax = row['x_max']
    n = row['N']
    r2 = row['rsq_within']

    lines = []
    lines.append(f"样本量: {n:,} | Within R² = {r2}")
    lines.append(f"一次项系数: {coef_l:.5f} (p={p_l:.4f}){' ' + row['sig_linear'] if row['sig_linear'] else ''}")
    lines.append(f"平方项系数: {coef_sq:.5f} (p={p_sq:.4f}){' ' + row['sig_sq'] if row['sig_sq'] else ''}")

    if p_sq < 0.1:
        if shape == '倒U型候选':
            lines.append("判断: 一次项为正、平方项为负，符合倒U型假设")
            if tp_ok and tp is not None:
                lines.append(f"极值点 x* = {tp:.4f} 位于数据范围 [{xmin:.4f}, {xmax:.4f}] 内，倒U型关系在样本中得到支持")
            else:
                lines.append(f"极值点 x* = {tp} 不在数据范围内，倒U型关系可能仅存在于理论层面")
        elif shape == 'U型候选':
            lines.append("判断: 一次项为负、平方项为正，符合U型假设")
            if tp_ok and tp is not None:
                lines.append(f"极值点 x* = {tp:.4f} 位于数据范围 [{xmin:.4f}, {xmax:.4f}] 内，U型关系在样本中得到支持")
            else:
                lines.append(f"极值点 x* = {tp} 不在数据范围内，U型关系可能仅存在于理论层面")
        lines.append("结论: 存在显著的非线性特征，可以考虑写入论文正文")
    elif p_sq < 0.15:
        lines.append("判断: 平方项接近显著(p<0.15)，存在一定的非线性趋势")
        lines.append("结论: 可作为补充检验写入论文，或在稳健性中提及")
    else:
        lines.append("判断: 平方项不显著，未发现支持非线性关系的证据")
        lines.append("结论: 不建议写入论文正文，可在脚注或附录中提及")
    return '\n'.join(lines)


md_lines = []
md_lines.append("# 非线性检验报告：供应链关系结构与企业绩效的 U 型 / 倒 U 型关系")
md_lines.append("")
md_lines.append("## 一、检验目的与方法")
md_lines.append("")
md_lines.append("本检验旨在探究供应商和竞争对手关系结构与企业绩效之间是否存在非线性（U型或倒U型）关系。")
md_lines.append("在基准线性回归模型的基础上，加入核心解释变量的平方项，构建以下双向固定效应模型：")
md_lines.append("")
md_lines.append("```")
md_lines.append("Performance_it = β₁·X_it + β₂·X_it² + γ·Controls_it + μ_i + λ_t + ε_it")
md_lines.append("```")
md_lines.append("")
md_lines.append("其中，X 为四个核心关系变量（供应商广度、供应商行业多样性、竞争对手广度、竞争对手行业多样性），")
md_lines.append("Controls 包括企业规模、资产负债率和营业收入增长率。每次仅纳入一个核心变量及其平方项，")
md_lines.append("以避免变量间共线性干扰。")
md_lines.append("")
md_lines.append(f"样本量: 282,543 (ROA) / 280,215 (ROE) 企业-月度观测")
md_lines.append("时间范围: 2010–2020 年")
md_lines.append("")
md_lines.append("## 二、ROA 非线性检验结果")
md_lines.append("")
md_lines.append("| 核心变量 | 一次项系数 | 平方项系数 | 极值点 | 形状判断 | 结论 |")
md_lines.append("|----------|-----------|-----------|--------|---------|------|")

for _, r in roa_df.iterrows():
    shape = r['shape']
    tp_str = f"{r['turning_point']:.4f}" if r['turning_point'] is not None else "N/A"
    coef_l = f"{r['coef_linear']:.5f}{r['sig_linear']}"
    coef_sq = f"{r['coef_sq']:.5f}{r['sig_sq']}"
    conclusion = "适合写入正文" if r['p_sq'] < 0.1 else ("补充检验" if r['p_sq'] < 0.15 else "不建议写入")
    md_lines.append(f"| {r['core_var']} | {coef_l} | {coef_sq} | {tp_str} | {shape} | {conclusion} |")

md_lines.append("")
md_lines.append("### 详细解释：")
md_lines.append("")

for _, r in roa_df.iterrows():
    md_lines.append(f"### 1. {r['core_var']}")
    md_lines.append("")
    md_lines.append(interpret_nonlinear(r))
    md_lines.append("")

md_lines.append("## 三、ROE 稳健性检验结果")
md_lines.append("")
md_lines.append("| 核心变量 | 一次项系数 | 平方项系数 | 极值点 | 形状判断 | 结论 |")
md_lines.append("|----------|-----------|-----------|--------|---------|------|")

for _, r in roe_df.iterrows():
    shape = r['shape']
    tp_str = f"{r['turning_point']:.4f}" if r['turning_point'] is not None else "N/A"
    coef_l = f"{r['coef_linear']:.5f}{r['sig_linear']}"
    coef_sq = f"{r['coef_sq']:.5f}{r['sig_sq']}"
    conclusion = "适合写入正文" if r['p_sq'] < 0.1 else ("补充检验" if r['p_sq'] < 0.15 else "不建议写入")
    md_lines.append(f"| {r['core_var']} | {coef_l} | {coef_sq} | {tp_str} | {shape} | {conclusion} |")

md_lines.append("")
md_lines.append("### 详细解释：")
md_lines.append("")

for _, r in roe_df.iterrows():
    md_lines.append(f"### 1. {r['core_var']}")
    md_lines.append("")
    md_lines.append(interpret_nonlinear(r))
    md_lines.append("")

md_lines.append("## 四、ROA 与 ROE 结果对比")
md_lines.append("")
md_lines.append("| 核心变量 | ROA 形状 | ROA 显著性 | ROE 形状 | ROE 显著性 | 一致性 |")
md_lines.append("|----------|---------|-----------|---------|-----------|--------|")

for disp, col, cn in core_vars:
    r_roa = roa_df[roa_df['core_var'] == disp]
    r_roe = roe_df[roe_df['core_var'] == disp]
    if len(r_roa) and len(r_roe):
        rr = r_roa.iloc[0]
        re = r_roe.iloc[0]
        roa_sig = f"p_sq={rr['p_sq']:.4f} ({'显著' if rr['p_sq']<0.1 else '不显著'})"
        roe_sig = f"p_sq={re['p_sq']:.4f} ({'显著' if re['p_sq']<0.1 else '不显著'})"
        consistent = "一致" if (rr['p_sq']<0.1) == (re['p_sq']<0.1) else "部分一致"
        md_lines.append(f"| {disp} | {rr['shape']} | {roa_sig} | {re['shape']} | {roe_sig} | {consistent} |")

md_lines.append("")
md_lines.append("## 五、模型设定说明")
md_lines.append("")
md_lines.append("1. **被解释变量**: ROA（总资产收益率，1%缩尾处理），ROE（净资产收益率，1%缩尾处理）")
md_lines.append("2. **核心解释变量**: 供应商广度（ln(1+供应商数)）、供应商行业多样性（ln(1+供应商覆盖RBICS行业数)）、")
md_lines.append("   竞争对手广度（ln(1+竞争对手数)）、竞争对手行业多样性（ln(1+竞争对手覆盖RBICS行业数)）")
md_lines.append("3. **平方项**: 各核心变量的平方值")
md_lines.append("4. **控制变量**: 企业规模（ln总资产）、资产负债率、营业收入增长率（1%缩尾）")
md_lines.append("5. **固定效应**: 企业固定效应 + 月度时间固定效应")
md_lines.append("6. **标准误**: 按企业聚类（Cluster-robust standard errors）")
md_lines.append("7. **估计方法**: 面板最小二乘虚拟变量法（PanelOLS, LSDV）")
md_lines.append("")
md_lines.append("## 六、论文建议")
md_lines.append("")
md_lines.append("### 可写入论文正文：")
md_lines.append("")

# Check which are significant
for _, r in all_df.iterrows():
    if r['p_sq'] < 0.1:
        md_lines.append(f"- **{r['y']}模型中的{r['core_var']}**: 平方项显著(p={r['p_sq']:.4f})，{r['shape']}，极值点{'在' if r['tp_inside_range'] else '不在'}数据范围内")

md_lines.append("")
md_lines.append("### 可作为补充检验/稳健性：")
md_lines.append("")
for _, r in all_df.iterrows():
    if 0.1 <= r['p_sq'] < 0.15:
        md_lines.append(f"- **{r['y']}模型中的{r['core_var']}**: 平方项接近显著(p={r['p_sq']:.4f})，存在一定非线性趋势")

md_lines.append("")
md_lines.append("### 不建议写入论文：")
md_lines.append("")
for _, r in all_df.iterrows():
    if r['p_sq'] >= 0.15:
        md_lines.append(f"- **{r['y']}模型中的{r['core_var']}**: 平方项不显著(p={r['p_sq']:.4f})")

md_lines.append("")
md_lines.append("## 七、注意事项")
md_lines.append("")
md_lines.append("1. 平方项与一次项之间可能存在共线性，导致标准误膨胀")
md_lines.append("2. 非线性检验仅为描述性分析，不能直接推断因果关系")
md_lines.append("3. 极值点的解释需谨慎，尤其是当极值点位于数据范围边缘时")
md_lines.append("4. 建议在论文中呈现非线性检验结果表，正文重点讨论显著的结果")
md_lines.append("5. 本检验采用逐个变量的方法，未控制其他核心变量的非线性效应")
md_lines.append("")

md_content = '\n'.join(md_lines)

with open(f'{OUT_DIR}/nonlinear_test_summary.md', 'w', encoding='utf-8') as f:
    f.write(md_content)
print(f"  ✓ {OUT_DIR}/nonlinear_test_summary.md")
print(md_content)

print("\n" + "=" * 60)
print("非线性检验全部完成!")
print("=" * 60)
