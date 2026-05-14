"""
事件研究图：逐年交互项检验
直接使用 PanelOLS(y, X, entity_effects=True) 而非公式
"""
import pandas as pd, numpy as np, os, warnings
warnings.filterwarnings('ignore')
from linearmodels.panel import PanelOLS
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

OUT_DIR  = '/Users/deesanyu/pythonproject/deesan_supply_0506v4/output/event_study'
FONT_PATH = '/Users/deesanyu/pythonproject/deesan_supply_0506v4/simhei.ttf'
os.makedirs(OUT_DIR, exist_ok=True)
zh_font = FontProperties(fname=FONT_PATH)
plt.rcParams['axes.unicode_minus'] = False

print("加载月度面板...")
mp = pd.read_csv('/Users/deesanyu/pythonproject/deesan_supply_0506v4/output/firm_monthly_panel.csv', low_memory=False)
mp['month_dt'] = pd.to_datetime(mp['month'])
mp['year'] = mp['month_dt'].dt.year
mp = mp.sort_values(['ISIN', 'month_dt']).set_index(['ISIN', 'month_dt'])

core_vars = ['sup_breadth', 'sup_ind_div', 'comp_breadth', 'comp_ind_div']
zh_names = ['供应商广度', '供应商行业多样性', '竞争对手广度', '竞争对手行业多样性']
controls = ['size', 'lev', 'growth_w']
all_years = sorted(mp['year'].unique())
BASE_YEAR = 2017

print(f"面板: {len(mp):,} 行, {mp.index.get_level_values(0).nunique():,} 企业")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

for idx, (core_col, zh_name) in enumerate(zip(core_vars, zh_names)):
    ax = axes[idx // 2, idx % 2]

    # 构建交互项
    d = mp[[core_col, 'roa_w'] + controls + ['year']].copy()
    for y in all_years:
        if y != BASE_YEAR:
            d[f'x_{y}'] = d[core_col] * (d['year'] == y).astype(float)

    x_cols = [f'x_{y}' for y in all_years if y != BASE_YEAR] + controls
    sub = d[['roa_w'] + x_cols].dropna()
    n = len(sub)
    print(f"\n{zh_name}: N={n:,}")

    try:
        mod = PanelOLS(sub['roa_w'], sub[x_cols], entity_effects=True, check_rank=False, drop_absorbed=True)
        res = mod.fit(cov_type='clustered', cluster_entity=True)

        years_plot, coefs_plot, cis_plot = [], [], []
        for y in all_years:
            if y == BASE_YEAR:
                years_plot.append(y); coefs_plot.append(0); cis_plot.append(0)
            else:
                vn = f'x_{y}'
                if vn in res.params:
                    c = res.params[vn]; se = res.std_errors[vn]
                    years_plot.append(y); coefs_plot.append(c); cis_plot.append(se * 1.96)

        coefs_arr, cis_arr = np.array(coefs_plot), np.array(cis_plot)

        ax.axhline(y=0, color='gray', ls='--', lw=1, alpha=0.7)
        ax.axvline(x=2017.5, color='red', ls='--', lw=1.5, alpha=0.5, label='贸易战')
        ax.plot(years_plot, coefs_arr, 'o-', color='steelblue', linewidth=2, markersize=8, zorder=3)
        ax.fill_between(years_plot, coefs_arr - cis_arr, coefs_arr + cis_arr,
                        color='steelblue', alpha=0.15)

        for yi, (y, c, ci) in enumerate(zip(years_plot, coefs_arr, cis_arr)):
            if abs(c) > ci:
                ax.annotate('*', (y, c + ci + 0.015), ha='center', fontsize=14,
                           color='red', fontweight='bold')

        ax.set_xlabel('年份', fontproperties=zh_font, fontsize=11)
        ax.set_ylabel('相对于2017年的效应差异', fontproperties=zh_font, fontsize=10)
        ax.set_title(f'{zh_name}的逐年效应(基准=2017)', fontproperties=zh_font, fontsize=13)
        ax.legend(prop=zh_font, fontsize=9)
        ax.set_xticks(all_years); ax.set_xticklabels(all_years, rotation=45)
        ax.grid(True, alpha=0.2)

        for y, c, ci in zip(years_plot, coefs_arr, cis_arr):
            star = '*' if abs(c) > ci else ' '
            print(f"  {y}: β={c:+.5f}{star} ±{ci:.5f}")

    except Exception as e:
        print(f"  ERROR: {e}")
        ax.text(0.5, 0.5, f'Error: {e}', ha='center', va='center', transform=ax.transAxes)

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/event_study.png', dpi=200, bbox_inches='tight')
plt.close()
print(f"\n✓ {OUT_DIR}/event_study.png")
print("完成!")
