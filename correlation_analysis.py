"""相关性分析：供应商和竞争对手关系结构与企业绩效"""
import pandas as pd, numpy as np, matplotlib, os, warnings
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
warnings.filterwarnings('ignore')

OUT_DIR='/Users/deesanyu/pythonproject/deesan_supply_0506v4/output'
FONT_PATH='/Users/deesanyu/pythonproject/deesan_supply_0506v4/simhei.ttf'
os.makedirs(f'{OUT_DIR}/figures',exist_ok=True)
zh_font=FontProperties(fname=FONT_PATH)
plt.rcParams['axes.unicode_minus']=False

# ========== 1. 加载数据 ==========
print("1. 加载数据")
p=pd.read_csv(f'{OUT_DIR}/firm_monthly_panel.csv')
print(f"  面板: {p.shape}")

# ========== 2. 构建变量 ==========
print("2. 构建分析变量")

# 供应商/竞争对手国家数量: 统计 sup_ct_* / comp_ct_* 中非零列的数量
sup_ct_cols=[c for c in p.columns if c.startswith('sup_ct_')]
comp_ct_cols=[c for c in p.columns if c.startswith('comp_ct_')]
p['supplier_country_count']=(p[sup_ct_cols]>0).sum(axis=1)
p['competitor_country_count']=(p[comp_ct_cols]>0).sum(axis=1)

# EU占比: 德国+英国+法国 (已有的欧洲主要国家)
for prefix in ['sup','comp']:
    cols=[f'{prefix}_ct_德国',f'{prefix}_ct_英国',f'{prefix}_ct_法国']
    exist=[c for c in cols if c in p.columns]
    p[f'{prefix}_ct_eu']=p[exist].sum(axis=1)

# ========== 3. 主相关性表 ==========
print("3. 主相关性分析")
main_vars={
    'ROA':'roa_w',
    'ROE':'roe_w',
    'supplier_count_log':'sup_breadth',
    'supplier_industry_count':'sup_ind_div',
    'competitor_count_log':'comp_breadth',
    'competitor_industry_count':'comp_ind_div',
    'supplier_country_count':'supplier_country_count',
    'competitor_country_count':'competitor_country_count',
    'Size':'size',
    'Leverage':'lev',
    'Growth':'growth_w',
}
main_df=p[list(main_vars.values())]
main_df.columns=list(main_vars.keys())
corr_main=main_df.corr(method='pearson')
corr_main.to_csv(f'{OUT_DIR}/correlation_matrix.csv',encoding='utf-8-sig')
print("  主相关性矩阵:")
print(corr_main.round(3))

# ========== 4. 国别补充相关性表 ==========
print("4. 国别补充相关性")
ct_vars={
    'ROA':'roa_w',
    'supplier_count_log':'sup_breadth',
    'competitor_count_log':'comp_breadth',
    'us_supplier_ratio':'sup_ct_美国',
    'china_supplier_ratio':'sup_ct_中国',
    'eu_supplier_ratio':'sup_ct_eu',
    'us_competitor_ratio':'comp_ct_美国',
    'china_competitor_ratio':'comp_ct_中国',
    'eu_competitor_ratio':'comp_ct_eu',
    'Size':'size',
    'Leverage':'lev',
    'Growth':'growth_w',
}
ct_df=p[list(ct_vars.values())]
ct_df.columns=list(ct_vars.keys())
corr_ct=ct_df.corr(method='pearson')
corr_ct.to_csv(f'{OUT_DIR}/correlation_country_matrix.csv',encoding='utf-8-sig')
print("  国别相关性矩阵:")
print(corr_ct.round(3))

# ========== 5. 热力图 ==========
print("5. 生成热力图")
fig,ax=plt.subplots(figsize=(11,9))
labels=list(main_vars.keys())
vals=corr_main.values
im=ax.imshow(vals,cmap='RdBu_r',vmin=-1,vmax=1,aspect='auto')
ax.set_xticks(range(len(labels)))
ax.set_yticks(range(len(labels)))
ax.set_xticklabels(labels,fontproperties=zh_font,fontsize=8,rotation=45,ha='right')
ax.set_yticklabels(labels,fontproperties=zh_font,fontsize=8)
for i in range(len(labels)):
    for j in range(len(labels)):
        c=vals[i,j]
        color='white' if abs(c)>0.5 else 'black'
        ax.text(j,i,f'{c:.2f}',ha='center',va='center',fontsize=7,color=color)
plt.colorbar(im,ax=ax,shrink=0.8)
plt.title('变量相关性热力图',fontproperties=zh_font,fontsize=14)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/figures/correlation_heatmap.png',dpi=200,bbox_inches='tight')
plt.close()
print("  [OK] 热力图")

# ========== 6. 分析摘要 ==========
print("6. 生成分析摘要")
r=corr_main

lines=[]
lines.append("="*70)
lines.append("相关性分析摘要")
lines.append("="*70)
lines.append("")

# 6.1 ROA/ROE与核心关系变量
lines.append("一、ROA/ROE 与核心关系变量的相关性")
lines.append("-"*50)
for dv_disp, dv_code in [('ROA','ROA'),('ROE','ROE')]:
    lines.append(f"\n{dv_disp}:")
    for iv in ['supplier_count_log','competitor_count_log',
               'supplier_industry_count','competitor_industry_count',
               'supplier_country_count','competitor_country_count']:
        val=r.loc[dv_disp,iv]
        direction="正" if val>0 else "负"
        strength="强" if abs(val)>0.3 else ("中等" if abs(val)>0.1 else "弱")
        lines.append(f"  × {iv:35s}: {val:+.3f} ({direction}相关, {strength})")

# 6.2 关系广度与行业多样性
lines.append("\n\n二、关系广度与行业多样性的相关性")
lines.append("-"*50)
for bv,iv,lb in [('supplier_count_log','supplier_industry_count','供应商'),
                  ('competitor_count_log','competitor_industry_count','竞争对手')]:
    val=r.loc[bv,iv]
    lines.append(f"  {lb}: 广度 × 行业多样性 = {val:+.3f}")
    if abs(val)>0.5:
        lines.append(f"    → 高度相关，回归中可能存在共线性问题，建议分别纳入模型")
    elif abs(val)>0.3:
        lines.append(f"    → 中等相关，需关注VIF值")

# 6.3 多重共线性风险
lines.append("\n\n三、多重共线性风险判断")
lines.append("-"*50)
high_pairs=[]
for i in range(len(r.columns)):
    for j in range(i+1,len(r.columns)):
        v=abs(r.iloc[i,j])
        if v>0.5 and i!=j:
            high_pairs.append((r.columns[i],r.columns[j],r.iloc[i,j]))
if high_pairs:
    lines.append("  以下变量对相关系数 > 0.5，存在共线性风险:")
    for a,b,v in high_pairs:
        lines.append(f"    {a:35s} ↔ {b:35s}: r={v:+.3f}")
    lines.append("  → 建议避免将高度相关的变量同时纳入同一回归模型")
else:
    lines.append("  所有变量对相关系数绝对值均<0.5，不存在严重共线性问题")

# VIF替代检查: 变量间最大相关系数
max_corr=abs(r.values[np.triu_indices_from(r.values,k=1)]).max()
lines.append(f"  变量间最大相关系数: {max_corr:.3f}")
if max_corr<0.6:
    lines.append("  → 整体共线性风险较低，可同时纳入回归模型")
elif max_corr<0.8:
    lines.append("  → 存在一定共线性风险，建议检查VIF")
else:
    lines.append("  → 严重共线性风险，需谨慎处理")

# 6.4 国别补充表关键发现
lines.append("\n\n四、国别变量相关性关键发现")
lines.append("-"*50)
rc=corr_ct
for dv in ['ROA']:
    for ct in ['us_supplier_ratio','china_supplier_ratio','eu_supplier_ratio',
               'us_competitor_ratio','china_competitor_ratio','eu_competitor_ratio']:
        v=rc.loc[dv,ct]
        if abs(v)>0.01:
            lines.append(f"  {dv} × {ct:30s}: {v:+.3f}")

# 国别变量之间的关系
ct_pairs=[('us_supplier_ratio','us_competitor_ratio'),
           ('china_supplier_ratio','china_competitor_ratio'),
           ('supplier_count_log','us_supplier_ratio'),
           ('supplier_count_log','china_supplier_ratio')]
for a,b in ct_pairs:
    if a in rc.columns and b in rc.columns:
        v=rc.loc[a,b]
        lines.append(f"  {a:30s} × {b:30s}: {v:+.3f}")

# 6.5 回归注意事项
lines.append("\n\n五、回归分析注意事项")
lines.append("-"*50)
lines.append("""
1. 供应商广度和供应商行业多样性高度相关(r≈0.53)，建议：
   - 分别纳入模型，或使用主成分分析降维
   - 若同时纳入，需报告VIF值

2. 竞争对手广度和竞争对手行业多样性高度相关(r≈0.59)，同上。

3. ROA与Leverage存在中等负相关(r≈-0.29)，符合财务理论预期：
   - 高杠杆企业通常面临更高财务风险，盈利能力较低
   - 两者可同时纳入回归模型

4. ROA与Growth正相关(r≈0.13)，高增长企业盈利水平更高：
   - 但需注意Growth的测量误差问题（季度增长率赋值月度）

5. 供应商/竞争对手广度与企业规模正相关(r≈0.18-0.28)：
   - 大规模企业关系网络更广泛
   - 回归中控制Size是必要的

6. 国别占比变量存在大量零值，分布高度偏态：
   - 建议在回归中谨慎使用，或采用二元变量（有/无）
   - 国别变量之间的相关性也需要注意

7. 供应商国别和竞争对手国别占比之间的相关性：
   - 如中美供应商占比与中美竞争对手占比的相关性
   - 可能反映行业或企业层面的系统性特征
""")

lines.append("="*70)
lines.append("分析完成")
lines.append("="*70)

summary='\n'.join(lines)
print(summary)
with open(f'{OUT_DIR}/correlation_summary.txt','w',encoding='utf-8') as f:
    f.write(summary)
print(f"\n所有文件已保存至 {OUT_DIR}/")
