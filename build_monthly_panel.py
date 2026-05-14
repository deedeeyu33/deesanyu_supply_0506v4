"""
构建企业月度面板数据 (高速版)
供应商和竞争对手关系结构对企业绩效的影响
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import os, re, warnings
warnings.filterwarnings('ignore')

DATA_DIR  = '/Users/deesanyu/pythonproject/deesan_supply_0506v4/data'
OUT_DIR   = '/Users/deesanyu/pythonproject/deesan_supply_0506v4/output'
FONT_PATH = '/Users/deesanyu/pythonproject/deesan_supply_0506v4/simhei.ttf'
os.makedirs(f'{OUT_DIR}/figures', exist_ok=True)
zh_font = FontProperties(fname=FONT_PATH)
plt.rcParams['axes.unicode_minus'] = False

# ========== 1. 加载关系数据 ==========
print("1. 加载关系数据")
rel = pd.read_csv(f'{DATA_DIR}/crossborder_relationships_mirrored_2010_2020.csv', low_memory=False)
rel['end_'] = rel['end_'].str.replace('4000-01-01', '2020-12-31', regex=False)
rel['start_'] = pd.to_datetime(rel['start_'], format='%Y-%m-%d', errors='coerce')
rel['end_'] = pd.to_datetime(rel['end_'], format='%Y-%m-%d %H:%M:%S', errors='coerce')

# 只保留供应商和竞争对手
rel = rel[rel['rel_type_broad'].isin(['SUPPLIER','COMPETITOR'])].copy()
rel = rel[['cn_company_isin','counterparty_isin','counterparty_country',
            'counterparty_region','rel_type_broad','start_','end_']].copy()
print(f"  供应商+竞争对手: {len(rel):,}, 企业: {rel['cn_company_isin'].nunique():,}")

# ========== 2. 转整数月编码 (2010-01=0 ... 2020-12=131) ==========
print("2. 转整数月编码")
def to_month_int(dt):
    return (dt.dt.year - 2010) * 12 + dt.dt.month - 1

rel['ms'] = to_month_int(rel['start_']).clip(0, 131)
rel['me'] = to_month_int(rel['end_']).clip(0, 131)
rel = rel[rel['ms'] <= rel['me']].copy()

# ========== 3. 用 numpy 快速展开 ==========
print("3. 展开关系到月 (numpy)")
n_arr = (rel['me'] - rel['ms'] + 1).values.astype(int)
print(f"  总关系-月: {n_arr.sum():,}")

idx = np.repeat(rel.index.values, n_arr)
exp = rel.loc[idx, ['cn_company_isin','counterparty_isin','counterparty_country',
                      'rel_type_broad']].copy()
exp['mi'] = rel.loc[idx, 'ms'].values + np.concatenate([np.arange(n) for n in n_arr])
print(f"  展开后: {len(exp):,} 行")

# ========== 4. 按类型拆分并聚合 ==========
print("4. 聚合月度关系变量")
is_sup = exp['rel_type_broad'] == 'SUPPLIER'

# 4a. 关系广度
sup_counts = exp[is_sup].groupby(['cn_company_isin','mi']).size()
comp_counts = exp[~is_sup].groupby(['cn_company_isin','mi']).size()

# 4b. 国家归类
def cg(c):
    c=str(c)
    for kw in ['美国','中国','日本','韩国','德国','法国','英国','香港','台湾',
               '新加坡','印度','荷兰','瑞士','瑞典','加拿大','澳大利亚','开曼',
               '百慕大','意大利','西班牙','巴西','俄罗斯','越南','泰国','马来西亚',
               '印度尼西亚','菲律宾','阿联酋','沙特','挪威','芬兰','丹麦','爱尔兰',
               '奥地利','比利时','葡萄牙','波兰','土耳其','南非','墨西哥','智利',
               '阿根廷','新西兰','卢森堡','以色列','埃及','尼日利亚','哥伦比亚',
               '巴拿马','塞浦路斯','希腊','毛里求斯','根西','泽西','英属维尔京']:
        if kw in c: return kw
    if '未知' in c: return '未知'
    return '其他'

exp['cg'] = exp['counterparty_country'].apply(cg)

# 国别月度统计 (只保留主要国家)
countries_of_interest = ['美国','中国','日本','韩国','德国','英国','法国',
                         '新加坡','香港','台湾','印度','荷兰','瑞士','瑞典',
                         '加拿大','澳大利亚','开曼群岛','百慕大','意大利',
                         '西班牙','巴西','俄罗斯','越南','泰国','马来西亚',
                         '印度尼西亚','菲律宾','阿联酋','沙特阿拉伯','挪威',
                         '芬兰','丹麦','爱尔兰','奥地利','比利时','葡萄牙',
                         '波兰','土耳其','南非','墨西哥','智利','阿根廷',
                         '新西兰','卢森堡','以色列','埃及','哥伦比亚',
                         '巴拿马','希腊','毛里求斯','根西岛','泽西岛',
                         '英属维尔京群岛','未知','其他']

# ========== 5. 构建完整面板 ==========
print("5. 构建完整面板结构")
all_firms = rel['cn_company_isin'].unique()
all_mi = np.arange(132)
firm_grid = pd.MultiIndex.from_product([all_firms, all_mi], names=['ISIN','mi']).to_frame(index=False)
print(f"  面板: {len(firm_grid):,} 行 ({len(all_firms)} 企业 x 132 月)")

# 合并广度
firm_grid = firm_grid.merge(sup_counts.reset_index(name='sup_raw').rename(columns={'cn_company_isin':'ISIN'}),
                             on=['ISIN','mi'], how='left')
firm_grid['sup_raw'] = firm_grid['sup_raw'].fillna(0).astype(int)

firm_grid = firm_grid.merge(comp_counts.reset_index(name='comp_raw').rename(columns={'cn_company_isin':'ISIN'}),
                             on=['ISIN','mi'], how='left')
firm_grid['comp_raw'] = firm_grid['comp_raw'].fillna(0).astype(int)

# 国别占比
for prefix, mask, label in [('sup', is_sup, '供应商'), ('comp', ~is_sup, '竞争对手')]:
    sub = exp[mask].copy()
    if len(sub):
        ct = sub.groupby(['cn_company_isin','mi','cg']).size().reset_index(name='n')
        ct['pct'] = ct.groupby(['cn_company_isin','mi'])['n'].transform(lambda x: x/x.sum()*100)
        # Pivot for key countries
        for c in countries_of_interest:
            vals = ct[ct['cg']==c][['cn_company_isin','mi','pct']].rename(
                columns={'cn_company_isin':'ISIN','pct':f'{prefix}_ct_{c}'})
            firm_grid = firm_grid.merge(vals, on=['ISIN','mi'], how='left')
            firm_grid[f'{prefix}_ct_{c}'] = firm_grid[f'{prefix}_ct_{c}'].fillna(0)

print(f"  关系面板: {firm_grid.shape}")

# ========== 6. 行业匹配 ==========
print("6. 行业匹配")
assets_raw = pd.read_excel(f'{DATA_DIR}/china-quarterly-assets.xlsx', header=None)
hdr = assets_raw.iloc[4]
isin_col = list(hdr).index('FE Isin')
ind_info = assets_raw.iloc[6:, [10,11,isin_col]].copy()
ind_info.columns = ['L1','L4','ISIN']
ind_info = ind_info.dropna(subset=['ISIN'])
ind_info = ind_info[ind_info['L1']!='@NA'].drop_duplicates('ISIN')

# 供应商行业多样性
sup_exp = exp[is_sup].merge(ind_info.rename(columns={'ISIN':'counterparty_isin'}),
                             on='counterparty_isin', how='left')
sup_l1 = sup_exp.dropna(subset=['L1']).groupby(['cn_company_isin','mi'])['L1'].nunique().reset_index(name='sup_ind_l1')
sup_l4 = sup_exp.dropna(subset=['L4']).groupby(['cn_company_isin','mi'])['L4'].nunique().reset_index(name='sup_ind_l4')

# 竞争对手行业多样性
comp_exp = exp[~is_sup].merge(ind_info.rename(columns={'ISIN':'counterparty_isin'}),
                               on='counterparty_isin', how='left')
comp_l1 = comp_exp.dropna(subset=['L1']).groupby(['cn_company_isin','mi'])['L1'].nunique().reset_index(name='comp_ind_l1')
comp_l4 = comp_exp.dropna(subset=['L4']).groupby(['cn_company_isin','mi'])['L4'].nunique().reset_index(name='comp_ind_l4')

for df_i, col in [(sup_l1,'sup_ind_l1'),(sup_l4,'sup_ind_l4'),
                   (comp_l1,'comp_ind_l1'),(comp_l4,'comp_ind_l4')]:
    firm_grid = firm_grid.merge(df_i.rename(columns={'cn_company_isin':'ISIN'}),
                                 on=['ISIN','mi'], how='left')
    firm_grid[col] = firm_grid[col].fillna(0).astype(int)

# 删除临时大对象
del exp, sup_exp, comp_exp

# ========== 7. 加载季度财务数据 ==========
print("7. 加载季度财务数据")
def load_q(filepath, vname):
    raw = pd.read_excel(filepath, header=None)
    h = raw.iloc[4]
    isin_pos = next(i for i,v in enumerate(h) if pd.notna(v) and 'Isin' in str(v))
    # 找到2010-2020的季度列, 建立 实际列名 -> 季度编码 的映射
    col_to_q = {}  # 'assets_2010_Q1' -> '2010Q1'
    q_to_col = {}  # '2010Q1' -> 'assets_2010_Q1'
    for i in range(14, isin_pos):
        v = str(h[i]) if pd.notna(h[i]) else ''
        m = re.search(r'(201[0-9]|2020).*?Q(\d)', v)
        if m:
            qcode = f"{m.group(1)}Q{m.group(2)}"
            col_to_q[v] = qcode
            q_to_col[qcode] = v

    d = raw.iloc[6:].copy()
    d.columns = list(h)
    d = d[d['Symbol'].notna()]

    id_cols = ['Symbol','Name']
    if 'FE Isin' in d.columns:
        id_cols.append('FE Isin')

    value_cols = list(col_to_q.keys())  # actual column names
    melted = d.melt(id_vars=id_cols, value_vars=value_cols,
                    var_name='qname', value_name=vname)

    # Map actual column name to quarter code
    melted['qcode'] = melted['qname'].map(col_to_q)
    melted[vname] = pd.to_numeric(melted[vname], errors='coerce')

    def qcode_to_int(q):
        m = re.search(r'(201[0-9]|2020)Q(\d)', str(q))
        if m:
            y, q_ = int(m.group(1)), int(m.group(2))
            return (y-2010)*4 + (q_-1)
        return -1

    melted['qi'] = melted['qcode'].apply(qcode_to_int)
    melted = melted[(melted['qi']>=0) & (melted['qi']<=43)].dropna(subset=[vname])
    return melted[['FE Isin','qi',vname]]

print("  总资产...")
a = load_q(f'{DATA_DIR}/china-quarterly-assets.xlsx', 'assets')
print("  ROA...")
r1 = load_q(f'{DATA_DIR}/china-quarterly-roa.xlsx', 'roa')
print("  ROE...")
r2 = load_q(f'{DATA_DIR}/china-quarterly-roe.xlsx', 'roe')
print("  负债...")
d = load_q(f'{DATA_DIR}/china-quarterly-debt.xlsx', 'debt')
print("  营收...")
s = load_q(f'{DATA_DIR}/china-quarterly-sales.xlsx', 'sales')

# 合并财务数据
fin = a.merge(r1, on=['FE Isin','qi'], how='outer')
for df_o in [r2, d, s]:
    vn = [c for c in df_o.columns if c not in ['FE Isin','qi']][0]
    fin = fin.merge(df_o, on=['FE Isin','qi'], how='outer')

print(f"  财务合并: {len(fin):,} 行")

# 展开季度到月度: 每个季度对应3个月 (qi*3, qi*3+1, qi*3+2)
fin_ex = fin.loc[fin.index.repeat(3)].copy()
fin_ex['mo'] = fin_ex.groupby(level=0).cumcount()
fin_ex['mi'] = fin_ex['qi'] * 3 + fin_ex['mo']
fin_ex = fin_ex[fin_ex['mi'].between(0, 131)]
print(f"  月度财务: {len(fin_ex):,} 行")

# ========== 8. 合并关系+财务 ==========
print("8. 合并面板")
panel = firm_grid.merge(fin_ex.rename(columns={'FE Isin':'ISIN'}),
                         on=['ISIN','mi'], how='left')
panel['month'] = pd.Timestamp('2010-01-01') + pd.to_timedelta(panel['mi'], unit='M')
print(f"  面板: {panel.shape}, 企业: {panel['ISIN'].nunique()}, "
      f"月份: {panel['month'].min().strftime('%Y-%m')}~{panel['month'].max().strftime('%Y-%m')}")

# ========== 9. 衍生变量 ==========
print("9. 计算衍生变量")
panel['sup_breadth'] = np.log1p(panel['sup_raw'])
panel['comp_breadth'] = np.log1p(panel['comp_raw'])
panel['sup_ind_div'] = np.log1p(panel['sup_ind_l4'])
panel['comp_ind_div'] = np.log1p(panel['comp_ind_l4'])
panel['sup_ind_div_l1'] = np.log1p(panel['sup_ind_l1'])
panel['comp_ind_div_l1'] = np.log1p(panel['comp_ind_l1'])

def winsor(s, l=0.01, u=0.99):
    lo, hi = s.quantile(l), s.quantile(u)
    return s.clip(lo, hi)

panel['roa_w'] = winsor(panel['roa'])
panel['roe_w'] = winsor(panel['roe'])
panel['size'] = np.log(panel['assets'].clip(lower=1))
panel['lev'] = (panel['debt']/panel['assets']).replace([np.inf,-np.inf], np.nan)

# 销售增长率 (YoY quarterly)
fin_q = fin.sort_values(['FE Isin','qi'])
fin_q['sales_l4'] = fin_q.groupby('FE Isin')['sales'].shift(4)
fin_q['growth'] = ((fin_q['sales']-fin_q['sales_l4'])/fin_q['sales_l4'])

fin_q_ex = fin_q.loc[fin_q.index.repeat(3)].copy()
fin_q_ex['mo'] = fin_q_ex.groupby(level=0).cumcount()
fin_q_ex['mi'] = fin_q_ex['qi']*3 + fin_q_ex['mo']
fin_q_ex = fin_q_ex[fin_q_ex['mi'].between(0,131)]

panel = panel.merge(fin_q_ex[['FE Isin','mi','growth']].rename(columns={'FE Isin':'ISIN'}),
                     on=['ISIN','mi'], how='left')
panel['growth'] = panel['growth'].replace([np.inf,-np.inf], np.nan)
panel['growth_w'] = winsor(panel['growth'])

# ========== 10. 保存 ==========
print("10. 保存")
panel.to_csv(f'{OUT_DIR}/firm_monthly_panel.csv', index=False, encoding='utf-8-sig')
print(f"  已保存: {OUT_DIR}/firm_monthly_panel.csv")

# ========== 11. 描述性统计 ==========
print("11. 描述性统计")
svars = ['sup_breadth','comp_breadth','sup_ind_div','comp_ind_div',
         'roa_w','roe_w','size','lev','growth_w']

desc = panel[svars].describe().T[['count','mean','std','min','25%','50%','75%','max']]
desc.columns = ['N','Mean','SD','Min','P25','P50','P75','Max']
desc.to_csv(f'{OUT_DIR}/desc_core.csv', encoding='utf-8-sig')
print(desc.round(3))

panel['year'] = panel['month'].dt.year
print(f"\n企业: {panel['ISIN'].nunique()}, 观测: {len(panel):,}, "
      f"年份: {panel['year'].min()}-{panel['year'].max()}")

# 国别
for prefix, cdf, label in [('sup', sup_counts, '供应商'), ('comp', comp_counts, '竞争对手')]:
    pass  # computed above

# 相关性矩阵
corr = panel[svars].corr()
corr.to_csv(f'{OUT_DIR}/corr_matrix.csv', encoding='utf-8-sig')

# ========== 12. 可视化 ==========
print("12. 可视化")
titles = ['供应商广度','竞争对手广度','供应商行业多样性','竞争对手行业多样性',
          'ROA(缩尾)','ROE(缩尾)','企业规模','资产负债率','增长率(缩尾)']

fig, axes = plt.subplots(3,3,figsize=(15,12))
for i,v in enumerate(svars):
    ax = axes[i//3,i%3]
    d = panel[v].dropna()
    lo, hi = d.quantile(0.01), d.quantile(0.99)
    if hi>lo: d = d.clip(lo,hi)
    ax.hist(d, bins=50, ec='white', alpha=0.7, color='steelblue')
    ax.axvline(d.mean(), color='red', ls='--', lw=1, label=f'μ={d.mean():.3f}')
    ax.axvline(d.median(), color='green', ls=':', lw=1, label=f'Med={d.median():.3f}')
    ax.set_xlabel(titles[i], fontproperties=zh_font, fontsize=9)
    ax.set_ylabel('频数', fontproperties=zh_font, fontsize=9)
    ax.legend(prop=zh_font, fontsize=7)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/figures/dist.png', dpi=150, bbox_inches='tight')
plt.close()
print("  [OK] 分布图")

# 国别分布
sup_ct_all = rel[rel['rel_type_broad']=='SUPPLIER']['counterparty_country'].apply(cg).value_counts()
comp_ct_all = rel[rel['rel_type_broad']=='COMPETITOR']['counterparty_country'].apply(cg).value_counts()

fig, axes = plt.subplots(1,2,figsize=(16,7))
for ax, ct, label in [(axes[0], sup_ct_all, '供应商'), (axes[1], comp_ct_all, '竞争对手')]:
    t15 = ct.head(15)
    ax.barh(range(len(t15)), t15.values, color='steelblue', alpha=0.7)
    ax.set_yticks(range(len(t15))); ax.set_yticklabels(t15.index, fontproperties=zh_font, fontsize=9)
    ax.set_xlabel('数量', fontproperties=zh_font, fontsize=11)
    ax.set_title(f'{label}国别分布', fontproperties=zh_font, fontsize=13)
    ax.invert_yaxis()
    mx = max(t15.values)
    for i,(v,p) in enumerate(zip(t15.values, (t15/t15.sum()*100).values)):
        ax.text(v+mx*0.01, i, f'{p:.1f}%', va='center', fontsize=8)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/figures/country.png', dpi=150, bbox_inches='tight')
plt.close()
print("  [OK] 国别图")

# 趋势
fig, axes = plt.subplots(1,2,figsize=(14,5))
for ax, col, label in [(axes[0],'sup_breadth','供应商'),(axes[1],'comp_breadth','竞争对手')]:
    y = panel.groupby('year')[col].mean()
    ax.plot(y.index, y.values, 'o-', color='steelblue')
    ax.set_xlabel('年份', fontproperties=zh_font, fontsize=11)
    ax.set_ylabel(f'{label}广度(均值)', fontproperties=zh_font, fontsize=11)
    ax.set_title(f'{label}广度趋势', fontproperties=zh_font, fontsize=13)
    ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/figures/trend.png', dpi=150, bbox_inches='tight')
plt.close()
print("  [OK] 趋势图")

# 散点图
fig, axes = plt.subplots(1,2,figsize=(14,6))
s1 = panel[['sup_breadth','roa_w']].dropna().sample(min(10000,len(panel)))
axes[0].scatter(s1['sup_breadth'], s1['roa_w'], alpha=0.3, s=1, c='steelblue')
axes[0].set_xlabel('供应商广度', fontproperties=zh_font)
axes[0].set_ylabel('ROA', fontproperties=zh_font)
s2 = panel[['comp_breadth','roa_w']].dropna().sample(min(10000,len(panel)))
axes[1].scatter(s2['comp_breadth'], s2['roa_w'], alpha=0.3, s=1, c='coral')
axes[1].set_xlabel('竞争对手广度', fontproperties=zh_font)
axes[1].set_ylabel('ROA', fontproperties=zh_font)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/figures/scatter.png', dpi=150, bbox_inches='tight')
plt.close()
print("  [OK] 散点图")

# 热力图
fig, ax = plt.subplots(figsize=(10,8))
im = ax.imshow(corr.values, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
ax.set_xticks(range(len(corr))); ax.set_yticks(range(len(corr)))
ax.set_xticklabels(titles, fontproperties=zh_font, fontsize=7, rotation=45, ha='right')
ax.set_yticklabels(titles, fontproperties=zh_font, fontsize=7)
for i in range(len(corr)):
    for j in range(len(corr)):
        ax.text(j,i,f'{corr.values[i,j]:.2f}', ha='center',va='center',fontsize=7)
plt.colorbar(im, ax=ax, shrink=0.8)
plt.title('相关性矩阵', fontproperties=zh_font, fontsize=14)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/figures/heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print("  [OK] 热力图")

print(f"\n完成! 输出: {OUT_DIR}/")
