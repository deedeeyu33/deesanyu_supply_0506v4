"""第二步：合并财务数据 (逐个文件处理, 内存优化)"""
import pandas as pd, numpy as np, matplotlib, os, re, gc, warnings
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
warnings.filterwarnings('ignore')

DATA_DIR='/Users/deesanyu/pythonproject/deesan_supply_0506v4/data'
OUT_DIR='/Users/deesanyu/pythonproject/deesan_supply_0506v4/output'
FONT_PATH='/Users/deesanyu/pythonproject/deesan_supply_0506v4/simhei.ttf'
os.makedirs(f'{OUT_DIR}/figures',exist_ok=True)
zh_font=FontProperties(fname=FONT_PATH)
plt.rcParams['axes.unicode_minus']=False

def load_q(fp,vn):
    """加载季度数据并展开到月度, 返回 (FE Isin, mi, value)"""
    raw=pd.read_excel(fp,header=None); h=raw.iloc[4]
    ip=next(i for i,v in enumerate(h) if pd.notna(v) and 'Isin' in str(v))
    cm={}
    for i in range(14,ip):
        v=str(h[i]) if pd.notna(h[i]) else ''
        m=re.search(r'(201[0-9]|2020).*?Q(\d)',v)
        if m: cm[v]=f"{m.group(1)}Q{m.group(2)}"
    d=raw.iloc[6:].copy(); d.columns=list(h); d=d[d['Symbol'].notna()]
    ids=['Symbol','Name']; has_isin='FE Isin' in d.columns
    if has_isin: ids.append('FE Isin')
    melt=d.melt(id_vars=ids,value_vars=list(cm.keys()),var_name='qn',value_name=vn)
    del d,raw; gc.collect()
    melt['qc']=melt['qn'].map(cm)
    melt[vn]=pd.to_numeric(melt[vn],errors='coerce')
    def q2i(q):
        m=re.search(r'(201[0-9]|2020)Q(\d)',str(q))
        if m: y,q_=int(m.group(1)),int(m.group(2)); return (y-2010)*4+(q_-1)
        return -1
    melt['qi']=melt['qc'].apply(q2i)
    melt=melt[(melt['qi']>=0)&(melt['qi']<=43)].dropna(subset=[vn])
    # 展开到月度
    melt=melt.loc[melt.index.repeat(3)].copy()
    melt['mo']=melt.groupby(level=0).cumcount()
    melt['mi']=melt['qi']*3+melt['mo']
    result=melt[melt['mi'].between(0,131)][['FE Isin','mi',vn]]
    del melt; gc.collect()
    return result

# 1. 加载关系面板
print("1. 加载关系面板")
panel=pd.read_csv(f'{OUT_DIR}/panel_relationship.csv')
print(f"  关系面板: {panel.shape}, 企业: {panel['ISIN'].nunique()}")

# 2. 逐个处理财务文件并合并到面板
print("2. 逐个处理财务数据")
for fp,vn in [
    (f'{DATA_DIR}/china-quarterly-assets.xlsx','assets'),
    (f'{DATA_DIR}/china-quarterly-roa.xlsx','roa'),
    (f'{DATA_DIR}/china-quarterly-roe.xlsx','roe'),
    (f'{DATA_DIR}/china-quarterly-debt.xlsx','debt'),
]:
    print(f"  {vn}...",end=' ',flush=True)
    fin=load_q(fp,vn)
    panel=panel.merge(fin.rename(columns={'FE Isin':'ISIN'}),on=['ISIN','mi'],how='left')
    del fin; gc.collect()
    print(f"done ({panel.shape})")

# 3. 单独处理sales (需要用于growth计算)
print("  sales...",end=' ',flush=True)
sales=load_q(f'{DATA_DIR}/china-quarterly-sales.xlsx','sales')
panel=panel.merge(sales.rename(columns={'FE Isin':'ISIN'}),on=['ISIN','mi'],how='left')
del sales; gc.collect()
print(f"done ({panel.shape})")

print(f"  面板: {panel.shape}")

# 4. 衍生变量
print("3. 衍生变量")
panel['sup_breadth']=np.log1p(panel['sup_raw'])
panel['comp_breadth']=np.log1p(panel['comp_raw'])
panel['sup_ind_div']=np.log1p(panel['sup_ind_l4'])
panel['comp_ind_div']=np.log1p(panel['comp_ind_l4'])

def w(s,l=0.01,u=0.99):
    lo,hi=s.quantile(l),s.quantile(u)
    return s.clip(lo,hi)

panel['roa_w']=w(panel['roa'])
panel['roe_w']=w(panel['roe'])
panel['size']=np.log(panel['assets'].clip(lower=1))
panel['lev']=(panel['debt']/panel['assets']).replace([np.inf,-np.inf],np.nan)

# 销售增长率: 用季度原始数据算YoY, 再展开
print("  增长率...")
s2=pd.read_excel(f'{DATA_DIR}/china-quarterly-sales.xlsx',header=None)
h2=s2.iloc[4]; ip2=next(i for i,v in enumerate(h2) if pd.notna(v) and 'Isin' in str(v))
cm2={}
for i in range(14,ip2):
    v=str(h2[i]) if pd.notna(h2[i]) else ''
    m=re.search(r'(201[0-9]|2020).*?Q(\d)',v)
    if m: cm2[v]=f"{m.group(1)}Q{m.group(2)}"
d2=s2.iloc[6:].copy(); d2.columns=list(h2); d2=d2[d2['Symbol'].notna()]
m2=d2.melt(id_vars=['Symbol','Name','FE Isin'],value_vars=list(cm2.keys()),var_name='qn',value_name='sales')
del d2,s2; gc.collect()
m2['qc']=m2['qn'].map(cm2)
m2['sales']=pd.to_numeric(m2['sales'],errors='coerce')
def q2i(q):
    m=re.search(r'(201[0-9]|2020)Q(\d)',str(q))
    if m: return (int(m.group(1))-2010)*4+(int(m.group(2))-1)
    return -1
m2['qi']=m2['qc'].apply(q2i)
m2=m2[(m2['qi']>=0)&(m2['qi']<=43)].dropna(subset=['sales'])
m2=m2.sort_values(['FE Isin','qi'])
m2['s_l4']=m2.groupby('FE Isin')['sales'].shift(4)
m2['growth']=(m2['sales']-m2['s_l4'])/m2['s_l4']
# 展开到月度
m2['mi']=m2['qi']*3
m2x=m2.loc[m2.index.repeat(3)].copy()
m2x['mo']=m2x.groupby(level=0).cumcount(); m2x['mi']=m2x['mi']+m2x['mo']
m2x=m2x[m2x['mi'].between(0,131)]
panel=panel.merge(m2x[['FE Isin','mi','growth']].rename(columns={'FE Isin':'ISIN'}),on=['ISIN','mi'],how='left')
del m2,m2x; gc.collect()
panel['growth']=panel['growth'].replace([np.inf,-np.inf],np.nan)
panel['growth_w']=w(panel['growth'])

# 月份
all_months = pd.date_range('2010-01-01', periods=132, freq='MS')
panel['month'] = all_months[panel['mi'].values]
panel['year']=panel['month'].dt.year

# 5. 保存
print("4. 保存")
panel.to_csv(f'{OUT_DIR}/firm_monthly_panel.csv',index=False,encoding='utf-8-sig')
print(f"  面板: {panel.shape}")

# 6. 描述性统计
print("5. 描述性统计")
svars=['sup_breadth','comp_breadth','sup_ind_div','comp_ind_div',
       'roa_w','roe_w','size','lev','growth_w']
desc=panel[svars].describe().T[['count','mean','std','min','25%','50%','75%','max']]
desc.columns=['N','Mean','SD','Min','P25','P50','P75','Max']
desc.to_csv(f'{OUT_DIR}/desc_core.csv',encoding='utf-8-sig')
print(desc.round(3))
print(f"\n企业: {panel['ISIN'].nunique():,} | 观测: {len(panel):,} | "
      f"年份: {panel['year'].min()}-{panel['year'].max()}")

panel.groupby('year').agg(**{'企业数':('ISIN','nunique'),'观测数':('ISIN','count')}
).to_csv(f'{OUT_DIR}/year_dist.csv',encoding='utf-8-sig')

corr=panel[svars].corr()
corr.to_csv(f'{OUT_DIR}/corr_matrix.csv',encoding='utf-8-sig')
print(corr.round(3))

# 7. 可视化
print("6. 可视化")
titles=['供应商广度','竞争对手广度','供应商行业多样性','竞争对手行业多样性',
        'ROA(缩尾)','ROE(缩尾)','企业规模','资产负债率','增长率(缩尾)']

# 7.1 分布
fig,axs=plt.subplots(3,3,figsize=(15,12))
for i,v in enumerate(svars):
    ax=axs[i//3,i%3]; d=panel[v].dropna()
    lo,hi=d.quantile(.01),d.quantile(.99)
    if hi>lo: d=d.clip(lo,hi)
    ax.hist(d,bins=50,ec='white',alpha=.7,color='steelblue')
    ax.axvline(d.mean(),color='red',ls='--',lw=1,label=f'μ={d.mean():.3f}')
    ax.axvline(d.median(),color='green',ls=':',lw=1,label=f'M={d.median():.3f}')
    ax.set_xlabel(titles[i],fontproperties=zh_font,fontsize=9)
    ax.set_ylabel('频数',fontproperties=zh_font,fontsize=9)
    ax.legend(prop=zh_font,fontsize=7)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/figures/dist.png',dpi=150,bbox_inches='tight'); plt.close()
print("  [OK] 分布")

# 7.2 国别
rel=pd.read_csv(f'{DATA_DIR}/crossborder_relationships_mirrored_2010_2020.csv',low_memory=False,
                usecols=['counterparty_country','rel_type_broad'])
rel=rel[rel['rel_type_broad'].isin(['SUPPLIER','COMPETITOR'])]
def cg(c):
    c=str(c)
    for k in ['美国','中国','日本','韩国','德国','英国','法国','新加坡','香港','台湾',
              '印度','荷兰','瑞士','瑞典','加拿大','澳大利亚','开曼','百慕大','意大利','西班牙','巴西','俄罗斯']:
        if k in c: return k
    if '未知' in c: return '未知'
    return '其他'
rel['cg']=rel['counterparty_country'].apply(cg)
fig,axs=plt.subplots(1,2,figsize=(16,7))
for ax,lb,rt in [(axs[0],'供应商','SUPPLIER'),(axs[1],'竞争对手','COMPETITOR')]:
    ct=rel[rel['rel_type_broad']==rt]['cg'].value_counts().head(15)
    ax.barh(range(len(ct)),ct.values,color='steelblue',alpha=.7)
    ax.set_yticks(range(len(ct))); ax.set_yticklabels(ct.index,fontproperties=zh_font,fontsize=9)
    ax.set_xlabel('数量',fontproperties=zh_font,fontsize=11)
    ax.set_title(f'{lb}国别分布',fontproperties=zh_font,fontsize=13)
    ax.invert_yaxis(); mx=max(ct.values)
    for i,(v,p) in enumerate(zip(ct.values,(ct/ct.sum()*100).values)):
        ax.text(v+mx*0.01,i,f'{p:.1f}%',va='center',fontsize=8)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/figures/country.png',dpi=150,bbox_inches='tight'); plt.close()
del rel; gc.collect()
print("  [OK] 国别")

# 7.3 趋势
fig,axs=plt.subplots(1,2,figsize=(14,5))
for ax,col,lb in [(axs[0],'sup_breadth','供应商'),(axs[1],'comp_breadth','竞争对手')]:
    y=panel.groupby('year')[col].mean()
    ax.plot(y.index,y.values,'o-',color='steelblue')
    ax.set_xlabel('年份',fontproperties=zh_font); ax.set_ylabel(f'{lb}广度(均值)',fontproperties=zh_font)
    ax.set_title(f'{lb}广度趋势',fontproperties=zh_font,fontsize=13); ax.grid(True,alpha=.3)
plt.tight_layout(); plt.savefig(f'{OUT_DIR}/figures/trend.png',dpi=150,bbox_inches='tight'); plt.close()
print("  [OK] 趋势")

# 7.4 散点
fig,axs=plt.subplots(1,2,figsize=(14,6))
s1=panel[['sup_breadth','roa_w']].dropna().sample(min(10000,len(panel)))
axs[0].scatter(s1['sup_breadth'],s1['roa_w'],alpha=.3,s=1,c='steelblue')
axs[0].set_xlabel('供应商广度',fontproperties=zh_font); axs[0].set_ylabel('ROA',fontproperties=zh_font)
s2=panel[['comp_breadth','roa_w']].dropna().sample(min(10000,len(panel)))
axs[1].scatter(s2['comp_breadth'],s2['roa_w'],alpha=.3,s=1,c='coral')
axs[1].set_xlabel('竞争对手广度',fontproperties=zh_font); axs[1].set_ylabel('ROA',fontproperties=zh_font)
plt.tight_layout(); plt.savefig(f'{OUT_DIR}/figures/scatter.png',dpi=150,bbox_inches='tight'); plt.close()
print("  [OK] 散点")

# 7.5 热力图
fig,ax=plt.subplots(figsize=(10,8))
im=ax.imshow(corr.values,cmap='RdBu_r',vmin=-1,vmax=1,aspect='auto')
ax.set_xticks(range(len(corr))); ax.set_yticks(range(len(corr)))
ax.set_xticklabels(titles,fontproperties=zh_font,fontsize=7,rotation=45,ha='right')
ax.set_yticklabels(titles,fontproperties=zh_font,fontsize=7)
for i in range(len(corr)):
    for j in range(len(corr)):
        ax.text(j,i,f'{corr.values[i,j]:.2f}',ha='center',va='center',fontsize=7)
plt.colorbar(im,ax=ax,shrink=.8)
plt.title('相关性矩阵',fontproperties=zh_font,fontsize=14)
plt.tight_layout(); plt.savefig(f'{OUT_DIR}/figures/heatmap.png',dpi=150,bbox_inches='tight'); plt.close()
print("  [OK] 热力图")

print(f"\n完成! 所有输出在 {OUT_DIR}/")
