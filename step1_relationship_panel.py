"""第一步：构建关系面板 (供应商/竞争对手) - 内存优化版"""
import pandas as pd, numpy as np, os, gc, re, warnings
warnings.filterwarnings('ignore')
DATA_DIR='/Users/deesanyu/pythonproject/deesan_supply_0506v4/data'
OUT_DIR='/Users/deesanyu/pythonproject/deesan_supply_0506v4/output'

# 1. 加载关系数据
print("1. 加载关系数据")
rel=pd.read_csv(f'{DATA_DIR}/crossborder_relationships_mirrored_2010_2020.csv', low_memory=False,
                usecols=['cn_company_isin','counterparty_isin','counterparty_country',
                         'counterparty_region','rel_type_broad','start_','end_'])
rel['end_']=rel['end_'].str.replace('4000-01-01','2020-12-31',regex=False)
rel['start_']=pd.to_datetime(rel['start_'],format='%Y-%m-%d',errors='coerce')
rel['end_']=pd.to_datetime(rel['end_'],format='%Y-%m-%d %H:%M:%S',errors='coerce')
rel=rel[rel['rel_type_broad'].isin(['SUPPLIER','COMPETITOR'])].copy()
ms=(rel['start_'].dt.year-2010)*12+rel['start_'].dt.month-1
me=(rel['end_'].dt.year-2010)*12+rel['end_'].dt.month-1
rel['ms'],rel['me']=ms.clip(0,131),me.clip(0,131)
rel=rel[rel['ms']<=rel['me']].copy()
print(f"  关系: {len(rel):,}, 企业: {rel['cn_company_isin'].nunique():,}")

# 2. 快速展开 (numpy)
print("2. 展开")
n=(rel['me']-rel['ms']+1).values.astype(int)
print(f"  总: {n.sum():,}")
idx=np.repeat(rel.index.values,n)
exp=rel.loc[idx,['cn_company_isin','counterparty_isin','counterparty_country','rel_type_broad']].copy()
exp['mi']=rel.loc[idx,'ms'].values+np.concatenate([np.arange(x) for x in n])
del rel; gc.collect()
print(f"  展开: {len(exp):,}")

# 3. 聚合
print("3. 聚合")
# 4. 国别
def cg(c):
    c=str(c)
    for k in ['美国','中国','日本','韩国','德国','英国','法国','新加坡','香港','台湾',
              '印度','荷兰','瑞士','瑞典','加拿大','澳大利亚','开曼','百慕大','意大利',
              '西班牙','巴西','俄罗斯']:
        if k in c: return k
    if '未知' in c: return '未知'
    return '其他'

exp['cg']=exp['counterparty_country'].apply(cg)
top_ct=['美国','中国','日本','韩国','德国','英国','法国','新加坡','香港','台湾','其他']

sup=exp[exp['rel_type_broad']=='SUPPLIER']; comp=exp[exp['rel_type_broad']=='COMPETITOR']
sup_cnt=sup.groupby(['cn_company_isin','mi']).size().reset_index(name='sup_raw')
comp_cnt=comp.groupby(['cn_company_isin','mi']).size().reset_index(name='comp_raw')

# 5. 完整面板
print("4. 构建面板")
firms=exp['cn_company_isin'].unique()
grid=pd.MultiIndex.from_product([firms,np.arange(132)],names=['ISIN','mi']).to_frame(index=False)
grid=grid.merge(sup_cnt.rename(columns={'cn_company_isin':'ISIN'}),on=['ISIN','mi'],how='left')
grid['sup_raw']=grid['sup_raw'].fillna(0).astype(int)
grid=grid.merge(comp_cnt.rename(columns={'cn_company_isin':'ISIN'}),on=['ISIN','mi'],how='left')
grid['comp_raw']=grid['comp_raw'].fillna(0).astype(int)

# 国别合并 (简洁版)
for prefix,sub in [('sup',sup),('comp',comp)]:
    ct=sub.groupby(['cn_company_isin','mi','cg']).size().reset_index(name='n')
    ct['pct']=ct.groupby(['cn_company_isin','mi'])['n'].transform(lambda x:x/x.sum()*100)
    for c in top_ct:
        v=ct[ct['cg']==c][['cn_company_isin','mi','pct']].rename(columns={'cn_company_isin':'ISIN','pct':f'{prefix}_ct_{c}'})
        grid=grid.merge(v,on=['ISIN','mi'],how='left')
        grid[f'{prefix}_ct_{c}']=grid[f'{prefix}_ct_{c}'].fillna(0)
del sup,comp; gc.collect()

# 6. 行业 (在删除 exp 之前做)
print("5. 行业")
ar=pd.read_excel(f'{DATA_DIR}/china-quarterly-assets.xlsx',header=None)
h=ar.iloc[4]; ic=list(h).index('FE Isin')
ind=ar.iloc[6:,[10,11,ic]].copy()
ind.columns=['L1','L4','ISIN']; ind=ind.dropna(subset=['ISIN'])
ind=ind[ind['L1']!='@NA'].drop_duplicates('ISIN')
del ar; gc.collect()

# 供应商行业
si=exp[exp['rel_type_broad']=='SUPPLIER'].merge(ind.rename(columns={'ISIN':'counterparty_isin'}),on='counterparty_isin',how='left')
for lev,col in [('L1','sup_ind_l1'),('L4','sup_ind_l4')]:
    g=si.dropna(subset=[lev]).groupby(['cn_company_isin','mi'])[lev].nunique().reset_index(name=col)
    grid=grid.merge(g.rename(columns={'cn_company_isin':'ISIN'}),on=['ISIN','mi'],how='left')
    grid[col]=grid[col].fillna(0).astype(int)
del si; gc.collect()

# 竞争对手行业
ci=exp[exp['rel_type_broad']=='COMPETITOR'].merge(ind.rename(columns={'ISIN':'counterparty_isin'}),on='counterparty_isin',how='left')
for lev,col in [('L1','comp_ind_l1'),('L4','comp_ind_l4')]:
    g=ci.dropna(subset=[lev]).groupby(['cn_company_isin','mi'])[lev].nunique().reset_index(name=col)
    grid=grid.merge(g.rename(columns={'cn_company_isin':'ISIN'}),on=['ISIN','mi'],how='left')
    grid[col]=grid[col].fillna(0).astype(int)
del ci,exp,ind; gc.collect()

# 保存
grid.to_csv(f'{OUT_DIR}/panel_relationship.csv',index=False,encoding='utf-8-sig')
print(f"关系面板保存: {grid.shape}, 内存约{grid.memory_usage(deep=True).sum()/1024/1024:.0f}MB")
