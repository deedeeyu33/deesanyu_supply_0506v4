"""生成描述性分析详细报告"""
import pandas as pd, numpy as np

p = pd.read_csv('/Users/deesanyu/pythonproject/deesan_supply_0506v4/output/firm_monthly_panel.csv')

print("="*70)
print("一、核心变量描述性统计")
print("="*70)
svars = ['sup_raw','comp_raw','sup_ind_l4','comp_ind_l4',
         'sup_breadth','comp_breadth','sup_ind_div','comp_ind_div',
         'roa','roa_w','roe','roe_w','size','lev','growth','growth_w']
for v in svars:
    d = p[v].dropna()
    print(f'{v:15s} N={len(d):>8,} Mean={d.mean():>8.3f} SD={d.std():>8.3f} '
          f'Min={d.min():>10.3f} P25={d.quantile(.25):>8.3f} P50={d.quantile(.5):>8.3f} '
          f'P75={d.quantile(.75):>8.3f} Max={d.max():>10.3f}')

print("\n"+"="*70)
print("二、缩尾效果")
print("="*70)
for o,w in [('roa','roa_w'),('roe','roe_w')]:
    od=p[o].dropna(); wd=p[w].dropna()
    print(f'{o}:  [{od.min():.2f}, {od.max():.2f}] -> {w}: [{wd.min():.2f}, {wd.max():.2f}]')

print("\n"+"="*70)
print("三、国别分布")
print("="*70)
rel=pd.read_csv('/Users/deesanyu/pythonproject/deesan_supply_0506v4/data/crossborder_relationships_mirrored_2010_2020.csv',
                low_memory=False, usecols=['counterparty_country','rel_type_broad'])
rel=rel[rel['rel_type_broad'].isin(['SUPPLIER','COMPETITOR'])]
def cg(c):
    c=str(c)
    for k in ['美国','中国','日本','韩国','德国','英国','法国','新加坡','香港','台湾',
              '印度','荷兰','瑞士','瑞典','加拿大','澳大利亚','开曼','百慕大','意大利',
              '西班牙','巴西','俄罗斯']:
        if k in c: return k
    if '未知' in c: return '未知'
    return '其他'
rel['cg']=rel['counterparty_country'].apply(cg)
for rt,lb in [('SUPPLIER','供应商'),('COMPETITOR','竞争对手')]:
    d=rel[rel['rel_type_broad']==rt]['cg'].value_counts()
    t=d.sum()
    print(f'\n{lb} (共{t:,}条):')
    for k,v in d.head(12).items():
        print(f'  {k:12s} {v:>8,} ({v/t*100:.1f}%)')

print("\n"+"="*70)
print("四、关系覆盖情况")
print("="*70)
fs=p.groupby('ISIN')['sup_raw'].max(); fc=p.groupby('ISIN')['comp_raw'].max()
print(f'有过供应商的企业: {(fs>0).sum()}/{len(fs)} ({(fs>0).sum()/len(fs)*100:.1f}%)')
print(f'有过竞争对手的企业: {(fc>0).sum()}/{len(fc)} ({(fc>0).sum()/len(fc)*100:.1f}%)')
print(f'月度有供应商: {(p["sup_raw"]>0).sum()/len(p)*100:.1f}%')
print(f'月度有竞争对手: {(p["comp_raw"]>0).sum()/len(p)*100:.1f}%')
print(f'月度有供应商or竞争对手: {((p["sup_raw"]>0)|(p["comp_raw"]>0)).sum()/len(p)*100:.1f}%')

print("\n"+"="*70)
print("五、行业分布")
print("="*70)
ar=pd.read_excel('/Users/deesanyu/pythonproject/deesan_supply_0506v4/data/china-quarterly-assets.xlsx',header=None)
h=ar.iloc[4]; ic=list(h).index('FE Isin')
ind=ar.iloc[6:,[10,ic]].copy()
ind.columns=['Industry','ISIN']; ind=ind.dropna(subset=['ISIN'])
ind=ind[ind['Industry']!='@NA'].drop_duplicates('ISIN')
ins = ind[ind['ISIN'].isin(p['ISIN'].unique())]
print(ins['Industry'].value_counts().to_string())

print("\n"+"="*70)
print("六、相关性矩阵")
print("="*70)
sv2=['sup_breadth','comp_breadth','sup_ind_div','comp_ind_div','roa_w','roe_w','size','lev','growth_w']
corr=p[sv2].corr()
print(corr.round(3))

print("\n"+"="*70)
print("七、供应商广度分布")
print("="*70)
sb=p['sup_raw']
print(f'供应商数量分布:')
print(f'  0家: {(sb==0).sum()/len(sb)*100:.1f}%')
print(f'  1家: {(sb==1).sum()/len(sb)*100:.1f}%')
print(f'  2家: {(sb==2).sum()/len(sb)*100:.1f}%')
print(f'  3-5家: {((sb>=3)&(sb<=5)).sum()/len(sb)*100:.1f}%')
print(f'  6-10家: {((sb>=6)&(sb<=10)).sum()/len(sb)*100:.1f}%')
print(f'  10+家: {(sb>10).sum()/len(sb)*100:.1f}%')
print(f'  最大供应商数: {sb.max()}')

cb=p['comp_raw']
print(f'\n竞争对手数量分布:')
print(f'  0家: {(cb==0).sum()/len(cb)*100:.1f}%')
print(f'  1家: {(cb==1).sum()/len(cb)*100:.1f}%')
print(f'  2家: {(cb==2).sum()/len(cb)*100:.1f}%')
print(f'  3-5家: {((cb>=3)&(cb<=5)).sum()/len(cb)*100:.1f}%')
print(f'  6-10家: {((cb>=6)&(cb<=10)).sum()/len(cb)*100:.1f}%')
print(f'  10+家: {(cb>10).sum()/len(cb)*100:.1f}%')
print(f'  最大竞争对手数: {cb.max()}')
