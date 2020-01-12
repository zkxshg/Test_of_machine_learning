import pandas as pd
import numpy as np

# 生成 6*4 矩陣
dates = pd.date_range('20190101', periods=6)
df = pd.DataFrame(np.arange(24).reshape((6, 4)), index=dates, columns=['A', 'B', 'C', 'D'])
print(df.columns)  # df 的屬性名

# 1.基於 column/變量名 選擇列
print(df.A)
print(df['B'])
# 選擇多列
cols = ['A', 'D']
print(df[cols])

# 2.基於 slice/index 選擇多行/列
print(df[2:5])
print(df[0:5])
print(df['B'][2:5])

# 3.基於ioc-標籤選擇數據
print(df.loc['20190104'])
print(df.loc['20190104':'20190106'])
print(df.loc[:,'A'])
print(df.loc[:,['A', 'D']])
print(df.loc['20190104':'20190106', ['A', 'D']])

# 4.基於iloc-位置選擇數據
print(df.iloc[4, 2])
print(df.iloc[2:5, 2])
print(df.iloc[[1, 3, 5], 2:4])

# 5.基於ix-混合選擇數據
print(df.ix[1:4, ['A','D']])

# 6.基於Boolean indexing進行選擇
print(df[df.B < 10])
