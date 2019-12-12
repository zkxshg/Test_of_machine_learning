# ===========1. 合併list=============
l1 = [1, 2, 3]
l2 = ['a', 'b', 'c']
l3 = l1 + l2  # 運算符重載
l1.extend(l2)  # 覆蓋l1
l1[len(l1): len(l1)] = l2  # 插入指定位置

# ===========2. 數字運算=============
# 整除: //
10 // 3  # 3
10 / 3  # 3.3333333333333335
10. // 2  # 5.0
# 取模(mod): %
10 % 2  # 0
2 % 10  # 2  
10 % 2.2  # 1.1999999999999993
# 冪運算：**
2 ** 3  # 8
-2 ** 2  # -4
(-2) ** 2  # 4
# 十六進制
print(0xBA)  # 186
# 八進制
print(0o111)  # 73
# 二進制
print(0b111)  # 7

# ===========3. 複製list=============
# 錯誤複製：相同id
a = [1, 2, 3]
b = a
print(id(a) == id(b))  # True
# 切片複製
c = a[:]
print(id(a) == id(c))  # False
# 列表重構造
d = list(a)
print(id(a) == id(d))  # False
# 其他複製方法
e = a * 1
f = copy.copy(a)

# 高維數組完全複製方法
g = copy.deepcopy(a)

# ===========4. 讀取文件並識別行首是否為數字=============
# 一次性讀取
f = open(path)
f.readlines()
f.close()
# 逐行讀取
f = open(path)
f.readline()
f.readline()
f.close()
# 循環讀取
f = open(path)
for line in f:
  if line[0:2] == "EOF":
    break
  elif line[0].isdigit():
    loc = line.split()
    self.city_location.append([int(loc[1]), int(loc[2])])
