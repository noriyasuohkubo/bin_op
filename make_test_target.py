f = open('/home/reicou/tmp_ub2.txt', 'r')
strs = []
for s in f.readlines():
  tmp = s.strip()
  if tmp != "":
    strs.append(str(tmp))
print(set(strs))
f.close()