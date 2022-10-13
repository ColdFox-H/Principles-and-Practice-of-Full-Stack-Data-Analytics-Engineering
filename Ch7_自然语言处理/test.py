# -*- coding: cp936 -*-
import trees
import json
 
fr = open(r'ID3data.txt')
 
listWm = [inst.strip().split('\t') for inst in fr.readlines()]
labels = ['第一列', '第二列', '第三列', '第四列']
Trees = trees.createTree(listWm, labels)
 
print(Trees, encoding="cp936", ensure_ascii=False)
