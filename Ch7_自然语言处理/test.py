# -*- coding: cp936 -*-
import trees
import json
 
fr = open(r'ID3data.txt')
 
listWm = [inst.strip().split('\t') for inst in fr.readlines()]
labels = ['��һ��', '�ڶ���', '������', '������']
Trees = trees.createTree(listWm, labels)
 
print(Trees, encoding="cp936", ensure_ascii=False)
