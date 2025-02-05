#!/usr/bin/python
# -*- coding: utf-8 -*-

#以中心原子为第0层，沿共价键距离依次展开，生成每一层的原子索引
##包含H原子

#输入rdkit mol
#输入中心原子Index
#输入展开层数layer

#输出以set为元素的list
##list格式为[set(第零层原子索引(即中心原子)),set([第一层原子索引]),set([第二层原子索引]),...]
##列表索引即为层数

from rdkit import Chem

def L405m(mol,Index,layer):
	
	atoms=mol.GetAtoms()
	
	ECSet_List=[set([Index])]
	
	for i in range(layer):
		Connect_Set=set([])
		for x in ECSet_List[i]:	
			neighbors=atoms[int(x)].GetNeighbors()	
			for atom in neighbors:
				Connect_Set.add(atom.GetIdx())
		ECSet_List.append(Connect_Set)
		
	for i in range(layer+1):
		for x in ECSet_List[:i]:
			ECSet_List[i]=ECSet_List[i]-x
		
	return ECSet_List