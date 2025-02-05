#!/usr/bin/python
# -*- coding: utf-8 -*-

#由MOPAC输出文件.out获得分子的xyz

#输入MOPAC输出文件路径
#输入待计算分子的SMILES

#输出pandas DataFrame格式
##每行数据index为SMILES_xyz_AtomIndex
##列索引为SMILES，MOPAC_OutFileName，Atom_Index，x，y，z

import numpy as np
import pandas as pd
from rdkit import Chem

def L101(MOPAC_OutFileName,SMILES):
	
	df=pd.DataFrame({
		'SMILES':[],
		'MOPAC_OutFileName':[],
		'Atom_Index':[],
		'From':[],
		'x':[],
		'y':[],
		'z':[]
		}).astype('object')
									
	mol=Chem.MolFromSmiles(SMILES)
	mol=Chem.AddHs(mol)
	atoms=mol.GetAtoms()
	
	for atom in atoms:
		
		Atom_Index=atom.GetIdx()
		
		with open(MOPAC_OutFileName) as xyz:
			line=xyz.readline()
			while line!='                             CARTESIAN COORDINATES\n':
				line=xyz.readline()
			line=xyz.readline()
			for i in range(Atom_Index):
				xyz.readline()
			xyz.read(12)
			x=xyz.read(16)
			y=xyz.read(16)
			z=xyz.read(16)
			
		df.loc[str(SMILES)+'_xyz_'+str(Atom_Index),['SMILES','MOPAC_OutFileName','Atom_Index','From','x','y','z']]=[SMILES,MOPAC_OutFileName,Atom_Index,'MOPAC_out',float(x),float(y),float(z)]
	
	return df
	
	
if __name__ == '__main__':
	
	Data=pd.read_csv('D:/LSY/HSPOC-outSample/HSPOC-outSample.csv')
	MOPAC_outPATH='D:/LSY/HSPOC-outSample/'											
	MOPAC_outfile='mopac_out/'
	
	df_list=[]
	for smi in set(Data['SMILES'].values.tolist()):
	
		print('\nGenerate:\t\t'+smi)
		df=L101(MOPAC_outPATH+MOPAC_outfile+Data[Data['SMILES']==smi]['filename'].values.tolist()[0]+'.out',smi)
		df_list.append(df)
		print('Done:\t\t'+smi+'\n')
		
	DF=pd.concat(df_list,axis=0,join='outer')
	DF.to_csv('D:/LSY/xyz_HSPOC-outSample.csv',index=0)