#!/usr/bin/python
# -*- coding: utf-8 -*-

from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem import rdDepictor

rdDepictor.SetPreferCoordGen(True)
from rdkit import Chem


# 通过RDKit生成带索引的分子图示，此索引与H-SPOC中mopac模式下的分子索引一致
def Get_svg(PATH, name, mol):

    d2d = rdMolDraw2D.MolDraw2DSVG(350, 300)
    d2d.drawOptions().addAtomIndices = True
    d2d.DrawMolecule(mol)
    d2d.FinishDrawing()
    with open(PATH + "/" + name + ".svg", "w") as f:
        f.write(d2d.GetDrawingText())
    print("DONE:" + name)


####
if __name__ == "__main__":

    PATH = "./Pred/"
    names = "test"

    Smi_List = [
        r"[O-]S(N[C@]1(CCC2=CC=CC=C2)C[S+](C1)[O-])(=[N+](C)C)=O",
    ]

    for i in range(len(Smi_List)):

        mol = Chem.MolFromSmiles(Smi_List[i])
        mol = Chem.AddHs(mol)

        Get_svg(PATH, names + str(i), mol)
