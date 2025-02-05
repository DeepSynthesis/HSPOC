#!/usr/bin/python
# -*- coding: utf-8 -*-


# 将.xyz文件转化为MOPAC程序的输入文件
def GetMop(xyz_file, outfile, cmd):

    with open(outfile, "w") as mopacfile:
        mopacfile.write(cmd + "\n")
        with open(xyz_file) as xyzfile:
            lines = xyzfile.readlines()
            for line in lines:
                mopacfile.write(line)


# if __name__ == "__main__":

#     xyz_files = [""]
#     outfiles = [""]

#     command = "PM6-DH+ OPT "
#     Add_command = ""

#     for i in range(len(xyz_files)):
#         GetMop(xyz_files[i], outfiles[i], command + Add_command)

#     mol = Chem.MolFromSmiles(smi)
#     mol = Chem.AddHs(mol)
#     atoms = mol.GetAtoms()
#     charge = sum([atom.GetFormalCharge() for atom in atoms])
#     if charge == 0:
#         Add_command = ""
#     else:
#         Add_command = "CHARGE=" + str(charge)
