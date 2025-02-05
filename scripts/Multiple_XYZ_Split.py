#!/usr/bin/python3

import os, sys

current_path = os.getcwd()
print(
    "~~~~ Working at the current working directory... ~~~~\n{0}\n".format(current_path)
)
input_file_name = sys.argv[1]

print("The filename of *.xyz file to be splitted is:\n{0}\n".format(input_file_name))
with open(input_file_name, "r") as inputs:
    input_content = inputs.readlines()
    total_atom_number = int(input_content[0])
    total_lines = len(input_content)
    total_frame_number = int(total_lines / (total_atom_number + 2))

    print("----------  information of the {0} file  ----------".format(input_file_name))
    print("=> Total frame number: {0:21d}".format(total_frame_number))
    print("=> Total atom number in each frame: {0:8d}\n".format(total_atom_number))
    input("Press any key to start the splitting process... ^_^")

    xyz_container = [[] for i in range(total_frame_number)]
    line_flags = [i * (total_atom_number + 2) + 1 for i in range(0, total_frame_number)]
    frame = 0
    for i in line_flags:
        start_line = i + 2
        end_line = start_line + total_atom_number - 1
        for index in range(start_line - 1, end_line):
            xyz_container[frame].append(input_content[index])
        frame += 1

for i in range(total_frame_number):
    output_filename = input_file_name[:-4] + "_{0}.xyz".format(str(i + 1).zfill(4))
    output_foldername = input_file_name[:-4] + "_{0}".format(str(i + 1).zfill(4))
    os.mkdir(current_path + os.path.sep + output_foldername)
    outputs = open(output_foldername + os.path.sep + output_filename, "w")
    outputs.write(str(total_atom_number) + "\n")
    outputs.write(output_filename[:-4] + "\n")
    outputs.writelines(xyz_container[i])
    outputs.close()

print(
    "\n######  Total {0} *.xyz files have been generated  ######".format(
        total_frame_number
    )
)
