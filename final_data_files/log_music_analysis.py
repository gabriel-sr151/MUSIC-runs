# script to extract the cells where questrevert is activated from the log_music files 
import numpy as np
from numpy import *
from os import path
import re
home = path.expanduser("~")



def find_warning(filename):

    tau_pattern = r'Done time step \d+/\d+ tau = ([\d.]+) fm/c'
    warning_pattern = r'ieta = (\d+), ix = (\d+), iy = (\d+)'

    warned_cells = []
    current_tau = None

    with open(filename, 'r') as file:

        for line in file:

            tau_match = re.search(tau_pattern, line)

            if tau_match: 

                current_tau = float(tau_match.group(1))

                continue

            warning_match = re.search(warning_pattern, line)

            if warning_match and current_tau is not None:

                ieta, ix, iy = warning_match.groups()

                warned_cells.append([current_tau, ieta, ix, iy])

    return warned_cells


working_folder = path.join(home, "MUSIC/final_data_files/acausality-w-shear/run6-echo+v2")


if __name__ == '__main__':

    filename = path.join(working_folder, "log_music.txt")
    
    #'./acausality-w-shear/run6-echo+v2/log_music.txt'

    extracted_data = find_warning(filename)

    for entry in extracted_data:

        print(f"tau={entry[0]} fm/c, ieta={entry[1]}, ix={entry[2]}, iy={entry[3]}")            


