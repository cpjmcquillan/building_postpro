import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys

from csv_to_pandas import csv_to_pandas

folder_path = sys.argv[1]

probes = ['windward', 'roof', 'wake', 'ground']
colours = ['b', 'r', 'g', 'c', 'm', 'y']

for probe in probes:
    data = csv_to_pandas(folder_path, 'cp_{}.csv'.format(probe))

    direction_headers = {column: "{}_y/h".format(column.split('_')[1].split(':')[0])
                         for column in data.columns if 'Direction' in column}
    cp_headers = {column: '{}_Cp'.format(column.split('_')[1].split(':')[0])
                  for column in data.columns if 'Pressure Coefficient' in column}
    data.rename(columns=direction_headers, inplace=True)
    data.rename(columns=cp_headers, inplace=True)

    splitter = '=' if probe is 'ground' else 'b'

    locations = [location.split('_')[0] for location in cp_headers.values()]
    locations.sort(key=lambda x: x.split(splitter)[1])
    locations.sort(key=lambda x: len(x))

    for location in locations:
        row = int(np.ceil(float(location.split(splitter)[1]) / 2.0))
        colour = colours[row - 1]
        linestyle = 'solid' if probe is 'ground' \
            else 'dashed' if not float(location.split(splitter)[1]) % 2 else 'solid'
        plt.plot(data['{}_y/h'.format(location)], data['{}_Cp'.format(location)],
                 label=location, c=colour, ls=linestyle)

    plt.legend(loc='best', ncol=3, fontsize='small')
    plt.title(probe)
    plt.ylabel('Cp')
    plt.xlabel('y/h')
    plt.savefig(os.path.join(folder_path, 'plots', 'cp_{}.png'.format(probe)))
    plt.gcf().clear()
