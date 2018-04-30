import matplotlib.pyplot as plt
import pandas as pd
import os
import sys

folder_path = sys.argv[1]

domain_comparison = os.path.join(folder_path, 'domain_comparison.xlsx')
writer = pd.ExcelWriter(domain_comparison, engine='xlsxwriter')

domain_heights = [6, 9, 12]

excel_files = dict()
for height in domain_heights:
    file_path = os.path.join(folder_path, '6x6x{}_buildings_20h_bias'.format(height), 'csvs', '_validation.xlsx')
    buffer_file = pd.ExcelFile(file_path)
    excel_files[height] = buffer_file

sheet_names = buffer_file.sheet_names

for sheet in sheet_names:
    df = pd.DataFrame()
    variable, probe = sheet.split('_')
    for height, excel_file in excel_files.iteritems():
        data = pd.read_excel(excel_file, sheet)
        df['z={}h_z_norm'.format(height)] = data['unique_normz']
        df['z={}h_{}_norm'.format(height, variable)] = data['{}_norm'.format(variable)]
        plt.plot(data['{}_norm'.format(variable)], data['unique_normz'], label='RANS z={}h'.format(height))

    df['diplos_star_normz'] = data['diplos_star_normz']
    df['diplos_of_normz'] = data['diplos_of_normz']
    df['diplos_star_{}_norm'.format(variable)] = data['diplos_star_{}_norm'.format(variable)]
    df['diplos_of_{}_norm'.format(variable)] = data['diplos_of_{}_norm'.format(variable)]
    df.to_excel(writer, sheet)

    plt.plot(df['diplos_star_{}_norm'.format(variable)], df['diplos_star_normz'], label='DIPLOS StarCCM+')
    plt.plot(df['diplos_of_{}_norm'.format(variable)], df['diplos_of_normz'], label='DIPLOS OpenFOAM')
    plt.title(probe)
    plt.ylabel('z/h')
    plt.ylim(ymin=0)
    order = '^2' if len(variable) > 1 else ''
    xlabel = '{}/u_tau{}'.format(variable, order)
    plt.xlabel(xlabel)
    plt.legend(loc='upper left', ncol=2, fontsize='small')
    plot_path = os.path.join(folder_path, 'domain_plots')
    if not os.path.exists(plot_path):
        os.makedirs(plot_path)
    plt.savefig(os.path.join(plot_path, '{}.png'.format(sheet)))
    plt.gcf().clear()
