import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import sys
from csv_to_pandas import csv_to_pandas

logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger('error_file')


class BuildingsValidation(object):
    """
    Plot generation for buildings CFD case.
        1. Read Star CCM+ XY plot CSVs.
        2. Output Excel file containing a worksheet for each probe region (behind, top, gap).
        3. Output png plots comparing averaged data for the run against DIPLOS data.

    DIPLOS data for probe regions present in run folder.
    Code present in run folder.
    Star CCM+ data an average of all probe locations included in XY plot csv.
        (Currently searches for maximum of n probes where n is the number of buildings).
    Relies heavily on XY plot CSVs outputted correctly.
    Catch all errors and pass to log to be solved at later date.
    """
    def __init__(self, folder_path,
                 create_workbook=True, create_plots=True, create_residuals=True, create_periodic=True,
                 ref_or_tau=False, n_buildings=6):
        self.folder_path = folder_path
        self.create_workbook = create_workbook
        self.create_plots = create_plots
        self.create_residuals = create_residuals
        self.create_periodic = create_periodic
        self.ref_or_tau = ref_or_tau    # ref is True
        self.u_ref = None
        self.diplos_star_u_ref = None
        self.diplos_of_u_ref = None
        self.n_buildings = n_buildings
        self.probe_regions = ['behind', 'gap', 'top', 'intersection']
        self.variables = {"u": 'umean', "v": 'vmean', "w": 'wmean',
                          "u'u'": 'var_u', "v'v'": 'var_v', "w'w'": 'var_w',
                          "-u'v'": 'var_uv', "-u'w'": 'var_uw', "-v'w'": 'var_vw'}
        self.star_velocities = {'u': "Velocity[i] (m/s)",
                                'v': "Velocity[j] (m/s)",
                                'w': "Velocity[k] (m/s)"}
        self.direction = "normz"
        self.star_direction = "Direction [0,0,1] (m)"
        self.diplos_direction = 'z/h'

    def __call__(self, *args, **kwargs):
        if self.create_workbook:
            output_excel_file = os.path.join(self.folder_path, '_validation.xlsx')
            self.writer = pd.ExcelWriter(output_excel_file, engine='xlsxwriter')

        for probe_region in self.probe_regions:
            star_csvs_to_read = self._gen_csv_list(probe_region=probe_region)
            try:
                self.diplos_star_data = self._read_diplos_data(probe_region, True)
                self.diplos_of_data = self._read_diplos_data(probe_region, False)
            except Exception:
                logging.exception('Error reading DIPLOS data')
            try:
                self._get_u_ref(probe_region)
            except Exception:
                logging.exception('Error reading u velocity data for {}'.format(probe_region))

            for csv in star_csvs_to_read:
                try:
                    self._read_variable_data(probe_region, csv)
                except Exception:
                    logging.exception('Error reading variable data for {}'.format(csv))

        if self.create_residuals:
            try:
                self._monitors('Residuals')
                self._monitors('Drag Coefficient Monitor Plot')
            except Exception:
                logging.exception('Error reading residuals csv.')

        if self.create_periodic:
            try:
                self._periodic('periodic_check_vmag_io.csv')
                self._periodic('periodic_check_vmag_lr.csv')
            except Exception:
                logging.exception('Error reading periodic BC check data.')

    def _gen_csv_list(self, probe_region):
        return ['{}_{}.csv'.format(variable, probe_region)
                for variable in self.variables.keys()]

    def _read_diplos_data(self, probe_region, star):
        if star:
            return csv_to_pandas(self.folder_path, 'diplos_star_{}.csv'.format(probe_region))
        else:
            return csv_to_pandas(self.folder_path, 'diplos_of_{}.csv'.format(probe_region))

    def _get_u_ref(self, probe_region):
        if not self.ref_or_tau:
            self.diplos_star_u_ref = self.diplos_star_data['utau'][0]
            self.diplos_of_u_ref = self.diplos_of_data['utau'][0]
            self.u_ref = self.diplos_star_data['utau'][0]
        else:
            self.diplos_star_u_ref = self.diplos_star_data['uref'][0]
            self.diplos_of_u_ref = self.diplos_of_data['uref'][0]
            u_data = self._format_data(probe_region, 'u_{}.csv'.format(probe_region))
            z_index = pd.Index(u_data[self.direction]).get_loc(2.0, method='nearest')
            self.u_ref = u_data['u_avg'][z_index]

    def _format_data(self, probe_region, csv):
        variable = csv.split('_')[0]
        old_var = variable if variable not in self.star_velocities.keys() else self.star_velocities[variable]
        raw_data = csv_to_pandas(self.folder_path, csv)

        if raw_data is not None:
            new_columns = {"{}_b{}: {}".format(probe_region.capitalize(), probe, old_var):
                           "b{}_{}".format(probe, variable) for probe in range(1, self.n_buildings + 1)}
            direction_columns = {"{}_b{}: {}".format(probe_region.capitalize(), probe, self.star_direction):
                                 "b{}_{}".format(probe, self.direction) for probe in range(1, self.n_buildings + 1)}

            raw_data.rename(columns=new_columns, inplace=True)
            raw_data.rename(columns=direction_columns, inplace=True)

            var_col_list = [v for k, v in new_columns.iteritems() if v in raw_data.columns.values]
            dir_col_list = [v for k, v in direction_columns.iteritems() if v in raw_data.columns.values]

            unique_loc = pd.unique(raw_data[dir_col_list].values.ravel('K'))
            unique_loc = unique_loc[~np.isnan(unique_loc)]
            unique_loc_avg = np.zeros(unique_loc.size)

            for idx, loc in enumerate(unique_loc):
                var_values = []
                for col in direction_columns.values():
                    if col in raw_data.columns.values:
                        var_col = col.split(self.direction)[0] + variable
                        raw_data[col] = raw_data[col].apply(lambda x: round(x, 2))
                        try:
                            loc_index = pd.Index(raw_data[col]).get_loc(np.round(loc, decimals=2))
                        except KeyError:
                            pass
                        else:
                            var_values.append(raw_data[var_col][loc_index])
                avg_values = np.mean(var_values)
                unique_loc_avg[idx] = avg_values

            # for col in direction_columns.values():
            #     if col in raw_data.columns.values:
            #         var_col = col.split(self.direction)[0] + variable
            #         null_count = raw_data[col].isnull().sum()
            #         if null_count == 0 and self.direction not in raw_data.columns.values:
            #             raw_data[self.direction] = raw_data[col]
            #         raw_data[col] = raw_data[col].shift(null_count)
            #         raw_data[var_col] = raw_data[var_col].shift(null_count)

            raw_data['unique_{}'.format(self.direction)] = unique_loc
            raw_data['{}_avg'.format(variable)] = unique_loc_avg
            # raw_data['{}_avg'.format(variable)] = raw_data[var_col_list].mean(axis=1)
        return raw_data

    def _read_variable_data(self, probe_region, csv):
        variable = csv.split('_')[0]
        raw_data = self._format_data(probe_region, csv)
        if raw_data is not None:
            norm = self.u_ref ** 2 if len(variable) > 1 else self.u_ref
            diplos_star_norm = self.diplos_star_u_ref ** 2 if len(variable) > 1 else self.diplos_star_u_ref
            diplos_star_norm = diplos_star_norm * -1.0 if '-' in variable else diplos_star_norm
            diplos_of_norm = self.diplos_of_u_ref ** 2 if len(variable) > 1 else self.diplos_of_u_ref
            diplos_of_norm = diplos_of_norm * -1.0 if '-' in variable else diplos_of_norm

            raw_data['{}_norm'.format(variable)] = raw_data['{}_avg'.format(variable)] / norm
            for building in range(1, self.n_buildings + 1):
                raw_data['b{}_{}_norm'.format(building, variable)] = \
                    raw_data['b{}_{}'.format(building, variable)] / norm
            raw_data['u_ref'] = self.u_ref
            raw_data['diplos_star_{}'.format(self.direction)] = self.diplos_star_data['z/h']
            raw_data['diplos_of_{}'.format(self.direction)] = self.diplos_of_data['z/h']
            raw_data['diplos_star_{}_norm'.format(variable)] = self.diplos_star_data[
                                                                   self.variables[variable]] / diplos_star_norm
            raw_data['diplos_of_{}_norm'.format(variable)] = self.diplos_of_data[
                                                                 self.variables[variable]] / diplos_of_norm
            raw_data['diplos_star_u_ref'] = self.diplos_star_u_ref
            raw_data['diplos_of_u_ref'] = self.diplos_of_u_ref
            raw_data.sort_values('unique_normz')

            if self.create_workbook:
                raw_data.to_excel(self.writer, '{}_{}'.format(variable, probe_region))

            if self.create_plots:
                plot_path = self._create_plots_dir()
                x_diplos_star = raw_data['diplos_star_{}_norm'.format(variable)]
                y_diplos_star = raw_data['diplos_star_{}'.format(self.direction)]
                x_diplos_of = raw_data['diplos_of_{}_norm'.format(variable)]
                y_diplos_of = raw_data['diplos_of_{}'.format(self.direction)]
                x = raw_data['{}_norm'.format(variable)]
                y = raw_data['unique_{}'.format(self.direction)]

                plt.plot(x, y, 'k-', label='RANS Star CCM+')
                plt.plot(x_diplos_star, y_diplos_star, 'r-', label='DIPLOS Star CCM+')
                plt.plot(x_diplos_of, y_diplos_of, 'b-', label='DIPLOS OpenFOAM')
                plt.title(probe_region)
                plt.legend(loc='best', fontsize='small')
                x_label_power = '^2' if len(variable) > 1 else ''
                x_label = '{}/u_z=2h{}'.format(variable,
                                               x_label_power) if self.ref_or_tau else '{}/u_tau{}'.format(variable,
                                                                                                          x_label_power)
                plt.xlabel(x_label)
                plt.ylabel(self.diplos_direction)
                plt.ylim(ymin=0)
                plot_file = '{}_{}.png'.format(variable, probe_region)
                plt.savefig(os.path.join(plot_path, plot_file))
                plt.gcf().clear()

                for building in range(1, self.n_buildings + 1):
                    x = raw_data['b{}_{}_norm'.format(building, variable)]
                    y = raw_data['b{}_{}'.format(building, self.direction)]
                    plt.plot(x, y, label='b{}'.format(building))
                plt.title(probe_region)
                plt.legend(loc='best', ncol=2, fontsize='small')
                plt.xlabel(x_label)
                plt.ylabel(self.diplos_direction)
                plt.ylim(ymin=0)
                plot_file = '{}_{}_all.png'.format(variable, probe_region)
                plt.savefig(os.path.join(plot_path, plot_file))
                plt.gcf().clear()

    def _monitors(self, name):
        residual_data = csv_to_pandas(self.folder_path, '{}.csv'.format(name))
        plot_path = self._create_plots_dir()
        iterations = residual_data['Iteration']
        for residual in residual_data.columns.values:
            if residual != 'Iteration':
                residual_label = residual.split(':')[0]
                plt.semilogy(iterations, residual_data[residual], label=residual_label)

        plt.title(name)
        plt.legend(loc='best', ncol=2, fontsize='small')
        plt.xlabel('Iteration')
        plt.savefig(os.path.join(plot_path, '{}.png'.format(name)))
        plt.gcf().clear()

    def _periodic(self, filename):
        data = csv_to_pandas(self.folder_path, filename)
        plot_path = self._create_plots_dir()
        x_keys = [column for column in data.columns.values if 'direction' in column.lower()]
        y_keys = [column for column in data.columns.values if 'direction' not in column.lower()]
        for i, key in enumerate(x_keys):
            plt.plot(data[key], data[y_keys[i]], label=key.split(':')[0])
            direction = 'x' if '[1,0,0]' in key else 'y' if '[0,1,0]' in key else 'z'
            plt.xlabel('{}/h'.format(direction))
            plt.ylabel('vmag')
            plt.legend(loc='best', fontsize='small')
        plt.savefig(os.path.join(plot_path, '{}.png'.format(filename.split('.')[0])))
        plt.gcf().clear()

    def _create_plots_dir(self):
        plot_path = os.path.join(self.folder_path, 'plots')
        if not os.path.exists(plot_path):
            os.makedirs(plot_path)
        return plot_path


if __name__ == '__main__':
    input_folder_path = sys.argv[1]
    six_buildings_validation = BuildingsValidation(input_folder_path)
    six_buildings_validation()
