import pandas as pd
import os
import sys


def csv_to_pandas(folder_path, filename):
    file_path = os.path.join(folder_path, filename)
    try:
        data = pd.read_csv(file_path, delimiter=',', header=0)
    except IOError:
        data = None
    return data


if __name__ == '__main__':
    folder = sys.argv[1]
    csv_file = sys.argv[2]
    print('Opening {}...'.format(csv_file))
    csv_to_pandas(folder, csv_file)

