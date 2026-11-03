'''
TDMS loader module.

This module provides functions to load and process TDMS files.
'''

import pandas as pd

from nptdms import TdmsFile
from pathlib import Path

class LoadLIBSData:
    '''
    This class loads LIBS TDMS data.
    '''
    def __init__(self, file_path: str | Path):
        self.file_path = file_path
        self.tdms_file = TdmsFile.read(file_path)

    def to_df(self) -> pd.DataFrame:
        '''
        Converts the loaded LIBS TDMS data into a pandas DataFrame.
        '''
        data = {}
        for channel in self.tdms_file.channels():
            data[channel.name] = channel.data
        return pd.DataFrame(data)

    def param_table(self, df, start=0, end=31):
        '''
        Prints a table of parameters from the DataFrame.

        NOTE: This assumes the standard format from Applied Spectra LIBS data files.
        In this case, the param table starts at index 0 to 31. Adjust accordingly if your data follows a different format.
        '''
        params = df.iloc[start:end]
        names = params.iloc[:, 0].str.lstrip("'").tolist()
        values = [f'{v:g}' for v in params.iloc[:, 1]]

        name_w = max(len('Param'), *map(len, names))
        value_w = max(len('Value'), *map(len, values))
        print(f"{'Param':<{name_w}} | {'Value':>{value_w}}")
        print(f"{'-' * name_w}-+-{'-' * value_w}")
        for name, value in zip(names, values):
            print(f'{name:<{name_w}} | {value:>{value_w}}')

