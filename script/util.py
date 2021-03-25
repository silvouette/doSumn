import pandas as pd
pd.options.mode.chained_assignment = None
pd.option_context('display.max_rows', None, 'display.max_columns', None)

def get_parser_data(filename):
    data = pd.read_csv('../' + filename)
    return data