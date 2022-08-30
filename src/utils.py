import os
import numpy as np
from netCDF4 import Dataset
from src.variables import (
    VAR2DSKEY,
    FD_VAR2FILE,
    EA_VAR2FILE,
    KO_VAR2FILE,
)

def read_gk2a_data(var, datetime, base_path, area='fulldisk'):
    
    if area == 'fulldisk':
        filename = FD_VAR2FILE[var]
    elif area == 'eastasia':
        filename = EA_VAR2FILE[var]
    elif area == 'korea':
        filename = KO_VAR2FILE[var]
    else:
        raise ValueError('Invalid area, choose one of [fulldisk, eastasia, korea]')
        
    try: 
        ds = Dataset(os.path.join(base_path, filename.format(target_datetime=datetime)))
    except Exception as e:
        print(e)
        raise
    
    result_dict = {}
    for ds_var in VAR2DSKEY[var]:
        result_dict.update({ds_var: np.array(ds[ds_var][:])})
    return ds, result_dict