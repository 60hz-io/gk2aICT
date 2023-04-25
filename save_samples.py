import os
import glob
import re
import json
import base64
import argparse
import numpy as np
from datetime import (
    datetime, 
    timedelta,
    date,
)
from netCDF4 import Dataset
from src.config import (
    LOCAL_GK2A_DIR,
)
from src.variables import (
    VAR2DSKEY,
    FD_VAR2FILE,
)
from src.tools.clip import (
    GK2AFDProcessor,
)
from src.utils import (
    read_gk2a_data,
    convert_output_format,
    get_gk2a_target,
)
from src.tools.cloud_albedo import get_cloud_albedo

jeju_range = { # TODO: get range from arguments
    'ullatitude': 35.0,
    'ullongitude': 125.0,
    'lrlatitude': 32.0,
    'lrlongitude': 128.0,
}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--varname', type=str,
                        help='gk2a data name')
    parser.add_argument('--directory', type=str,
                        help='directory to save file')
    parser.add_argument('--obs_start', type=str, 
                        help='start time, yyyymmddHHMM')
    parser.add_argument('--obs_end', type=str,
                        help='start time, yyyymmddHHMM')
    parser.add_argument('--data_root', type=str, default='s3') # or local
    parser.add_argument('--area', type=str, default='fd')
    return parser
    

# see example skt_sample file
# def get_gk2a_data(varname, obs_datetime, target_range, base_path, area, data_root = ):
#     ds, data = read_gk2a_data(varname, obs_datetime, base_path=base_path, area=area)
#     proc = GK2AFDProcessor(size=ds.dimensions['xdim'].size)
#     for ds_var in VAR2DSKEY[varname]:
#         data.update({ds_var: proc.cut_with_latlon(data[ds_var], **target_range)})
#         if ds_var == 'CF': # convert to calculate cloud albedo
#             data['CF'] = data['CF']/100
#     return data
    
    
if __name__ == '__main__':
    parser = parse_args()
    args = parser.parse_args()
    varname = args.varname.lower()
    target_dir = args.directory
    obs_start = datetime.strptime(args.obs_start, '%Y%m%d%H%M')
    obs_end = datetime.strptime(args.obs_end, '%Y%m%d%H%M')
    data_root = args.data_root.lower()
    area = args.area.lower()
    # if data_root == 's3':
    #     base_path = S3_GK2A_DIR
    # if data_root == 'local':
    #     base_path = os.path.join(LOCAL_GK2A_DIR)
    # target_range = {  # TODO: get range from argument
    #     'ullatitude': 36.0,
    #     'ullongitude': 123.5,
    #     'lrlatitude': 31.0,
    #     'lrlongitude': 130.0
    # }
    target_range = { # east asia
        'ullatitude': 53,
        'ullongitude': 77,
        'lrlatitude': 11,
        'lrlongitude': 135,
    }
    print(target_range)

    for ii in range(0, int((obs_end-obs_start).total_seconds())+600, 600):
        try:
            target_obs = (obs_start + timedelta(minutes=ii/60)).strftime('%Y%m%d%H%M')
            data = get_gk2a_target(varname, target_obs, target_range)
        except Exception as ex:
            print(ex)
            print('No data:', varname, 'at', target_obs)
        else:
            res = {}
            for kk, vv in data.items():
                if kk == 'resolution':
                    continue
                print(target_obs, varname, kk)
                var_dir = os.path.join(target_dir, varname)
                os.makedirs(var_dir, exist_ok=True)
                res.update({kk:convert_output_format(data[kk])})
                
            print(res.keys())
            with open(
                os.path.join(var_dir, f'gk2a_{varname}_{target_obs}.json'), 'w'
            ) as f:
                json.dump(res, f)
        
    print(target_dir)
    print(obs_start)
    print(obs_end)