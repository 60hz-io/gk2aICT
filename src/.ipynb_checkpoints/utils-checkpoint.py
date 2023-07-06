import os
import base64
import s3fs
import numpy as np
from netCDF4 import Dataset
from src.config import (
    LOCAL_GK2A_DIR,
    S3_GK2A_DIR,
)
from src.variables import (
    VAR2DSKEY,
    FD_VAR2FILE,
    EA_VAR2FILE,
    KO_VAR2FILE,
)
from src.tools.clip import GK2AFDProcessor


def convert_output_format(gk2a_array):
    """
    Used in make skt samples
    Convert 2d numpy array to json data with base64 encoding
    gk2a_array: 2d numpy array
    """
    return({
        'dtype': str(gk2a_array.dtype),
        'shape': gk2a_array.shape,
        'data': base64.b64encode(np.ascontiguousarray(gk2a_array)).decode('utf8')
    })


def convert_gk2a2output(dict_input):
    """
    Used in API server response
    Convert 2d numpy array to json data with base64 encoding
    dict_result: dictionary having 2d numpy array
    """
    dict_output = {}
    for kk, vv in dict_input.items():
        if kk == 'resolution':
            continue
        dict_output.update({
            kk:{
                'dtype': str(vv.dtype),
                'shape': vv.shape,
                'resolution': dict_input['resolution'],
                'data': base64.b64encode(np.ascontiguousarray(vv)).decode('utf8')
            }
        })
    
    return dict_output


def get_gk2a_target(
    variable, 
    obs_datetime, 
    target_range={
        'ullatitude': 43.0,
        'ullongitude': 124.0,
        'lrlatitude': 33.0,
        'lrlongitude': 132.0,
    },
    data_root='s3',
):
    '''
    variable: name of target gk2a variable name
    obs_datetime: observed time
    target_range: spatial range for clipping
        - default: ko_range = {
            'ullatitude': 43.0,
            'ullongitude': 124.0,
            'lrlatitude': 33.0,
            'lrlongitude': 132.0,
        }
    data_root: s3 or local(workbench)
    '''
    # base_path: data directory path according to data_Root
    #     - s3://60hz.data/kmipa/gk2a if data_root == s3
    #     - GK2A_DATA_PATH if data_root == local
    if data_root == 'local':
        base_path = LOCAL_GK2A_DIR,
        ds, data = read_gk2a_data(
            variable,
            obs_datetime,
            base_path
        )
    elif data_root == 's3':
        base_path = os.path.join("s3://", S3_GK2A_DIR)
        ds, data = read_gk2a_data_s3(
            variable, 
            obs_datetime,
            base_path
        )
    proc = GK2AFDProcessor(size=ds.dimensions['xdim'].size)
    for ds_var in VAR2DSKEY[variable]:
        data.update({ds_var: proc.cut_with_latlon(data[ds_var], **target_range)})
        if ds_var == 'CF': # convert to calculate cloud albedo
            data['CF'] = data['CF']/100
    ds.close()
    data.update({'resolution':proc.resolution})
    return data


def read_gk2a_data(variable, datetime, base_path, area='fulldisk'):
    
    if area == 'fulldisk':
        filename = FD_VAR2FILE[variable]
        prefix = os.path.join(base_path, "fd")
    elif area == 'eastasia':
        filename = EA_VAR2FILE[variable]
        prefix = os.path.join(base_path, 'ea')
    elif area == 'korea':
        filename = KO_VAR2FILE[variable]
        prefix = os.path.join(base_path, 'ko')
    else:
        raise ValueError('Invalid area, choose one of [fulldisk, eastasia, korea]')
        
    try: 
        ds = Dataset(os.path.join(prefix, filename.format(target_datetime=datetime)))
    except Exception as e:
        print(e)
        raise
    
    result_dict = {}
    for ds_var in VAR2DSKEY[variable]:
        result_dict.update({ds_var: np.array(ds[ds_var][:])})
    return ds, result_dict


def read_gk2a_data_s3(variable, datetime, base_path, area='fulldisk'):
    s3 = s3fs.S3FileSystem()
    if area == 'fulldisk':
        filename = FD_VAR2FILE[variable].format(target_datetime=datetime)
        s3_prefix = os.path.join(base_path, "fd")
    elif area == 'eastasia':
        filename = EA_VAR2FILE[variable].format(target_datetime=datetime)
        s3_prefix = os.path.join(base_path, "ea")
    elif area == 'korea':
        filename = KO_VAR2FILE[variable].format(target_datetime=datetime)
        s3_prefix = os.path.join(base_path, "ko")
    else:
        raise ValueError('Invalid area, choose one of [fulldisk, eastasia, korea}]')
    
    date = datetime[:8]
    try:
        with s3.open(
            os.path.join(s3_prefix, variable, date, filename),
        ) as file:
            ds = Dataset(
                "dummy.nc",
                mode="r",
                memory=file.read()
            )
    except Exception as e:
        print(e)
        raise
    
    result_dict = {}
    for ds_var in VAR2DSKEY[variable]:
        result_dict.update({ds_var: np.array(ds[ds_var][:])})
    return ds, result_dict