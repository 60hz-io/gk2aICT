import numpy as np

def get_cloud_albedo(data_dict, cf_threshold=0.0):
    assert all(x in data_dict.keys() for x in ['RSR', 'DSR', 'ASR', 'CF'])
    data = data_dict.copy()
    data['ISR'] = 1.29744*(data['RSR']+data['ASR'])
    data['CAL'] = np.clip(1 - data['DSR']/data['ISR'], 0, 1)
    data['CAL'][data['CF'] <= cf_threshold] = data['CF'][data['CF'] <= cf_threshold]
    
    cloud_albedo = data['CAL'] / (data['CF']+1.0e-6)
    # cloud_albedo[data['CF'] <= cf_threshold] = data['CF'][data['CF'] <= cf_threshold]
    
    return cloud_albedo, data