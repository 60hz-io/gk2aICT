
import numpy as np
import pandas as pd
import cv2
import pickle

from PIL import Image
from scipy.ndimage import gaussian_filter
from datetime import datetime, timedelta

global _BOUNDARY 
_BOUNDARY = 250

class CloudFlowCalculator(object):

    def __init__(self):
        return None
   
    def cloud_amount_preproc(self,cloud_amount,resize=(1000, 800), normalize=True):
        cloud_amount[cloud_amount>1] = 0
        if normalize:
            cloud_amount = (cloud_amount - cloud_amount.min()) / (cloud_amount.max() - cloud_amount.min()) * 255
        # resize image
        img = Image.fromarray(cloud_amount.astype('uint8')).resize(resize[::-1])
        gk2a_preproc = np.array(img)

        return gk2a_preproc

    def warp_flow(self, img, flow):
        h, w = flow.shape[:2]
        flow = -flow
        flow[:,:,0] += np.arange(w)
        flow[:,:,1] += np.arange(h)[:,np.newaxis]
        res = cv2.remap(img, flow, None, cv2.INTER_CUBIC, borderMode=cv2.BORDER_DEFAULT)

        return res

    def cal_optical_flow(self,optical_flow, gk2a_preproc, ca_pred_2d_path, time_interval=15 , f_interval=120, l_interval=240):        
        prev = gk2a_preproc[-1]
        forwrd = gk2a_preproc[-2]
        forwrd2 = gk2a_preproc[-3]

        f_time = int(f_interval / time_interval)
        l_time = int(l_interval / time_interval)

        print("-="*20+'-\n')
        print(f'   First Prediction  : + {f_interval} minutes')
        print(f'   Last  Prediction  : + {l_interval} minutes\n')
        print("-="*20+'-\n')

        if ((f_interval%time_interval)!=0)|((l_interval%time_interval)!=0):
            raise ValueError(
                "Time interval needs to be a divisor of 'first prediction time and last prediction time'!"
            )
        raw_prev = prev.copy()
        bf_flow = optical_flow.calc(forwrd2, forwrd, None)
        flow = optical_flow.calc(forwrd, prev, None)
        acc = time_interval/10*(flow - bf_flow)
        flow *= time_interval/10

        raw_predict = []

        for i in range(l_time):
            print(f'   Calculating + {15*(i+1)} minutes Cloud Amount\n')

            raw_prev = self.warp_flow(raw_prev, flow) 
            flow += acc
            if i >= (f_time-1):
                raw_predict.append(raw_prev)
        predict = [
            (gaussian_filter(
            cur_predict, sigma = 1, truncate = 4)[_BOUNDARY:-_BOUNDARY, _BOUNDARY:-_BOUNDARY]/255).round(2)
            for cur_predict in raw_predict
            ]
        with open(ca_pred_2d_path, 'wb') as f:
            pickle.dump(predict, f)

    #method : ['OBS', 'Plant']
    def pick_var_target(self,ca_path,ca_pred_2d_path,meta_path,
        str_time, lat_clip, lon_clip, time_interval=15 , f_interval=120, l_interval=240):
        stn_meta = pd.read_pickle(meta_path)
        with open(ca_pred_2d_path, 'rb') as f:
            ca_pred_2d = pickle.load(f)
        ca_pred = pd.DataFrame({'stn_id':[],
                               'stn_nm' :[],
                               'latitude':[],
                               'longitude':[],
                               'altitude' :[],
                               'dt' : []})
        stn_meta['nrow'] = np.NaN
        stn_meta['ncol'] = np.NaN

        for m,cur_result in enumerate(ca_pred_2d):
            for i, cur_row in stn_meta.iterrows():
                distance = (lat_clip - cur_row['latitude'] )**2 + (lon_clip - cur_row['longitude'])**2
                target_rc = np.where(distance == distance.min())
                print(f'{cur_row.stn_nm} y,x : {target_rc[0][0]}, {target_rc[1][0]}')
                stn_meta['nrow'][i]= target_rc[0][0]
                stn_meta['ncol'][i] = target_rc[1][0]
                cur_time = pd.to_datetime(str_time, format='%Y-%m-%d %H:%M').tz_localize('Asia/Seoul')+ timedelta(minutes=f_interval + m*time_interval)
                cur_ca  = cur_result[target_rc[0][0],target_rc[1][0]]
                ca_pred = ca_pred.append(pd.DataFrame([[cur_row.stn_id, cur_row.stn_nm,
                                                        cur_row.latitude,cur_row.longitude,cur_row.altitude,
                                                        cur_time,cur_ca]], 
                                                        columns=['stn_id', 'stn_nm','latitude','longitude', 'altitude','dt','cloud_index']))
        ca_pred = ca_pred.set_index('dt')
        stn_meta[['nrow','ncol']] = stn_meta[['nrow','ncol']].astype(int)
        ca_pred.to_pickle(ca_path)
        stn_meta.to_pickle(meta_path)

    def __call__(self):
        return None