
import os
import glob
import re
from datetime import datetime, timedelta
from PIL import Image

import numpy as np
import pandas as pd
import pickle
from netCDF4 import Dataset
import joblib
import cv2
from scipy.ndimage import gaussian_filter
from pvlib.location import Location

from pycaret.regression import *
import joblib

global _LATLON_DIR
global _BOUNDARY 

_LATLON_DIR = '/mnt/sdb1/wscho/data_for_research/ICTgk2a/latlon/'
_BOUNDARY = 250

class GK2ABaseProcessor(object):
   


    def main():
        
        return None
        return None
        return None


    @staticmethod
    def resolution_from_filename(gk2a_filename):
        re_result = gk2a_filename[-23:-16]
        resolution_str = re_result[2:5]
        return float(resolution_str)*0.1


    def cut_with_latlon(
        self,
        array, 
        ullatitude, 
        ullongitude, 
        lrlatitude, 
        lrlongitude,
        boundary = True,
    ):
        arr = np.array(array)
        

        (ulrow, ulcol) = self.rowcol_from_latlon(ullatitude, ullongitude) 
        (lrrow, lrcol) = self.rowcol_from_latlon(lrlatitude, lrlongitude) 
        
        ulrow = int(np.floor(ulrow))
        ulcol = int(np.floor(ulcol))
        lrrow = int(np.ceil(lrrow))
        lrcol = int(np.ceil(lrcol)) 

        if boundary:
            ulrow  -=  _BOUNDARY
            ulcol  -=  _BOUNDARY 
            lrrow  +=  _BOUNDARY 
            lrcol  +=  _BOUNDARY 
        
        clip = np.zeros((self.index_max, self.index_max))
        
        if ( 
            (ulcol <= lrcol)
            and (ulrow <= lrrow) 
            and (0 <= ulcol) 
            and (lrcol < self.index_max) 
            and (0 <= ulrow) 
            and (lrrow < self.index_max) 
           ): 
            clip = arr[ulrow:lrrow, ulcol:lrcol]
            ulrow  +=  _BOUNDARY
            ulcol  +=  _BOUNDARY 
            lrrow  -=  _BOUNDARY 
            lrcol  -=  _BOUNDARY 
            self.lat_clip = self.lat[ulrow:lrrow, ulcol:lrcol]
            self.lon_clip = self.lon[ulrow:lrrow, ulcol:lrcol]
        else:
            raise ValueError(
                "Invalid arguments, check [(ulcol <= lrcol), (ulrow <= lrrow), " +\
                "(0 <= ulcol), (lrcol < index_max), (0 <= ulrow), (lrrow < index_max)]"
            )
        return clip


    def latlon_from_rowcol(self, idx_row, idx_col):
        """ returns latitude and longitude from index of row and column of an array """
        nlat = self.lat[idx_row, idx_col] 
        nlon = self.lon[idx_row, idx_col] 
        
        return (nlat, nlon)
    

    def rowcol_from_latlon(self, latitude, longitude): 
        """ returns index of row and column from given latitude and longitude """
        distance = (self.lat - latitude )**2 + (self.lon - longitude)**2
        target_rc = np.where(distance == distance.min())
        nrow = target_rc[0]
        ncol = target_rc[1]
        
        return (nrow, ncol)


    def get_gk2a_var(self,file_key,var_name,base_dir,str_time):

        time_org     =  datetime.strptime(str_time, "%Y-%m-%d %H:%M")
        time_data   = time_org - timedelta(hours=9)
        
        str_time_data  = time_data.strftime("%Y%m%d%H%M")
        
        var_path    =  os.path.join(base_dir, str_time_data[:-4], 'LE2' , file_key.lower())
        gk2a_data   =  glob.glob(os.path.join(var_path, f'*{str_time_data}*.nc'))[0]
        print(f"using data named \n'{gk2a_data}' \nfor time '{time_org}'")

        ds = Dataset(gk2a_data)
        var = ds[var_name]

        return var


    def get_gk2a_var_set(self,file_key,var_name,gk2a_path,base_dir,str_time):

        var_set = []

        for dt in range(3):
            cur_time = datetime.strptime(str_time, "%Y-%m-%d %H:%M")  -  timedelta(minutes=(20-10*dt))
            cur_str_time = cur_time.strftime("%Y-%m-%d %H:%M")
            var_set.append(self.get_gk2a_var(file_key,var_name,base_dir,cur_str_time))

        return var_set


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
            cur_predict, sigma=1,truncate = 4)[_BOUNDARY:-_BOUNDARY,_BOUNDARY:-_BOUNDARY]/255).round(2)
            for cur_predict in raw_predict
            ]
        with open(ca_pred_2d_path, 'wb') as f:
            pickle.dump(predict, f)


    #method : ['OBS', 'Plant']
    def pick_var_target(self,ca_path,ca_pred_2d_path,meta_path,
        str_time, time_interval=15 , f_interval=120, l_interval=240):

        lat_clip = self.lat_clip
        lon_clip = self.lon_clip

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


    #method : ['OBS', 'gk2a']
    def cal_clear_sky(self,cs_path, meta_path, input_path,method = 'gk2a'):

        ca_col = []
        if method=='OBS':
            ca_col = ['true_ghi']

        cs_data = pd.DataFrame()
        input_data = pd.read_pickle(input_path)
        meta_data = pd.read_pickle(meta_path)

        for i, row in meta_data.iterrows():
            stn_nm = row.stn_nm
            print(stn_nm)

            cur_input = input_data[['stn_id','stn_nm','latitude','longitude','altitude','cloud_index']+ca_col
                                   ][input_data.stn_nm == stn_nm]
            lat, lon, alt = row.latitude, row.longitude, row.altitude
            stn_loc = Location(lat, lon, altitude=alt, tz='Asia/Seoul')

            solpos = stn_loc.get_solarposition(cur_input.index)
            cs_irrads = stn_loc.get_clearsky(cur_input.index, solar_position=solpos)
            cs_irrads.columns = [f'cs_{c}' for c in cs_irrads.columns]
            cur_input = cur_input.join(solpos).join(cs_irrads)
            cs_data = cs_data.append(cur_input)

            # if method == 'OBS' run following process
            # replace irradiance of np.nan value measured at hour H with 0
            # if solar elevation for that time is less than 0
            # calculates gradient of the cumulative irradiance function
            # to get the instantaneous power output for each hour
            # convert MJ -> Watt (just converting unit in this step)

        cs_data.to_pickle(cs_path)


    def preproc_asos_ghi(self,cs_obs_path,cs_path, meta_path):
        
        print('.\n.\n.\nIf you did not use ASOS data, never use this function\n.\n.\n.\n')            
        asos_data = pd.read_pickle(cs_path)
        meta_data = pd.read_pickle(meta_path)
        asos_data_preproc = pd.DataFrame()

        for i, row in meta_data.iterrows():
            cur_asos = asos_data[asos_data.stn_nm == row.stn_nm]
            print(row.stn_nm)
            cur_asos.loc[cur_asos.true_ghi.isnull() & (cur_asos.elevation < 0), 'true_ghi'] = 0.
            cur_asos = cur_asos.sort_index()
            ghi_cumsum = cur_asos.groupby(cur_asos.index.date)['true_ghi'].cumsum()
            
            ghi_deriv = ghi_cumsum.groupby(ghi_cumsum.index.date).\
                                            apply(lambda x: pd.DataFrame(np.gradient(x, x.index.hour,edge_order = 1), index=x.index) if len(x)>1 else pd.DataFrame([np.NaN], index=x.index))[0].rename('true_ghi_corrected')
            
            cur_asos = cur_asos.join(ghi_deriv)
            cur_asos['true_ghi_corrected'] *= 277.7777
            asos_data_preproc = asos_data_preproc.append(cur_asos)

        asos_data_preproc.rename(columns = {'true_ghi' : 'true_ghi_cum',
                                        'true_ghi_corrected' : 'true_ghi',
                                        }, inplace = True)
        asos_data_preproc = asos_data_preproc.drop_duplicates()
        asos_data_preproc = asos_data_preproc[asos_data_preproc.true_ghi>0]
        asos_data_preproc.to_pickle(cs_obs_path)



    def prepare_features(self,input_path, cs_path, ca_pred_path =None,method = 'gk2a'):
        cs_data = pd.read_pickle(cs_path)
        input_data = cs_data.copy()

        # if method =='gk2a':
        #     gk2a_data = pd.read_pickle(ca_pred_path)
        #     input_data = gk2a_data.reset_index().merge(cs_data.reset_index(),on=['dt','stn_nm'])
        #     input_data = input_data.drop(columns = 'cloud_index').rename(columns = {'ca' : 'cloud_index'})
        if method == 'OBS':
            input_data['Cloud_OD'] = (input_data.true_ghi-input_data.cs_dhi)/(input_data.cs_dni)
            input_data = input_data[(input_data['Cloud_OD']>=0)&(input_data['Cloud_OD']<=1)]
        input_data['cloud_index'] *= 0.1
        input_data['cloud_index'] = input_data.cloud_index+0.05
        input_data['cloud_index'] = input_data['cloud_index'].clip(0,1)
        input_data['cal_Cloud_OD'] = ((1-input_data.cloud_index)/((1/np.cos(np.pi/180*input_data.zenith)) - (input_data.cloud_index))).clip(0,1)
        
        input_data.to_pickle(input_path)

    def fitting_ghi_ref(self,model_path, result_path, input_data,  model_name = 'lightgbm',trs_switch = True):
        target_col = 'Cloud_OD'
        errors = np.zeros(11)


        for z,cur_cloud_OD in enumerate(np.arange(0,1.01,0.1)):
            
            tmp_ghi = input_data[z]

            print('='*100)
            print(f'cloud index      =      {cur_cloud_OD}')
            
            #==========================================================================================================================================
            
            model_col = ['azimuth', 'zenith','cs_ghi','cs_dni', 'cs_dhi', 'cloud_index','latitude','longitude', 'altitude', 'cal_Cloud_OD', 'Cloud_OD']
            ghi_train = tmp_ghi[model_col]
            
            ghi_model = setup(session_id=1234, train_size=0.8, data = ghi_train, target = target_col, normalize=True, transformation=True, use_gpu=False, silent =True)
            ghi_model_res = compare_models(n_select=1,sort = 'MAE', include=[model_name])

            tuned_model = tune_model(ghi_model_res)
            final_model = finalize_model(tuned_model)
                        

            save_model(final_model,model_path.format(od = z))
            result = predict_model(final_model, data=ghi_train)
            result.to_pickle(result_path.format(od = z))
            y = result.Label
            x = result.Cloud_OD
            errors[z] = np.mean(abs(y-x))
            print(errors[z])


class GK2AFDProcessor(GK2ABaseProcessor):
    
    DEG2RAD = 3.14159265358979 / 180.0 
    
    def __init__(self, resolution=None, gk2a_filename=None, size=None):

        if resolution:
            self.resolution = resolution    
        elif gk2a_filename:
            self.resolution = self.resolution_from_filename(gk2a_filename)
        elif size:
            self.resolution = self.resolution_from_size(size)
        else:
            raise ValueError(
                "Should input one of resolution and gk2a_filename"
            )
        if (self.resolution == 0.5):
            self.COFF = 11000.5
            self.CFAC = 8.170135561335742e7
            self.LOFF = 11000.5
            self.LFAC = 8.170135561335742e7
            self.index_max = 22000
        elif (self.resolution == 1.0):
            self.COFF = 5500.5
            self.CFAC = 4.0850677806678705e7
            self.LOFF = 5500.5
            self.LFAC = 4.0850677806678705e7
            self.index_max = 11000
        elif (self.resolution == 2.0):
            self.COFF = 2750.5
            self.CFAC = 2.0425338903339352e7
            self.LOFF = 2750.5
            self.LFAC = 2.0425338903339352e7
            self.index_max = 5500
        else:
            raise ValueError(
                "Invalid resolution, which should be one of [0.5, 1.0, 2.0]"
            )
        self.sub_lon = 128.2 
        self.sub_lon = self.sub_lon * self.DEG2RAD

        latlons = Dataset(glob.glob(os.path.join(_LATLON_DIR, '*fd*.nc'))[0])
        self.lat = latlons['lat'][:]
        self.lon = latlons['lon'][:]
    
    
    @staticmethod
    def resolution_from_size(size):
        #""" if you do not know resolution, use size of full disk data to find resolution"""
        if size == 22000:
            return np.float(0.5)
        if size == 11000:
            return np.float(1.0)
        if size == 5500:
            return np.float(2.0)
        
            
        def latlon_from_rowcol_fd_old(self, idx_row, idx_col):
            """ returns latitude and longitude from index of row and column of an array """
            x = self.DEG2RAD * ( (idx_col - self.COFF)*2**16 / self.CFAC )
            y = self.DEG2RAD * ( (idx_row - self.LOFF)*2**16 / self.LFAC )
            Sd = np.sqrt( (42164.0*np.cos(x)*np.cos(y))**2 - (np.cos(y)**2 + 1.006739501*np.sin(y)**2)*1737122264)
            Sn = (42164.0*np.cos(x)*np.cos(y)-Sd) / (np.cos(y)**2 + 1.006739501*np.sin(y)**2)
            S1 = 42164.0 - ( Sn * np.cos(x) * np.cos(y) )
            S2 = Sn * ( np.sin(x) * np.cos(y) )
            S3 = -Sn * np.sin(y)
            Sxy = np.sqrt( ((S1*S1)+(S2*S2)) )

            nlon = (np.arctan(S2/S1)+self.sub_lon)/self.DEG2RAD 
            nlat = np.arctan( ( 1.006739501 *S3)/Sxy)/self.DEG2RAD
            
            return (nlat, nlon)
        
        def rowcol_from_latlon_fd_old(self, latitude, longitude): 
        #""" returns index of row and column from given latitude and longitude """
            latitude = latitude * self.DEG2RAD 
            longitude = longitude * self.DEG2RAD
            c_lat = np.arctan(0.993305616*np.tan(latitude))
            RL = 6356.7523 / np.sqrt( 1.0 - 0.00669438444 * np.cos(c_lat)**2.0 ) 
            R1 = 42164.0 - RL * np.cos(c_lat) * np.cos(longitude - self.sub_lon)
            R2 = -RL * np.cos(c_lat) *np.sin(longitude - self.sub_lon)
            R3 = RL* np.sin(c_lat)
            Rn = np.sqrt(R1**2.0 + R2**2.0 + R3**2.0 )
            x = np.arctan(-R2 / R1) / self.DEG2RAD 
            y = np.arcsin(-R3 / Rn) / self.DEG2RAD 

            ncol = self.COFF + (x * 2.0**(-16) * self.CFAC) 
            nrow = self.LOFF + (y * 2.0**(-16) * self.LFAC)

            return (nrow, ncol)



class GK2AEAProcessor(GK2ABaseProcessor):

    DEG2RAD = 3.14159265358979 / 180.0

    def __init__(self, resolution=None, gk2a_filename=None, size=None):
        
        if resolution:
            self.resolution = resolution    
        elif gk2a_filename:
            self.resolution = self.resolution_from_filename(gk2a_filename)
        elif size:
            self.resolution = self.resolution_from_size(size)
        else:
            raise ValueError(
                "Should input one of resolution and gk2a_filename"
            )
            
        if (self.resolution == 0.5):
            self.index_max = 12000
        elif (self.resolution == 1.0):
            self.index_max = 6000
        elif (self.resolution == 2.0):
            self.index_max = 3000
        else:
            raise ValueError(
                "Invalid resolution, which should be one of [0.5, 1.0, 2.0]"
            )
            
        latlons = Dataset(glob.glob(os.path.join(_LATLON_DIR, '*ea*.nc'))[0])
        self.lat = latlons['lat'][:]
        self.lon = latlons['lon'][:]

    
    @staticmethod
    def resolution_from_size(size):
        """ if you do not know resolution, use size of full disk data to find resolution"""
        if size == 12000:
            return np.float(0.5)
        if size == 6000:
            return np.float(1.0)
        if size == 3000:
            return np.float(2.0)



class GK2AKOProcessor(GK2ABaseProcessor):
    
    _LATLON_DIR = '/mnt/sdb1/wscho/data_for_research/ICTgk2a/latlon/'
    DEG2RAD = 3.14159265358979 / 180.0 
    
    def __init__(self, resolution=None, gk2a_filename=None, size=None):
        
        if resolution:
            self.resolution = resolution    
        elif gk2a_filename:
            self.resolution = self.resolution_from_filename(gk2a_filename)
        elif size:
            self.resolution = self.resolution_from_size(size)
        else:
            raise ValueError(
                "Should input one of resolution and gk2a_filename"
            )
            
        if (self.resolution == 0.5):
            self.index_max = 2400
        elif (self.resolution == 1.0):
            self.index_max = 1200
        elif (self.resolution == 2.0):
            self.index_max = 900
        else:
            raise ValueError(
                "Invalid resolution, which should be one of [0.5, 1.0, 2.0]"
            )
            
        latlons = Dataset(glob.glob(os.path.join(_LATLON_DIR, '*ko*.nc'))[0])
        self.lat = latlons['lat'][:]
        self.lon = latlons['lon'][:]
    
    @staticmethod
    def resolution_from_size(size):
        """ if you do not know resolution, use size of full disk data to find resolution"""
        if size == 2400:
            return np.float(0.5)
        if size == 1800:
            return np.float(1.0)
        if size == 900:
            return np.float(2.0)