import numpy as np
import pandas as pd
import pickle
import joblib

from pycaret.regression import *
from pvlib.location import Location

class ShortTermGHIForecaster(object):

    def __init__(self):
        return None
    
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
                                            apply(lambda x: pd.DataFrame(np.gradient(x, x.index.hour,edge_order = 1), index=x.index)\
                                                   if len(x)>1 else pd.DataFrame([np.NaN], index=x.index))[0].rename('true_ghi_corrected')
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
        """ if method =='gk2a':
             gk2a_data = pd.read_pickle(ca_pred_path)
             input_data = gk2a_data.reset_index().merge(cs_data.reset_index(),on=['dt','stn_nm'])
            input_data = input_data.drop(columns = 'cloud_index').rename(columns = {'ca' : 'cloud_index'})"""
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

            model_col = ['azimuth', 'zenith','cs_ghi','cs_dni', 'cs_dhi', 'cloud_index','latitude','longitude', 'altitude', 'cal_Cloud_OD', 'Cloud_OD']
            ghi_train = tmp_ghi[model_col]
            ghi_model = setup(session_id=1234, train_size=0.8, data=ghi_train, target=target_col, normalize=True, transformation=True, use_gpu=False, silent =True)
            ghi_model_res = compare_models(n_select=1, sort='MAE', include=[model_name])
            tuned_model = tune_model(ghi_model_res)
            final_model = finalize_model(tuned_model)  
            save_model(final_model,model_path.format(od = z))

            result = predict_model(final_model, data=ghi_train)
            result.to_pickle(result_path.format(od = z))
            y = result.Label
            x = result.Cloud_OD
            errors[z] = np.mean(abs(y-x))
            print(errors[z])

    def __call__(self):
        return None