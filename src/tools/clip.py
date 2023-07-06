
import os
import glob
import numpy as np
from datetime import datetime, timedelta
from netCDF4 import Dataset
from PIL import Image

global _LATLON_DIR
global _BOUNDARY 

_LATLON_DIR = '/mnt/sdb1/wscho/data_for_research/ICTgk2a/latlon/'
_BOUNDARY = 250

class GK2ABaseProcessor(object):

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

        time_org = datetime.strptime(str_time, "%Y-%m-%d %H:%M")
        time_data = time_org - timedelta(hours=9)
        
        str_time_data = time_data.strftime("%Y%m%d%H%M")
        
        var_path = os.path.join(base_dir, str_time_data[:-4], 'LE2' , file_key.lower())
        gk2a_data = glob.glob(os.path.join(var_path, f'*{str_time_data}*.nc'))[0]
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