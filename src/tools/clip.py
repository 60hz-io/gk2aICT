import re
import numpy as np

class GK2ABaseProcessor(object):
    def main():
        return None

class GK2AFDProcessor(object):
    
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
    
    @staticmethod
    def resolution_from_filename(gk2a_filename):
        re_result = re.findall('fd[0-9]*ge', gk2a_filename)[0]
        resolution_str = re_result[2:5]
        return float(resolution_str)*0.1
    
    @staticmethod
    def resolution_from_size(size):
        """ if you do not know resolution, use size of full disk data to find resolution"""
        if size == 22000:
            return np.float(0.5)
        if size == 11000:
            return np.float(1.0)
        if size == 5500:
            return np.float(2.0)
        
    def latlon_from_rowcol(self, idx_row, idx_col):
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
    
    def rowcol_from_latlon(self, latitude, longitude): 
        """ returns index of row and column from given latitude and longitude """
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
    
    
    def cut_with_latlon(
        self,
        array, 
        ullatitude, 
        ullongitude, 
        lrlatitude, 
        lrlongitude,
    ):
        arr = np.array(array)
        (ulrow, ulcol) = self.rowcol_from_latlon(ullatitude, ullongitude) 
        (lrrow, lrcol) = self.rowcol_from_latlon(lrlatitude, lrlongitude) 
        
        ulrow = int(np.floor(ulrow))
        ulcol = int(np.floor(ulcol))
        lrrow = int(np.ceil(lrrow))
        lrcol = int(np.ceil(lrcol)) 

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
        else:
            raise ValueError(
                "Invalid arguments, check [(ulcol <= lrcol), (ulrow <= lrrow), " +\
                "(0 <= ulcol), (lrcol < index_max), (0 <= ulrow), (lrrow < index_max)]"
            )
        return clip