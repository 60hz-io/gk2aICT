=======================
GK2A Satellite Data API
=======================

This API Service is to provide GK2A satellite data and secondary processed data which is expected to be used in various areas


1. Introduction
^^^^^^^^^^^^^^^^

   Request GK2A satellite data and secondary processed data (cloud albedo, cloud motion, and etc.)
   Provide clip interesting regions via latitude and longitude, and resolution manipulation)

      - **url**: ```http://apialpha.60hz.io/gk2a/```
      - **output type**: ```json```


2. Request Parameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
     
   ===============  ============  ==========================================================================
    Arguments        Required      Description
   ===============  ============  ==========================================================================
    variable         True          Name of data interested in
    obs_datetime     True          Datetime of data interested in (YYYYMMDDhhmm)
    ullatitude       False         Latitude at upper-left corner 
    ullongitude      False         Longitude at upper-left corner
    lrlatitude       False         Latitude at lower-right corner
    lrlongitude      False         Longitude at lower-right corner
    resolution       False         Resolution reduction/magnification ratio compared to original data
   ===============  ============  ==========================================================================


3. Response
^^^^^^^^^^^^^^^^^^^^^^^^

   ==================  =====================================================================================
    Keys                Description
   ==================  =====================================================================================
    dtype               Type of data
    shape               Original shape of data, decoded result of 'data' should be reshpaed by this
    data                8-bit encoded numpy array data, should be decoded using 'base64' or other libraries
   ==================  =====================================================================================



^^^^^^^^^^^^^^^^


Sample code (python3)
======================

.. code::

   import requests
   import base64
   import numpy as np

   url = 'http://apialpha.60hz.io/data'
   params = {'variables':'cloudalbedo', 'obs_datetime':'202111180550'}

   response = requests.get(base_url, params=params)
   data = response.json()
   result = np.frombuffer(
      base64.decodebytes(data['data'].encode('utf8')),
      dtype=data['dtype']
   ).reshape(data['shape'])

   print(result)




.. 99. References
.. ^^^^^^^^^^^^^^^^^^^^^^^

.. https://nmsc.kma.go.kr/enhome/html/main/main.do
