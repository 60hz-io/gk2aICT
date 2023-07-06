def down_gk2a(date, lv, ch, area, down_dir):                                                                            
    import urllib.request as rq
    import datetime as dt
    import os
    
    ################################
    
    st = dt.datetime.strptime(str(date), "%Y%m%d%H%M")    
    key = 'NMSC57f2a0f803744c94a364f188f5e233d4'                
    url = "http://api.nmsc.kma.go.kr:9080/api/GK2A/" + lv + '/' + ch +'/'+  area  + '/' + 'data?date=' + st.strftime("%Y%m%d%H%M") + '&key='   
    # $input_url="http://api.nmsc.kma.go.kr:8080/api/GK2A/LE1B/VI004/EA/data?date=202007021604&req_div=oper01"
    url = url + key  
    
    request = rq.Request(url)
    response = rq.urlopen(request)
    rescode = response.getcode()
    
    #폴더 만들기

    down_dir = os.path.join(down_dir, st.strftime("%Y%m%d"), lv , ch.lower())        
    
    if not os.path.isdir(down_dir):
        os.makedirs(down_dir)
        
    if rescode == 200:
        fn = response.headers.get_filename()
        rq.urlretrieve(url, os.path.join(down_dir, fn))
        print('Complete to download: ' + fn)
    else:                                                                                                                             
        print('Error code: ' + str(rescode)) 

