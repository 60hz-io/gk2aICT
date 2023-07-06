import cv2

def downsize_img(img_arr, scale=None, size=None):
    """ make image resolution down to given scale or size """

    assert img_arr.shape[0] > 1
    assert img_arr.shape[1] > 1
    
    if scale is not None:
        assert scale > 0
        assert scale < 1
        out_arr = cv2.resize(
            img_arr,
            dsize=None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_AREA
        )
        
        return out_arr
    
    if size is not None:
        assert img_arr.shape[0] >= size[0]
        assert img_arr.shape[1] >= size[1]
        assert size[0] > 0 and size[1] > 0
        
        out_arr = cv2.resize(
            img_arr,
            dsize=size,
            interpolation=cv2.INTER_AREA
        )
        
        return out_arr
    
    raise ValueError("""
        Invalid input arguments, please input scale factor or size of desired output
    """)  
    
    
    
def upsize_img(img_arr, scale=None, size=None):
    """ make image resolution up to given scale or size """
    
    if scale is not None:
        assert scale > 1
        assert scale < 1000
        out_arr = cv2.resize(
            img_arr,
            dsize=None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_CUBIC
        )
        
        return out_arr
    
    if size is not None:
        assert img_arr.shape[0] <= size[0]
        assert img_arr.shape[1] <= size[1]
        assert size[0] > 0 and size[1] > 0
        
        out_arr = cv2.resize(
            img_arr,
            dsize=size,
            interpolation=cv2.INTER_CUBIC
        )
        
        return out_arr
    
    raise ValueError("""
        Invalid input arguments, please input scale factor or size of desired output
    """)  