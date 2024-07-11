import math
import numpy # to use the numpy version of **, not sure import is a must or not
from numpy import cos, sin, tan, pi
from pyproj import transform, Proj
import pandas as pd

import warnings
warnings.filterwarnings("ignore")


def twd97_to_lonlat(twd97_x: float = 174458.0, twd97_y: float = 2525824.0) -> tuple:
    '''
    This program transforms TWD97 to WGS84 projections.
    "transform" and "Proj" are the required function from the 
    package "pyproj".

    Parameters
    ----------
    twd97_x : float
        TWD97 coord system. The default is 174458.0.
    twd97_y : float
        TWD97 coord system. The default is 2525824.0.

    Returns
    -------
    tuple
        (longitude, latitude)
    '''

    # Define the TWD97 and WGS84 projections
    twd97 = Proj(init='epsg:3826')  # TWD97 (EPSG:3826)
    wgs84 = Proj(init='epsg:4326')  # WGS84 (EPSG:4326)

    return transform(twd97, wgs84, twd97_x, twd97_y)

def twd97_to_lonlat_pt(twd97_x=174458.0,twd97_y=2525824.0):
    """
    Found this program from https://tylerastro.medium.com/twd97-to-longitude-latitude-dde820d83405
    There are also twd67_to_twd97 and lonlat_to_twd97 in the website.

    Parameters
    ----------
    twd97_x : float
        TWD97 coord system. The default is 174458.0.
    twd97_y : float
        TWD97 coord system. The default is 2525824.0.

    Returns
    -------
    list
        [longitude, latitude]
    """
    
    a = 6378137
    b = 6356752.314245
    long_0 = 121 * math.pi / 180.0
    k0 = 0.9999
    dx = 250000
    dy = 0
    
    e = math.pow((1-math.pow(b, 2)/math.pow(a,2)), 0.5)
    
    twd97_x -= dx
    twd97_y -= dy
    
    M = twd97_y / k0
    
    mu = M / ( a*(1-math.pow(e, 2)/4 - 3*math.pow(e,4)/64 - 5 * math.pow(e, 6)/256))
    e1 = (1.0 - pow((1   - pow(e, 2)), 0.5)) / (1.0 +math.pow((1.0 -math.pow(e,2)), 0.5))

    j1 = 3*e1/2-27*math.pow(e1,3)/32
    j2 = 21 * math.pow(e1,2)/16 - 55 * math.pow(e1, 4)/32
    j3 = 151 * math.pow(e1, 3)/96
    j4 = 1097 * math.pow(e1, 4)/512
    
    fp = mu + j1 * math.sin(2*mu) + j2 * math.sin(4* mu) + j3 * math.sin(6*mu) + j4 * math.sin(8* mu)
    
    e2 = math.pow((e*a/b),2)
    c1 = math.pow(e2*math.cos(fp),2)
    t1 = math.pow(math.tan(fp),2)
    r1 = a * (1-math.pow(e,2)) / math.pow( (1-math.pow(e,2)* math.pow(math.sin(fp),2)), (3/2))
    n1 = a / math.pow((1-math.pow(e,2)*math.pow(math.sin(fp),2)),0.5)
    d = twd97_x / (n1*k0)
    
    q1 = n1* math.tan(fp) / r1
    q2 = math.pow(d,2)/2
    q3 = ( 5 + 3 * t1 + 10 * c1 - 4 * math.pow(c1,2) - 9 * e2 ) * math.pow(d,4)/24
    q4 = (61 + 90 * t1 + 298 * c1 + 45 * math.pow(t1,2) - 3 * math.pow(c1,2) - 252 * e2) * math.pow(d,6)/720
    lat = fp - q1 * (q2 - q3 + q4)
    
    
    q5 = d
    q6 = (1+2*t1+c1) * math.pow(d,3) / 6
    q7 = (5 - 2 * c1 + 28 * t1 - 3 * math.pow(c1,2) + 8 * e2 + 24 * math.pow(t1,2)) * math.pow(d,5) / 120
    lon = long_0 + (q5 - q6 + q7) / math.cos(fp)
    
    lat = (lat*180) / math.pi
    lon = (lon*180) / math.pi
    return [lon, lat]

def twd97_to_lonlat_srs(twd97_x: pd.Series,twd97_y: pd.Series) -> tuple:
    """
    This is the rewrite version of twd97_to_lonlat_pt to convert series of 
    longitudes and latitudes.

    Parameters
    ----------
    twd97_x : pd.Series
        TWD97 coord system.
    twd97_y : pd.Series
        TWD97 coord system.

    Returns
    -------
    tuple of pd.Series
        (lon_new, lat_new)
    """
    
    a = 6378137
    b = 6356752.314245
    long_0 = 121 * pi / 180.0
    k0 = 0.9999
    dx = 250000
    dy = 0
    
    e = (1 - b**2 / a**2) ** 0.5
    
    twd97_x = twd97_x - dx # srs
    twd97_y = twd97_y - dy # srs
    # breakpoint()

    M = twd97_y / k0 # srs
    
    denom = (a * (1 - e**2 / 4 - 3 * e**4 / 64 - 5 * e**6 /256))
    mu = M / denom # srs
    e1 = (1 - (1 - e**2)**0.5) / (1 + (1- e**2)**0.5)
    
    j1 = 3 * e1 / 2 - 27 * e1**3/32
    j2 = 21 * e1**2 / 16 - 55 * e1**4 /32
    j3 = 151 * e1**3 / 96
    j4 = 1097 * e1**4 / 512
    
    fp = mu + j1*sin(2*mu) + j2*sin(4*mu) + j3*sin(6*mu) + j4*sin(8*mu) # srs
    
    e2 = (e * a / b)**2
    c1 = e2 * cos(fp)**2 # srs
    t1 = tan(fp)**2 # srs
    r1 = a * (1 - e**2) / (1 - e**2 * sin(fp)**2)**1.5 # srs
    n1 = a / (1 - e**2 * sin(fp)**2)**0.5 # srs
    d = twd97_x / (n1*k0) # srs
    
    q1 = n1 * tan(fp) / r1 # srs
    q2 = d**2 / 2 # srs
    q3 = (5 + 3 * t1 + 10 * c1 - 4 * c1**2 - 9 * e2) * d**4 / 24 # srs
    q4 = (61 + 90 * t1 + 298 * c1 + 45 * t1**2 - 3 * c1**2 - 252 * e2) * d**6 / 720 # srs
    lat = fp - q1 * (q2 - q3 + q4) # srs
    
    q5 = d # srs
    q6 = (1 + 2 * t1 + c1) * d**3 / 6 # srs
    q7 = (5 - 2 * c1 + 28 * t1 - 3 * c1**2 + 8 * e2 + 24 * t1**2) * d**5 / 120 # srs
    lon = long_0 + (q5 - q6 + q7) / cos(fp) # srs
    
    lat = (lat * 180) / pi # srs
    lon = (lon * 180) / pi # srs

    return lon, lat


def main():
    # ===============
    # TWD987 TO WGS84
    # ===============
    # -- in sum, the two ways have no significant difference
    # -- get the same result upto 6 decimal points
    # TESTING PACKAGE
    test_pairs = [
        {"twd97_x": 302997, "twd97_y": 2772651},
        {"twd97_x": 303273, "twd97_y": 2771417},
        {"twd97_x": 304406, "twd97_y": 2772017}
    ]
    for test_pair in test_pairs:
        ll = twd97_to_lonlat(**test_pair)
        print(ll)

    # TESTING FORMULAS - SINGLE POINT
    # twd97_to_lonlat_pt(**{"twd97_x": 302997, "twd97_y": 2772651})

    # TESTING FORMULAS - ARRAYS
    test_df = {
        "x": [302997, 303273, 304406],
        "y": [2772651, 2771417, 2772017]
    }
    test_df = pd.DataFrame(test_df)
    new_coords = twd97_to_lonlat_srs(test_df["x"], test_df["y"])
    print(new_coords)

if __name__ == "__main__":
    main()