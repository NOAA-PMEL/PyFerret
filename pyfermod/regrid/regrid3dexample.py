'''
A Python program for testing the regrid3d.CurvRect3DRegridder class "by hand"
and also serves as a coding example of using this class.

@author: Karl Smith
'''

import numpy
import ESMP
from esmpcontrol import ESMPControl
from regrid3d import CurvRect3DRegridder


GULF_MEX_LONS = (262.0, 278.0, 0.5)
GULF_MEX_LATS = (18.0, 31.0, 0.5)
# Gulf of Mexico Relief of the Surface of the Earth (meters) 98W:82W:0.5 (columns), 18N:31N:0.5 (rows)
GULF_MEX_ROSE = (
      ( 1627.1,  2130.9,  1317.9,   373.8,   125.7,    45.0,    25.0,    20.0,    25.0,    25.0,
          15.0,    15.0,    25.0,    25.0,    40.0,    74.3,   169.7,   179.7,   151.3,    23.0,
          -3.2, -1638.9, -3668.0, -4098.9, -3075.3, -3017.9, -4554.6, -4969.7, -4961.6, -5008.2,
       -4759.7, -4932.3, -4016.7, ),
      ( 1951.7,  1669.0,   685.7,    63.4,    50.7,   779.4,   400.0,   -70.7,   -49.0,   -16.2,
          -6.1,     2.0,    -9.6,     5.0,    69.7,    80.1,   194.7,   210.2,   110.6,     6.1,
          -0.3,  -192.5, -1587.3, -4357.8, -4401.9, -4397.7, -2980.1, -2035.6,  -463.5, -2265.7,
       -4933.4, -4959.4, -4490.4, ),
      ( 2220.5,  2654.2,  1232.2,    60.8,   -10.0,  -217.9, -1170.2,  -807.9,  -599.1,  -388.8,
        -101.1,   -40.1,   -14.0,    -8.1,    19.9,    65.0,   139.4,   175.8,   115.2,    40.3,
           0.0,  -220.1,  -919.6, -4284.3, -4502.4, -4474.9, -4418.9, -3648.0, -2258.9, -2581.0,
       -2998.4, -3578.1, -3493.9, ),
      ( 2593.1,  2368.9,  1097.8,    29.9,  -225.9, -1889.3, -2357.3, -1504.2, -1161.4,  -939.7,
        -702.9,  -141.0,   -47.9,   -29.0,   -13.1,    29.7,    94.8,   150.0,   110.0,    70.5,
           0.0,  -340.6, -1551.1, -4353.5, -4516.0, -4534.5, -4455.3, -4438.0, -4153.9, -3078.7,
       -3004.0, -3228.8, -3595.4, ),
      ( 1944.0,   923.6,   191.8,   -12.8, -1599.3, -2398.4, -2802.9, -1996.6, -1597.9, -1039.4,
       -1061.3, -1345.8,   -62.4,   -25.9,   -21.1,    -9.9,    79.4,   120.0,    65.3,    30.5,
          -1.0,  -278.2,  -994.7, -1794.1, -4315.4, -4438.8, -4437.1, -4443.0, -4378.4, -3605.0,
       -3977.6, -4239.2, -2402.8, ),
      (  459.9,   210.9,    -9.9,  -588.9, -1913.3, -2591.4, -3053.4, -2975.4, -2016.1, -1992.1,
       -1619.8, -1926.1,   -38.2,   -23.6,   -23.0,    -0.1,    30.0,    30.2,    24.8,    15.0,
          10.0,     5.0,  -262.5, -1178.6, -2949.8, -4621.6, -4421.9, -4450.7, -4401.8, -4361.9,
       -4401.3, -4437.3, -3876.6, ),
      (  108.9,    23.3,   -97.1, -1136.2, -2098.3, -2602.7, -3192.4, -3178.5, -2608.1, -2588.1,
       -2597.9,  -660.0,    -9.9,   -40.1,   -29.0,   -10.1,     5.0,    10.0,    10.0,    10.0,
          10.0,    10.0,     0.0,  -276.1, -1413.3, -4101.5, -3835.0, -4424.2, -4415.9, -4443.6,
       -4443.1, -4460.6, -4399.8, ),
      (   69.7,   -10.0,  -402.6, -1785.3, -2391.5, -2800.2, -3278.6, -3408.0, -3111.9, -3002.1,
       -3014.9,  -782.6,   -44.6,   -17.9,   -21.9,   -29.0,   -21.0,   -14.0,   -12.0,     0.0,
           0.0,   -10.0,   -10.0,  -103.2, -1775.1, -2001.9, -2965.9, -3711.9, -3032.2, -2801.0,
          -3.1,  -920.6, -1273.1, ),
      (   64.0,   -10.0, -1313.9, -2190.4, -2769.9, -2899.3, -3317.6, -3536.4, -3564.0, -3364.1,
       -3331.0, -1715.7,   -13.2,   -47.9,   -19.7,   -46.2,   -45.9,   -40.0,   -33.1,    -3.1,
         -24.9,    -4.2,   -10.0,  -202.8, -1409.1, -1982.8,  -298.8,     0.0,  -575.5,  -339.9,
          -3.9,   -10.0,   -10.0, ),
      (   -8.5,   -10.0, -1714.1, -2321.6, -2790.2, -3072.8, -3446.8, -3621.8, -3639.0, -3635.0,
       -3581.5, -3569.9, -3367.2, -1055.4,  -227.7,  -117.0,   -53.2,   -47.7,   -61.2,   -55.2,
         -53.0,   -50.0,  -133.0,  -507.9, -1065.2, -1840.9, -2201.4,  -907.3,   192.0,    32.1,
          -2.2,   -10.0,   -10.0, ),
      (    7.2,   -45.9, -1526.8, -2196.7, -2588.5, -3178.9, -3593.6, -3630.1, -3623.9, -3628.0,
       -3621.0, -3619.0, -3709.8, -3908.8, -3558.9, -1507.3,  -237.2,   -45.2,    -4.3,   -62.2,
         -48.1,   -32.1,  -394.2,  -807.6, -1193.3, -1139.0, -2075.7, -2395.8, -2128.4, -1034.6,
          42.3,   112.0,   125.4, ),
      (  134.6,   -39.1, -1078.0, -2179.4, -2595.1, -3002.4, -3565.4, -3618.1, -3666.8, -3679.0,
       -3648.2, -3598.0, -3597.9, -3602.0, -3695.7, -3695.5, -1106.2,  -725.6,   -12.7,   -10.0,
         -92.3,  -151.6,  -591.2,  -977.7, -2939.2, -1435.4, -3011.0, -2948.8, -2399.3, -2182.2,
       -1800.0, -1789.0, -1570.9, ),
      (   55.0,   -35.2,  -838.4, -1609.6, -2488.7, -3268.8, -3594.0, -3613.1, -3724.5, -3727.1,
       -3714.9, -3671.4, -3739.6, -3718.0, -3655.8, -3607.0, -3826.2, -3652.7, -3069.2,  -352.8,
        -181.9,  -396.3,  -994.3, -1254.2, -3228.1, -3401.8, -3468.6, -2829.9, -2184.4, -1017.2,
       -1006.6,  -808.0, -1059.1, ),
      (   -8.7,   -18.7,  -141.0, -1008.6, -2264.6, -3205.6, -3619.4, -3620.2, -3654.8, -3655.1,
       -3631.0, -3622.0, -3611.0, -3609.0, -3616.1, -3611.1, -3600.8, -3851.1, -3622.7, -2515.4,
       -1578.2, -1492.1, -1769.6, -3051.5, -3383.7, -3415.2, -3402.4, -3268.5, -1954.1,  -192.3,
         -40.4,   -13.9,   -34.0, ),
      (    2.1,   -10.0,   -65.5,  -487.2, -1596.1, -2703.7, -3579.7, -3627.0, -3628.0, -3612.1,
       -3609.9, -3578.6, -3542.4, -3511.9, -3587.7, -3583.2, -3545.1, -3585.9, -3544.5, -3302.6,
       -3388.8, -3189.2, -3241.8, -3287.6, -3321.9, -3363.9, -3249.2, -2189.4,  -127.6,   -33.0,
         -43.8,   -35.9,   -10.0, ),
      (   17.1,     2.0,   -25.7,  -129.9, -1088.0, -1432.3, -3386.1, -3450.0, -3524.9, -3534.9,
       -3478.5, -3322.9, -3172.3, -3173.0, -3485.8, -3474.6, -3377.6, -3338.1, -3346.9, -3323.2,
       -3332.8, -3348.0, -3326.2, -3248.5, -3205.8, -3292.8, -3125.2,  -576.5,   -10.0,   -95.3,
         -54.2,   -34.1,   -10.1, ),
      (   17.2,     5.1,   -26.6,   -16.6,  -984.4, -1684.9, -2579.5, -3005.0, -3172.7, -3123.2,
       -2726.5, -2209.1, -2458.3, -2407.9, -3268.3, -3300.9, -3207.9, -3135.3, -3060.5, -2991.1,
       -3029.2, -3116.5, -3162.7, -3184.2, -3205.8, -3252.9, -2952.0,  -322.8,   -16.8,  -100.2,
         -44.2,   -30.0,   -12.1, ),
      (   13.1,   -10.0,   -29.0,   -98.8, -1000.0, -1589.9, -1674.0, -1989.6, -1577.0, -1851.7,
       -1797.0, -1904.0, -1999.0, -2290.0, -2206.1, -2795.8, -2937.9, -2917.3, -2723.1, -2597.1,
       -2752.5, -2909.1, -2999.6, -3094.5, -3177.6, -3381.8, -2460.4,  -200.7,   -85.4,   -73.2,
         -36.1,   -16.1,     0.9, ),
      (   14.0,    -9.7,   -41.8,  -140.8,  -789.2, -1386.5, -1399.6, -1211.4, -1293.0, -1023.3,
       -1212.6, -1503.1, -1733.2, -1991.2, -1894.8, -1864.4, -2391.9, -2511.8, -2282.3, -2586.4,
       -2724.9, -2879.5, -2959.6, -3129.6, -3203.8, -3205.9,  -817.6,  -102.9,   -89.2,   -50.3,
         -33.1,    -9.1,     7.8, ),
      (   19.3,    -9.6,   -28.7,   -67.7,  -205.8,  -796.4,  -994.9,  -950.1,  -881.5,  -601.2,
        -796.9,  -869.5,  -814.3, -1010.9, -1083.3, -1199.0, -1203.9, -1873.3, -1800.9, -2018.7,
       -2598.2, -2898.5, -3047.9, -3258.1, -3224.9, -1994.5,  -406.1,  -116.8,   -65.2,   -39.1,
         -17.2,    13.8,    20.2, ),
      (   41.6,    14.1,    -5.8,   -28.8,   -43.9,   -61.6,   -58.8,   -89.9,   -80.0,  -100.0,
        -103.2,  -105.9,  -134.1,  -203.9,  -211.7,  -228.9,  -537.9, -1026.6, -1397.3, -2180.6,
       -2476.4, -2800.7, -2668.6, -2293.7, -1026.3,  -600.2,  -230.9,   -80.6,   -44.1,   -29.0,
          -9.1,     8.9,    38.4, ),
      (  105.0,    44.1,     7.2,     0.0,    -7.9,   -27.9,   -31.0,   -28.0,   -41.9,   -47.0,
         -47.0,   -49.0,   -49.0,   -39.0,   -29.0,   -37.9,  -196.6,  -395.4,  -885.8, -1677.7,
       -2276.7, -1953.6,  -873.2,  -570.8,  -303.7,  -200.0,  -100.1,   -50.1,   -33.1,   -23.1,
         -10.1,    36.6,    35.8, ),
      (   85.3,    81.1,    31.2,    15.0,    11.1,     2.1,    -7.0,   -17.9,   -19.0,   -20.0,
         -22.0,   -25.0,   -18.0,   -10.0,    -3.0,    -8.0,   -24.8,   -11.4,   -77.7,  -502.6,
       -1330.0, -1583.2,  -704.2,  -377.0,  -219.5,   -99.1,   -42.1,   -34.1,   -28.0,   -14.1,
          -4.1,    16.4,    25.0, ),
      (  189.7,    83.3,    79.0,    45.2,    29.9,    15.0,   -10.7,    -1.3,   -12.0,    -9.0,
         -12.0,    -9.0,    -1.0,    -1.0,     1.0,     1.0,    -0.3,    -9.0,   -13.9,   -42.7,
         -51.4,  -109.8,  -397.0,  -187.9,   -58.6,   -20.2,    -9.0,   -20.1,   -19.0,    -4.2,
          11.8,    24.0,    36.4, ),
      (  210.0,   148.8,   104.6,    76.4,    56.2,    26.2,    14.1,     8.1,     1.1,     2.9,
           5.0,     3.1,     6.0,     1.0,     3.0,     3.0,    -9.6,    -0.3,   -10.0,   -24.8,
         -28.1,   -28.0,   -95.3,   -58.0,   -31.1,     4.8,     8.9,     0.1,    -1.0,    16.0,
          12.0,    32.1,    24.7, ),
      (  259.9,   163.1,   111.3,    66.4,   105.2,    97.8,    20.5,    30.1,    20.2,    17.1,
          12.0,    14.0,     8.0,     7.0,     9.0,     6.1,    15.9,    20.1,     9.0,    14.8,
          -1.0,     4.2,    -0.8,     3.8,    17.9,    46.0,    53.6,    21.5,    39.1,    32.4,
          34.1,    31.9,    19.2, ),
      (  272.2,   149.6,   114.9,   130.2,    88.9,    55.0,    95.1,    92.3,    93.1,    58.1,
          52.1,    23.1,    11.9,    71.1,    66.6,    91.5,    70.5,    76.5,    59.7,    54.1,
          11.9,    68.2,    64.6,    76.9,    37.2,    37.3,    44.9,    63.2,    76.9,    67.6,
          51.2,    33.1,     9.2, ),
)


def createExampleCurvData():
    '''
    Creates and returns example longitude, latitudes, depth, and data
    for a curvilinear grid.  Assigns grid center point data[i,j,k]
        = -2 * sin(lon[i,j,k]) * cos(lat[i,j,k]) / log(depth[i,j,k] + 1.0)
            for valid center points
        = 1.0E20 for invalid center points

    Arguments:
        None
    Returns:
        (corner_lons, corner_lats, corner_depths, corner_invalid,
         center_lons, center_lats, center_depths, center_invalid, data) where:
        corner_lons:    numpy 3D array of curvilinear corner point longitude coordinates
        corner_lats:    numpy 3D array of curvilinear corner point latitude coordinates
        corner_depths:  numpy 3D array of curvilinear corner point depth coordinates
        corner_invalid: numpy 3D array of logicals indicating invalid corner points
        center_lons:    numpy 3D array of curvilinear center point longitude coordinates
        center_lats:    numpy 3D array of curvilinear center point latitude coordinates
        center_depths:  numpy 3D array of curvilinear center point depth coordinates
        center_invalid: numpy 3D array of logicals indicating invalid center points
        data:           numpy 3D array of curvilinear center point data values
    '''
    # Longitudes for the Gulf of Mexico ROSE data
    start = GULF_MEX_LONS[0]
    end = GULF_MEX_LONS[1]
    delta = GULF_MEX_LONS[2]
    # numpy.arange does not include the ending value
    end += 0.5 * delta
    center_lons = numpy.arange(start, end, delta)
    corner_lons = numpy.arange(start - 0.5 * delta, end + 0.5 * delta, delta)

    # Latitudes for the Gulf of Mexico ROSE data
    start = GULF_MEX_LATS[0]
    end = GULF_MEX_LATS[1]
    delta = GULF_MEX_LATS[2]
    # numpy.arange does not include the ending value
    end += 0.5 * delta
    center_lats = numpy.arange(start, end, delta)
    corner_lats = numpy.arange(start - 0.5 * delta, end + 0.5 * delta, delta)

    # arbitrary depths as sigma coordinates
    center_sigmas = numpy.array(    (0.200, 0.030, 0.075, 0.150, 0.300, 0.450, 0.600, 0.800) )
    corner_sigmas = numpy.array( (0.015, 0.025, 0.053, 0.113, 0.225, 0.375, 0.525, 0.700, 0.900) )
    center_shape = (center_lons.shape[0], center_lats.shape[0], center_sigmas.shape[0])
    corner_shape = (corner_lons.shape[0], corner_lats.shape[0], corner_sigmas.shape[0])

    # Expand longitudes, latitudes, and sigmas to 3D arrays
    center_lons = numpy.tile(center_lons, center_shape[1] * center_shape[2]) \
                       .reshape(center_shape, order='F')
    center_lats = numpy.tile(numpy.repeat(center_lats, center_shape[0]),
                             center_shape[2]) \
                       .reshape(center_shape, order='F')
    center_sigmas = numpy.repeat(center_sigmas, center_shape[0] * center_shape[1]) \
                         .reshape(center_shape, order='F')
    corner_lons = numpy.tile(corner_lons, corner_shape[1] * corner_shape[2]) \
                       .reshape(corner_shape, order='F')
    corner_lats = numpy.tile(numpy.repeat(corner_lats, corner_shape[0]),
                             corner_shape[2]) \
                       .reshape(corner_shape, order='F')
    corner_sigmas = numpy.repeat(corner_sigmas, corner_shape[0] * corner_shape[1]) \
                         .reshape(corner_shape, order='F')

    # Add a zeta coordinate (sea swell above earth radius) - less than 1.0 meter
    zeta = -1.0 * numpy.sin(numpy.deg2rad(center_lons)) * numpy.cos(numpy.deg2rad(center_lats))

    # Convert sigma, zeta coordinates to center depths
    # Modify the depths so nothing more than 325 meters deep to better match rectilinear data
    gulf_mex_center_rose = numpy.array(GULF_MEX_ROSE).flatten()
    gulf_mex_center_rose[ (gulf_mex_center_rose < -325.0) ] = -325.0
    gulf_mex_center_rose = numpy.tile(gulf_mex_center_rose, center_shape[2]) \
                                .reshape(center_shape, order='F')
    center_invalid = ( gulf_mex_center_rose > -10.0 )
    center_depths = center_sigmas * (zeta - gulf_mex_center_rose) - zeta

    # Compute the data values for these center points
    insea = numpy.logical_not(center_invalid)
    data = numpy.empty(center_shape, dtype=numpy.float64, order='F')
    data[center_invalid] = 1.0E20
    data[insea] = -2.0 * numpy.sin(numpy.deg2rad(center_lons[insea] + 20.0)) \
                       * numpy.cos(numpy.deg2rad(center_lats[insea])) \
                       / numpy.log10(center_depths[insea] + 10.0)

    # Convert sigma, zeta coordinates to corner depths
    # Estimate the corner depths using the center depths;
    # edges not too critical since most are over land
    gulf_mex_corner_rose = numpy.zeros((corner_shape[0], corner_shape[1]),
                                       dtype=numpy.float64, order='F')
    gulf_mex_corner_rose[1:-1,1:-1] = ( gulf_mex_center_rose[:-1, :-1, 0] + \
                                        gulf_mex_center_rose[1:,  :-1, 0] + \
                                        gulf_mex_center_rose[1:,  1:,  0] + \
                                        gulf_mex_center_rose[:-1, 1:,  0] ) / 4.0
    gulf_mex_corner_rose[ 0, 1:-1] = 2.0 * gulf_mex_corner_rose[ 1, 1:-1] - gulf_mex_corner_rose[ 2, 1:-1]
    gulf_mex_corner_rose[-1, 1:-1] = 2.0 * gulf_mex_corner_rose[-2, 1:-1] - gulf_mex_corner_rose[-3, 1:-1]
    gulf_mex_corner_rose[:,  0] = 2.0 * gulf_mex_corner_rose[:,  1] - gulf_mex_corner_rose[:,  2]
    gulf_mex_corner_rose[:, -1] = 2.0 * gulf_mex_corner_rose[:, -2] - gulf_mex_corner_rose[:, -3]
    gulf_mex_corner_rose = numpy.tile(gulf_mex_corner_rose.flatten('F'), corner_shape[2]) \
                                .reshape(corner_shape, order='F')
    zeta = -1.0 * numpy.sin(numpy.deg2rad(corner_lons)) * numpy.cos(numpy.deg2rad(corner_lats))
    corner_invalid = ( gulf_mex_corner_rose > -10.0 )
    corner_depths = corner_sigmas * (zeta - gulf_mex_corner_rose) - zeta

    return (corner_lons, corner_lats, corner_depths, corner_invalid,
            center_lons, center_lats, center_depths, center_invalid, data)


def createExampleRectData():
    '''
    Creates and returns example longitude, latitudes, depth, and data
    for a rectilinear grid.  Covers approximately the same region given
    by createExampleCurvData.  Assigns grid center point data[i,j,k]
        = -2 * sin(lon[i]) * cos(lat[j]) / log(depth[k] + 1.0)
            for valid center points
        = 1.0E34 for invalid center points

    Arguments:
        None
    Returns:
        (corner_lons, corner_lats, corner_depths,
         center_lons, center_lats, center_depths, data) where:
        corner_lons:    numpy 1D array of rectilinear corner plane longitudes
        corner_lats:    numpy 1D array of rectilinear corner plane latitudes
        corner_depths:  numpy 1D array of rectilinear corner plane depths
        corner_invalid: numpy 3D array of logicals indicating invalid corner points
        center_lons:    numpy 1D array of rectilienar center plane longitudes
        center_lats:    numpy 1D array of rectilinear center plane latitudes
        center_depths:  numpy 1D array of rectilinear center plane depths
        center_invalid: numpy 3D array of logicals indicating invalid center points
        data:           numpy 3D array of rectilinear center point data values
    '''
    # Longitudes for the Gulf of Mexico ROSE data
    start = GULF_MEX_LONS[0]
    end = GULF_MEX_LONS[1]
    delta = GULF_MEX_LONS[2]
    # numpy.arange does not include the ending value
    end += 0.5 * delta
    center_lons = numpy.arange(start, end, delta)
    corner_lons = numpy.arange(start - 0.5 * delta, end + 0.5 * delta, delta)
    # Latitudes for the Gulf of Mexico ROSE data
    start = GULF_MEX_LATS[0]
    end = GULF_MEX_LATS[1]
    delta = GULF_MEX_LATS[2]
    # numpy.arange does not include the ending value
    end += 0.5 * delta
    center_lats = numpy.arange(start, end, delta)
    corner_lats = numpy.arange(start - 0.5 * delta, end + 0.5 * delta, delta)
    # Arbitrarily chosen depths
    center_depths = numpy.array(   (10.0,  25.0,  50.0, 100.0, 150.0, 250.0) )
    # Edges for these irregularily spaced depths      
    corner_depths = numpy.array( (5.0,  17.5,  37.5,  75.0, 125.0, 200.0, 300.0) )

    # Create the data values
    center_shape = (center_lons.shape[0], center_lats.shape[0], center_depths.shape[0])
    center_3d_lons = numpy.tile(center_lons, center_shape[1] * center_shape[2]) \
                          .reshape(center_shape, order='F')
    center_3d_lats = numpy.tile(numpy.repeat(center_lats, center_shape[0]),
                                center_shape[2]) \
                          .reshape(center_shape, order='F')
    center_3d_depths = numpy.repeat(center_depths, center_shape[0] * center_shape[1]) \
                            .reshape(center_shape, order='F')
    data = -2.0 * numpy.sin(numpy.deg2rad(center_3d_lons + 20.0)) \
                * numpy.cos(numpy.deg2rad(center_3d_lats)) \
                / numpy.log10(center_3d_depths + 10.0)
           
    # Use GULD_MEX_ROSE as a land mask to mark undefined data
    gulf_mex_center_rose = numpy.array(GULF_MEX_ROSE).flatten()
    gulf_mex_center_rose = numpy.tile(gulf_mex_center_rose, center_shape[2]) \
                                .reshape(center_shape, order='F')
    center_invalid = numpy.logical_or( ( gulf_mex_center_rose > -10.0 ),
                                       ( center_3d_depths > -gulf_mex_center_rose ) )
    data[ center_invalid ] = 1.0E34

    # Convert sigma, zeta coordaintes in corner depths
    # Estimate the corner depths using the center depths;
    # edges not too critical since most are over land
    corner_shape = (corner_lons.shape[0], corner_lats.shape[0], corner_depths.shape[0])
    corner_3d_depths = numpy.repeat(corner_depths, corner_shape[0] * corner_shape[1]) \
                            .reshape(corner_shape, order='F')
    gulf_mex_corner_rose = numpy.zeros((corner_shape[0], corner_shape[1]),
                                       dtype=numpy.float64, order='F')
    gulf_mex_corner_rose[1:-1,1:-1] = ( gulf_mex_center_rose[:-1, :-1, 0] + \
                                        gulf_mex_center_rose[1:,  :-1, 0] + \
                                        gulf_mex_center_rose[1:,  1:,  0] + \
                                        gulf_mex_center_rose[:-1, 1:,  0] ) / 4.0
    gulf_mex_corner_rose[ 0, 1:-1] = 2.0 * gulf_mex_corner_rose[ 1, 1:-1] - gulf_mex_corner_rose[ 2, 1:-1]
    gulf_mex_corner_rose[-1, 1:-1] = 2.0 * gulf_mex_corner_rose[-2, 1:-1] - gulf_mex_corner_rose[-3, 1:-1]
    gulf_mex_corner_rose[:,  0] = 2.0 * gulf_mex_corner_rose[:,  1] - gulf_mex_corner_rose[:,  2]
    gulf_mex_corner_rose[:, -1] = 2.0 * gulf_mex_corner_rose[:, -2] - gulf_mex_corner_rose[:, -3]
    gulf_mex_corner_rose = numpy.tile(gulf_mex_corner_rose.flatten('F'), corner_shape[2]) \
                                .reshape(corner_shape, order='F')
    corner_invalid = numpy.logical_or( ( gulf_mex_corner_rose > -10.0 ),
                                       ( corner_3d_depths > -gulf_mex_corner_rose ) )

    return (corner_lons, corner_lats, corner_depths, corner_invalid,
            center_lons, center_lats, center_depths, center_invalid, data)


def printDiffs(grid_lons, grid_lats, grid_depths, undef_val, max_negl,
                 expect_data, found_data):
    '''
    Prints significant differences between expect_data and found_data
    along with the location of these differences

    Arguments:
        grid_lons:   numpy 3D array of grid longitudes
        grid_lats:   numpy 3D array of grid latitudes
        grid_depths: numpy 3D array of grid depths
        undef_val:   numpy array of one value; the undefined data value
        max_negl:    maximumum negligible absolute difference
        expect_data: numpy 3D array of expected data values
        found_data:  numpy 3D array of data values to check
    Returns:
        None
    Raises:
        ValueError:  if the array shapes do not match
    '''
    if len(grid_lons.shape) != 3:
        raise ValueError("grid_lons is not 3D")
    if grid_lats.shape != grid_lons.shape:
        raise ValueError("grid_lats.shape != grid_lons.shape")
    if grid_depths.shape != grid_lons.shape:
        raise ValueError("grid_depth.shape != grid_lons.shape")
    if expect_data.shape != grid_lons.shape:
        raise ValueError("expect_data.shape != grid_lons.shape")
    if found_data.shape != grid_lons.shape:
        raise ValueError("found_data.shape != grid_lons.shape")
    different = ( numpy.abs(expect_data - found_data) > max_negl )
    diff_lons = grid_lons[different]
    diff_lats = grid_lats[different]
    diff_depths = grid_depths[different]
    diff_expect = expect_data[different]
    diff_found = found_data[different]
    diff_list = [ ]
    for (lon, lat, depth, expect, found) in \
            zip(diff_lons, diff_lats, diff_depths, diff_expect, diff_found):
        if expect == undef_val:
            # most serious - should have been masked out
            diff_list.append([2, lon, lat, depth, expect, found])
        elif found == undef_val:
            # least serious - destination not covered by source
            diff_list.append([0, lon, lat, depth, expect, found])
        else:
            # might be of concern
            diff_list.append([1, lon, lat, depth, expect, found])
    # order primarily from least to most serious, 
    # secondarily smallest to largest longitude,
    # thirdly smallest to largest latitude
    # fourthly highest to deepest
    diff_list.sort()
    num_not_undef = 0
    num_undef = 0
    num_diff = 0
    for (_, lon, lat, depth, expect, found) in diff_list:
        if expect == undef_val:
            num_not_undef += 1
            print "lon = %#7.3f, lat = %7.3f, depth = %7.2f, expect =  undef, " \
                  "found = %#6.3f" % (lon, lat, depth, found)
        elif found == undef_val:
            num_undef += 1
            # print "lon = %#7.3f, lat = %7.3f, depth = %7.2f, expect = %#6.3f, " \
            #       "found =  undef" % (lon, lat, depth, expect)
        else:
            num_diff += 1
            print "lon = %#7.3f, lat = %7.3f, depth = %7.2f, expect = %#6.3f, " \
                  "found = %#6.3f, diff = %#6.3f" \
                  % (lon, lat, depth, expect, found, found - expect)
    print "%3d undefined when defined might be expected" % num_undef
    print "%3d with absolute difference > %#.3f" % (num_diff, max_negl)
    print "%3d defined when undefined expected" % num_not_undef
    print "%3d values in the grid" \
            % (expect_data.shape[0] * expect_data.shape[1] * expect_data.shape[2])


if __name__ == '__main__':
    try:
        while True:
            print 'cw2r: curvilinear with corners to rectilinear'
            print 'co2r: curvilinear without corners to rectilinear'
            print 'r2cw: rectilinear to curvilinear with corners'
            print 'r2co: rectilinear to curvilinear without corners'
            print 'Ctrl-D to quit'
            direction = raw_input('Regrid test to run? ')
            direction = direction.strip().lower()
            if direction in ('cw2r', 'co2r', 'r2cw', 'r2co'):
                break
    except EOFError:
        raise SystemExit(0)

    # Synthesize curvilinear test data
    (curv_corner_lons, curv_corner_lats, curv_corner_depths, curv_corner_ignore,
     curv_center_lons, curv_center_lats, curv_center_depths, curv_center_ignore,
     curv_data) = createExampleCurvData()

    # Synthesize rectilinear test data
    (rect_corner_lons, rect_corner_lats, rect_corner_depths, rect_corner_ignore,
     rect_center_lons, rect_center_lats, rect_center_depths, rect_center_ignore,
     rect_data) = createExampleRectData()

    undef_val = numpy.array([-1.0E10], dtype=numpy.float64)

    # Create the expected results on the curvilinear grid
    curv_expect_data = curv_data.copy()
    curv_expect_data[curv_center_ignore] = undef_val

    # Create the expected results on the rectilinear grid
    rect_expect_data = rect_data.copy()
    rect_expect_data[rect_center_ignore] = undef_val

    # Initialize ESMP
    if not ESMPControl().startCheckESMP():
        raise RuntimeError("Unexpected failure to start ESMP")

    # Create the regridder
    regridder = CurvRect3DRegridder()

    if direction in ('cw2r', 'r2cw'):
        # Create the curvilinear grid with corner and center points
        regridder.createCurvGrid(curv_center_lons, curv_center_lats, curv_center_depths,
                                 curv_center_ignore, True, curv_corner_lons, curv_corner_lats,
                                 curv_corner_depths, curv_corner_ignore)
    elif direction in ('co2r', 'r2co'):
        # Create the curvilinear grid with only center points
        regridder.createCurvGrid(curv_center_lons, curv_center_lats, curv_center_depths,
                                 curv_center_ignore, True)
    else:
        raise ValueError("unexpected direction of %s" % direction)

    # Create the rectilinear grid with corner and center points
    regridder.createRectGrid(rect_center_lons, rect_center_lats, rect_center_depths,
                             rect_center_ignore, True, rect_corner_lons, rect_corner_lats,
                             rect_corner_depths, rect_corner_ignore)

    if direction in ('cw2r', 'co2r'):

        print ""
        if direction == 'cw2r':
            print "Examining rectilinear results from curvilinear with corners"
        else:
            print "Examining rectilinear results from curvilinear without corners"

        # Create the curvilinear source field
        regridder.assignCurvField(curv_data)

        # Create the rectilinear destination field
        regridder.assignRectField()

        # Generate the 3D rectilinear longitude, latitude, and depth arrays
        # only to simplify printing differences; not used in regridding
        rect_center_shape = (rect_center_lons.shape[0], rect_center_lats.shape[0],
                             rect_center_depths.shape[0])
        rect_3d_center_lons = numpy.tile(rect_center_lons,
                                         rect_center_shape[1] * rect_center_shape[2]) \
                                   .reshape(rect_center_shape, order='F')
        rect_3d_center_lats = numpy.tile(numpy.repeat(rect_center_lats,
                                                      rect_center_shape[0]),
                                         rect_center_shape[2]) \
                                   .reshape(rect_center_shape, order='F')
        rect_3d_center_depths = numpy.repeat(rect_center_depths,
                                             rect_center_shape[0] * rect_center_shape[1]) \
                                     .reshape(rect_center_shape, order='F')

        if direction == 'cw2r':
            # Regrid from curvilinear to rectilinear using the conserve method
            # Corners required for this method
            rect_regrid_data = regridder.regridCurvToRect(undef_val, ESMP.ESMP_REGRIDMETHOD_CONSERVE)
            # Print the differences between the expected and regrid data
            print ""
            print "analytic (expect) versus conserve regridded (found) differences"
            printDiffs(rect_3d_center_lons,
                         rect_3d_center_lats,
                         rect_3d_center_depths,
                         undef_val, 0.001,
                         rect_expect_data,
                         rect_regrid_data)

        # Regrid from curvilinear to rectilinear using the bilinear method
        rect_regrid_data = regridder.regridCurvToRect(undef_val, ESMP.ESMP_REGRIDMETHOD_BILINEAR)
        # Print the differences between the expected and regrid data
        print ""
        print "analytic (expect) versus bilinear regridded (found) differences"
        printDiffs(rect_3d_center_lons,
                     rect_3d_center_lats,
                     rect_3d_center_depths,
                     undef_val, 0.12,
                     rect_expect_data,
                     rect_regrid_data)

        # Regrid from curvilinear to rectilinear using the patch method
        rect_regrid_data = regridder.regridCurvToRect(undef_val, ESMP.ESMP_REGRIDMETHOD_PATCH)
        # Print the differences between the expected and regrid data
        print ""
        print "analytic (expect) versus patch regridded (found) differences"
        printDiffs(rect_3d_center_lons,
                     rect_3d_center_lats,
                     rect_3d_center_depths,
                     undef_val, 1.0, 
                     rect_expect_data,
                     rect_regrid_data)

    elif direction in ('r2cw', 'r2co'):

        print ""
        if direction == 'r2cw':
            print "Examining curvilinear with corners results from rectilinear"
        else:
            print "Examining curvilinear without corners results from rectilinear"

        # Create the rectilinear source field
        regridder.assignRectField(rect_data)

        # Create the curvilinear destination field
        regridder.assignCurvField(None)

        if direction == 'r2cw':
            # Regrid from rectilinear to curvilinear using the conserve method
            # Corners required for this method
            curv_regrid_data = regridder.regridRectToCurv(undef_val, ESMP.ESMP_REGRIDMETHOD_CONSERVE)
            # Print the differences between the expected and regrid data
            print ""
            print "analytic (expect) versus conserve regridded (found) differences"
            printDiffs(curv_center_lons[:,:,1:-1],
                         curv_center_lats[:,:,1:-1],
                         curv_center_depths[:,:,1:-1],
                         undef_val, 0.001,
                         curv_expect_data[:,:,1:-1],
                         curv_regrid_data[:,:,1:-1])

        # Regrid from rectilinear to curvilinear using the bilinear method
        curv_regrid_data = regridder.regridRectToCurv(undef_val, ESMP.ESMP_REGRIDMETHOD_BILINEAR)
        # Print the differences between the expected and regrid data
        print ""
        print "analytic (expect) versus bilinear regridded (found) differences"
        printDiffs(curv_center_lons[:,:,1:-1],
                     curv_center_lats[:,:,1:-1],
                     curv_center_depths[:,:,1:-1],
                     undef_val, 0.25,
                     curv_expect_data[:,:,1:-1],
                     curv_regrid_data[:,:,1:-1])

        # Regrid from rectilinear to curvilinear using the patch method
        curv_regrid_data = regridder.regridRectToCurv(undef_val, ESMP.ESMP_REGRIDMETHOD_PATCH)
        # Print the differences between the expected and regrid data
        print ""
        print "analytic (expect) versus patch regridded (found) differences"
        printDiffs(curv_center_lons[:,:,1:-1],
                     curv_center_lats[:,:,1:-1],
                     curv_center_depths[:,:,1:-1],
                     undef_val, 1.1,
                     curv_expect_data[:,:,1:-1],
                     curv_regrid_data[:,:,1:-1])

    else:
        raise ValueError("unexpected direction of %s" % direction)

    # Done with this regridder
    regridder.finalize()

    # Done with ESMP    
    ESMPControl().stopESMP(True)

