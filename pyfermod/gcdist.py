"""
Module for computing and using great circle distances
between points given in longitudes and latitudes.
"""

import math
import numpy
import numpy.random

# Equitorial radius of the earth in kilometers
EARTH_ER = 6378.137

# Authalic radius of the earth in kilometers
EARTH_AR = 6371.007

# Meridional radius of the earth in kilometers
EARTH_MR = 6367.449

# Polar radius of the earth in kilometers
EARTH_PR = 6356.752

DEG2RAD = math.pi / 180.0
RAD2DEG = 180.0 / math.pi
KM2MI = 0.6213712
MI2KM = 1.609344

def lonlatdist(pt1lon, pt1lat, pt2lon, pt2lat):
    """
    Compute the great circle distance between two points
    on a sphere using the haversine formula.

    Arguments:
        pt1lon - longitude(s) of the first point
        pt1lat - latitude(s) of the first point
        pt2lon - longitude(s) of the second point
        pt2lat - latitude(s) of the second point
    Returns:
        The great circle distance(s) in degrees [0.0, 180.0]
    """
    lon1 = numpy.deg2rad(numpy.array(pt1lon, dtype=float))
    lat1 = numpy.deg2rad(numpy.array(pt1lat, dtype=float))
    lon2 = numpy.deg2rad(numpy.array(pt2lon, dtype=float))
    lat2 = numpy.deg2rad(numpy.array(pt2lat, dtype=float))
    dellat = numpy.power(numpy.sin(0.5 * (lat2 - lat1)), 2.0)
    dellon = numpy.cos(lat1) * numpy.cos(lat2) * \
             numpy.power(numpy.sin(0.5 * (lon2 - lon1)), 2.0)
    dist = 2.0 * numpy.arcsin(numpy.power(dellon + dellat, 0.5))
    return numpy.rad2deg(dist)


def equidistscatter(min_lon, min_lat, max_lon, max_lat, min_gcdist, dfactor=5.0):
    """
    Create a roughly equidistance set of points in a specified region.

    This is done by creating a dense "grid" of points, then repeatedly 
    randomly selecting a point from that collection and eliminating 
    points too close to that selected point.  For the special cases 
    where min_lon and max_lon, or min_lat and max_lat, are very close 
    relative to min_gcdist, the maximum number of evenly spaced points 
    that can be put on the line described is computed and assigned.

    Arguments:
        min_lon - minimum longitude of the region
        min_lat - minimum latitude of the region
        max_lon - maximum longitude of the region
        max_lat - maximum latitude of the region
        min_gcdist - minimum distance, in great circle degrees,
                  between returned points
        dfactor - the number of axis points in the dense "grid"
                  compared to the desired "grid".  Larger value will 
                  generally increase the uniformity of the returned 
                  points but will also increase the time required 
                  for the calculation.
    Returns:
        (pt_lons, pt_lats) - ptlons is an array of longitudes and ptlats 
                  is an array of latitudes of (somewhat random) points in
                  the specified region that are roughly equidistance from 
                  each other but not closer than min_gcdist to each other.
    """
    lonmin = float(min_lon)
    lonmax = float(max_lon)
    if math.fabs(lonmax - lonmin) > 180.0:
        raise ValueError("Difference between max_lon and min_lon is more than 180.0")
    latmin = float(min_lat)
    if math.fabs(latmin) > 90.0:
        raise ValueError("min_lat is not in [-90.0,90.0]")
    latmax = float(max_lat)
    if math.fabs(latmax) > 90.0:
        raise ValueError("max_lat is not in [-90.0,90.0]")
    mindeg = float(min_gcdist)
    if (mindeg <= 0.0) or (mindeg >= 90.0):
        raise ValueError("min_gcdist is not in (0.0,90.0)")
    dfact = float(dfactor)
    if dfact < 1.0:
        raise ValueError("dfactor is less than one");

    # If lonmin is relatively close to lonmax, directly
    # compute the points.  Distance on a meridian is the 
    # differnce in latitudes.
    if math.fabs(lonmax - lonmin) < (0.05 * mindeg):
        lon = 0.5 * (lonmax + lonmin)
        dellat = mindeg
        numlats = int( (math.fabs(latmax - latmin) + dellat) / dellat )
        if latmax < latmin:
            dellat *= -1.0
        hdiff = 0.5 * ( (latmax - latmin) - (numlats - 1) * dellat )
        latvals = numpy.linspace(latmin + hdiff, latmax - hdiff, numlats)
        lonvals = numpy.ones((numlats,), dtype=float) * lon
        return (lonvals, latvals)

    # If latmin is relatively close to latmax, directly
    # compute the points.  Distance depends on the latitude
    # as well as the differnce in longitudes.
    if math.fabs(latmax - latmin) < (0.05 * mindeg):
        lat = 0.5 * (latmax + latmin)
        numer = math.sin(0.5 * DEG2RAD * mindeg)
        denom = math.cos(lat * DEG2RAD)
        if numer < denom:
            dellon = math.asin(numer / denom) * 2.0 * RAD2DEG
            numlons = int( (math.fabs(lonmax - lonmin) + dellon) / dellon )
        else:
            # everything too close to a pole - just select one point
            dellon = 180.0
            numlons = 1
        if lonmax < lonmin:
            dellon *= -1.0
        hdiff = 0.5 * ( (lonmax - lonmin) - (numlons - 1) * dellon )
        lonvals = numpy.linspace(lonmin + hdiff, lonmax - hdiff, numlons)
        latvals = numpy.ones((numlons,), dtype=float) * lat
        return (lonvals, latvals)

    # Get the number of latitudes for the dense grid
    # Always user latmin and latmax, even if they are too close
    dellat = mindeg / dfact
    numlats = int( (math.fabs(latmax - latmin) + dellat) / dellat )
    if numlats < 2:
         numlats = 2
    latvals = numpy.linspace(latmin, latmax, numlats)

    # Create the dense grid of longitudes and latitudes
    denslons = [ ]
    denslats = [ ]
    numer = math.sin(0.5 * DEG2RAD * mindeg / dfact)
    for lat in latvals:
        # Get the number of longitudes for the dense grid
        # Always use lonmin and lonmax, even if they are too close
        denom = math.cos(lat * DEG2RAD)
        if numer < denom:
            dellon = math.asin(numer / denom) * 2.0 * RAD2DEG
            numlons = int( (math.fabs(lonmax - lonmin) + dellon) / dellon )
            if numlons < 2:
                numlons = 2
        else:
            # too close to a pole
            numlons = 2
        lonvals = numpy.linspace(lonmin, lonmax, numlons)
        # Add each lon,lat pair to the dense grid
        for lon in lonvals:
            denslons.append(lon)
            denslats.append(lat)
    denslons = numpy.array(denslons)
    denslats = numpy.array(denslats)

    # create a random permutation of the indices to use for the selection order
    availinds = numpy.random.permutation(len(denslats))
    selectinds = [ ]
    while len(availinds) > 0:
        # Get the index of the next available point
        ind = availinds[0]
        selectinds.append(ind)
        # Compute the distance of the available points to the selected point
        gcdists = lonlatdist(denslons[ind], denslats[ind], denslons[availinds], denslats[availinds])
        # Remove indices of any remaining points too close to this point
        availinds = availinds[ gcdists >= mindeg ]
    # sort the selected indices so the longitudes and latitudes have some order
    selectinds = numpy.sort(selectinds)
    # get the selected longitudes and latitudes
    selectlons = denslons[selectinds]
    selectlats = denslats[selectinds]
    # return the selected longitudes and latitudes arrays
    return (selectlons, selectlats)


if __name__ == "__main__":
    # Test lonlatdist
    tenten = numpy.linspace(0.0,90.0,10)
    # On the equator, distance = delta longitude
    dists = lonlatdist(0.0, 0.0, tenten, 0.0)
    if not numpy.allclose(dists, tenten):
        raise ValueError("Equatorial distances FAIL; expect: %s; found: %s" % (str(tenten), str(dists)))
    print "Equatorial distance: PASS"
    print

    # On any meridian, distance = delta latitude
    dists = lonlatdist(20.0, 0.0, 20.0, tenten)
    if not numpy.allclose(dists, tenten):
        raise ValueError("Meridional distances FAIL; expect: %s; found: %s" % (str(tenten), str(dists)))
    print "Meridional distance: PASS"
    print

    # Play with some distances between cities
    seattle =  (122.0 + (20.0 / 60.0), 47.0 + (37.0 / 60.0))
    portland = (122.0 + (41.0 / 60.0), 45.0 + (31.0 / 60.0))
    spokane =  (117.0 + (26.0 / 60.0), 47.0 + (40.0 / 60.0))
    austin =   ( 97.0 + (45.0 / 60.0), 30.0 + (15.0 / 60.0))
    houston =  ( 95.0 + (23.0 / 60.0), 29.0 + (46.0 / 60.0))
    dallas =   ( 96.0 + (48.0 / 60.0), 32.0 + (47.0 / 60.0))

    lons = ( seattle[0], portland[0], spokane[0] )
    lons1, lons2 = numpy.meshgrid(lons, lons)
    lats = ( seattle[1], portland[1], spokane[1] )
    lats1, lats2 = numpy.meshgrid(lats, lats)
    dists = lonlatdist(lons1, lats1, lons2, lats2)
    dists *= DEG2RAD * EARTH_MR * KM2MI
    print "Seattle:  %.2f W, %.2f N" % seattle
    print "Portland: %.2f W, %.2f N" % portland
    print "Spokane:  %.2f W, %.2f N" % spokane
    print "Computed distances (mi)"
    print "              Seattle     Portland    Spokane"
    print "   Seattle     %5.0f       %5.0f       %5.0f" % (dists[0,0], dists[0,1], dists[0,2])
    print "   Portland    %5.0f       %5.0f       %5.0f" % (dists[1,0], dists[1,1], dists[1,2])
    print "   Spokane     %5.0f       %5.0f       %5.0f" % (dists[2,0], dists[2,1], dists[2,2])
    print

    lons = ( austin[0], houston[0], dallas[0] )
    lons1, lons2 = numpy.meshgrid(lons, lons)
    lats = ( austin[1], houston[1], dallas[1] )
    lats1, lats2 = numpy.meshgrid(lats, lats)
    dists = lonlatdist(lons1, lats1, lons2, lats2)
    dists *= DEG2RAD * EARTH_MR * KM2MI
    print "Austin:  %.2f W, %.2f N" % austin
    print "Houston: %.2f W, %.2f N" % houston
    print "Dallas:  %.2f W, %.2f N" % dallas
    print "Computed distances (mi)"
    print "              Austin      Houston     Dallas"
    print "   Austin      %5.0f       %5.0f       %5.0f" % (dists[0,0], dists[0,1], dists[0,2])
    print "   Houston     %5.0f       %5.0f       %5.0f" % (dists[1,0], dists[1,1], dists[1,2])
    print "   Dallas      %5.0f       %5.0f       %5.0f" % (dists[2,0], dists[2,1], dists[2,2])
    print

    # Test equdistscatter
    lons, lats = equidistscatter(0.0, 0.0, 0.0, 0.0, 1.0)
    if (lons.shape != (1,)) or (lons[0] != 0.0) or \
       (lats.shape != (1,)) or (lats[0] != 0.0):
        raise ValueError("Equidistscatter single-point FAIL; \n" \
                         "  expect: ([0.0],[0.0]), \n" \
                         "  found (%s,%s)" % (str(lons), str(lats)))
    print "Equidistscatter single-point PASS"
    print

    lons, lats = equidistscatter(0.0, 90.0, 90.0, 90.0, 1.0)
    if (lons.shape != (1,)) or (lons[0] != 45.0) or \
       (lats.shape != (1,)) or (lats[0] != 90.0):
        raise ValueError("Equidistscatter pole-point FAIL; \n" \
                         "  expect: ([45.0],[90.0]), \n" \
                         "  found (%s,%s)" % (str(lons), str(lats)))
    print "Equidistscatter pole-point PASS"
    print

    lons, lats = equidistscatter(0.0, 0.0, 90.0, 0.0, 1.0)
    if not numpy.all( lats == 0.0 ):
        raise ValueError("Equidistscatter equitorial FAIL; \n" \
                         "  expect: all zero latititudes, \n" \
                         "  found %s" % str(lats))
    deltas = lons[1:] - lons[:-1]
    if not numpy.all( deltas >= 1.0 ):
        raise ValueError("Equidistscatter equitorial FAIL; \n" \
                         "  expect: longitudes monotonic increasing by at least 1.0 degrees, \n" \
                         "  found %s" % str(lons))
    if not numpy.all( deltas < 1.0001 ):
        raise ValueError("Equidistscatter equitorial FAIL; \n" \
                         "  expect: longitudes monotonic increasing by less than 1.0001 degrees, \n" \
                         "  found %s" % str(lons))
    print "Equidistscatter equitorial PASS"
    print

    lons, lats = equidistscatter(0.0, 0.0, 0.0, 90.0, 1.0)
    if not numpy.all( lons == 0.0 ):
        raise ValueError("Equidistscatter meridional FAIL; \n" \
                         "  expect: all zero longitudes, \n" \
                         "  found %s" % str(lons))
    deltas = lats[1:] - lats[:-1]
    if not numpy.all( deltas >= 1.0 ):
        raise ValueError("Equidistscatter meridional FAIL; \n" \
                         "  expect: latitudes monotonic increasing by at least 1.0 degrees, \n" \
                         "  found %s" % str(lats))
    if not numpy.all( deltas < 1.0001 ):
        raise ValueError("Equidistscatter meridional FAIL; \n" \
                         "  expect: latitudes monotonic increasing by less than 1.0001 degrees, \n" \
                         "  found %s" % str(lats))
    print "Equidistscatter meridional PASS"
    print

    lons, lats = equidistscatter(0.0, 0.0, 90.0, 90.0, 5.0, 15.0)
    nndists = [ ]
    for j in xrange(len(lons)):
        gcdists = lonlatdist(lons[j], lats[j], lons, lats)
        gcdists[j] = 180.0
        if not numpy.all( gcdists >= 5.0 ):
            raise ValueError("Equidistscatter region FAIL; \n" \
                             "  expect distances[%d] >= 2.0, \n" \
                             "  found %s" % (j, str(gcdists)))
        nndists.append(gcdists.min())
    nndists = numpy.array(nndists)
    if not numpy.all( nndists < 10.0 ):
        raise ValueError("Equidistscatter region FAIL; \n" \
                         "  expect nearest neighbor distances < 10.0, \n" \
                         "  found %s" % str(nndists))
    print "Nearest neighbor distances: \n" \
          "    min = %f, max = %f, mean = %f, stdev = %f" % \
          (nndists.min(), nndists.max(), nndists.mean(), nndists.std())

    print "Equidistscatter region PASS"
    print
