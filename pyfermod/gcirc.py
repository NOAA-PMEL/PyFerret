"""
Module of functions involving great circles
(thus assuming spheroid model of the earth)
with points given in longitudes and latitudes.
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


def lonlatdistance(pt1lon, pt1lat, pt2lon, pt2lat):
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


def lonlatintersect(gc1lon1, gc1lat1, gc1lon2, gc1lat2,
                    gc2lon1, gc2lat1, gc2lon2, gc2lat2):
    """
    Compute the intersections of two great circles.  Uses the line of
    intersection between the two planes of the great circles.

    Arguments:
        gc1lon1 - longitude(s) of the first point on the first great circle
        gc1lat1 - latitude(s) of the first point on the first great circle
        gc1lon2 - longitude(s) of the second point on the first great circle
        gc1lat2 - latitude(s) of the second point on the first great circle
        gc2lon1 - longitude(s) of the first point on the second great circle
        gc2lat1 - latitude(s) of the first point on the second great circle
        gc2lon2 - longitude(s) of the second point on the second great circle
        gc2lat2 - latitude(s) of the second point on the second great circle
    Returns:
        ( (pt1lon, pt1lat), (pt2lon, pt2lat) ) - the longitudes and latitudes
                  of the two intersections of the two great circles.  NaN will 
                  be returned for both longitudes and latitudes if a great 
                  circle is not well-defined, or the two great-circles coincide.

    If arrays are given, all arguments must be arrays with the same shape.
    The four returned arrays will have this same shape.
    """
    # Convert longitudes and latitudes to points on a unit sphere
    ptlonr = numpy.deg2rad(numpy.array(gc1lon1, dtype=float))
    ptlatr = numpy.deg2rad(numpy.array(gc1lat1, dtype=float))
    gcz = numpy.sin(ptlatr)
    coslat = numpy.cos(ptlatr)
    gcy = coslat * numpy.sin(ptlonr)
    gcx = coslat * numpy.cos(ptlonr)
    gc1xyz1 = numpy.array((gcx, gcy, gcz))
    #
    ptlonr = numpy.deg2rad(numpy.array(gc1lon2, dtype=float))
    ptlatr = numpy.deg2rad(numpy.array(gc1lat2, dtype=float))
    gcz = numpy.sin(ptlatr)
    coslat = numpy.cos(ptlatr)
    gcy = coslat * numpy.sin(ptlonr)
    gcx = coslat * numpy.cos(ptlonr)
    gc1xyz2 = numpy.array((gcx, gcy, gcz))
    #
    ptlonr = numpy.deg2rad(numpy.array(gc2lon1, dtype=float))
    ptlatr = numpy.deg2rad(numpy.array(gc2lat1, dtype=float))
    gcz = numpy.sin(ptlatr)
    coslat = numpy.cos(ptlatr)
    gcy = coslat * numpy.sin(ptlonr)
    gcx = coslat * numpy.cos(ptlonr)
    gc2xyz1 = numpy.array((gcx, gcy, gcz))
    #
    ptlonr = numpy.deg2rad(numpy.array(gc2lon2, dtype=float))
    ptlatr = numpy.deg2rad(numpy.array(gc2lat2, dtype=float))
    gcz = numpy.sin(ptlatr)
    coslat = numpy.cos(ptlatr)
    gcy = coslat * numpy.sin(ptlonr)
    gcx = coslat * numpy.cos(ptlonr)
    gc2xyz2 = numpy.array((gcx, gcy, gcz))
    # Get the unit-perpendicular to the plane going through
    # the origin and the two points on each great circle.
    # If the norm of the cross product is too small, 
    # the great circle is not well-defined, so zero it out.
    gc1pp = numpy.cross(gc1xyz1, gc1xyz2, axis=0)
    norm = (gc1pp[0]**2 + gc1pp[1]**2 + gc1pp[2]**2)**0.5
    if len(norm.shape) == 0:
        if numpy.fabs(norm) < 1.0E-15:
            norm = 0.0
    else:
        norm[ numpy.fabs(norm) < 1.0E-15 ] = 0.0
    gc1pp /= norm
    gc2pp = numpy.cross(gc2xyz1, gc2xyz2, axis=0)
    norm = (gc2pp[0]**2 + gc2pp[1]**2 + gc2pp[2]**2)**0.5
    if len(norm.shape) == 0:
        if numpy.fabs(norm) < 1.0E-15:
            norm = 0.0
    else:
        norm[ numpy.fabs(norm) < 1.0E-15 ] = 0.0
    gc2pp /= norm
    # The line of intersection of the two planes is perpendicular
    # to the two plane-perpendiculars and goes through the origin.
    # Points of intersection are the points on this line one unit
    # from the origin.
    # If the norm of the cross product is too small,
    # the two planes coincide.
    pt1xyz = numpy.cross(gc1pp, gc2pp, axis=0)
    norm = (pt1xyz[0]**2 + pt1xyz[1]**2 + pt1xyz[2]**2)**0.5
    if len(norm.shape) == 0:
        if numpy.fabs(norm) < 1.0E-15:
            norm = 0.0
    else:
        norm[ numpy.fabs(norm) < 1.0E-15 ] = 0.0
    pt1xyz /= norm
    pt2xyz = -1.0 * pt1xyz
    # Convert back to longitudes and latitudes
    pt1lats = numpy.rad2deg(numpy.arcsin(pt1xyz[2]))
    pt1lons = numpy.rad2deg(numpy.arctan2(pt1xyz[1], pt1xyz[0]))
    pt2lats = numpy.rad2deg(numpy.arcsin(pt2xyz[2]))
    pt2lons = numpy.rad2deg(numpy.arctan2(pt2xyz[1], pt2xyz[0]))
    return ( (pt1lons, pt1lats), (pt2lons, pt2lats) )


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
    # Always use latmin and latmax, even if they are too close
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
        gcdists = lonlatdistance(denslons[ind], denslats[ind],
                                 denslons[availinds], denslats[availinds])
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
    # Test lonlatdistance
    tenten = numpy.linspace(0.0,90.0,10)
    # On the equator, distance = delta longitude
    dists = lonlatdistance(0.0, 0.0, tenten, 0.0)
    if not numpy.allclose(dists, tenten):
        raise ValueError("Equatorial distances FAIL; expect: %s; found: %s" % (str(tenten), str(dists)))
    print "Equatorial distance: PASS"
    print

    # On any meridian, distance = delta latitude
    dists = lonlatdistance(20.0, 0.0, 20.0, tenten)
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
    dists = lonlatdistance(lons1, lats1, lons2, lats2)
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
    dists = lonlatdistance(lons1, lats1, lons2, lats2)
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

    # Test lonlatintersect
    # Intersections of the equator with meridians
    arr000 = numpy.zeros((10,), dtype=float)
    arrn90 = -90.0 * numpy.ones((10,), dtype=float)
    ((pt1lons, pt1lats), (pt2lons, pt2lats)) = \
            lonlatintersect(arr000, arr000, tenten, arr000, \
                            arr000, arrn90, tenten, tenten)
    # First of the first great circle and last of the second great circle are not well-defined
    expvalid = numpy.array([ True ] + ([ False ]*8) + [ True ])
    validity = numpy.isnan(pt1lons)
    if not numpy.allclose(validity, expvalid):
        raise ValueError("Validity of pt1lons: expect: %s, found %s" % \
                          (str(expvalid), str(validity)))
    validity = numpy.isnan(pt1lats)
    if not numpy.allclose(validity, expvalid):
        raise ValueError("Validity of pt1lats: expect: %s, found %s" % \
                          (str(expvalid), str(validity)))
    validity = numpy.isnan(pt2lons)
    if not numpy.allclose(validity, expvalid):
        raise ValueError("Validity of pt2lons: expect: %s, found %s" % \
                          (str(expvalid), str(validity)))
    validity = numpy.isnan(pt2lats)
    if not numpy.allclose(validity, expvalid):
        raise ValueError("Validity of pt2lats: expect: %s, found %s" % \
                          (str(expvalid), str(validity)))
    if not numpy.allclose(pt1lons[1:-1], tenten[1:-1]):
        raise ValueError("Valid pt1lons: expect: %s, found %s" %\
                          (str(arr000[1:-1]), str(pt1lons[1:-1])))
    if not numpy.allclose(pt1lats[1:-1], arr000[1:-1]):
        raise ValueError("Valid pt1lats: expect: %s, found %s" %\
                          (str(arr000[1:-1]), str(pt1lats[1:-1])))
    if not numpy.allclose(pt2lons[1:-1], tenten[1:-1]-180.0):
        raise ValueError("Valid pt2lons: expect: %s, found %s" %\
                          (str(tenten[1:-1]-180.0), str(pt2lons[1:-1])))
    if not numpy.allclose(pt2lats[1:-1], arr000[1:-1]):
        raise ValueError("Valid pt2lats: expect: %s, found %s" %\
                          (str(arr000[1:-1]), str(pt2lats[1:-1])))
    print "Equator/meridian intersections PASS"
    print

    ((pt1lons, pt1lats), (pt2lons, pt2lats)) = \
            lonlatintersect( 0.0, 89.99, 180.0, 89.99,
                            90.0, 89.99, -90.0, 89.99)
    # longitudes could actually be anything, but this algorithm gives 45.0 and -135.0
    if (abs(pt1lons -  45.0) > 1.0E-8) or (abs(pt1lats - 90.0) > 1.0E-8) or \
       (abs(pt2lons + 135.0) > 1.0E-8) or (abs(pt2lats + 90.0) > 1.0E-8):
        raise ValueError("Mini north pole cross intersections: expect: %s, found %s" % \
                         (str([45.0, 90.0, 135.0, -90.0]),
                          str([float(pt1lons), float(pt1lats),
                               float(pt2lons), float(pt2lats)])))
    print "Mini north pole cross intersections PASS"
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
        gcdists = lonlatdistance(lons[j], lats[j], lons, lats)
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
