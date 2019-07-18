
	SUBROUTINE EARTH_DEL (lons, lats, num, bad_lon, bad_lat, result, errtxt)

* Ansley Manke 2/27/2019
* This computation is pulled out of what was the external function earth_distance, for general
* use for instance in transformations for DSG data.

* Compute the point-to-point-distance in Km along the path given by longitudes, latitudes

* lons, lats	1-D arrays with the longitude and latitude values
* num		Number of data points
* bad_lon	Bad-data flags on input
* bad_lat
* result	result array
* errtxt	Error message if bad-data found in the locations

	INTEGER num
	REAL lons(*), lats(*), result(*), bad_lon, bad_lat
	CHARACTER*(*) errtxt

	INTEGER i
	REAL	dlo, dhi, ed_pd, ed_rearth, 
     .		ed_lat1, ed_lat2, ed_lon1, ed_lon2, 
     .		ed_x1, ed_x2, ed_y1, ed_y2, 
     .		ed_z1, ed_z2, ed_d1, ed_angle

* To first point, no distance
	
	result(1) = 0.
	IF (num .EQ. 1) RETURN

* initialize

	errtxt = ' '
	ed_pd = 4.*atan(1.)/180.
	ed_rearth = 6371.2 

* At each i, compute the distance from the previous point to this one

* pre-check that the location arrays have no missing values

	DO i = 1, num
	   IF (lons(i) .EQ. bad_lon) THEN
	      errtxt = 'missing value found in longitudes'
	   ENDIF
	ENDDO
    
	DO i = 1, num
	   IF (lats(i) .EQ. bad_lat) THEN
	      errtxt = 'missing value found in latitudes'
	   ENDIF
	ENDDO
	

	DO i = 2, num
	
	   result(i) = 0.
	   IF (lons(i) .EQ. bad_lon) CYCLE
	   IF (lats(i) .EQ. bad_lat) CYCLE

	   ed_lat1 = 90 - lats(i-1)
	   ed_lat2 = 90 - lats(i)
	   ed_lon1 = lons(i-1)
	   ed_lon2 = lons(i)

	   ed_x1 = SIN( ed_pd*ed_lat1 )* COS( ed_pd* ed_lon1 )
	   ed_y1 = SIN( ed_pd*ed_lat1 )* SIN( ed_pd* ed_lon1 )
	   ed_z1 = COS( ed_pd*ed_lat1 )
	   ed_x2 = SIN( ed_pd*ed_lat2 )* COS( ed_pd* ed_lon2 )
	   ed_y2 = SIN( ed_pd*ed_lat2 )* SIN( ed_pd* ed_lon2 )
	   ed_z2 = COS( ed_pd*ed_lat2 )

	   ed_d1 = (ed_x1 - ed_x2)**2 + (ed_y1 - ed_y2)**2  +  (ed_z1 - ed_z2)**2
	   ed_angle = ACOS(1 - ed_d1/2)	   ! angular distance between the points

	   result(i) = ed_rearth* ed_angle

	ENDDO

 5000	RETURN
	END
