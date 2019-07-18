
	SUBROUTINE BOXES_DSG (orientation, flen, unit, 
     .		dsgcoords, latitudes, box)

        INCLUDE 'tmap_dims.parm'
	INCLUDE	'ferret.parm'

	INTEGER orientation, flen
	REAL	unit, dsgcoords(*), latitudes(*), box(*)

* orientation - is the dsg-grid orientation
* flen        - is the lenght of this feature
* dsgcoords   - are the coordinates in the dsg-orientation direction: longitudes
*               for trajectory datasets, depth/alt for profiles, time for timeseries
* latitudes   - for trajectories, the latitude data, otherwise unused.
* box         - box-sizes, returned 
* 

* Local declarations
	INTEGER i
	REAL	coord, cnext, upper, lower, km2m
	CHARACTER*80 errstr


	IF (flen .EQ. 1) THEN
	   box(1) = 0.
	   GOTO 5000
	ENDIF

* Weighted computation:
*   For trajectories the distance along the lon-lat line between points
*   and integrals with great-circle-distance weighting. For integrals,
*   convert to meters.
*   In Z and T use distance between, constructing a box

	IF ( (orientation .EQ. x_dim) ) THEN

* EARTH_DEL routine returns distance in Km, return units of meters if units are used.
	   km2m = 1000.
 
	   CALL EARTH_DEL( dsgcoords, latitudes, flen, 
     .                     bad_val4, bad_val4, box, errstr)
	   box(1) = box(2)
	   IF (unit .NE. 1.) unit = km2m
	      
	ELSE

* Get and save box sizes for all feature coordinates
* Profile coords may run lowest to highest so use ABS

	   coord = dsgcoords(1)
	   cnext = dsgcoords(2)
	   box(1) = ABS(cnext-coord)  
	   lower = (cnext+coord)/2

	   DO i = 2, flen - 1
	      coord = dsgcoords(i)
	      cnext = dsgcoords(i+1)

	      upper = (cnext+coord)/2
	      box(i) = ABS(upper-lower)  

	      lower = upper
	   ENDDO
           box(flen) = ABS(cnext - coord)  

	ENDIF

 5000	RETURN
	END
