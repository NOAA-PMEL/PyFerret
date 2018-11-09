	SUBROUTINE SHRINK_FOR_MODULO (cx, msrc, do_cdf, do_shrink)

* If the region was specified with world coordinates, e.g. X=0:360, but this makes
* the region just one cell too large to be corrrectly a modulo axis, then shrink
* the data by one, when writing a netCDF file. This way if the intent was, for
* instance, to shift a 20:380 axis to 0:360, we don't lose the correct modulo
* behavior and length.
* - If the region is larger than one cell over the modulo length, 
*   then the axis will be written as a non-modulo axis.
* - If the region was specified using indices, then the limits will be 
*   as specified, a non-modulo axis.

* Flag do_shrink is returned TRUE if one or more axes are shrunk


        IMPLICIT NONE
	include 'tmap_dims.parm'
	include	'ferret.parm'
	include 'xvariables.cmn'
	include 'xcontext.cmn'

* Calling argument declarations	
	LOGICAL do_cdf, do_shrink
	INTEGER cx, msrc

* Local declarations	
	LOGICAL ITSA_MODULO_AXIS
	INTEGER	grid, idim, dimsiz, ax_min_ss, ax_max_ss
	   
	do_shrink = .FALSE.
	IF (.NOT. do_cdf) RETURN

* Adjust the context index range, only in the limted case defined above.

	grid = cx_grid ( cx )
	DO 100 idim = 1, nferdims
	   IF (.NOT.  ITSA_MODULO_AXIS( cx, idim )) GOTO 100
	   IF ( cx_by_ss(idim,cx) ) GOTO 100

	   dimsiz = mr_hi_ss(msrc,idim) - mr_lo_ss(msrc,idim)
	   CALL GRID_SUBSCRIPT_EXTREMES_NO_MOD(ax_min_ss, ax_max_ss, grid, idim)
	   IF (ax_max_ss-ax_min_ss .EQ. dimsiz-1) THEN
	      do_shrink = .TRUE.
	      cx_hi_ss(cx,idim) = cx_hi_ss(cx,idim) - 1
	   ENDIF
 100	CONTINUE
	      
	RETURN
	END
