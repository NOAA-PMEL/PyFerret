	REAL FUNCTION CX_WW_POS( idim, cx, isubsc ) 

* return a single value representing the position of a context along an axis
* if the axis has been reduced to a point by, say, averaging then return the
* midpoint.
* If no reasonable position can be found then return bad_val4
 
#ifdef unix
	include 'ferret_cmn/ferret.parm'
	include 'ferret_cmn/xcontext.cmn'
#else
	INCLUDE 'FERRET_CMN:FERRET.PARM'
	INCLUDE 'FERRET_CMN:XCONTEXT.CMN'
#endif

* calling argument declarations:
	INTEGER		idim, cx, isubsc

* internal variable declarations:
	INTEGER	grid
	REAL*8	TM_WORLD

* initialize
	grid = cx_grid(cx)

	IF ( cx_lo_ss(cx,idim) .NE. unspecified_int4 ) THEN
* ... send back exact position
	   CX_WW_POS = TM_WORLD(isubsc,grid,idim,box_middle)
	ELSEIF ( cx_lo_ww(idim,cx) .NE. unspecified_val8 ) THEN
* ... send back midpoint of range
	   CX_WW_POS = (cx_lo_ww(idim,cx)+cx_hi_ww(idim,cx)) / 2.
	ELSE
* ... no valid position ...
	   CX_WW_POS = bad_val4
	ENDIF

	RETURN
	END
