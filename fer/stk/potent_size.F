	INTEGER FUNCTION POTENT_SIZE ( cx )

* determine the "potential" size of context cx - that is, the total number of
* values that will need to be accessed in order to perform the indicated
* compressing transformations (transf. that reduce axis to a point)

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* V200:  6/8/89
*	8/29/89 - special treatment of interpolation

#ifdef unix
	include 'ferret_cmn/ferret.parm'
	include 'ferret_cmn/xcontext.cmn'
#else
	INCLUDE 'FERRET_CMN:FERRET.PARM'
	INCLUDE 'FERRET_CMN:XCONTEXT.CMN'
#endif

* calling argument declarations:
	INTEGER	cx

* internal variable declarations:
	INTEGER	CGRID_SIZE, idim, status

* make a copy of the context in a scratch buffer
	CALL TRANSFER_CONTEXT( cx, cx_buff )

* expand all the compressing transformations
	DO 100 idim = 1, 4
	   IF ( cx_trans(idim,cx) .EQ. trans_interpolate ) THEN
	      cx_trans(idim,cx_buff) = trans_no_transform
	      cx_lo_ss(cx_buff,idim) = 1
	      cx_hi_ss(cx_buff,idim) = 2
	   ELSEIF ( cx_trans(idim,cx) .GT. trans_compress_code ) THEN
	      cx_trans(idim,cx_buff) = trans_no_transform
	      cx_by_ss(idim,cx_buff) = .FALSE.
	      CALL FLESH_OUT_AXIS( idim, cx_buff, status )	! ignore status
	   ENDIF
 100	CONTINUE

* now, what's the size of the result ?
	POTENT_SIZE = CGRID_SIZE( cx_buff )

	RETURN
	END
