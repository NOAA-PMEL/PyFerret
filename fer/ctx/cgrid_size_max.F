	INTEGER FUNCTION CGRID_SIZE_MAX ( cx )

* determine the number of values used on the grid
* of the specified context

* unlike CGRID_SIZE which uses "1" as the size of unspecified regions this
* one uses CAXIS_LEN - the maximum size.  A kludge in guessing this way.

* another kludge is needed for abstract variables -- since the abstract
* axis is deliberately defined to be "arbitrarily" long.  Use a length of
* 1 in this case

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program

* V420: 11/95 - from CGRID_SIZE

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
	INTEGER	CX_DIM_LEN, CAXIS_LEN, CGRID_AXIS, idim, i4d, axlen

* to accomodate the ABSTRACT axis
* (inefficient -- "mline_abstract" should be saved by DEFINE_SPECIAL_GRIDS)

* initialize
	i4d = 1

	DO 100 idim = 1, 4
	   IF (cx_lo_ss(cx,idim) .NE. unspecified_int4) THEN
	      i4d = i4d * CX_DIM_LEN(idim,cx)
	   ELSE
	      axlen = CAXIS_LEN(idim,cx)
	      IF ( axlen .EQ. 9999999 ) THEN
	         axlen = 1	! kludge trap for ABSTRACT axis length
	      ELSEIF ( axlen .GT. 5000 ) THEN
	         axlen = 5000	! kludge 
	      ENDIF
	      i4d = i4d * axlen
	   ENDIF
 100	CONTINUE

	CGRID_SIZE_MAX = i4d

	RETURN
	END
