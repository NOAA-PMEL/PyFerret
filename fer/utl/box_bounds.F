	SUBROUTINE BOX_BOUNDS( x_or_y, val1, n, bounds, valid )

* PLOT+ routine for use inside FERRET:
* Supply to PLOT+ the bounds of n grid boxes from X or Y axis of the contour
* data currently in PLOT+.  Pointers to the appropriate FERRET axes are in
* COMMON.

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system

* for FERRET V2.00: 12/21/89
* V3.00 changed REAL FUNCTION to SUBROUTINE (noticed by f77 3.2)

#ifdef unix
	include 'ferret_cmn/ferret.parm'
	include 'ferret_cmn/xbox_bounds.cmn'
#else
	INCLUDE 'FERRET_CMN:FERRET.PARM'
	INCLUDE 'FERRET_CMN:XBOX_BOUNDS.CMN'
#endif

* calling argument declarations:
	LOGICAL		valid
	INTEGER		n
	REAL		val1, bounds(n+1)
	CHARACTER	x_or_y*1

* internal variable declarations:
	INTEGER	ISUBSCRIPT, i, idim, isub1, isub0
	DOUBLE PRECISION TM_WORLD

* initialize
	IF ( x_or_y .EQ. 'X' ) THEN
	   idim = bb_xaxis
	ELSEIF ( x_or_y .EQ. 'Y' ) THEN
	   idim = bb_yaxis
	ENDIF

* determine starting subscript
	isub1 = ISUBSCRIPT ( DBLE(val1), bb_grid, idim, round_dn )
	isub0 = isub1 - 1

* pass back box edges
	DO 100 i = 1, n
 100	bounds(i) = SNGL( TM_WORLD (isub0+i,bb_grid,idim,box_lo_lim) )
	bounds(n+1) = SNGL( TM_WORLD (isub0+n,bb_grid,idim,box_hi_lim) )

* always (the appearance of) success
	valid = .TRUE.
	RETURN
	END
