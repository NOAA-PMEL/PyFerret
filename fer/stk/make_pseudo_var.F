	SUBROUTINE MAKE_PSEUDO_VAR ( cx, idim, pvar, mr, lo_ss, hi_ss )

* create a "pseudo-variable" - numerical grid of data based solely on the
* underlying grid (not involving external data)

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 - 3/2/87
* V200:  6/20/89 - 4D-symmetrical version

#ifdef unix
	include	'ferret_cmn/ferret.parm'
	include	'ferret_cmn/errmsg.parm'
	include 'ferret_cmn/xvariables.cmn'
	include 'ferret_cmn/xcontext.cmn'
#else
	INCLUDE	'FERRET_CMN:FERRET.PARM'
	INCLUDE	'FERRET_CMN:ERRMSG.PARM'
	INCLUDE 'FERRET_CMN:XVARIABLES.CMN'
	INCLUDE 'FERRET_CMN:XCONTEXT.CMN'
#endif

* calling argument declarations:
	INTEGER	cx, mr, idim, lo_ss, hi_ss
	REAL	pvar( lo_ss:hi_ss )

* internal variable declarations:
	INTEGER	grid, pv, i
	REAL	BOX_SIZE
	REAL*8	TM_WORLD

* initialize

* defining grid for output data
	grid = cx_grid( cx )

	pv = (mr_variable(mr)-1) / 4  + 1
	IF ( pv .GT. 3 )  STOP 'MK_PSEUDO_VAR'	! diag

	GOTO (  100, 200, 300 ) pv

* "I", "J", "K", "L"
 100	DO 110 i = lo_ss, hi_ss
 110	pvar( i ) = i
	RETURN

* "X", "Y", "Z", "T"
 200	DO 210 i = lo_ss, hi_ss
 210	pvar( i ) = TM_WORLD( i,grid,idim,box_middle )
	RETURN

* "XBOX", "YBOX", "ZBOX", "TBOX"
 300	DO 310 i = lo_ss, hi_ss
 310	pvar( i ) = BOX_SIZE( i, grid, idim )
	RETURN

	END
