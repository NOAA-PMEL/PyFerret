	SUBROUTINE EXTRACT_LINE( cx, src, msrc, line, idim, ndim, n, status )

* extract a single line of data from a memory variable grid.  Determine
* which line to extract from the given context

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 - 7/8/87
* V200:  5/19/89 - 4D symmetrical
*	10/16/89 - modified array declarations using call to *_SUB
*	  5/9/90 - allow a multi-dimensional region as a line for PLOT/VS
*	         - return number of dims

#ifdef unix
	include 'ferret_cmn/ferret.parm'
	include 'ferret_cmn/errmsg.parm'
	include 'ferret_cmn/xvariables.cmn'
	include 'ferret_cmn/xcontext.cmn'
#else
	INCLUDE 'FERRET_CMN:FERRET.PARM'
	INCLUDE 'FERRET_CMN:ERRMSG.PARM'
	INCLUDE 'FERRET_CMN:XVARIABLES.CMN'
	INCLUDE 'FERRET_CMN:XCONTEXT.CMN'
#endif

* calling argument declarations:
	INTEGER	cx, msrc, idim, ndim, n, status
	REAL    src(*), line(*)

* internal variable declarations:
	INTEGER	CGRID_SIZE, dim(4)

* determine 4D shape of context
	CALL GET_CX_DIMS( cx, ndim, dim )

* have we been given a reasonable context ?
! eliminated 5/90
!	IF ( ndim .NE. 1 ) CALL ERRMSG
!     .		( ferr_dim_underspec, status, 'can only PLOT lines', *5000 )

* determine the axis orientation and length
	idim = dim(1)
	n    = CGRID_SIZE( cx )

* extract the line
	CALL EXTRACT_LINE_SUB(	cx, line, src,
     .				mr_lo_s1(msrc), mr_hi_s1(msrc),
     .				mr_lo_s2(msrc), mr_hi_s2(msrc),
     .				mr_lo_s3(msrc), mr_hi_s3(msrc),
     .				mr_lo_s4(msrc), mr_hi_s4(msrc)  )

* successful completion
	status = ferr_ok
	RETURN

! 5000	RETURN
	END
