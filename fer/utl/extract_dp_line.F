	SUBROUTINE EXTRACT_DP_LINE( cx, grid_data, dp_line, idim, n )

* extract a single line of data from a memory variable grid along axis idim
* no error checking is done

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* V200:  9/20/89 - 4D symmetrical

#ifdef unix
	include 'ferret_cmn/ferret.parm'
	include 'ferret_cmn/errmsg.parm'
	include 'ferret_cmn/xvariables.cmn'
	include 'ferret_cmn/xcontext.cmn'
	include 'ferret_cmn/xmem_subsc.cmn'
#else
	INCLUDE 'FERRET_CMN:FERRET.PARM'
	INCLUDE 'FERRET_CMN:ERRMSG.PARM'
	INCLUDE 'FERRET_CMN:XVARIABLES.CMN'
	INCLUDE 'FERRET_CMN:XCONTEXT.CMN'
	INCLUDE 'FERRET_CMN:XMEM_SUBSC.CMN'
#endif

* calling argument declarations:
	INTEGER	cx, mv, idim, n
	REAL    grid_data( m1lox:m1hix,m1loy:m1hiy,m1loz:m1hiz,m1lot:m1hit )
	REAL*8	dp_line(n)

* internal variable declarations:
	INTEGER	iout, i, j, k, l

* extract the line
	iout = 0
	DO 200 l = cx_lo_s4(cx), cx_hi_s4(cx)
	DO 200 k = cx_lo_s3(cx), cx_hi_s3(cx)
	DO 200 j = cx_lo_s2(cx), cx_hi_s2(cx)
	DO 200 i = cx_lo_s1(cx), cx_hi_s1(cx)

	   iout = iout + 1
	   dp_line( iout ) = grid_data( i,j,k,l )

 200	CONTINUE

	RETURN
	END
