	SUBROUTINE DEL_CX_DIM( idim, cx )

* eliminate all data specifying an axis in a context

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 - 9/23/87
* V200:  5/19/89 - reordered _ss dims

#ifdef unix
	include 'ferret_cmn/ferret.parm'
	include 'ferret_cmn/xcontext.cmn'
#else
	INCLUDE 'FERRET_CMN:FERRET.PARM'
	INCLUDE 'FERRET_CMN:XCONTEXT.CMN'
#endif

* calling argument declarations
	INTEGER	cx, idim

	cx_lo_ss    ( cx, idim ) = unspecified_int4
	cx_hi_ss    ( cx, idim ) = unspecified_int4
	cx_lo_ww    ( idim, cx ) = unspecified_val8
	cx_hi_ww    ( idim, cx ) = unspecified_val8
	cx_trans    ( idim, cx ) = trans_no_transform
	cx_trans_arg( idim, cx ) = bad_val4

	RETURN
	END
