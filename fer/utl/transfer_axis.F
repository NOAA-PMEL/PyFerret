	SUBROUTINE TRANSFER_AXIS ( idim, source, dest )

* transfer region and transformation information about one axis from one
* context buffer to another

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* V200:  5/31/89 - extracted from TRANSFER_CONTEXT

#ifdef unix
	include 'ferret_cmn/ferret.parm'
	include 'ferret_cmn/xcontext.cmn'
#else
	INCLUDE 'FERRET_CMN:FERRET.PARM'
	INCLUDE 'FERRET_CMN:XCONTEXT.CMN'
#endif

	INTEGER	source, dest, idim

	cx_lo_ss ( dest, idim )   = cx_lo_ss ( source, idim )
	cx_hi_ss ( dest, idim )   = cx_hi_ss ( source, idim )
	cx_lo_ww ( idim, dest )   = cx_lo_ww ( idim, source )
	cx_hi_ww ( idim, dest )   = cx_hi_ww ( idim, source )
	cx_trans ( idim, dest )   = cx_trans ( idim, source )
	cx_by_ss ( idim, dest )   = cx_by_ss ( idim, source )
	cx_given ( idim, dest )   = cx_given ( idim, source )
	cx_trans_arg(idim, dest)  = cx_trans_arg(idim, source)

	RETURN
	END
