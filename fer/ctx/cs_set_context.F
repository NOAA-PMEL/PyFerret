	SUBROUTINE CS_SET_CONTEXT

* transfer the information indicated in the current control stack level into
* the default ("LAST") context

* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system

* revision 0.0 - 5/1/87
* V200:  6/15/89 - reordered lo/hi_ss args
* 9/97 *kob* - need to reset cx_calendar to false if cx_lo/hi_ww are
*              not used
* 12/97 *kob* - and set  cx_calendar to true if they are used....
* internal variable declarations:
	INTEGER	idim

#ifdef unix
	include 'ferret_cmn/ferret.parm'
	include 'ferret_cmn/xcontext.cmn'
	include 'ferret_cmn/xcontrol.cmn'
#else
	INCLUDE 'FERRET_CMN:FERRET.PARM'
	INCLUDE 'FERRET_CMN:XCONTEXT.CMN'
	INCLUDE 'FERRET_CMN:XCONTROL.CMN'
#endif

* initialize
	idim = cs_axis( csp )

* set up by subscript or by world coordinates
	cx_by_ss( idim, cx_last ) = cs_by_ss( csp )

	IF ( cs_by_ss( csp ) ) THEN
	   cx_lo_ss( cx_last, idim ) = cs_phase( csp )
	   cx_hi_ss( cx_last, idim ) = cs_phase( csp )
	   cx_lo_ww( idim, cx_last ) = unspecified_val8
	   cx_hi_ww( idim, cx_last ) = unspecified_val8
* kob 9/97 - need to reset cx_calendar as well
	   IF (idim .eq. t_dim) 
     .                cx_calendar(cx_last) = .FALSE.
	ELSE
* kob 12/97 - i need to set cx_calendar to true here
 	   cx_lo_ss( cx_last, idim ) = unspecified_int4
	   cx_hi_ss( cx_last, idim ) = unspecified_int4
	   cx_lo_ww( idim, cx_last ) = cs_phase( csp )
	   cx_hi_ww( idim, cx_last ) = cs_phase( csp )
	   IF (idim .eq. t_dim) 
     .                cx_calendar(cx_last) = cs_is_calendar(csp)
	ENDIF
	   
	RETURN
	END

