      SUBROUTINE MERGE_AXIS_WITH_OFFSETS( dflt_cx, mrg_cx, com_cx, 
     .     idim, lo_off, hi_off )

* Take an argument axis which has offsets and merge it into the
* result context.

*     JC_TODO: Explanation of logic and scenarios

* programmer - jonathan callahan
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
*
* V500:  4/9/98   - extracted from merge_context
	
#ifdef unix
	include 'ferret_cmn/ferret.parm'
	include	'ferret_cmn/xcontext.cmn'
	include	'ferret_cmn/xvariables.cmn'
#else
	INCLUDE 'FERRET_CMN:FERRET.PARM'
	INCLUDE	'FERRET_CMN:XCONTEXT.CMN'
	INCLUDE	'FERRET_CMN:XVARIABLES.CMN'
#endif

* calling argument declarations
	INTEGER	dflt_cx, mrg_cx, com_cx, idim, lo_off(4), hi_off(4)

* local variable declarations
	INTEGER	temp_ss, ui4


*     do the following for cx_lo_ss and then cx_hi_ss
*     If the default context is undefined
*        get argument context (undo offsets)
*        If merge context is undefined:    use arg context
*        Else:                             use intersection of mrg_cx and arg cx
*        Endif
*     Else context is defined
*        merge context = default context
*        prefill with missing value flag
*     Endif

        ui4 = unspecified_int4

        IF ( cx_lo_ss(dflt_cx,idim) .EQ. ui4 ) THEN
           temp_ss = cx_lo_ss(com_cx, idim) - lo_off(idim)
           IF ( cx_lo_ss(mrg_cx,idim) .EQ. ui4 ) THEN
              cx_lo_ss(mrg_cx,idim) = temp_ss
           ELSE
              cx_lo_ss(mrg_cx,idim) = 
     .             MIN(cx_lo_ss(mrg_cx,idim),temp_ss)
           ENDIF

        ELSE
           cx_lo_ss(mrg_cx,idim) = cx_lo_ss(dflt_cx,idim)
*     prefill with bad flags

        ENDIF
        
        IF ( cx_hi_ss(dflt_cx,idim) .EQ. ui4 ) THEN
           temp_ss = cx_hi_ss(com_cx,idim) - hi_off(idim)
           IF ( cx_hi_ss(mrg_cx,idim) .EQ. ui4 ) THEN
              cx_hi_ss(mrg_cx,idim) = temp_ss
           ELSE
              cx_hi_ss(mrg_cx,idim) = 
     .             MIN(cx_hi_ss(mrg_cx,idim),temp_ss)
           ENDIF

        ELSE
           cx_hi_ss(mrg_cx,idim) = cx_hi_ss(dflt_cx,idim)
*     prefill with bad flags

        ENDIF


*     JC_TODO: should I implement pieces of code from TRANSFER_AXIS?
*
*	cx_lo_ss ( dest, idim )   = cx_lo_ss ( source, idim )
*	cx_hi_ss ( dest, idim )   = cx_hi_ss ( source, idim )
*	cx_lo_ww ( idim, dest )   = cx_lo_ww ( idim, source )
*	cx_hi_ww ( idim, dest )   = cx_hi_ww ( idim, source )
*	cx_trans ( idim, dest )   = cx_trans ( idim, source )
*	cx_by_ss ( idim, dest )   = cx_by_ss ( idim, source )
*	cx_given ( idim, dest )   = cx_given ( idim, source )
*	cx_trans_arg(idim, dest)  = cx_trans_arg(idim, source)
** 8/97 *kob* 	
*	IF (idim .EQ. t_dim) 
*     .        cx_calendar(dest) = cx_calendar(source)


* successful completion
 100	RETURN
	END
