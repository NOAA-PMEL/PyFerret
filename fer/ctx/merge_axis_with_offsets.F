      SUBROUTINE MERGE_AXIS_WITH_OFFSETS( dflt_cx, mrg_cx, com_cx, 
     .     idim, lo_off, hi_off )

*
*
*  This software was developed by the Thermal Modeling and Analysis
*  Project(TMAP) of the National Oceanographic and Atmospheric
*  Administration's (NOAA) Pacific Marine Environmental Lab(PMEL),
*  hereafter referred to as NOAA/PMEL/TMAP.
*
*  Access and use of this software shall impose the following
*  obligations and understandings on the user. The user is granted the
*  right, without any fee or cost, to use, copy, modify, alter, enhance
*  and distribute this software, and any derivative works thereof, and
*  its supporting documentation for any purpose whatsoever, provided
*  that this entire notice appears in all copies of the software,
*  derivative works and supporting documentation.  Further, the user
*  agrees to credit NOAA/PMEL/TMAP in any publications that result from
*  the use of this software or in any product that includes this
*  software. The names TMAP, NOAA and/or PMEL, however, may not be used
*  in any advertising or publicity to endorse or promote any products
*  or commercial entity unless specific written permission is obtained
*  from NOAA/PMEL/TMAP. The user also understands that NOAA/PMEL/TMAP
*  is not obligated to provide the user with any support, consulting,
*  training or assistance of any kind with regard to the use, operation
*  and performance of this software nor to provide the user with any
*  updates, revisions, new versions or "bug fixes".
*
*  THIS SOFTWARE IS PROVIDED BY NOAA/PMEL/TMAP "AS IS" AND ANY EXPRESS
*  OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
*  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
*  ARE DISCLAIMED. IN NO EVENT SHALL NOAA/PMEL/TMAP BE LIABLE FOR ANY SPECIAL,
*  INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER
*  RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF
*  CONTRACT, NEGLIGENCE OR OTHER TORTUOUS ACTION, ARISING OUT OF OR IN
*  CONNECTION WITH THE ACCESS, USE OR PERFORMANCE OF THIS SOFTWARE. 
*
* Take an argument axis which has offsets and merge it into the
* result context.

*     JC_TODO: Explanation of logic and scenarios

* programmer - jonathan callahan
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
*
* V500:  4/9/98   - extracted from merge_context
	
#ifdef unix
	include 'ferret.parm'
	include	'xcontext.cmn'
	include	'xvariables.cmn'
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
