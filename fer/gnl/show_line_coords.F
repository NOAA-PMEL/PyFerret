	SUBROUTINE SHOW_LINE_COORDS( grid, cx, idim )

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
*
* describe the axes of a grid and then give coordinate data about those
* axes as indicated in cx

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program

* V50 - *sh* 4/99 code extracted from SHOW_GRID
* V530 - *sh* 1/01 - show lower limit value of grid box
* 560 *acm* 4/04 - stride/modulo fixes: use NINT to convert REAL*8 to integer
* v580 *acm* 11/04 Store cal_id in cx_cal_id, the calendar ID used when  
*                  computing world coordinates
* V581 *acm*  6/05 Fix bug in show axis/t with non-standard calendars; RESET_SECS_FROM_BC
* V581 *acm*  6/05 Fix call to TM_GET_CALENDAR_ID (did this ever work with the
*                  argument grid rather than calendar name?!)
* V601 *acm*  3/08 Allow more figs when writing values with negative exponents.
* V650  Write to show_lun which may be set to a file unit number rather than std out
*       *acm* 3/12 6D Ferret, changed a comment only
*       *acm* 4/12 6D Ferret: time axis may be in t or f direction. The time 
*                  dimension is now an arg to SECS_TO_TSTEP.
* V683  *acm*10/12 Improve handling of F axes and info about the calendar of the grid.
* V695  *acm* 2/15 cx_calendar stores info about calendar formatting of T and F axes
* V71 12/16 *acm* For ticket 2158: working with true monthly time axes. Return integer
*                  monthly values in these listings

        IMPLICIT NONE
	include 'tmap_dims.parm'
#	include "tmap_dset.parm"
	include 'xtm_grid.cmn_text'
	external xgt_grid_data
	include	'ferret.parm'
	include	'slash.parm'
	include 'errmsg.parm'
	include 'xprog_state.cmn'
	include 'xtext_info.cmn'
	include 'xcontext.cmn'
	include 'xinterrupt.cmn'
	include 'xrisc.cmn'

* calling argument declarations
	INTEGER grid, cx, idim

* local variable declarations:
	LOGICAL	GEOG_LABEL, ITSA_TRUEMONTH_AXIS, calendar, true_month
	INTEGER ISUBSCR_CX, TM_LENSTR, TM_LENSTR1,
     .          TM_GET_CALENDAR_ID,
     .		lo, hi, i, line, loest, hiest, dlen,
     .		ndec, status, delta, exlen, cal_id,
     .          prev_cal_id, less_figs
	CHARACTER TM_FMT*48, buff*14, pt_str*28, lo_str*28, extra*24
	REAL	BOX_SIZE
	REAL*8	TM_WORLD, SECS_TO_TSTEP,
     .          pt_val8, lo_val8, start, end, new_ww, new_lo, new_hi, timefac


* local parameter declarations:
	INTEGER		sig_figs
	PARAMETER     ( sig_figs = 7 )   ! was 5 1/95

* initialize
	line = grid_line( idim, grid )
	CALL TM_MONTH_TIME_FACTOR (grid, true_month, timefac)

* ignore bogus lines (should never occur ...)
	IF ( line .EQ. munknown
     .  .OR. line .EQ. mnormal ) RETURN

	calendar = (idim.EQ.t_dim .OR. idim.EQ.f_dim) .AND. GEOG_LABEL( idim, grid )	
	IF ( calendar ) THEN
	   exlen = INDEX(line_units(line), '(') - 1
	   IF (exlen .LE. 0) exlen = TM_LENSTR(line_units(line))
	   extra = 'TSTEP ('//line_units(line)(:exlen)
	   extra = extra(:TM_LENSTR(extra))//')'
	ELSE
	   extra = ' '
	ENDIF
	exlen = TM_LENSTR1(extra)
	risc_buff = ' '
	WRITE ( risc_buff, 3030 ) ss_dim_name(idim),
     .				  ww_dim_name(idim),
     .				  ww_dim_name(idim),
     .				  ww_dim_name(idim), extra
	CALL SPLIT_LIST(pttmode_explct, show_lun, ' ', 1)
	CALL SPLIT_LIST(pttmode_explct, show_lun, risc_buff, 0)
* ... determine range of data to display
	IF ( cx_lo_ss( cx, idim ) .EQ. unspecified_int4
     . .AND. cx_lo_ww( idim, cx ) .EQ. unspecified_val8 )
     .									THEN
	   lo    = 1
	   hi    = line_dim( line )
	ELSEIF ( cx_by_ss( idim, cx ) ) THEN
	   CALL GRID_SUBSCRIPT_EXTREMES( loest, hiest, grid, idim )
	   lo = MAX( loest, cx_lo_ss( cx, idim ) )
	   hi = MIN( hiest, cx_hi_ss( cx, idim ) )
	ELSE
* kob - 1/98 -  cx_lo/hi_ww need to be translated to tsteps, before calling
*         ISUBSCR_CX, for the t dimension when cx_calendar is true.
*         similar to what goes on in flesh_out_axis. 
*
* cx_lo_ww was set without knowledge of the axis, so if this is a non-std calendar
* need to reset according to the actual calendar type

	   IF ( (idim.EQ.t_dim .OR. idim.EQ.f_dim) .AND. cx_calendar(idim,cx) ) THEN


	      IF (line_t0(line) .EQ. char_init20) GOTO 110  ! Not a calendar ax (e.g. f axis)
              cal_id = TM_GET_CALENDAR_ID(line_cal_name(line))
              cx_cal_id(cx) = cal_id

	      start = cx_lo_ww(idim,cx)
	      end   = cx_hi_ww(idim,cx)
              prev_cal_id = 1

	      IF ( (idim.EQ.t_dim .OR. idim.EQ.f_dim)  .AND.  cal_id .NE. prev_cal_id .AND.
     .             start .LT. 0. .AND. start .NE. unspecified_val8.AND.
     .             end   .LT. 0. .AND. end   .NE. unspecified_val8) THEN
	         CALL RESET_SECS_FROM_BC (-start, new_ww, prev_cal_id, 
     .                                    cal_id, status)
	         cx_lo_ww(idim,cx) = new_ww   ! not negative as we now know the axis.

	         CALL RESET_SECS_FROM_BC (-end, new_ww, prev_cal_id, 
     .                                    cal_id, status)
	         IF ( status .NE. ferr_ok ) RETURN
	         cx_hi_ww(idim,cx) = new_ww

              ELSE IF ( (idim.EQ.t_dim .OR. idim.EQ.f_dim)  .AND. 
     .              cal_id .EQ. prev_cal_id .AND.
     .             start .LT. 0. .AND. start .NE. unspecified_val8.AND.
     .             end   .LT. 0. .AND. end   .NE. unspecified_val8) THEN
	         cx_lo_ww(idim,cx) = -1*cx_lo_ww(idim,cx)
	         cx_hi_ww(idim,cx) = -1*cx_hi_ww(idim,cx)
              ENDIF

              cx_lo_ww(idim,cx) =
     .            SECS_TO_TSTEP( grid, idim, cx_lo_ww(idim,cx) )
              cx_hi_ww(idim,cx) =
     .		   SECS_TO_TSTEP( grid, idim, cx_hi_ww(idim,cx) )
 
* Convert to months

	          cx_lo_ww(idim,cx) = cx_lo_ww(idim,cx)/ timefac
	          cx_hi_ww(idim,cx) = cx_hi_ww(idim,cx)/ timefac

              IF (true_month) THEN
c          cx_lo_ww(idim,cx) = INT(cx_lo_ww(idim,cx))
c	          cx_hi_ww(idim,cx) = INT(cx_lo_ww(idim,cx)+1) 
	          CALL TM_TCOORD_TRUMONTH (line, cx_lo_ww(idim,cx),  
     .		    box_middle, new_lo)
	          CALL TM_TCOORD_TRUMONTH (line, cx_hi_ww(idim,cx),  
     .		    box_middle, new_hi)
     
* Convert to days
	          cx_lo_ww(idim,cx) = new_lo* timefac
	          cx_hi_ww(idim,cx) = new_hi* timefac

              ENDIF

* reset cx_calendar now that cx_lo/hi_ww have been converted
              cx_calendar(idim,cx) = .FALSE.
	   ENDIF
 110	   CONTINUE
	   lo = ISUBSCR_CX( cx_lo_ww(idim,cx),
     .			grid, idim, cx, round_dn )
	   hi = ISUBSCR_CX( cx_hi_ww(idim,cx),
     .			grid, idim, cx, round_up )
	ENDIF

* determine delta increment for listing
	IF ( cx_delta( idim, cx ) .NE. unspecified_val8 ) THEN
	   delta = NINT(cx_delta( idim, cx ))
	ELSE
	   delta = 1
	ENDIF

* write lines of coord and box size data
	IF ( calendar ) THEN
	   ndec = 6	! was 5 1/95
	ELSE
	   ndec = ABS(sig_figs)
	ENDIF
	IF (true_month) ndec = 2

	DO 120 i = lo, hi, delta
	   IF ( interrupted ) CALL ERRMSG
     .		( ferr_interrupt, status, ' ', *1000 )
	   lo_val8 = TM_WORLD( i, grid, idim, box_lo_lim )
	   pt_val8 = TM_WORLD( i, grid, idim, box_middle )
	   CALL TRANSLATE_TO_WORLD( pt_val8,idim,grid,ndec,pt_str )
	   
	   IF (true_month) THEN  ! show in months
	      CALL TRANSLATE_TO_WORLD( lo_val8,idim,grid,ndec+1,lo_str )
	      pt_val8 = INT((pt_val8+1)/timefac)
	      
	   ELSE
	      CALL TRANSLATE_TO_WORLD( lo_val8,idim,grid,ndec,lo_str )
	   ENDIF

	   IF ( calendar ) extra = 
     .	         	TM_FMT( (pt_val8), sig_figs, 14, dlen )
           buff = TM_FMT(
     .                       BOX_SIZE(i,grid,idim), sig_figs, 9, dlen )
           IF (dlen .GT. 9) THEN
              less_figs = sig_figs - 3
              buff = TM_FMT(
     .            BOX_SIZE(i,grid,idim), less_figs, 9, dlen )
           ENDIF
	   risc_buff = ' '
	   WRITE ( risc_buff, 3040 ) ! 3040
     .			i, pt_str,
     .			buff, lo_str,
     .			extra(:exlen)

	   CALL SPLIT_LIST(pttmode_explct, show_lun, risc_buff, 0)
 120	CONTINUE

 3030	FORMAT ( T8,A1,T14,A1,T38,A1,'BOX  ',T52,A1,'BOXLO',T70,A28 )
 3040   FORMAT ( T3,I6,'> ',T12,A28,T38,A10,T49,A28,T75,A14 )

* successful completion
 1000	RETURN
	END
