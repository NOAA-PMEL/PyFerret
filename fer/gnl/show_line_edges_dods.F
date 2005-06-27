	SUBROUTINE show_line_edges_dods( grid, cx, idim,
     .       workspace, clobber, filename, istat)

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
* Write binary data for DODS output listing the axis coordinates

* V554 - *acm* 9/03  from show_line_coords
* *acm* v580 11/04 Store cal_id in cx_cal_id, the calendar ID used when  
*                  computing world coordinates
* V581 *acm*  6/05 Fix bug in show axis/t with non-standard calendars; RESET_SECS_FROM_BC
* V581 *acm*  6/05 Fix call to TM_GET_CALENDAR_ID (did this ever work with the
*                  argument grid rather than calendar name?!)


	include 'tmap_dims.parm'
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
        LOGICAL clobber
	INTEGER istat, grid, cx, idim
        REAL*8  workspace(*)
        CHARACTER*(*) filename

* local variable declarations:
        LOGICAL  swap
	INTEGER ISUBSCR_CX, TM_LENSTR1, TM_GET_CALENDAR_ID,
     .		lo, hi, i, line, loest, hiest, cal_id,
     .		status, delta, slen, size, clobber_flag, swap_flag,
     .          axis, prev_cal_id
	REAL*8	TM_WORLD, SECS_TO_TSTEP,
     .           pt_val8, start, end, new_ww


* #ifdef for endianness info.
#ifdef sun
	include 'xmachine_byte.cmn'
#else
	include 'xmachine_int1.cmn'
#endif



* BYTEORDER - get byteorder of this architecture
 	IF (active_cpu .EQ. cptype_sun) THEN
	   swap_flag = 0
	ELSE
	   swap_flag = 1
	ENDIF


* initialize
	line = grid_line( idim, grid )

* ignore bogus lines (should never occur ...)
	IF ( line .EQ. munknown
     .  .OR. line .EQ. mnormal ) RETURN


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
	   IF ( idim.EQ.t_dim .AND. cx_calendar(cx) ) THEN
              axis = grid_line(t_dim, grid)
              cal_id = TM_GET_CALENDAR_ID(line_cal_name(axis))
              cx_cal_id(cx) = cal_id

	      start = cx_lo_ww(idim,cx)
	      end   = cx_hi_ww(idim,cx)
              prev_cal_id = 1

	      IF ( idim .EQ. t_dim  .AND.  cal_id .NE. prev_cal_id .AND.
     .             start .LT. 0. .AND. start .NE. unspecified_val8.AND.
     .             end   .LT. 0. .AND. end   .NE. unspecified_val8) THEN
	         CALL RESET_SECS_FROM_BC (-start, new_ww, prev_cal_id, 
     .                                    cal_id, status)
	         cx_lo_ww(idim,cx) = new_ww   ! not negative as we now know the axis.

	         CALL RESET_SECS_FROM_BC (-end, new_ww, prev_cal_id, 
     .                                    cal_id, status)
	         IF ( status .NE. ferr_ok ) RETURN

	         cx_hi_ww(idim,cx) = new_ww
                 
              ELSE IF ( idim .EQ. t_dim  .AND. 
     .              cal_id .EQ. prev_cal_id .AND.
     .             start .LT. 0. .AND. start .NE. unspecified_val8.AND.
     .             end   .LT. 0. .AND. end   .NE. unspecified_val8) THEN
	         cx_lo_ww(idim,cx) = -1*cx_lo_ww(idim,cx)
	         cx_hi_ww(idim,cx) = -1*cx_hi_ww(idim,cx)

              ENDIF

              cx_lo_ww(idim,cx) =
     .            SECS_TO_TSTEP( grid, cx_lo_ww(idim,cx) )
              cx_hi_ww(idim,cx) =
     .		   SECS_TO_TSTEP( grid, cx_hi_ww(idim,cx) )
 

* reset cx_calendar now that cx_lo/hi_ww have been converted
              cx_calendar(cx) = .FALSE.
	   ENDIF
	   lo = ISUBSCR_CX( cx_lo_ww(idim,cx),
     .			grid, idim, cx, round_dn )
	   hi = ISUBSCR_CX( cx_hi_ww(idim,cx),
     .			grid, idim, cx, round_up )
	ENDIF

* determine delta increment for listing
	IF ( cx_delta( idim, cx ) .NE. unspecified_val8 ) THEN
	   delta =  cx_delta( idim, cx )
	ELSE
	   delta = 1
	ENDIF

* write lines of coord and box size data
        size = 0
        i = lo
	pt_val8 = TM_WORLD( i, grid, idim, box_lo_lim )
        size = size + 1
        workspace(size) = pt_val8
	DO 120 i = lo, hi, delta
	   IF ( interrupted ) CALL ERRMSG
     .		( ferr_interrupt, status, ' ', *1000 )
	   pt_val8 = TM_WORLD( i, grid, idim, box_hi_lim )
           size = size + 1
           workspace(size) = pt_val8

 120	CONTINUE

         slen = TM_LENSTR1(filename)
         clobber_flag = 0
         IF (clobber) clobber_flag = 1
         CALL write_dods_double (filename, slen, clobber_flag,  
     .                   swap_flag, size, workspace)

* successful completion
 1000	RETURN


 5010   istat = 0
        RETURN
        END
