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
* describe the 4 axes of a grid and then give coordinate data about those
* axes as indicated in cx

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program

* V50 - *sh* 4/99 code extracted from SHOW_GRID

#ifdef unix
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
#else
	INCLUDE 'TMAP_FORMAT:TMAP_DIMS.PARM'
	INCLUDE 'TMAP_FORMAT:XTM_GRID.CMN'
	INCLUDE	'FERRET_CMN:FERRET.PARM'
	INCLUDE	'FERRET_CMN:SLASH.PARM'
	INCLUDE 'FERRET_CMN:ERRMSG.PARM'
	INCLUDE 'FERRET_CMN:XPROG_STATE.CMN'
	INCLUDE 'FERRET_CMN:XTEXT_INFO.CMN'
	INCLUDE 'FERRET_CMN:XCONTEXT.CMN'
	INCLUDE 'FERRET_CMN:XINTERRUPT.CMN'
#endif

* calling argument declarations
	INTEGER grid, cx, idim

* local variable declarations:
	LOGICAL	GEOG_LABEL, calendar
	INTEGER ISUBSCR_CX, TM_LENSTR, 
     .		lo, hi, i, line, loest, hiest, dlen,
     .		ndec, status, delta
	CHARACTER TM_FMT*14, buff*14, lo_str*20, hi_str*20, extra*30
	REAL	BOX_SIZE
	REAL*8	TM_WORLD, lo_val8, SECS_TO_TSTEP

* local parameter declarations:
	INTEGER		sig_figs
	PARAMETER     ( sig_figs = 7 )   ! was 5 1/95


* initialize
	line = grid_line( idim, grid )

* ignore bogus lines (should never occur ...)
	IF ( line .EQ. munknown
     .  .OR. line .EQ. mnormal ) RETURN

	calendar = idim .EQ. t_dim .AND. GEOG_LABEL( idim, grid )	
	IF ( calendar ) THEN
	   extra = 'TIME_STEP ('//line_units(line)
	   extra = extra(:TM_LENSTR(extra))//')'
	ELSE
	   extra = ' '
	ENDIF
	risc_buff = ' '
	WRITE ( risc_buff, 3030 ) ss_dim_name(idim),
     .				  ww_dim_name(idim), extra
	CALL SPLIT_LIST(pttmode_explct, ttout_lun, ' ', 1)
	CALL SPLIT_LIST(pttmode_explct, ttout_lun, risc_buff, 0)
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
              cx_lo_ww(idim,cx) =
     .          SECS_TO_TSTEP( grid, -cx_lo_ww(idim,cx) )
              cx_hi_ww(idim,cx) =
     .		SECS_TO_TSTEP( grid, -cx_hi_ww(idim,cx) )
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
	DO 120 i = lo, hi, delta
	   IF ( interrupted ) CALL ERRMSG
     .		( ferr_interrupt, status, ' ', *1000 )
	   lo_val8 = TM_WORLD( i, grid, idim, box_middle )
	   IF ( calendar ) THEN
	      ndec = 6	! was 5 1/95
	   ELSE
	      ndec = -sig_figs
	   ENDIF
	   CALL TRANSLATE_TO_WORLD( lo_val8,idim,grid,ndec,lo_str )
	   IF ( calendar ) extra = 
     .	         	TM_FMT( SNGL(lo_val8), sig_figs, 14, dlen )
           buff = TM_FMT(
     .                       BOX_SIZE(i,grid,idim), sig_figs, 14, dlen )
	   risc_buff = ' '
	   WRITE ( risc_buff, 3040 ) 
     .			i, lo_str,
     .			buff,
     .			extra
	   CALL SPLIT_LIST(pttmode_explct, ttout_lun, risc_buff, 0)
 120	CONTINUE

 3030	FORMAT ( T12,A1,T18,A1,T36,'BOX_SIZE',T51,A28 )
 3040	FORMAT ( T7,I6,'> ',T16,A20,T38,A14,T53,A14 )

* successful completion
 1000	RETURN
	END
