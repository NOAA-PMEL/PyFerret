	SUBROUTINE SHOW_GRID( grid, cx )

* describe the 4 axes of a grid and then give coordinate data about those
* axes as indicated in cx

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 - 9/23/87
* revision 0.1 - 2/18/88 - checks subscript limits on axes
*			   and allows non-geographical, +/- oriented axes
* revision 0.2 - 3/18/88 - decimal places automated for sig. figs.
*			 - time step written in calendar mode
*			 - interruptable; - axis unit labels via AX_TITLE
* revision 1.0 - 8/1/88  - created LINE_FACTS and removed half of code to it
* revision 1.1 - 8/16/88 - eliminated units column, added delta in coord display
* V200:  5/19/89 - reordered lo/hi_ss arguments
*        9/22/89 - change to LINE_FACTS call for abstract axis orientations
* Unix/RISC port 3/18/91 *sh*: can't use function in WRITE list
*                            - corrected use of risc_buff (was right justified)
* V300:  5/28/93 - bug fix: incorrect branch for mnormal or munknown axis
* V320: 12/29/94 - use SPLIT_LIST to redirect tty output for GUI
*	  1/5/95 - changed default precision from 5 to 7  (and for
*		   dates to seconds) 
* kob*   9/97 - replaced call to ISUBSCRIPT w/ call to ISUBSCR_CX in order
*               to have access to context information - needed for negative
*               time step processing
* *kob*  1/98 - add calls to convert cx_lo/hi_ww to tsteps before calling
*               ISUBSCR_CX

#ifdef unix
	include 'tmap_format/tmap_dims.parm'
	include 'tmap_format/xtm_grid.cmn_text'
	external xgt_grid_data
	include	'ferret_cmn/ferret.parm'
	include	'ferret_cmn/slash.parm'
	include 'ferret_cmn/errmsg.parm'
	include 'ferret_cmn/xprog_state.cmn'
	include 'ferret_cmn/xtext_info.cmn'
	include 'ferret_cmn/xcontext.cmn'
	include 'ferret_cmn/xinterrupt.cmn'
	include 'ferret_cmn/xrisc.cmn'
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
	INTEGER grid, cx

* local variable declarations:
	LOGICAL	GEOG_LABEL, calendar
	INTEGER ISUBSCR_CX, TM_LENSTR, 
     .		lo, hi, idim, i, line, loest, hiest, dlen,
     .		ndec, status, delta
	CHARACTER TM_FMT*14, buff*14, lo_str*20, hi_str*20, extra*30
	REAL	BOX_SIZE
	REAL*8	TM_WORLD, lo_val8, SECS_TO_TSTEP

* local parameter declarations:
	INTEGER		sig_figs
	PARAMETER     ( sig_figs = 7 )   ! was 5 1/95

* list and describe the 4 axes of the grid
	risc_buff = ' '
	WRITE ( risc_buff, 3000 )
	CALL SPLIT_LIST(pttmode_explct, ttout_lun, risc_buff, 0)
	DO 110 idim = 1, 4
	   CALL LINE_FACTS( ttout_lun, idim, grid, idim )
 110	CONTINUE

* list the data on any axis that has been especially requested
*		via (eg.) /X=lo:hi or /X alone
	DO 130 idim = 1, 4
	   IF ( qual_given( slash_show_grid_x0+idim ) .GT. 0
     .	  .OR.  qual_given( slash_show_grid_i0+idim ) .GT. 0 ) THEN
	      line = grid_line( idim, grid )
	      IF ( line .EQ. munknown
     .        .OR. line .EQ. mnormal ) GOTO 130    ! changed from 120 5/93
	      calendar = idim .EQ. t_dim .AND. GEOG_LABEL( idim, grid )	
	      IF ( calendar ) THEN
	         extra = 'TIME_STEP ('//line_units(line)
	         extra = extra(:TM_LENSTR(extra))//')'
	      ELSE
	         extra = ' '
	      ENDIF
	      risc_buff = ' '
	      WRITE ( risc_buff, 3030 )	ss_dim_name(idim),
     .					ww_dim_name(idim), extra
	      CALL SPLIT_LIST(pttmode_explct, ttout_lun, ' ', 1)
	      CALL SPLIT_LIST(pttmode_explct, ttout_lun, risc_buff, 0)
* ... determine range of data to display
	      IF ( cx_lo_ss( cx, idim ) .EQ. unspecified_int4
     .	     .AND. cx_lo_ww( idim, cx ) .EQ. unspecified_val8 )
     .									THEN
	         lo    = 1
	         hi    = line_dim( line )
	      ELSEIF ( cx_by_ss( idim, cx ) ) THEN
	         CALL GRID_SUBSCRIPT_EXTREMES( loest, hiest, grid, idim )
	         lo = MAX( loest, cx_lo_ss( cx, idim ) )
	         hi = MIN( hiest, cx_hi_ss( cx, idim ) )
	      ELSE
* kob - 1/98 -  cx_lo/hi_ww need to be translated to tsteps, before calling
*               ISUBSCR_CX, for the t dimension when cx_calendar is true.
*               similar to what goes on in flesh_out_axis. 
*
		 IF ( idim.EQ.t_dim .AND. cx_calendar(cx) ) THEN
                    cx_lo_ww(idim,cx) =
     .                SECS_TO_TSTEP( grid, -cx_lo_ww(idim,cx) )
                    cx_hi_ww(idim,cx) =
     .		      SECS_TO_TSTEP( grid, -cx_hi_ww(idim,cx) )
* reset cx_calendar now that cx_lo/hi_ww have been converted
                    cx_calendar(cx) = .FALSE.
		 ENDIF
	         lo = ISUBSCR_CX( cx_lo_ww(idim,cx),
     .						grid, idim, cx, round_dn )
	         hi = ISUBSCR_CX( cx_hi_ww(idim,cx),
     .						grid, idim, cx, round_up )
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
     .			( ferr_interrupt, status, ' ', *1000 )
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
 120	      CONTINUE
	   ENDIF
 130	CONTINUE

 3000	FORMAT (T2,'name',T13,'axis',T31,'# pts',T39,'start',T60,
     .          'end' )
 3030	FORMAT ( T12,A1,T18,A1,T36,'BOX_SIZE',T51,A28 )
 3040	FORMAT ( T7,I6,'> ',T16,A20,T38,A14,T53,A14 )

* successful completion
 1000	RETURN
	END
