	SUBROUTINE XEQ_PLOT( memory )

* execute the plot command

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 - 7/8/87  - based on XEQ_CONTOUR rev. 0.3
* revision 0.1 - 7/28/87 - added CALL DISP_PREP
* revision 0.2 - 3/2/88  - added safety check for data range prior to plotting
* revision 0.3 - 4/12/88 - added /NOLABELS
* revision 0.4 - 4/28/88 - moved rev. 0.2 check to PLOT_SET_UP
* revision 0.5 - 8/2/88  - added mode wait
* revision 1.0 - 9/21/88 - do_labels --> no_labels: /NOLABEL processing chged
* V200:   1/8/89 - using keys - major rewrite
*	 3/13/90 - check status from DISP_PREP
*	  5/8/90 - allow PLOT/VS to overlay 2D fields
* V230:  7/23/92 - when /LINE_STYLE is given /SYMBOLS must default to OFF
* V301:	  1/7/94 - bug fix: (see DISP_DATA_SET_UP) must unprotect plot data
*        1/27/94 - raise limit on /LINE=# from num_plain (6) to num_fancy_lines
* V312: 5/94 - array "memory" as a calling argument
* V320:   1/20/94 - changes to SAVE_FRAME call reflecting C-based procedures
*	  1/27/95 - default style for /VS is now /SYMBOLS_ONLY
* V450:  5/29/97 - added /NOAXES qualifier

#ifdef unix
	include 'tmap_format/tmap_dims.parm'
	include	'ferret_cmn/ferret.parm'
	include 'ferret_cmn/errmsg.parm'
	include 'ferret_cmn/plot_setup.parm'
	include 'ferret_cmn/slash.parm'
	include 'ferret_cmn/xcontext.cmn'
	include 'ferret_cmn/xvariables.cmn'
	include 'ferret_cmn/xprog_state.cmn'
	include 'ferret_cmn/xplot_state.cmn'
	include 'ferret_cmn/xplot_setup.cmn'
#else
	INCLUDE 'TMAP_FORMAT:TMAP_DIMS.PARM'
	INCLUDE	'FERRET_CMN:FERRET.PARM'
	INCLUDE 'FERRET_CMN:ERRMSG.PARM'
	INCLUDE 'FERRET_CMN:PLOT_SETUP.PARM'
	INCLUDE 'FERRET_CMN:SLASH.PARM'
	INCLUDE 'FERRET_CMN:XCONTEXT.CMN'
	INCLUDE 'FERRET_CMN:XVARIABLES.CMN'
	INCLUDE 'FERRET_CMN:XPROG_STATE.CMN'
	INCLUDE 'FERRET_CMN:XPLOT_STATE.CMN'
	INCLUDE 'FERRET_CMN:XPLOT_SETUP.CMN'
#endif
	
* calling argument declarations:
	REAL	memory( mem_blk_size, max_mem_blks )

* internal variable declarations:
	LOGICAL		overlay, transpz, versus, no_labels, set_up, noaxes
	INTEGER		symbol, line_type, status, mr, cx, ipl,
     .			loc, limit, slen,
     .			ax1_blks, ax1_start,
     .			ax2_blks, ax2_start
	REAL		val
	CHARACTER	LEFINT*4

* set up the graphics environment and get the data
	CALL START_PPLUS
	CALL DISP_DATA_SET_UP( memory,
     .		ax1_start, ax1_blks, ax2_start, ax2_blks, status )
	IF ( status .NE. ferr_ok ) GOTO 5100

* initialize
	overlay   = qual_given( slash_plot_over_plot ).GT.0
	no_labels = qual_given( slash_plot_nolabels  ).GT.0
	transpz   = qual_given( slash_plot_transpz   ).GT.0
	versus    = qual_given( slash_plot_vs	     ).GT.0
	set_up	  = qual_given( slash_shad_set_up    ).GT.0
	noaxes    = qual_given( slash_noaxes         ).GT.0
	symbol	  = unspecified_int4
	line_type = unspecified_int4
	IF ( .NOT.overlay ) CALL DISP_INIT(no_labels,is_cx,num_uvars_in_cmnd)
	CALL DISP_PREP( status )
	IF ( status .NE. ferr_ok ) GOTO 5000

* change default style to symbols, only if PLOT/VS (1/27/95)
	IF ( versus ) THEN
	   symbol = qual_on
	   line_type = qual_off
	ENDIF

* decode qualifiers
* ... PLOT/SYMBOL=n
	loc = qual_given( slash_plot_symb )
	IF ( loc .GT. 0  ) THEN
	   CALL EQUAL_VAL( cmnd_buff(qual_start(loc):qual_end(loc)),
     .			      val, status )	! syntax check
	   IF ( status .NE. ferr_ok ) GOTO 5000
	   IF ( val .NE. unspecified_val4 ) THEN
	      limit = max_sym 
	      IF ( val.LE. 0.0 .OR. val.GT.limit ) GOTO 5200
	      symbol = NINT( val )	! user-specified symbol
	   ELSE
	      symbol = qual_on		! auto selection of symbols
	   ENDIF
	   line_type = qual_off		! default to no line w/ symbol	      
	ENDIF
* ... PLOT/LINE_STYLE=n
	loc = qual_given( slash_plot_line )
	IF ( loc .GT. 0  ) THEN
	   CALL EQUAL_VAL( cmnd_buff(qual_start(loc):qual_end(loc)),
     .			      val, status )	! syntax check
	   IF ( status .NE. ferr_ok ) GOTO 5000
	   IF ( val .NE. unspecified_val4 ) THEN
	      limit = num_fancy_lines 			!was num_plain 1/94
	      IF ( val.LE. 0.0 .OR. val.GT.limit ) GOTO 5200
	      line_type = NINT( val )	! user-specified line style
	   ELSE
	      line_type = qual_on		! auto selection of line style
	   ENDIF
* ... default to no symbols when /LINE_STYLE, alone, is given
* 	or for PLOT/VS/LINE  (1/27/95)
           IF ( symbol .EQ. unspecified_int4
     .	   .OR. (versus.AND.qual_given(slash_plot_symb).EQ.0) ) 
     .							symbol = qual_off
	ENDIF

* set up the axes and title and load the data into PPLUS
	CALL PLOT_SET_UP   ( 	memory,
     .				overlay,
     .				transpz,
     .				versus,
     .				no_labels,
     .				symbol,
     .				line_type,
     .				is_mr,
     .				is_cx,
     .				num_uvars_in_cmnd,
     .				memory( 1, ax1_start ),
     .				memory( 1, ax2_start ),
     .				status )
	IF ( status .NE. ferr_ok ) GOTO 5000

* turn off axes? (5/29/97)
	IF ( noaxes ) CALL PPL_AXES_OFF(set_up)

* /set_up only requested ?
	IF ( set_up ) GOTO 1000

* plot it 
	IF ( overlay ) THEN
	   CALL PPLCMD ( ' ', ' ', 0, 'PLOT/OVER/' //ppl_wait//' ', 1, 1 )
	ELSE
	   CALL PPLCMD ( ' ', ' ', 0, 'PLOT/LATCH/'//ppl_wait//' ', 1, 1 )
	ENDIF

* restore axes? (5/29/97)
	IF ( noaxes ) CALL PPL_AXES_RESTORE

* interrupt occurred during plot ?
	IF ( ppl_interrupted ) GOTO 1000

* capture the bit map ?
	IF ( qual_given( slash_plot_frame ) .GT. 0 ) THEN
	   CALL SAVE_FRAME( frame_file, "HDF", status )
	   IF ( status .NE. ferr_ok ) GOTO 5000
	ENDIF

* release the memory used for axis buffers
 1000	CALL FREE_MEMORY( ax2_start, ax2_blks )
 	CALL FREE_MEMORY( ax1_start, ax2_blks )
	CALL UNPROTECT_CMND_DATA			! 1/94
 5100	RETURN

* error exits
 5000	GOTO 1000
 5200	CALL ERRMSG( ferr_out_of_range, status, 
     .		cmnd_buff(qual_start(loc):qual_end(loc))//pCR//
     .		'Legal range is 1 to '//LEFINT(limit,slen), *1000 )
	END
