	SUBROUTINE XEQ_CANCEL( memory )

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
* execute the CANCEL command

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 - 4/3/86
* revision 0.1 - 12/15/86
* revision 0.2 - 3/20/87 - changes for new TMAP library
* revision 0.3 - 5/1/87  - added CANCEL MOVIE, CANCEL LIST/OUT, CANCEL REGION
*			   and CANCEL EXPRESSION
* revision 0.4 - 8/29/87 - updated CANCEL LIST for unformatted option
* revision 0.5 - 9/11/87 - updated CANCEL LIST for BIBO option
* revision 0.6 - 9/16/87 - added CANCEL GRID (not imp)
* revision 0.7 - 9/23/87 - modified CANCEL REGION for predefined regions
*			   and added CANCEL LIST/HEAD
* revision 0.8 - 10/5/87 - modified CANCEL LIST to toggle SET LIST
*			   and added CANCEL LIST/APPEND
* revision 0.9 - 3/7/88  - added list_tmap
* revision 1.0 - 6/29/88 - added CANCEL MEMORY
*			 - CANCEL LIST returns filename to 'AUTO'
*			 - added CANCEL WINDOW
* revision 1.1 - 7/2/88  - CANCEL DATA now wipes out all memory variables that
*			   depend on a climatological air temp. field
* revision 1.2 - 8/2/88  - added mode wait
* revision 1.3 - 9/1/88  - added CANCEL VIEWPORT name
* revision 1.4 - 12/2/88 - bug fix: cancel mem only to max_mr_avail
* revision 1.5 - 1/10/89 - added CANCEL MODE REJECT, corrected last_mode
* revision 1.6 - 2/16/89 - ds_open --> ds_name.EQ.char_init30 to test open-ness
* revision 1.7 - 2/17/89 - GT_CLOSE_SET --> TM_CLOSE_SET   
* revision 1.8 - 4/1/89  - MODE WAIT directly sets PPLUS variable deflts
* V200:  5/24/89 - layout changes, only
* 		 - deleted variables are flagged in mr_protected
*		 - added CANCEL VARIABLE
*		 - CANCEL DATA purges all user defined variables
*		 - added laser movie output
*	10/20/89 - CANCEL DATA removed to CANCEL_DATA_SET, if only one set left
*		 - then cancel data points to it
*	  1/5/90 - added CANCEL MODE metafile
*	 3/13/90 - call CANCEL_WINDOW even if in batch mode
* Unix/RISC port - 4/25/91 *sh*: increased ds_name size
*                              : bug fix in CANCEL MODE: mode_GKS --> pmode_GKS
* V230:5/14/92 - REMOTE_X mode was replaced by the REFRESH mode
*       6/8/92 - added CANCEL ALIAS
*       6/9/92 - CANCEL LIST uses list_fmt_type
*              - added tmap_diag_on to mode diagnostic
*      8/25/92 - added CANCEL AXIS/MODULO
*      11/4/92 - CANCEL VARIABLE to handle long names
*     1/14/93  - gutted CANCEL MOVIE
* V300:2/10/93 - CANCEL MODE META will also issue PPL CLSPLT 
*      5/12/93 - added MODE PPLLIST and MODE JOURNAL filename
*      5/24/93 - added CANCEL DATA/NOERROR
*      6/16/93 - added CANCEL MODE GUI
*      2/24/94 - incorporate a stack of past mode states
* V312: 5/94 - array "memory" as a calling argument
* V314:9/1/94 *kob* IBM port - Need to make logical comparisons with NEQ 
*			       or NEQV
* V320: 12/29/94 - use SPLIT_LIST to redirect tty output for GUI
*	 1/18/94 - check CANCEL MODE VERIFY for "ALWAYS" state
*	 2/28/95 - update TMAP COMMON alt_messages when mode_gui is CANCELLED
* V400:  6/22/95 - added CANCEL SYMBOL
* V411:   9/1/95 - Added support for  CAN VAR/D=dset var
*	  10/95  - buffer READ(*) for AIX port
* V420:    4/96  - use arg_to_item to make comma-separated line blank sep
*		 - templates in CANCEL VAR, SYMBOL, ALIAS, VIEW
*Linux Port 1/97 *kob* - Added a preprocessor include for tmap_dset.parm
*			  because it needed a preprocessor.
* V450:    7/97  - update CANCEL VAR/ALL so it handles child vars and dynamic
*                  grids/axes
* V510: 5/00 *sh* - added CANCEL AXIS and CANCEL GRID
* v540 *acm* 10/01 increase length of ds_name
* v540 *acm* 10/01 remove VMS include statements
* v541 *acm*  6/02 Add CANCEL AXIS/ALL
*             6/02 For irregular axes, on CANCEL AXIS axname, recover line_mem
*                  space with call to PACK_LINE_STORAGE
* v541 *acm*  7/02 Capitalize routine names in CALL PACK_LINE_STORAGE, DEALLO_ALL_AXES.
* v551 *acm*  1/03 Bug fix: datasets starting with digit not correctly cancelled.
* V570: *acm* 5/04 - add MODE GRATICULE[:argument] 
*       *acm* 6/04 - CANCEL AXIS/DEPTH as inverse to SET AXIS/DEPTH
* V600: *acm* 2/06 - fix bug 1392: cancel dataset whose name starts with digit.
* V600 *acm* 10/05 CANCEL ATTRIBUTE[/ALL][/OUTPUT] varname.attname
* V600 *acm* 5/06 MODE LINECOLORS User can change the number of line colors.
* V600 *acm* 6/06 additional declarations found by Solaris compiler
* V602 *jli* 7/06 added CANCEL AXIS/STRIDE
* V601 *acm*11/06 Fix bug 1470: on a CANCEL AXIS/MODULO or CANCEL AXIS/DEPTH, change
*                 values in the attribute structure for the axis.
* V602: *acm* 1/07 Changes for metafile-batch mode: When in gif mode or batch mode, 
*                  ignore CANCEL MODE META
* V602  2/07 *acm* Fix bug 1492, changing attributes of coordinate variables 
* v604 *acm* 7/07 increase length of
* V606  8/07 *acm* Send informational and error messages that are returned from 
*                  commands via SPLIT_LIST to std error rather than std out.
* V610  *acm* 3/08 Fixes for metafile batch mode
* V63   *acm* 9/09 Fix bug 1691: deallocate use of dynamic axis when it is found in a grid.
*                  If more than one grid uses the dyn axis, deallocate use of axis for each.
* V66   *kms* 5/10 add CANCEL REDIRECT for stdout/stderr redirection

	include 'tmap_dims.parm'
#	include "tmap_dset.parm"
	include 'tmap_errors.parm'
	include 'xdset_info.cmn_text'
	external xdset_info_data
        include 'xdiag_ctrl.cmn_text'
        include 'xtm_grid.cmn_text'
        external xgt_grid_data
	include 'xalt_messages.cmn_text'
	include	'ferret.parm'
	include	'slash.parm'
	include 'errmsg.parm'
	include 'movies.parm'
	include 'plot_setup.parm'
	include 'xprog_state.cmn'
	include 'xplot_state.cmn'
	include 'xvariables.cmn'
	include 'xcontext.cmn'
	include 'xfr_grid.cmn'
	include 'xvideo.cmn'
	include 'xcommand.cmn'
	include	'xcontrol.cmn'
	include	'pltcom_dat.decl'	! with gksopn
	include	'PLTCOM.DAT'	! with gksopn
	include	'switch_inc.decl'	! with deflts
	include	'SWITCH.INC'	! with deflts
        include 'LUNITS.INC'          ! with lttout
        include 'lunits_inc.decl'

* calling argument declarations:
	REAL	memory( mem_blk_size, max_mem_blks )

* local variable declarations:
	INTEGER		REGION_NUMBER, FIND_DSET_NUMBER,
     .                  TM_GET_LINENUM, TM_GET_GRIDNUM, 
     .			TM_GET_GRID_OF_LINE, TM_LENSTR1, STR_UPCASE,
     .                  STR_SAME,
     .			ivp, i, blen, ier, i1, num_protected_axes,
     .			status, variable, category,
     .			idim, iqual, iset, item, ivar, nth, mr,
     .			mode, cx, axis, grid, mod_cx,
     .                  TM_LENSTR, len_test, len_mchars, varid,
     .                  dset, cat, var, mods_cx, iflag, loc, 
     .                  nvdims, vdims(8), nvatts, vtype, all_outflag, 
     .                  slen, iatt, igrd, iline, new_att_modulo_len,
     .			cache_size, cache_nelems, cache_preemption
	LOGICAL		MATCH4, TM_DIGIT_TEST, IS_TEMPLATE, MATCH_TEMPLATE,
     .			ITS_GIF_GRAPHICS, ITS_BATCH_GRAPHICS,
     .			report_error, slash_all, del_perm, del_temp,
     .                  perm_var, all_dims, coordvar, do_err, 
     .                  canout, new_att_modulo, its_gif, batch, 
     .                  redir_cancel_stdout, redir_cancel_stderr

	CHARACTER	arg*4, buff*512, arg30*30, varname*512,
     .                  attname*128, show_str*12, buff2*3, DUMMY*512

* local parameter declarations:
	INTEGER		slash_del_temp, slash_del_perm
	PARAMETER     ( slash_del_temp = 1+1,
     .			slash_del_perm = 2+1 )

* command given with /ALL ?
	slash_all = .FALSE.
	DO 10 iqual = 1, num_qualifiers
	IF ( qualifier_list( iqual ) .EQ. 1 ) slash_all = .TRUE.
 10	CONTINUE

* if the user entered a blank-separated list insteas of comma-separated
* then make it behave like a comma-separated list (4/96)
	CALL ARG_TO_ITEM

* select subcommand
	GOTO ( 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000,
     .        1100,1200,1300,1400,1500,1600,1700, 1800 ) subcmnd_num

* CANCEL '    '
* arrival at this point usually means an illegal subcommand was given
* and was assumed to be an argument by the command parser
 100	IF ( num_args .GE. 1 ) THEN
	   CALL ERRMSG( ferr_invalid_subcmnd,status,
     .				cmnd_buff( arg_start(1):arg_end(1) ), *5000 )
	ELSE
	   GOTO 5010
	ENDIF
	RETURN

* CANCEL WINDOW
 200	CALL CANCEL_WINDOW( status )
	RETURN

* CANCEL REGION[/ALL/X/Y/Z/T] [region_name]
 300	IF ( slash_all .AND.
     .	     (num_args.GT.0 .OR. num_qualifiers.GT.1) ) GOTO 5310
	all_dims = num_qualifiers .EQ. 0
	IF ( slash_all ) THEN
* ... CANCEL REGION/ALL
	   DO 302 idim = 1, 4
 302	   CALL DEL_CX_DIM( idim, cx_last )
	   DO 304 cx = 0, min_context, -1
 304	   cx_name( cx ) = unspecified_name4
	ELSEIF ( num_args .EQ. 0 ) THEN
* ... CANCEL REGION[/I/J/K/L]
	   DO 310 idim = 1, 4
	      IF ( all_dims
     .	      .OR. qual_given(idim+1) .GT. 0		! /I/J/K or /L
     .	      .OR. qual_given(idim+5) .GT. 0 ) THEN	! /X/Y/Z or /T
	         CALL DEL_CX_DIM( idim, cx_last )
	      ENDIF
 310	   CONTINUE
	ELSE
* ... CANCEL REGION[/I/J/K/L] name
	   DO 340 item = 1, num_items
	      arg = cmnd_buff( item_start(item):item_end(item) )
	      cx = REGION_NUMBER( arg )
	      IF ( cx .EQ. unspecified_int4 ) CALL ERRMSG
     .			( ferr_unknown_arg, status, arg, *5000 )
	      IF ( all_dims ) THEN
	         cx_name( cx ) = unspecified_name4
	      ELSE
	         DO 320 idim = 1, 4
	            IF ( qual_given(idim+1) .GT. 0		! /I/J/K or /L
     .	            .OR. qual_given(idim+5) .GT. 0 ) THEN	! /X/Y/Z or /T
	               CALL DEL_CX_DIM( idim, cx )
	               cx_given( idim, cx ) = .FALSE.
	            ENDIF
 320	         CONTINUE
	      ENDIF	         
 340	   CONTINUE
	ENDIF
	RETURN

* CANCEL MEMORY /ALL /TEMP[default] or /PERM
 400	del_perm = slash_all .OR. qual_given( slash_del_perm ) .GT. 0
	del_temp = slash_all .OR. qual_given( slash_del_temp ) .GT. 0
     .			     .OR. .NOT.del_perm
	   DO 410 mr = 1,max_mr_avail
	      IF ( mr_protected( mr ) .EQ. mr_deleted ) GOTO 410
	      perm_var = mr_protected(mr) .EQ. mr_perm_protected
	      IF ( del_perm .AND. perm_var
     .	      .OR. del_temp .AND. .NOT.perm_var )
     .				CALL DELETE_VARIABLE( mr )
 410	   CONTINUE
	IF ( del_perm ) CALL SPLIT_LIST(pttmode_ops, err_lun, 
     .			' PERMANENT data cleared from memory', 0)
	IF ( del_temp ) CALL SPLIT_LIST(pttmode_ops, err_lun, 
     .			' TEMPORARY data cleared from memory', 0) 
	RETURN

* CANCEL EXPRESSION
 500	CALL DELETE_OLD_EXPR
	num_uvars_in_cmnd = cmnd_uvars_not_given
	RETURN

* CANCEL LIST
 600	IF ( slash_all .OR. qual_given( slash_canc_list_prec   ) .GT. 0 )
     .	   list_digits = default_output_precision
	IF ( slash_all .OR. qual_given( slash_canc_list_out ) .GT. 0 ) THEN
	   list_file       = 'AUTO'
	ENDIF
	IF ( slash_all .OR. qual_given( slash_canc_list_append ) .GT. 0 )
     .	   list_append = .FALSE.
	IF ( slash_all .OR. qual_given( slash_canc_list_form ) .GT. 0 ) THEN
	   list_format_given = .FALSE.
	   list_fmt_type     = plist_default
	ENDIF
	IF ( qual_given( slash_canc_list_heading ) .GT. 0 ) THEN
	   list_heading = .FALSE.
	ELSEIF ( slash_all ) THEN
	   list_heading = .TRUE.		! /ALL restores default state
	ENDIF

	IF (slash_all) CALL NETCDF4_CANCEL_SETTINGS
	RETURN

* CANCEL DATA_SET data_set_number1,number2,...
 700	report_error = qual_given(slash_canc_data_noerr) .LE. 0      ! 5/93
 	IF ( num_items .GT. 0 ) THEN
	   DO 710 item = 1, num_items
	      buff = cmnd_buff(item_start(item):item_end(item))
	      blen = item_end(item)-item_start(item)+1
* interpret and check the data set specified
	      IF ( TM_DIGIT_TEST(buff) ) THEN
	         arg30 = buff(:blen)			! 10/95 for AIX

*  (1/03 *acm*) See xeq_set
#ifdef MANDATORY_FORMAT_WIDTHS
	         READ ( arg30,*,ERR=610 ) iset
#else
	         READ ( arg30,'(I)',ERR=610 ) iset
#endif

* ... check for valid number given
	         IF ( iset .LT. 1 .OR. iset .GT. maxdsets ) GOTO 5030
	         IF (  ds_name( iset ) .EQ. char_init1024) THEN
	            IF ( report_error ) THEN
	               GOTO 5030
	            ELSE
	               GOTO 710
	            ENDIF
	         ENDIF
	         GOTO 705

*     argument starts with a number but is not a number.
*     Check for a valid dataset name

 610	         iset = FIND_DSET_NUMBER( buff(:blen) )
	         IF (  iset .EQ. unspecified_int4 )then
	            IF ( report_error ) THEN
	               GOTO 5030
   	            ELSE
	               GOTO 710
	            ENDIF
	         ENDIF

	      ELSE
  	         iset = FIND_DSET_NUMBER( buff(:blen) )
	         IF (  iset .EQ. unspecified_int4 )then
	            IF ( report_error ) THEN
	               GOTO 5030
	            ELSE
	               GOTO 710
	            ENDIF
	         ENDIF
	      ENDIF

 705	      CALL CANCEL_DATA_SET( iset, status )
 710	   CONTINUE
* ... if only one set remains then point to it
	   iset = unspecified_int4
	   DO 715 i = 1, maxdsets
	     IF ( ds_name(i) .EQ. char_init1024) GOTO 715
	     IF ( iset .EQ. unspecified_int4 ) THEN
	         iset = i
	     ELSE
	         GOTO 790
	     ENDIF
 715	   CONTINUE
	   cx_data_set( cx_last ) = iset

	ELSEIF ( slash_all ) THEN
	   DO 720 iset = 1, maxdsets
	      IF ( ds_name(iset) .EQ. char_init1024) GOTO 720
	      CALL CANCEL_DATA_SET( iset, status )
 720	   CONTINUE
	ELSE
	   GOTO 5010
	ENDIF
 790	RETURN

* CANCEL MODE mode_name
* loop through the list of mode names given
 800	IF ( num_items .EQ. 0 ) GOTO 5010
	DO 820 item = 1, num_items
* identify the specified mode to CANCEL
           len_test = TM_LENSTR(cmnd_buff( item_start(item):item_end(item) ))
	   DO 810 mode = 1, max_modes
              len_mchars = TM_LENSTR(mode_name( mode)(:4))
	      IF ( MATCH4( cmnd_buff( item_start(item):item_end(item) ),
     .			  len_test, mode_name( mode),len_mchars )) THEN
	         IF ( mode.EQ.pmode_GKS .AND. gksopn ) GOTO 5820
* ... ignore MODE VERIFY from a GO file if MODE VERIFY:ALWAYS is set
	         IF ( mode.EQ.pmode_verify
     .	        .AND. mode_arg(mode,1).EQ.3
     .	        .AND. csp .GT. 0 ) GOTO 820
* ... push the mode stack adding on a FALSE (because it's CANCEL)
	         DO 806 i = mode_stack_size, 2, -1
	            mode_state(mode,i) = mode_state(mode,i-1) 
		    mode_arg  (mode,i) = mode_arg  (mode,i-1)
 806	         CONTINUE
		 mode_state( mode,1 ) = .FALSE.
* ... WAIT requires a string to communicate with PPLUS
	         IF ( mode .EQ. pmode_wait ) THEN
	            ppl_wait = 'NOWAIT'
	            deflts(5) = .FALSE.
* ... when REJECT is changed transformed and regridded variables may be wrong
	         ELSEIF ( mode .EQ. pmode_reject
     .		.AND. (mode_state(mode,1) .NEQV. mode_state(mode,2))) THEN
	            CALL DELETE_TRANSFORMED_VARS( status )
	            IF ( status .NE. ferr_ok ) RETURN
* ... mode metafile may require immediate communication with PPLUS
	         ELSEIF ( mode .EQ. pmode_metafile ) THEN
                 
*     When in gif mode or batch mode, ignore CANCEL MODE META
	            its_gif = ITS_GIF_GRAPHICS()
	            batch = ITS_BATCH_GRAPHICS()
                    IF ( its_gif ) THEN
                       mode_metafile = .FALSE.
                    ELSE IF ( batch ) THEN
                       mode_metafile = .FALSE.
 	               IF ( gksopn ) THEN
                          CALL SEND_PLTYPE( mode_metafile )
                          CALL PPLCMD ( from, line, 0, 'CLSPLT', 1, 1 )  ! 2/93
                       ENDIF
                    ELSE
 	               IF ( gksopn ) THEN
                          CALL SEND_PLTYPE( mode_metafile )
                          CALL PPLCMD ( from, line, 0, 'CLSPLT', 1, 1 )  ! 2/93
                       ENDIF
                    ENDIF

* ... mode REFRESH sends message to gnu READLINE routine to turn off refresh
	         ELSEIF ( mode .EQ. pmode_Xwindows ) THEN
	            CALL TM_SET_FREE_EVENT(0)
* ... mode diagnostic sends message to TMAP library
                 ELSEIF ( mode .EQ. pmode_diagnostic ) THEN
                    tmap_diag_on = .FALSE.
* ... mode journal - turn off journaling
                 ELSEIF ( mode .EQ. pmode_journal ) THEN
	            IF ( jrnl_lun .NE. unspecified_int4 )
     .                  CLOSE( UNIT=jrnl_lun, ERR = 5830 )
		    jrnl_lun = unspecified_int4
* ... mode ppllist - restore PPLUS listings to screen
                 ELSEIF ( mode .EQ. pmode_ppllist ) THEN
	            IF ( jrnl_lun .NE. 6 )
     .                  CLOSE( UNIT=lttout, ERR = 5830 )
		    lttout = 6
* ... mode graticule - turn off automatic graticules
                 ELSEIF ( mode .EQ. pmode_grat ) THEN
	            grat_on = .FALSE.
                    mode_grat_buff = ' '
* ... mode linecolors - resets number of line colors to 6
*                       the mode itself is always .TRUE.
                 ELSEIF ( mode .EQ. pmode_linecolors ) THEN
                    num_linecolors  = 6
		    mode_arg( mode,1 ) = num_linecolors
		    mode_state( mode,1 ) = .TRUE.

                    WRITE (show_str, 2000) mode_arg(pmode_linecolors, 1)
 2000               FORMAT (I3)
                    IF (no_plot_yet) THEN
                       CALL START_PPLUS
                       CALL PPLCMD (from, line, 0, 
     .                   'LINECOLORS '//show_str(1:3)//' 1', 1, 1)
                       CALL DISP_RESET
                    ELSE
                       CALL PPLCMD (from, line, 0, 
     .                   'LINECOLORS '//show_str(1:3)//' 0', 1, 1)
                       CALL WARN ('New number of line colors will take '//
     .                  'effect on next new window: SET WIN/NEW or '//
     .                  'SET WIN/CLEAR')
                    ENDIF

* ... mode pplgui - close reply pipe
                 ELSEIF ( mode .EQ. pmode_gui ) THEN
	            alt_messages = .FALSE.
		    CONTINUE
!	            CALL GUI_PIPE_CLOSE( status )
!		    IF ( status .NE. ferr_ok ) GOTO 5000
	         ENDIF
		 GOTO 820
	      ENDIF
 810	   CONTINUE
* unidentifiable mode name
	   GOTO 5020
 820	CONTINUE
	RETURN

* CANCEL MOVIE
 900	IF ( frame_on ) THEN
	   IF ( frame_file .eq. plaser_flag ) THEN
	      CALL ODR_OP( 'RC', -1, -1 )	! clear record mode
	      CALL LASER_STATUS( buff, status )
	      IF ( status .NE. ferr_OK ) GOTO 5090
	      CALL ODR_QUIT
	      CALL LASER_STATUS( buff, status )
	      IF ( status .NE. ferr_OK ) GOTO 5090
	   ELSE
              frame_file = ' '
	   ENDIF
	ENDIF
	frame_on = .FALSE.
	RETURN

* CANCEL VIEWPORT
 1000	IF ( num_items .EQ. 0 ) THEN
* return to default(full) viewport
           batch = ITS_BATCH_GRAPHICS()
           IF (batch) THEN
              IF ( gksopn ) THEN
                 CALL SEND_PLTYPE( mode_metafile )
                 CALL PPLCMD ( from, line, 0, 'CLSPLT', 1, 1 )  ! 2/93
              ENDIF
           ENDIF
	   CALL SET_VIEWPORT( mvp_dflt )	   
	ELSE
* cancel a viewport definition
	   DO 1050 item = 1, num_items
	      i1 = STR_UPCASE(buff,
     .			cmnd_buff(item_start(item):item_end(item)))
	      DO 1040 ivp = 1, max_viewport
	         IF ( vp_name(ivp).EQ.unspecified_name4 ) GOTO 1040
	         IF ( MATCH_TEMPLATE(vp_name(ivp),buff) ) THEN
	            IF (ivp.EQ.mvp_dflt) GOTO 5040
	            vp_name( ivp ) = unspecified_name4
	            IF ( vp_num .EQ. ivp ) vp_num = mvp_dflt
	         ENDIF
 1040	      CONTINUE
 1050	   CONTINUE
	ENDIF
	RETURN

* CANCEL VARIABLE
* ... data set specified?
* /D=xxx: has the user requested CANCEL VAR/D= ?
 1100	iqual = qual_given( slash_canc_var_data )
	IF ( iqual .GT. 0 ) THEN
	   CALL EQUAL_STRING( cmnd_buff(qual_start(iqual):qual_end(iqual)),
     .			      buff, status )
	   IF ( status .NE. ferr_ok ) RETURN
	   IF ( buff .EQ. ' ' ) THEN
	      iset = pdset_irrelevant	! "/D", alone for default-to-dset
	      buff = 'No current data set'	! ... for error reporting
	      IF ( iset .EQ. unspecified_int4) GOTO 5130
	   ELSE
	      iset = FIND_DSET_NUMBER( buff )
	      IF ( iset .EQ. unspecified_int4) GOTO 5130
	   ENDIF
	ELSE
	   iset = unspecified_int4
	ENDIF
	IF ( slash_all
     .	.OR. (num_items.EQ.0 .AND. iset.NE.unspecified_int4) ) THEN
	   CALL PURGE_ALL_UVARS
	   DO 1110 ivar = 1, max_uvar
	   IF (iset.EQ.unspecified_int4 .OR. iset.EQ.uvar_dset(ivar))
     .				CALL DELETE_USER_VAR(ivar)     ! 7/97
 1110	   CONTINUE
	   num_uvars_in_cmnd = cmnd_uvars_not_given  ! renders EX#n deleted 
	ELSE
	   IF (num_items .EQ. 0) GOTO 5010	   
	   DO 1120 item = 1, num_items
	      i1 = STR_UPCASE(buff,
     .			cmnd_buff(item_start(item):item_end(item)))
* ... decode specified variable name of form VNAME[D=iset] (9/95)
* ... Note: an "odd" behavior for the user will occur if the variable "A"
*	    is defined BOTH as a LET (global) and LET/D (global).  In this
*	    case CAN VAR A will delete only the LET definition.  CAN VAR A
*	    given twice will cancel both.  I regard this as acceptable -- the
*	    case described here is pretty pathological, anyway.
	      IF ( IS_TEMPLATE(buff) ) THEN
	         DO 1115 ivar = 1, max_uvar
	            IF ( uvar_name_code(ivar)(1:3) .NE. 'EX#'
     .	   .AND. (iset.EQ.unspecified_int4.OR.iset.EQ.uvar_dset(ivar)))
     .								THEN
		       IF (MATCH_TEMPLATE(uvar_name_code(ivar),buff)) THEN
* ... note: calling PURGE_ALL_UVARS for each individual deleted uvar is
*		inefficient ... but cautious. (minimum labor implementation)
	                  CALL PURGE_ALL_UVARS
			  CALL DELETE_USER_VAR(ivar)         ! 7/97
	               ENDIF
	            ENDIF
1115	         CONTINUE
	      ELSE
	         CALL PARSE_NAM_DSET( buff, cx_last, iset,
     .			     category, variable, mod_cx, status )
	         IF ( status .NE. ferr_ok ) RETURN
	         IF ( variable .EQ. munknown_var_name ) GOTO 5110
	         IF ( category .NE. cat_user_var ) GOTO 5120
	         CALL PURGE_ALL_UVARS
		 CALL DELETE_USER_VAR(variable)         ! 7/97
	      ENDIF
 1120	   CONTINUE
	ENDIF
	RETURN

* CANCEL AXIS/MODULO
 1200	IF ( qual_given(slash_canc_axis_modulo) .GT. 0 ) THEN

           new_att_modulo = .FALSE.
           new_att_modulo_len = 0.

	   DO 1220 item = 1, num_items
              axis = TM_GET_LINENUM(
     .				cmnd_buff(item_start(item):item_end(item)) )
              IF ( axis .EQ. unspecified_int4 ) GOTO 6210
              IF ( line_modulo(axis) ) THEN
                 line_modulo( axis ) = .FALSE.
* ... wipe memory clear of stored variables - this could change all definitions
*      ( could be more selective for efficiency by checking to see where this
*      axis gets used )
                 DO 1210 i = 1,max_mr_avail
 1210            IF ( mr_protected( i ) .NE. mr_deleted ) 
     .                                CALL DELETE_VARIABLE( i )
              ENDIF
 1220	   CONTINUE

* CANCEL AXIS/ALL

        ELSE IF ( qual_given(slash_canc_axis_all) .GT. 0 ) THEN 	
	   CALL DEALLO_ALL_AXES

* CANCEL AXIS/DEPTH

        ELSE IF ( qual_given(slash_canc_axis_depth) .GT. 0 ) THEN 
        
	   DO 1222 item = 1, num_items
              axis = TM_GET_LINENUM(
     .				cmnd_buff(item_start(item):item_end(item)) )
              IF ( axis .EQ. unspecified_int4 ) GOTO 6210
              IF ( line_direction(axis) .EQ. 'UD' ) THEN
                 line_direction(axis) = 'DU'
              ENDIF

 1222	   CONTINUE

* CANCEL AXIS/STRIDE
        ELSE IF ( qual_given(slash_canc_axis_stride) .GT. 0 ) THEN
           
* ...   Elimate all cached memory
           CALL PURGE_ALL_MEMORY

* ...   Eliminate all memory-resident user-defined vars and their
*       uvar_grid 's 
           CALL PURGE_ALL_UVARS
* ...   
           DO 1223 item = 1, num_items 
              axis = TM_GET_LINENUM(
     .                          cmnd_buff(item_start(item):item_end(item)) )

* ...   Check it is legal to issue CANCEL AXIS/STRIDE
              IF ( axis .EQ. unspecified_int4 ) GOTO 6210
 
*       not legal if axis no longer exists
              IF( line_name(axis) .EQ. char_init16) GOTO 6310

*       not legal if this is not a strided axis
              IF( line_parent(axis) .EQ. 0) GOTO 6320

              igrd = 0
              iline= 0

* ...   Loop through all dynamic grids and find which uses this axis

 6001         CALL TM_NEXT_DYN_GRID( igrd, *6199)

                 DO 6002 idim=1,4
                    iline = grid_line(idim,igrd)

                    IF ( iline .EQ. axis ) THEN

* ...   Reinsert the native axis (the parent of this axis)
                       grid_line(idim,igrd) = line_parent(axis)

* ...   Deallocate this axis
                       line_keep_flag(axis) = .FALSE. 
                       CALL TM_DEALLO_DYN_LINE(axis)

                    ENDIF
 6002            CONTINUE


              GOTO 6001
 6199         CONTINUE 


 1223      CONTINUE 

        ELSE

* ... CANCEL AXIS
	   num_protected_axes = TM_GET_LINENUM( 'EZ' )  ! top protected axis
	   DO 1230 item = 1, num_items
              axis = TM_GET_LINENUM(
     .				cmnd_buff(item_start(item):item_end(item)) )
              IF ( axis .EQ. unspecified_int4 ) GOTO 6210
	      IF ( axis .LE. num_protected_axes ) GOTO 6212
	      line_keep_flag(axis) = .FALSE.  ! see tm_garb_col_grids.F
	      IF (line_use_cnt(axis) .LE. 0) THEN
	         IF ( axis .LE. max_lines ) THEN
                    IF (.NOT. line_regular(axis))
     .                  CALL PACK_LINE_STORAGE(axis)
		    line_name(axis) = char_init16
	         ELSE
	            CALL TM_DEALLO_DYN_LINE(axis)
	         ENDIF
	      ELSE
*     ... cannot delete - in use.  Locate the grid that is using it.
	         grid = TM_GET_GRID_OF_LINE( axis )
	         CALL WARN( 'Not deleted: '//line_name(axis))
	         IF ( grid .NE. unspecified_int4 ) THEN
	            CALL WARN( 
     .		      'Axis is in use by grid '//grid_name(grid))
	         ELSE
	            CALL ERRMSG(ferr_internal, status,
     .				'axis use count err', *5000)
	         ENDIF
	      ENDIF
 1230	   CONTINUE
        ENDIF

* ... Check the attribute structure for this axis name and change 
*     attribute values as needed.

       buff2 = ' '
       CALL RESET_AX_ATT (line_name(axis), buff2, 
     .           buff2, buff2, new_att_modulo,
     .           new_att_modulo_len, line_direction(axis), 
     .           line_regular(axis) )

	RETURN

* CANCEL GRID
 1300   CONTINUE
	DO 1330 item = 1, num_items
           grid = TM_GET_GRIDNUM(
     .			cmnd_buff(item_start(item):item_end(item)) )
           IF (grid .EQ. unspecified_int4) GOTO 6215
* ... clear out any user variable dependencies and warn if used by a dataset
	   IF (grid_use_cnt(grid) .GE. 1 .OR. grid .LE. mgrid_tabstract) THEN
	      CALL PURGE_MR_GRID(grid, status)
	      IF ( status .NE. ferr_ok ) THEN
	          CALL WARN( 'Not deleted: '//grid_name(grid))
	          RETURN
	      ENDIF
* ... grids created by data sets without associated vars may get deleted by
*     PURGE_MR_GRID (e.g. handled as climatological_axes.cdf handles axes)
	      IF (grid_name(grid) .EQ. char_init16) RETURN
	   ENDIF
* ... remove the grid if possible
	   IF (grid_use_cnt(grid) .LE. 0) THEN
	      IF ( grid .LE. max_grids ) THEN
	         grid_name(grid) = char_init16
	         DO 1310 idim = 1, 4
 1310		 CALL TM_DEALLO_DYN_LINE( grid_line(idim,grid))
	      ELSE
	         CALL TM_DEALLO_DYN_GRID(grid)
	      ENDIF
	   ELSE
*     ... cannot delete - in use.
	      CALL WARN( 'Not deleted: '//grid_name(grid))
	   ENDIF
 1330	CONTINUE
	RETURN


* CANCEL ATTRIBUTE  /OUTPUT[=all]
 1400	 IF ( num_args .LT. 1 ) GOTO 5010

!
* ... get dset number if present:  CANCEL ATT/D=dset ...
	iqual = qual_given( slash_canc_attr_data ) 
        IF ( iqual .GT. 0 ) THEN
	   CALL EQUAL_STRING( cmnd_buff(qual_start(iqual):qual_end(iqual)),
     .			      show_str, status )  ! show_str is just a buffer
	   IF ( status .NE. ferr_ok ) RETURN
	   IF ( show_str .NE. ' ' ) THEN
	      dset = FIND_DSET_NUMBER( show_str )
	      IF ( dset .EQ. unspecified_int4) THEN
	         CALL WARN('Unknown data set: '
     .		 //show_str(:TM_LENSTR1(show_str)))
	         RETURN
	      ENDIF
	   ENDIF

        ELSE  ! get dataset from context or from [d=]
           buff = cmnd_buff(item_start(1):item_end(1))

           CALL GET_NEW_CX( cx_last, cx_cmnd, .TRUE., status )
           IF ( status .NE. ferr_ok ) GOTO 5000
 
           IF (buff(1:1) .EQ. '(') THEN
              dset = cx_data_set(cx_cmnd)   ! initial value to try; will return dset
              CALL ISIT_COORD_VAR ( buff, dset, 
     .              varname, coordvar, status )
              IF ( status .NE. ferr_ok ) THEN 
                 dset = -2  ! a user-defined axis (coordinate variable)
                 CALL ISIT_COORD_VAR ( buff, dset, 
     .              varname, coordvar, status )
                 IF ( status .NE. ferr_ok ) THEN
                    dset = cx_data_set(cx_cmnd)
                    GOTO 5920
                 ENDIF
              ENDIF

           ELSE

* Get the dataset if given as [d=]

	      IF ( INDEX(buff,'[') .GT. 0 ) THEN
                 dset = pdset_irrelevant	! default (not used ...)
                 CALL PARSE_NAM_DSET(buff, cx_last, dset, cat, var,
     .                 mods_cx, status)
              ELSE
                 cx = is_cx( 1 )
                 IF (cx .EQ. 0 .OR. cx .EQ. unspecified_int4)
     .                cx = cx_cmnd
                dset = cx_data_set(cx)
             ENDIF
	   ENDIF
        ENDIF

!---------

* /OUTPUT

* ... CANCEL ATT/OUTPUT varname.attname

        loc = qual_given( slash_canc_attr_output ) 
        canout = ( loc .GT. 0 )

       IF ( canout ) 
     .   CALL EQUAL_STRING( cmnd_buff(qual_start(loc):qual_end(loc)),
     .			    buff2, status )

* Get varname.attname and see if a dataset is specified

        buff = cmnd_buff(item_start(1):item_end(1))

!! get dataset from varname.attname
* ... evaluate the context without computing the expression to get the dataset

	CALL GET_NEW_CX( cx_last, cx_cmnd, .TRUE., status )
        IF ( status .NE. ferr_ok ) GOTO 5000
 
        IF (buff(1:1) .EQ. '(') THEN
           dset = cx_data_set(cx_cmnd)   ! initial value to try; will return dset
           CALL ISIT_COORD_VAR ( buff, dset, 
     .            varname, coordvar, status )
           IF ( status .NE. ferr_ok ) THEN 
              dset = -2  ! a user-defined axis (coordinate variable)
              CALL ISIT_COORD_VAR ( buff, dset, 
     .              varname, coordvar, status )
              IF ( status .NE. ferr_ok ) THEN
                 dset = cx_data_set(cx_cmnd)
                 GOTO 5920
              ENDIF
           ENDIF
        ENDIF
c
c        ELSE
c
c* Get the dataset if given as [d=]
c
c	   IF ( INDEX(buff,'[') .GT. 0 ) THEN
c               dset = pdset_irrelevant	! default (not used ...)
c               CALL PARSE_NAM_DSET(buff, cx_last, dset,
c     .				     cat, var, mods_cx, status)
c           ELSE
c              cx = is_cx( 1 )
c              IF (cx .EQ. 0 .OR. cx .EQ. unspecified_int4)
c     .             cx = cx_cmnd
c             dset = cx_data_set(cx)
c          ENDIF
c	ENDIF

        IF (dset .EQ. pdset_irrelevant .OR. dset .EQ. unspecified_int4)
     .        dset = cx_data_set(cx_cmnd) 
        IF (dset .EQ. unspecified_int4) dset = -1

* Check that the variable exists in the dataset

        IF (INDEX(buff, '.') .EQ. 0) THEN
           varname = buff
           CALL CD_GET_VAR_ID (dset, varname, varid, status)
        ELSE
           CALL BREAK_VARNAME( buff, dset, varname, attname, 
     .                         varid,  status )
        ENDIF
        IF (status .NE. ferr_ok) GOTO 6220

* If cancel/output varname.attname, set the attribute output flag to 0

        IF (canout) THEN
           iflag = 0

* Is the command CAN ATT/OUTPUT=all varname?
           slen = TM_LENSTR(buff2)
           IF (slen .GT. 0  .AND.  buff2(1:3) .EQ. 'ALL') THEN

* It must be just a varname, not varname.attname.
              CALL BREAK_VARATTNAME (buff, dset, varname, 
     .           attname, i, .FALSE., status)
              IF (attname .NE. '') GOTO 6240

              CALL CD_GET_VAR_INFO (dset, varid, varname, vtype,
     .                              nvdims, vdims, nvatts, coordvar, 
     .                              all_outflag, status)

* Set all attribute output flags to 0

              DO 1410 iatt = 1, nvatts
                 CALL CD_GET_VAR_ATT_NAME (dset, varid, iatt, attname, 
     .                                     status)
                 CALL CD_SET_ATT_FLAG(dset, varid, attname, iflag, status)
 1410         CONTINUE

           ELSE
* Not /OUT=ALL. For single attribute, see if it exists and if so change the 
* flag. The user may be wanting to turn off writing of an attribute that 
* Ferret creates, like the AXIS attribute, or bounds...
           
              CALL BREAK_VARATTNAME (buff, dset, varname, 
     .           attname, i, .FALSE., status)

              IF (attname .EQ. '') GOTO 6230

              IF (status .NE. ferr_ok) THEN
                 CALL CD_PUT_NEW_ATTR (dset, varid, attname, 5, 1, iflag,
     .              buff, 1., status)
              ELSE

* Reset an existing attribute output flag to 0
                 CALL CD_SET_ATT_FLAG(dset, varid, attname, iflag, status)
              ENDIF
           ENDIF

* canout is false. The command was CANCEL ATT varname.attname. Delete the attribute
        ELSE

* See if the attribute  exists for the variable
           do_err = .TRUE.
           dummy = varname  ! Save name for changing axis modulo or depth setting

           CALL BREAK_VARATTNAME ( buff, dset, varname, 
     .                 attname, varid, do_err, status )

           IF (attname .EQ. '') GOTO 6230

           IF (status .NE. ferr_ok) GOTO 6220 
           CALL CD_DELETE_ATTRIBUTE (dset, varid, attname, status )

* If this is an axis and we are changing modulo or depth setting, make the 
* setting in the Ferret definition of the axis.

           IF (coordvar) THEN
              IF ( STR_SAME(attname, 'modulo') .EQ. 0) THEN
                 new_att_modulo = .FALSE.
                 new_att_modulo_len = 0.

                 axis = TM_GET_LINENUM(dummy )
                 IF ( axis .EQ. unspecified_int4 ) GOTO 6210
                 IF ( line_modulo(axis) ) THEN
                    line_modulo( axis ) = .FALSE.
* ... wipe memory clear of stored variables - this could change all definitions
*      ( could be more selective for efficiency by checking to see where this
*      axis gets used )
                    DO 1420 i = 1,max_mr_avail
 1420               IF ( mr_protected( i ) .NE. mr_deleted ) 
     .                                CALL DELETE_VARIABLE( i )
                 ENDIF

              ENDIF
              IF ( STR_SAME(attname, 'depth') .EQ. 0) THEN

	         axis = TM_GET_LINENUM(dummy )
                 IF ( axis .EQ. unspecified_int4 ) GOTO 6210
                 IF ( line_direction(axis) .EQ. 'UD' ) THEN
                    line_direction(axis) = 'DU'
                 ENDIF
              ENDIF
           ENDIF   ! coordvar

        ENDIF

	RETURN
        
* CANCEL REDIRECT
 1500   redir_cancel_stdout = .FALSE.
        redir_cancel_stderr = .FALSE.
        IF ( num_args .GT. 0 ) THEN
*          Get all the redirect sources to cancel
           DO 1510 i = 1,num_args
              IF ( MATCH4( cmnd_buff(arg_start(i):arg_end(i) ), 
     .              arg_end(i) - arg_start(i) + 1, 'STDOUT', 6) ) THEN
                 redir_cancel_stdout = .TRUE.
              ELSE IF ( MATCH4( cmnd_buff(arg_start(i):arg_end(i) ), 
     .              arg_end(i) - arg_start(i) + 1, 'STDERR', 6) ) THEN
                 redir_cancel_stderr = .TRUE.
              ELSE
*                Unknown redirect source
                 CALL WARN('Ignored cancel redirect of unknown source: '
     .                   // cmnd_buff(arg_start(i):arg_end(i)) )
              ENDIF
 1510      CONTINUE
           IF ( redir_cancel_stdout .AND. 
     .          (redirect_stdout_flags .EQ. redirect_none) ) THEN
              CALL WARN('stdout is not redirected')
              redir_cancel_stdout = .FALSE.
           ENDIF
           IF ( redir_cancel_stderr .AND. 
     .          (redirect_stderr_flags .EQ. redirect_none) ) THEN
              CALL WARN('stderr is not redirected')
              redir_cancel_stderr = .FALSE.
           ENDIF
        ELSE
*          Default: cancel all active redirects
           IF ( redirect_stdout_flags .NE. redirect_none ) THEN
              redir_cancel_stdout = .TRUE.
           ENDIF
           IF ( redirect_stderr_flags .NE. redirect_none ) THEN
              redir_cancel_stderr = .TRUE.
           ENDIF
           IF ( (.NOT. redir_cancel_stdout) .AND. 
     .          (.NOT. redir_cancel_stderr) ) THEN
              CALL WARN('No active redirects to cancel')
           ENDIF
        ENDIF
*
        IF ( redir_cancel_stdout ) THEN
           IF ( (redirect_stdout_flags .EQ. redirect_file) .OR.
     .          (redirect_stdout_flags .EQ. redirect_file_tee) ) THEN
*             Check if stdout and stderr are sent to the same LUN
              IF ( (redirect_stderr_lun .NE. redirect_stdout_lun) .OR.
     .             ((redirect_stderr_flags .NE. redirect_file) .AND.
     .           (redirect_stderr_flags .NE. redirect_file_tee)) ) THEN
                 CLOSE(UNIT=redirect_stdout_lun)
              ENDIF
              redirect_stdout_lun = -1
           ENDIF
           redirect_stdout_flags = redirect_none
        ENDIF
*
        IF ( redir_cancel_stderr ) THEN
           IF ( (redirect_stderr_flags .EQ. redirect_file) .OR.
     .          (redirect_stderr_flags .EQ. redirect_file_tee) ) THEN
*             Check if stdout and stderr are sent to the same LUN
              IF ( (redirect_stdout_lun .NE. redirect_stderr_lun) .OR.
     .             ((redirect_stdout_flags .NE. redirect_file) .AND.
     .           (redirect_stdout_flags .NE. redirect_file_tee)) ) THEN
                 CLOSE(UNIT=redirect_stderr_lun)
              ENDIF
              redirect_stderr_lun = -1
           ENDIF
           redirect_stderr_flags = redirect_none
        ENDIF
*
        RETURN

* CANCEL ALIAS
 1600	IF ( slash_all ) THEN
           DO 1610 i = num_predefined_alias+1, total_num_alias
 1610      alias_name(i) = unspecified_name4
        ELSE
	   IF ( num_items .EQ. 0 ) GOTO 5010
	   DO 1630 item = 1, num_items
	      i1 = STR_UPCASE(buff,
     .			cmnd_buff(item_start(item):item_end(item)))
              DO 1620 i = 1, total_num_alias
                 IF ( alias(i) .EQ. unspecified_name4 ) GOTO 1620
                 IF ( MATCH_TEMPLATE(alias_name(i), buff) ) THEN
                    alias_name(i) = unspecified_name4
                 ENDIF
 1620         CONTINUE
 1630      CONTINUE
        ENDIF
	RETURN

* CANCEL SYMBOL
 1700	IF ( slash_all ) THEN
* cancel all symbols
* (this lovely piece of code is borrowed from the PLOT+ routine getcmd.F)
	   i=0
 1705      CALL LSTSYM(arg30,buff,blen,i,ier)	! get next symbol name (arg30)
           IF(ier.EQ.0)THEN
	      CALL DELSYM(arg30,ier)
	      GOTO 1705
	   ENDIF
	   RETURN
	ENDIF
* cancel the template-specified symbol(s)
	IF ( num_items .EQ. 0 ) GOTO 5010
* (this lovely piece of code is borrowed from the PLOT+ routine getcmd.F)
	i=0
 1710   CALL LSTSYM(arg30,buff,blen,i,ier)	! get next symbol name (arg30)
        IF(ier.EQ.0)THEN
* ... check the given templates 
	   DO 1720 i1 = 1, num_items
	      nth = STR_UPCASE(buff,
     .			cmnd_buff(item_start(i1):item_end(i1)))
	      IF ( MATCH_TEMPLATE(arg30,buff) ) CALL DELSYM(arg30,ier)
 1720	   CONTINUE
           GOTO 1710
	ENDIF
	RETURN


* CANCEL NCCACHE. Set all arguments of -1 to restore the default settings.

 1800	cache_size = -1
	cache_nelems = -1
	cache_preemption = -1
	CALL CD_SET_CHUNK_CACHE (cache_size, cache_nelems, 
     .     cache_preemption, status)
	CALL SPLIT_LIST(pttmode_ops, err_lun,
     .   ' Restoring default chunk cache settings', 0)

	RETURN


* error exits
 5000	RETURN
 5010	CALL ERRMSG( ferr_invalid_command, status,
     .			'CANCEL what?', *5000 )
 5020	CALL ERRMSG( ferr_invalid_command, status,
     .			'unknown mode:'//
     .			cmnd_buff( item_start(item):item_end(item) ), *5000 )
 5030	CALL ERRMSG( ferr_unknown_data_set, status, buff(:blen), *5000 )
 5040	CALL ERRMSG( ferr_invalid_command, status,
     .			'cannot cancel viewport:'//
     .			cmnd_buff( item_start(item):item_end(item) ), *5000 )
 5090	CALL ERRMSG( ferr_ODR_error, status, buff, *5000 )
 5110	CALL ERRMSG( ferr_unknown_variable, status, buff(:8), *5000 )
 5120	CALL ERRMSG( ferr_invalid_command, status,
     .		'can only CANCEL variables you DEFINE '//buff, *5000 )
 5130	CALL ERRMSG( ferr_unknown_data_set, status,
     .			buff(:TM_LENSTR1(buff)), *5000 )
 5310	CALL ERRMSG( ferr_invalid_command, status,
     .		'/ALL with named region or axis qualifier', *5000 )
 5400	CALL ERRMSG( ferr_invalid_command, status,
     .		'Unknown alias: '//
     .           cmnd_buff(item_start(item):item_end(item)), *5000 )
 5820	CALL ERRMSG( ferr_invalid_command, status,
     .		     'MODE GKS cannot be modified after the first GKS plot',
     .			*5000 )
 5830	CALL ERRMSG( ferr_erreq, status, 'Cannot close file', *5000 )
 5920   slen = TM_LENSTR(buff)
        CALL ERRMSG( ferr_invalid_command, status,
     .   'variable, axis, or attribute does not exist'//
     .    buff(:slen), *5000 )
 6210	CALL ERRMSG( ferr_invalid_command, status, 'unknown axis: '//
     .		     cmnd_buff(item_start(item):item_end(item)), *5000 )
 6212	CALL ERRMSG( ferr_invalid_command, status, 'protected axis: '//
     .		     cmnd_buff(item_start(item):item_end(item)), *5000 )
 6215	CALL ERRMSG( ferr_unknown_grid, status,
     .		     cmnd_buff(item_start(item):item_end(item)), *5000 )
 6220 CALL ERRMSG( ferr_unknown_variable, status, 
     .               'unknown variable or attribute: '//
     .		     cmnd_buff(item_start(1):item_end(1)), *5000 )
 6230	CALL ERRMSG( ferr_invalid_command, status,
     .	    'CANCEL ATTRIBUTE given with no attribute: '//
     .	    'Use CAN ATT/OUT=ALL varname, or CAN ATT[/OUT] varname.attname '
     .			, *5000 )
 6240	CALL ERRMSG( ferr_invalid_command, status,
     .	    'CANCEL ATTRIBUTE/ALL varname.attname: '//
     .	    'cannot specify attribute name '
     .			, *5000 )
 6310   CALL ERRMSG( ferr_invalid_command, status, 'axis no longer exists: '//
     .               cmnd_buff(item_start(item):item_end(item)), *5000 )

 6320   CALL ERRMSG( ferr_invalid_command, status, 'not a strided axis: '//
     .               cmnd_buff(item_start(item):item_end(item)), *5000 )
 
	END

