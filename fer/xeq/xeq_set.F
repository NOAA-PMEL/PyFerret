	SUBROUTINE XEQ_SET( memory, reconfig )

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
* execute the SET command

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 - 4/3/86
* revision 0.1 - 12/15/86 - implemented SET OUTPUT/FORMAT
* revision 0.2 - 3/20/87  - changes for new TMAP library
* revision 0.3 - 5/1/87   - added SET LIST/OUT/APPEND,
*			    SET MOVIE/OUT=file/APPEND title,
*			    SET REGION, SET EXPRESSION and 'SET what?' error
* revision 0.4 - 8/29/87  - added SET LIST/FORM=UNFORMATTED
* revision 0.5 - 9/9/87   - added SET LIST/FORM=BIBO
* revision 0.6 - 9/16/87  - added SET GRID (not implemented)
*			    moved unknown_qual_bad to FERRET.PARM
* revision 0.7 - 9/23/87  - condensed SET REGION via GET_NEW_cx
*			    and added SET LIST/HEAD
* revision 0.8 - 10/5/87  - revamped SET LIST logicals
* revision 0.9 - 10/22/87 - allowed SET DATA n
* revision 1.0 - 2/15/88  - parameters read from .DES files in SET DATA
* revision 1.1 - 3/18/88   - added list_tmap and SET MODE arguments
* revision 1.2 - 4/15/88   - added SET MODE/LAST
* revision 1.3 - 6/29/88 - allow SET LIST/FORMAT=GT equiv. to =TMAP
*			 - corrected READ (lun,*) bug in SET DATA Tname
*			 - added SET WINDOW
* revision 1.4 - 8/2/88  - added SET MODE WAIT
* revision 1.5 - 9/1/88  - added SET VIEWPORT
* revision 2.0 -11/17/88 - incorporated new TMAP library (handling time series)
* revision 2.1 - 1/10/89 - added SET MODE REJECT, corrected last_mode
* revision 2.2 - 2/16/89 - ds_open --> ds_name.EQ.char_init30 to test open-ness
* revision 2.3 - 2/17/89 - GT_CLOSE_SET --> TM_CLOSE_SET
* revision 2.4 - 4/1/89  - MODE WAIT directly sets PPLUS variable deflts
* V200:   9/1/89 - /FORMAT= code removed to EQUAL_FORMAT
*	10/13/89 - added /LASER option to movie stuff
*	10/23/89 - added SET DATA/EZ and SET VARIABLE
*	11/30/89 - added SET GRID (default for abstract variables)
*	 12/8/89 - added SET DATA/SAVE and /RESTORE
*	12/20/89 - added SET MODE Xwindows
*	  1/5/89 - added SET MODE metafile
*	  5/2/89 - changed code for SET REGION/DX = (bug in DEFINE REGION)
*	 6/11/90 - added CHECK_NEG_TAXIS to check for tstep<0
*	 8/31/90 - added SET MOVIE/START=#
* Unix/RISC port 4/25/91 *sh*: increased ds_name size
*                            : bug fix in SET MODE: mode_GKS --> pmode_GKS
* V230:  5/14/92 - REMOTE_X mode was replaced by the REFRESH mode
*         6/9/92 - MODE DIAGNOSTIC turns on TMAP diagnostics
*         6/?/92 - use list_fmt_type instead of list_unformatted
*        6/12/92 - added SET DATA/FORM=CDF
*        7/21/92 - /EZ optional if /FORMAT=unf or /FORMAT="()" given
*        7/23/92 - SET LIST to be case sensitive
*        8/25/92 - added SET AXIS/MODULO
*        1/14/93 - HDF movies: SET MOVIE/APPEND --> SET MOVIE/COMPRESS
* V300:  5/12/93 - file name args to SET MODE JOURNAL and SET MODE PPLLIST
*	 6/16/93 - mode GUI to open reply channel
* V301:  11/1/94 - bug: must open PLOT+ before MODE PPLLIST file is opened
*		   bug: buff2 enlarged (was 24) cuz ppllist name was truncated
*	 1/12/94 - SET MODE xxx pathname no longer requires quotes->"pathname"
*		   but SET MODE accepts only a single MODE at a time
*		   set mode meta will open PLOT+
*	 1/24/94 - turn on metafiles in DISP_INIT instead of here
*	           (resolve problem of metafiles created with wrong name)
*	 2/14/94 - fixed bug introduced 1/12/94 for SET MODE CAL:DAYS
*	 2/24/94 - incorporate a stack of past mode states
* V312: 5/94 - array "memory" as a calling argument
*	       added SET MEMORY mwords (and calling arg reconfig)
* V314: 8/26/94 *kob* IBM port : Logicalv var mode_state needs to be compared
*				 with NEQV instead of NE
* V320: 12/29/94 - use SPLIT_LIST to redirect tty output for GUI
*	 1/18/94 - added SET MODE VERIFY:ALWAYS
*	 1/29/95 - corrected error message about SET MEMORY/SIZE=
*	 2/28/95 - update TMAP COMMON alt_messages when mode_gui is SET
* V420:  9/27/95 - (for IBM/AIX compiler) eliminate READ(xxx,*) in SET DATA 
*Linux Port 1/97 *kob* - Added a preprocessor include for tmap_dset.parm
*			  because it needed a preprocessor.
*	    4/97 *kob* - added ifdef for MANDATORY_FORMAT_WIDTHS on internal
*			 read because linux/nag f90 needed a format width.
*          12/98 *js* - Disallow set mem command is in secure mode
* V500:  4/28/99 *sh* - added SET AXIS/DEPTH 
*	    5/99 *sh* - implemented SET DATA/ORDER=
*	    8/99 *sh* - bug fix: adjust grid_use_count
* V510: 3/00 *sh* - remove check for negative values on time axis
*		  - add qualifier /TREGULAR to SET DATA for netCDF data sets
*	4/00 *sh* - documentation enhancement, only
* V530:10/00 *sh* - added FILE/FORMAT=delimited/TYPE=/DELIMITERS=
* V533 6/01 *sh* - bug fix "SET GRID" w/out arg didn't check for -111
* v540 *acm* 10/01 increase length of ds_name

	include 'tmap_dims.parm'
	include 'tmap_errors.parm'
#	include "tmap_dset.parm"
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
	include 'gfdl_vms.parm'
	include 'plot_setup.parm'
	include 'xprog_state.cmn'
	include 'xplot_state.cmn'
	include 'xvariables.cmn'
	include 'xcontext.cmn'
	include 'xtext_info.cmn'
	include 'xrevision.cmn'
	include 'xtoday.cmn'
	include 'xvideo.cmn'
	include	'xfr_grid.cmn'
	include	'xcontrol.cmn'
	include	'pltcom_dat.decl'	! with gksopn
	include	'PLTCOM.DAT'	! with gksopn
	include	'switch_inc.decl'	! with deflts
	include	'SWITCH.INC'	! with deflts
        include 'LUNITS.INC'          ! with lttout
        include 'lunits_inc.decl'

* calling argument declarations:
	INTEGER	reconfig
	REAL	memory( mem_blk_size, max_mem_blks )

* local variable declarations:
	INTEGER		REGION_NUMBER, VIEWPORT_NUMBER, GRID_FROM_NAME,
     .                  TM_GET_LINENUM, LIB_GET_LUN,
     .			equal_pos, colon_pos, dset_num,
     .			status, tmap_status, mode, idim, iarg, item, lp, iv,
     .			cx, i, grid, axis, loc, perm(4)
	REAL		val_buf
	CHARACTER	title*128, subtitle*64, arg*4,
     .			buff1*24, buff2*80, data_form*3
	LOGICAL		TM_LEGAL_UNIX_NAME, MATCH4, TM_DIGIT,
     .                  set_last, any_delta, dset_cdf,
     .			do_laser_movie, past_laser_movie, start_frame,
     .                  do_ez, tregular

* local parameter declarations:
	INTEGER		slash_set_last
	PARAMETER     ( slash_set_last = 1 )

* functions
	LOGICAL is_secure

* initialize
	reconfig = 0   	! zero signals NO memory reconfigure

* select subcommand
	GOTO ( 100,200,300,400,500,600,700,800,900,1000,
     .        1100,1200,1300 ) subcmnd_num

* SET '    ' 
* arrival at this point usually means an illegal subcommand was given
* and was assumed to be an argument by the command parser
 100	IF ( num_args .GE. 1 ) THEN
	   CALL ERRMSG( ferr_invalid_subcmnd,status,
     .				cmnd_buff( arg_start(1):arg_end(1) ), *5000 )
	ELSE
	   GOTO 5100
	ENDIF
	RETURN

* SET WINDOW
 200	IF ( interactive ) CALL SET_WINDOW( status )
	RETURN

* SET REGION/I=ilow:ihigh/J=..., ... [reg_name]
* completely replace "last" context if requested
 300	IF ( num_args .EQ. 1 ) THEN
	   arg = cmnd_buff(arg_start(1):arg_end(1))
	   cx = REGION_NUMBER( arg )
	   IF ( cx .EQ. unspecified_int4 ) CALL ERRMSG
     .					( ferr_unknown_arg, status, arg, *5000 )
	   CALL TRANSFER_CONTEXT( cx, cx_last )
	ENDIF

* start with the default context from the last command
	CALL GET_NEW_CX( cx_last, cx_buff, status )
	IF ( status .NE. ferr_ok ) GOTO 5000

* apply delta context specifiers (DX=,DY=, etc) if any
	CALL GET_DELTA_CONTEXT( 10, any_delta, status )
	IF ( status .NE. ferr_ok ) GOTO 5000
	IF ( any_delta ) THEN
	   CALL APPLY_DELTA_CONTEXT( cx_buff, 'DEFAULT', status )
	   IF ( status .NE. ferr_ok ) GOTO 5000
	ENDIF

* illegal delta given ?
	DO 310 idim = 1, 4
 310	IF ( cx_given(idim,cx_buff)
     . .AND. cx_delta(idim,cx_buff) .NE. unspecified_val8 ) GOTO 5320

* wipe clean the history of the command defaults
	CALL FORGET_PAST_CONTEXT_MODS ( cx_buff )

* and since we completed with no errors - make it count
	CALL TRANSFER_CONTEXT ( cx_buff, cx_last )

	RETURN

* SET VIEWPORT
 400	IF ( num_args .EQ. 0 ) GOTO 5100
	item = 1
	iv = VIEWPORT_NUMBER( cmnd_buff(item_start(1):item_end(1)) )
	IF ( iv .EQ. unspecified_int4 ) GOTO 5020
	CALL SET_VIEWPORT( iv )
	RETURN

* SET EXPRESSION var_name1, var_name2, ...
 500	IF ( num_args .EQ. 0 ) GOTO 5100
	CALL DELETE_OLD_EXPR
	CALL BREAK_UP_EXPR( cmnd_buff( arg_start(1):len_cmnd ),
     .			    num_uvars_in_cmnd,
     .			    status )
	RETURN

* SET LIST
* ... SET LIST/PRECISION=#digits
 600	IF ( qual_given( slash_set_list_prec ) .GT. 0) THEN
	   lp = qual_given( slash_set_list_prec )	! list pos.
	   equal_pos = INDEX( cmnd_buff( qual_start(lp):qual_end(lp) ), '=' )
	   IF ( equal_pos .EQ. 0 ) THEN
	      list_digits = default_output_precision
	   ELSE
* syntax change 9/95 for IBM/AIX compatibility
	      buff1 = cmnd_buff( equal_pos+qual_start(lp):qual_end(lp) )
c *kob* 4/97
#ifdef MANDATORY_FORMAT_WIDTHS
	      READ ( buff1,*,ERR=5030 ) iv
#else
	      READ ( buff1,'(I)',ERR=5030 ) iv
#endif
	      list_digits = iv
	   ENDIF
	ENDIF
* ... SET LIST/HEADING
	IF ( qual_given( slash_set_list_heading ) .GT. 0 ) THEN
	   list_heading = .TRUE.
	ENDIF
* ... SET LIST/FILE=filename
        lp = qual_given( slash_set_list_out )
	IF ( lp .GT. 0 ) THEN
           CALL EQUAL_STR_LC( cmnd_buff(qual_start(lp):qual_end(lp)),
     .			      list_file, status )
           IF ( status .NE. ferr_ok ) RETURN
	   IF ( list_file .EQ. ' ' ) list_file = 'AUTO'
	ENDIF
* ... SET LIST/APPEND
	IF ( qual_given( slash_set_list_append ) .GT. 0 ) THEN
	   list_append = .TRUE.
	ENDIF
*  ... SET LIST/FORMAT=(FORTRAN_FORMAT), 'UNFORMATTED', 'BIBO', 'TMAP', 'GT'
	lp = qual_given( slash_set_list_form )
	IF ( lp .GT. 0 ) THEN
	   CALL EQUAL_FORMAT( cmnd_buff(qual_start(lp):qual_end(lp)), status )
	ENDIF
	RETURN

* SET DATA_SET data_set1, data_set2, ...
 700    ds_dset_type = ' '
 	IF ( qual_given(slash_set_data_save) .GT. 0
     .	.OR. qual_given(slash_set_data_restore) .GT. 0 ) THEN
	   IF ( num_items.GT.0 .OR. num_qualifiers.GT.1 ) GOTO 5710
	   IF ( qual_given(slash_set_data_save) .GT. 0 )
     .					saved_dset = cx_data_set(cx_last)
	   IF ( qual_given(slash_set_data_restore) .GT. 0 ) 
     .					cx_data_set(cx_last) = saved_dset
	   RETURN
	ENDIF

* initialize an EZ data set
* ... EZ may be implicit in /FORMAT="unformatted" or "(f77 format)"
        loc = qual_given(slash_set_data_form )
        IF ( loc .GT. 0 ) THEN
           CALL EQUAL_STRING(
     .                   cmnd_buff(qual_start(loc):qual_end(loc)),
     .                   data_form, status )              
           IF ( status .NE. ferr_ok ) GOTO 5000
           do_ez = data_form(:3) .EQ. 'UNF'
     .       .OR.  data_form(:3) .EQ. 'FRE'
     .       .OR.  data_form(:3) .EQ. 'DEL'
     .       .OR.  ( INDEX( data_form, '(' ) .GT. 0 )
        ELSE
           do_ez = .FALSE.
        ENDIF
        do_ez = do_ez .OR. ( qual_given(slash_set_data_ez) .GT. 0 )
	IF ( do_ez ) THEN
	   IF ( num_items .EQ. 0 ) THEN
	      CALL INIT_EZ_DSET( ' ', dset_num, status )
	      IF ( status .NE. ferr_ok ) RETURN
	   ELSE
	     DO 710 item = 1, num_items
	      CALL INIT_EZ_DSET( cmnd_buff(item_start(item):item_end(item)), 
     .				 dset_num, status )
	      IF ( status .NE. ferr_ok ) RETURN
* ... dummy calls to make EZ sets act like other sets
	      CALL GET_DSET_PARMS( dset_num, status )
	      CALL GET_AXIS_SPEED( dset_num )
	      CALL MAKE_CVAR_GRIDS( dset_num, status )
 710	     CONTINUE
	   ENDIF
* ... make this data set current
	   cx_data_set( cx_last )	=  dset_num
	   RETURN
	ENDIF

* initialize all non-EZ data sets	
	IF ( num_items .EQ. 0 ) GOTO 5100
	IF ( qual_given(slash_set_data_var  ) .GT. 0 
     .  .OR. qual_given(slash_set_data_grid ) .GT. 0 
     .  .OR. qual_given(slash_set_data_skip ) .GT. 0 ) GOTO 5700

* interpret the /TREGULAR qualifier (force reg T axis) for netCDF data sets
        loc = qual_given(slash_set_data_tregular )
	tregular = loc .GT. 0

* interpret the /ORDER permutation qualifier for a non-EZ data set
	DO 720 idim = 1,4
 720	perm(idim) = unspecified_int4 ! set default
        loc = qual_given(slash_set_data_order )
        IF ( loc .GT. 0 ) THEN
           CALL EQUAL_PERMUTE(
     .                   cmnd_buff(qual_start(loc):qual_end(loc)),
     .                   .FALSE., .FALSE., .TRUE., perm, status )
           IF ( status .NE. ferr_ok ) GOTO 5000
	ENDIF

	DO 750 item = 1, num_items

* check to see if data set was referred to by number
* 9/95 - replaced "*" with "(I)" read from separate buffer
	   IF ( TM_DIGIT(cmnd_buff(item_start(item):item_start(item)) )) THEN
	      buff1 = cmnd_buff(item_start(item):item_end(item))
* *kob* 4/97
#ifdef MANDATORY_FORMAT_WIDTHS
	      READ (buff1,*,ERR=740) dset_num
#else
	      READ (buff1,'(I)',ERR=740) dset_num
#endif
* ... check for valid number given
	      IF   ( dset_num .LT. 1
     .	      .OR.   dset_num .GT. maxdsets
     .	      .OR.   dset_num .GT. max_gfdl_dsets )		GOTO 5110
	      IF ( ds_name( dset_num ) .EQ. char_init256)	GOTO 5110
	      GOTO 745
	   ENDIF

* no! it wasn't given by number so open the data set
* is this a designated special format?
 740       loc = qual_given(slash_set_data_form )
           IF ( loc .GT. 0 ) THEN
              CALL EQUAL_STRING(
     .                   cmnd_buff(qual_start(loc):qual_end(loc)),
     .                   data_form, status )              
              IF ( status .NE. ferr_ok ) GOTO 5000
              dset_cdf = data_form .EQ. 'CDF'
              IF ( .NOT.dset_cdf ) GOTO 5710   ! only CDF allowed (so far)
              ds_dset_type = 'CDF'   ! used and cleared in tm_init_dset
           ENDIF

* initialize a TMAP-supported data set
 	   CALL TM_INIT_DSET( cmnd_buff( item_start(item):item_end(item) ),
     .			      perm, dset_num, tregular, tmap_status )
	   IF ( tmap_status .EQ. merr_no_action ) THEN
	      GOTO 745					! already init'ed
	   ELSEIF ( tmap_status .NE. merr_ok )	  THEN
	      GOTO 5090					! error in .DES file
	   ENDIF

* interpret special parameters in the descriptor
	   CALL GET_DSET_PARMS( dset_num, status )
	   IF ( status .NE. ferr_ok ) GOTO 790

* determine relative access speeds along each axis
	   CALL GET_AXIS_SPEED( dset_num )

* check for negative time step values (warning, only)
! 3/00	   CALL CHECK_NEG_TAXIS( dset_num )

* build completed ( with time axis ) grids for computable variables
	   CALL MAKE_CVAR_GRIDS( dset_num, status )
	   IF ( status .NE. ferr_ok ) GOTO 790

* make this data set current
 745	   cx_data_set( cx_last )	=  dset_num

* 10/01 allow /TITLE for NetCDF or DODS datasets.
           IF (qual_given(slash_set_data_title) .GT. 0) THEN
              loc = qual_given(slash_set_data_title )
              CALL EQUAL_STRING(
     .                   cmnd_buff(qual_start(loc):qual_end(loc)),
     .                   title, status )
              IF ( status .NE. ferr_ok ) GOTO 5000
              ds_title(dset_num) = title
           ENDIF

 750	CONTINUE
	RETURN


* error in descriptor not detected by GT_INIT_DSET
 790	CALL TM_CLOSE_SET( dset_num, tmap_status )
	RETURN

* SET MODE mode_name argument  (or mode_name:argument)
 800	set_last = qual_given( slash_set_last ) .GT. 0
	IF ( num_items .EQ. 0 ) GOTO 5100
* identify the specified mode to SET
* ... syntax can be SET MODE mode  or  SET MODE mode:arg  or  SET MODE mode arg
* ... since "arg" may contain "/" for a path name treat all text after the end
*     of the named mode as the argument  (modified this block 1/94)
	item = 1
	colon_pos = INDEX( cmnd_buff(item_start(1):item_end(1)), ':' )
	IF ( colon_pos .GT. 0 ) THEN
	   IF ( colon_pos .EQ. 1 ) GOTO 5010   ! colon as 1st character ???
	   buff1 = cmnd_buff( item_start(1):item_start(1)+colon_pos-2)
	   buff2 = cmnd_buff( item_start(1)+colon_pos:len_cmnd )
	ELSEIF ( arg_end(1) .EQ. len_cmnd ) THEN     ! no argument given
	   buff1 = cmnd_buff( item_start(1):item_end(1) )
	   buff2 = ' '
	ELSE
	   buff1 = cmnd_buff( item_start(1):item_end(1) )
	   buff2 = cmnd_buff( arg_end(1)+1:len_cmnd )
	ENDIF
	CALL LEFT_JUST( buff2, buff2, loc )
* ... remove surrounding quotation marks on argument, if any
	IF ( buff2(1:1) .EQ. '"' .AND. buff2(loc:loc) .EQ. '"'
     .	.AND. loc .GT.2  	) buff2 = buff2(2:loc-1)
* ... identify the mode name
	DO 810 mode = 1, max_modes
	   IF ( MATCH4( buff1, mode_name( mode ) )) THEN
	      IF ( mode.EQ.pmode_GKS .AND. gksopn ) GOTO 5820
* ... ignore MODE VERIFY from a GO file if MODE VERIFY:ALWAYS is set
	      IF ( mode.EQ.pmode_verify
     .	     .AND. mode_arg(mode,1).EQ.3
     .	     .AND. csp .GT. 0 ) GOTO 820
	      iarg = unspecified_int4
	      CALL GET_MODE_ARG( mode, buff2, iarg, status )
	      IF ( status .NE. ferr_ok ) GOTO 5000
	      IF ( set_last ) THEN
* ... pop the mode stack
	         DO 804 i = 2, mode_stack_size
	            mode_state(mode,i-1) = mode_state(mode,i) 
		    mode_arg  (mode,i-1) = mode_arg  (mode,i)
 804	         CONTINUE
	      ELSE
* ... push the mode stack adding on a TRUE (because it's SET)
	         DO 806 i = mode_stack_size, 2, -1
	            mode_state(mode,i) = mode_state(mode,i-1) 
		    mode_arg  (mode,i) = mode_arg  (mode,i-1)
 806	         CONTINUE
	         mode_state( mode, 1 ) = .TRUE.
	         IF ( iarg .NE. unspecified_int4 ) 
     .					mode_arg( mode,1 ) = iarg
	      ENDIF
* ... WAIT requires a string to communicate with PPLUS
	      IF ( mode .EQ. pmode_wait ) THEN
		 deflts(5) = mode_wait	! set PPLUS wait state
	         IF ( mode_wait ) THEN
	            ppl_wait = 'WAIT'
	         ELSE
	            ppl_wait = 'NOWAIT'
	         ENDIF
* ... when REJECT is changed transformed and regridded variables may be wrong
* *kob* mode_state needs to compared using NEQV because it is a logical 8/94
	      ELSEIF ( mode .EQ. pmode_reject
     .		    .AND. (mode_state(mode,1) .NEQV. mode_state(mode,2)) ) THEN
	         CALL DELETE_TRANSFORMED_VARS( status )
	         IF ( status .NE. ferr_ok ) RETURN
* ... mode REFRESH send a message to the gnu readline routine
	      ELSEIF ( mode .EQ. pmode_Xwindows ) THEN
	         CALL TM_SET_FREE_EVENT(1)
* ... mode metafile may require immediate communication with PPLUS
	      ELSEIF ( mode .EQ. pmode_metafile ) THEN
	         IF ( .NOT.pplus_started ) CALL START_PPLUS     ! 1/94
	  	 IF ( buff2 .NE. ' ' ) CALL PPLCMD
     .			( from, line, 0, 'PLTNME '//buff2, 1, 1 )
!	         CALL SEND_PLTYPE( mode_metafile )    ! done in DISP_INIT
* ... mode diagnostic sends message to TMAP library
	      ELSEIF ( mode .EQ. pmode_diagnostic ) THEN
	         tmap_diag_on = .TRUE.
* ... mode journal opens a journal file
	      ELSEIF ( mode .EQ. pmode_journal ) THEN
                 IF ( jrnl_lun .NE. unspecified_int4 )
     .                  CLOSE( UNIT=jrnl_lun, ERR = 5830 )
	         CALL INIT_JOURNAL( status )
                 IF ( status .NE. ferr_ok ) THEN
                    mode_journal = .FALSE.
		    GOTO 5840
	         ENDIF
* ... mode ppllist directs PLOT+ listings to a file
	      ELSEIF ( mode .EQ. pmode_ppllist ) THEN
	         IF ( .NOT.pplus_started ) CALL START_PPLUS
                 IF ( lttout .NE. 6 )
     .              CLOSE( UNIT=lttout, ERR = 5830 )
                 status = LIB_GET_LUN( lttout )
                 OPEN ( FILE=ppllist_file,
     .                  UNIT=lttout,
     .                  STATUS='UNKNOWN',
     .                  ERR = 5850 )
* ... mode GUI requires the reply pipe to be opened
              ELSEIF ( mode .EQ. pmode_gui ) THEN
	         alt_messages = .TRUE.   	! TMAP COMMON for error msgs
!                    CALL GUI_PIPE_OPEN( status )
!	             IF ( status .NE. ferr_ok ) GOTO 5000
	      ENDIF
	      GOTO 820
	   ENDIF
 810	CONTINUE
* unidentifiable mode name
	   GOTO 5020
 820	RETURN

* SET MOVIE[/FILE=filename] [title]
 900	past_laser_movie = frame_file .EQ. plaser_flag
	do_laser_movie = qual_given( slash_set_movie_laser ) .GT. 0
	IF ( frame_on ) THEN
	   IF ( past_laser_movie ) THEN
	      CALL ODR_OP( 'RC', -1, -1 )	! clear record mode
	      CALL LASER_STATUS( buff1, status )
	      IF ( status .NE. ferr_OK ) GOTO 5085
	      CALL ODR_QUIT
	      CALL LASER_STATUS( buff1, status )
	      IF ( status .NE. ferr_OK ) GOTO 5085
	   ENDIF
	ENDIF
	frame_on = .FALSE.

        lp = qual_given( slash_set_movie_out )
	IF ( lp .GT. 0 ) THEN
	   IF ( do_laser_movie ) GOTO 5082
           CALL EQUAL_STR_LC( cmnd_buff(qual_start(lp):qual_end(lp)),
     .			      title(1:80), status )
           IF ( status .NE. ferr_ok ) RETURN
           IF ( .NOT. TM_LEGAL_UNIX_NAME(title(1:80)) ) GOTO 5083
           frame_file = title(1:80)
	ENDIF

        lp = qual_given( slash_set_movie_start )
	start_frame = lp .GT. 0
	IF ( start_frame ) THEN
	   IF (.NOT.do_laser_movie) GOTO 5082
           CALL EQUAL_VAL( cmnd_buff(qual_start(lp):qual_end(lp)),
     .			   val_buf, status )
           IF ( status .NE. ferr_ok ) RETURN
	ENDIF

        lp = qual_given( slash_set_movie_compress )
	IF ( lp .GT. 0 ) THEN
           CALL EQUAL_STRING( cmnd_buff(qual_start(lp):qual_end(lp)),
     .                        buff1, status )
           IF ( status .NE. ferr_ok ) RETURN
           IF ( buff1(1:1) .EQ. 'N'
     .     .OR. buff1(1:3) .EQ. 'OFF' ) THEN
              frame_compress = ' '
	   ELSEIF ( buff1 .EQ. ' '
     .     .OR. buff1(1:2) .EQ. 'ON'
     .     .OR. buff1(1:1) .EQ. 'Y' ) THEN
	      frame_compress = 'RLE'
           ELSE
              GOTO 5084
	   ENDIF
        ENDIF
	IF ( num_args .GT. 0 ) THEN
	   title = cmnd_buff( arg_start(1) : len_cmnd )
	ELSE
	   title = 'GFDL movie output'
	ENDIF
	WRITE ( subtitle, 3000 ) program_name(1:len_program_name),
     .				 progname_mod(1:len_progname_mod),
     .				 revision_level	   
 3000	FORMAT(' Program ',A,1X,A,'  version',F5.2 )

	IF ( do_laser_movie ) THEN
	      CALL ODR_INIT
	      CALL LASER_STATUS( buff1, status )
	      IF ( status .NE. ferr_OK ) GOTO 5085
	      IF ( start_frame ) THEN
	         CALL ODR_OP( 'SR', INT(val_buf), -1 )
	         CALL LASER_STATUS( buff1, status )
	         IF ( status .NE. ferr_OK ) GOTO 5085
	      ENDIF
	      CALL ODR_OP( 'RM', -1, -1 )	! record mode:arbitrary # frames
	      CALL LASER_STATUS( buff1, status )
	      IF ( status .NE. ferr_OK ) GOTO 5085
	      frame_file = plaser_flag
	ENDIF
	frame_on = .TRUE.
	RETURN

* SET VARIABLE
 1000	IF ( num_items .NE. 1 ) GOTO 5100
	CALL SET_VAR( cmnd_buff(item_start(1):item_end(1)), status )
	RETURN

* SET GRID
* NOTE: for historical reasons mgrid_abstract is the name of the SET GRID
 1100	IF ( qual_given(slash_set_grid_save) .GT. 0 ) THEN
	   saved_abstract_grid = mgrid_abstract
	   RETURN
	ENDIF
	IF ( num_items .GT. 1 ) THEN
	   GOTO 5100
	ELSEIF ( num_items .EQ. 1 ) THEN
	   grid = GRID_FROM_NAME( cmnd_buff(item_start(1):item_end(1)),
     .			          cx_last, status )
	   IF ( status .NE. ferr_ok ) GOTO 5000
	ELSEIF ( qual_given(slash_set_grid_restore) .GT. 0 ) THEN
	   IF ( saved_abstract_grid.EQ.unspecified_int4) GOTO 6110
	   grid = saved_abstract_grid
	ELSE
* ... set the grid from last command
	   cx = is_cx( 1 )
	   IF ( cx.EQ.0 .OR. cx.EQ.unspecified_int4 )	GOTO 5120
	   grid = cx_grid( cx )
	   IF ( grid .EQ. unspecified_int4 )	GOTO 5120
	ENDIF
* ... adjust the use count
	CALL TM_USE_DYN_GRID(grid)
	CALL TM_DEALLO_DYN_GRID(mgrid_abstract)
* ... eliminate all memory-resident user-defined vars and their uvar_grid 's
	CALL PURGE_ALL_UVARS
	mgrid_abstract = grid
	RETURN

* SET AXIS  /MODULO and /DEPTH
 1200	axis = TM_GET_LINENUM( cmnd_buff(item_start(1):item_end(1)) )
        IF ( axis .EQ. unspecified_int4 ) GOTO 6210
* /MODULO
	IF ( qual_given(slash_set_axis_modulo) .GT. 0 ) THEN
           IF ( .NOT.line_modulo(axis) ) THEN
              line_modulo( axis ) = .TRUE.
* ... wipe memory clear of stored variables - this could change all definitions
*      ( could be more selective for efficiency by checking to see where this
*      axis gets used )
              DO 1210 i = 1,max_mr_avail
 1210         IF ( mr_protected( i ) .NE. mr_deleted ) 
     .                                CALL DELETE_VARIABLE( i )
           ENDIF
        ENDIF
* /DEPTH
	IF ( qual_given(slash_set_axis_depth) .GT. 0 ) THEN
           IF ( line_direction(axis) .EQ. 'DU' ) THEN
	      line_direction(axis) = 'UD'
           ENDIF
        ENDIF
	RETURN

* SET MEMORY/MWORDS=megawords
 1300	IF ( num_args .GT. 0 ) GOTO 5800
	if (is_secure()) then
	   call SPLIT_LIST(pttmode_help, ttout_lun,
     1	        'This command is not allowed.', 0)
	   return
	endif
        lp = qual_given( slash_set_memory_words )
	IF ( lp .GT. 0 ) THEN
           CALL EQUAL_VAL( cmnd_buff(qual_start(lp):qual_end(lp)),
     .			   val_buf, status )
           IF ( status .NE. ferr_ok ) RETURN
	ELSE
	   val_buf = 0
	ENDIF
	IF ( val_buf .GT. 0.0 ) THEN
* for neatness make mem_blk_size always be a multiple of 100
	   reconfig = INT(val_buf*1E6/(max_mem_blks*100))
     .					* max_mem_blks*100
	ENDIF
* clear the current contents of memory
	DO 1310 i = 1,max_mr_avail
	   IF ( mr_protected(i) .NE. mr_deleted ) CALL DELETE_VARIABLE(i)
 1310	CONTINUE
	CALL SPLIT_LIST(pttmode_ops, ttout_lun,
     .			' Cached data cleared from memory', 0)
	RETURN

* error exits
 5000	RETURN
 5010	CALL ERRMSG( ferr_syntax, status,
     .			cmnd_buff( item_start(item):item_end(item) ), *5000 )
 5020	CALL ERRMSG( ferr_invalid_command, status,
     .			'Unknown argument:'//
     .			cmnd_buff( item_start(item):item_end(item) ), *5000 )
 5030	CALL ERRMSG( ferr_syntax, status,
     .			cmnd_buff( qual_start(lp):qual_end(lp) ), *5000 )
 5060	CALL ERRMSG( ferr_erreq, status, list_file,  *5000 )
 5070	CALL ERRMSG( ferr_erreq, status, frame_file, *5000 )
 5080	CALL ERRMSG( status, status, frame_file, *5000 )
 5082	CALL ERRMSG( ferr_invalid_command, status,
     .			'laser movie or on-disk movie?', *5000 )
 5083	CALL ERRMSG( ferr_invalid_command, status, 
     .            'illegal file name: '//
     .            cmnd_buff(qual_start(lp):qual_end(lp)), *5000 )
 5084	CALL ERRMSG( ferr_invalid_command, status,
     .			'Unknown compression type: ON or OFF only', *5000 )
 5085	CALL ERRMSG( status, status, buff1, *5000 )
 5090	CALL ERRMSG( ferr_TMAP_error, status, ' ', *5000 )
 5100	CALL ERRMSG( ferr_invalid_command, status, 
     .			'SET what?', *5000 )
 5110	CALL ERRMSG( ferr_unknown_data_set, status,
     .			cmnd_buff( item_start(item):item_end(item) ), *5000 )
 5120	CALL ERRMSG( ferr_unknown_grid, status, 
     .			'nothing saved from past commands', *5000 )
 5320	CALL ERRMSG( ferr_invalid_command, status,
     .			'delta value not permitted on SET REGION'//
     .			pCR//'check '//ww_dim_name(idim)//' axis', *5000 )
 5700	CALL ERRMSG( ferr_invalid_command, status,
     .			'qualifier(s) valid only with SET DATA/EZ command',
     .			*5000 )
 5710	CALL ERRMSG( ferr_invalid_command, status,
     .		     'illegal qualifier or argument'//pCR//cmnd_buff(:len_cmnd),
     .			*5000 )
 5720	CALL ERRMSG( ferr_invalid_command, status,
     .		     '/ORDER must be permutation of XYZT (with minus signs)'//
     .			pCR//cmnd_buff(:len_cmnd), *5000 )
 5800	CALL ERRMSG( ferr_syntax, status,
     .		     'To reconfigure use SET MEMORY/SIZE=megawords'
     .		     //pCR//cmnd_buff(:len_cmnd), *5000 )
 5810	CALL ERRMSG( ferr_invalid_command, status,
     .		     'SET MODE REMOTE_X requires a node name argument',
     .			*5000 )
 5820	CALL ERRMSG( ferr_invalid_command, status,
     .		     'MODE GKS cannot be modified after the first GKS plot',
     .			*5000 )
 5830	CALL ERRMSG( ferr_erreq, status, 'Unable to close file', *5000 )
 5840	CALL ERRMSG( ferr_invalid_command, status,
     .		     'Unable to open journal file: '//journal_file,
     .			*5000 )
 5850	CALL ERRMSG( ferr_erreq, status, 
     .               'Unable to open '//ppllist_file,
     .			*5000 )
 6110	CALL ERRMSG( ferr_invalid_command, status,
     .		     'no grid has been saved', *5000 )
 6210	CALL ERRMSG( ferr_invalid_command, status,
     .		 'unknown axis: '//cmnd_buff(item_start(1):item_end(1)),
     .               *5000 )
	END

