    	SUBROUTINE XEQ_SHOW( memory )

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
* execute the SHOW command

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system

* revision 0.0 - 4/3/86
* revision 0.1 - 12/15/86
* revision 0.2 - 3/20/87 - changes for new TMAP library
* revision 0.3 - 5/1/87  - added SHOW MOVIE,SHOW LIST/OUT,
*			   SHOW EXPRESSION and SHOW REGION and "(default)" on
*			   SHOW DATA
* revision 0.4 - 5/1/87  - added SHOW VARIABLES/... and SHOW COMMANDS
* revision 0.5 - 6/30/87 - added variable subscript limits to SHOW VARI/DISK
* revision 0.6 - 7/15/87 - corrected arg 1 match on SHOW COMMAND name
* revision 0.7 - 8/29/87 - SHOW LIST updated for unformatted option
* revision 0.8 - 9/10/87 - SHOW LIST updated for BIBO option
* revision 0.9 - 9/16/87 - SHOW VAR --> SHOW DATA & SHOW MEMORY /added SHOW GRID
* revision 1.0 - 9/23/87 - SHOW REGION generalized to work with DEFINE REGION
* 			   and LIST heading option added
* revision 1.1 - 10/5/87 - revamped SHOW LIST logicals
* revision 1.2 - 2/18/88 - check for unspecified grid in SHOW GRID
* revision 1.3 - 3/25/88 - added list_TMAP, mode arguments and interrupts
* revision 1.4 - 4/7/88  - created sub. SHOW_DATA_SET - allow specified dset #
* revision 1.5 - 4/29/88 - allow specified variable in SHOW VAR
* revision 1.6 -  7/2/88 - modified SHOW LIST output
*			 - added SHOW WINDOWS
*			 - permitted SHOW MODE mode (named mode)
* revision 1.7 - 7/22/88 - added SHOW MEMORY/FREE using CHECK_MEMORY
* revision 1.8 - 7/28/88 - allow SHOW GRID to take grid or variable name with
*			   variable names having D= qualifiers
*			   SHOW DATA to accept name as well as number - default
*			   to /BRIEF
* revision 1.9 - 8/15/88 - added SHOW DATA/FILES
* revision 2.0 -  9/1/88 - added SHOW VIEWPORT
* revision 2.1 -11/23/88 - new TM IO library: FIND_LINE_NUMBER-->TM_GET_LINENUM
* revision 2.2 - 1/10/89 - SHOW GRID/ALL displays grid names, only
*			 - SHOW DATA defaults to name and variable list
* revision 2.3 - 2/16/89 - ds_open --> ds_name.EQ.char_init30 to test open-ness
* V200:  6/13/89 - SHOW_MEM_VARS doesn't require work space
*	10/20/89 - SHOW VARIABLES defaults to user-defined vars
*	10/29/89 - viewport 0 not showable
*	11/30/89 - SHOW GRID shows default and last
*	 12/6/89 - added mode_gks
* Unix/RISC port 4/25/91 *sh*: increased ds_name and xwind_node sizes
*                9/24/91 - increased size of "arg" to 8 characters for SHOW VAR
*                11/1/91 - 2 bug fixes: SHOW GRID
*                          1) check for unknown grid
*                          2) don't use comma item_list - it's used inside
* V230: 1/3/92 - added SHOW TRANSFORMS
*      5/14/92 - REMOTE_X mode was replaced by the REFRESH mode
*       6/8/92 - added SHOW ALIAS (allow arguments 7/29)
*       6/9/92 - SHOW LIST uses list_fmt_type
*      6/26/92 - commands, subcommands, and qualifiers 4-->8 characters
*     10/22/92 - added units on uvars, increased uvar name length
*     11/4+/92 - changed qualifiers and length of name on SHOW VIEWPORTS
*      11/9/92 - eliminate arguments from MODE GKS and MODE META
*      1/14/93 - updated SHOW MOVIE
* V300: 2/3/93 - added MODE VERIFY arguments
*      4/19/93 - old style vp's have negative vp_size and clip limits unknown
*      5/12/93 - added MODE PPLLIST and MODE JOURNAL filename
*      6/16/93 - added SHOW QUERIES
*       9/2/93 - reordered SHOW TRANSFORMS to group transforms better
* V301:1/12/94 - show MODE METAFILE filename
*      2/24/94 - incorporate a stack of past mode states
* V310:5/12/94 - fixed bug revealed by adding /ORDER to SET DATA (70th char)
* V320:5/17/94 - array "memory" is passed as argument
*	       - SHOW MEMORY now shows memory size, only
*     12/29/94 - added /FORMAT=STREAM to SHOW LIST 
* V320: 12/29/94 - use SPLIT_LIST to redirect tty output for GUI
*	 1/19/95 - explain "megaword" in SHOW MEMORY 
* V400:   2/6/95 - added SHOW SYMBOL
* V411:  7/95 - Added SHOW VAR output for  LET/D var = ...
* V420:  9/95 - added SHOW GRID/DYNAMIC
*	      - and SHOW LIST (tab and comma delimited formats)
*	11/95 - SHOW TRANSFORMS moved to 2 routines: SHOW_TRANSFORMS and
*						     SHOW_REGRID_TRANSFORMS
*	 4/96 - use arg_to_item to make comma-separated line blank sep
*	      - allow templates in SHOW VAR, GRID, AXIS, SYMBOL, ALIAS, VIEW
*		by using (case-sensitive) MATCH_TEMPLATE
*	      - changed SHOW VIEW and ALIAS from comma-separated to blank sep
*	      - added multi-argument capability to SHOW AXIS
*Linux Port 1/97 *kob* - Added a preprocessor include for tmap_dset.parm
*			  because it needed a preprocessor.
* V420:  1/97 - bug fix: SHOW MEMORY/ALL is ignoring /ALL
* V450:  7/97 - don't show child variables in SHOW VARIABLE
*             - added SHOW FUNCTION -- with template matching (7/22)
* 97.07.31 *jc* - added external function code to SHOW FUNCTIONS
* *kob*  10/97 - added check for string length of zero at 1750.  This error
*                caught by new DEC f77 version
* V500 - *kob* 3/99 - initialiaze line_name w/ 64 char string to match new
*                     length 
*        3/99 *jc*  - added a blank line between internal and external functions
*                   - added /EXTERNAL qualifier to SHOW FUNCTION
*                   - fixed EFCN_MATCH_TEMPLATE to behave like MATCH_TEMPLATE
*                     so that multiple functions are shown

#ifdef unix
	include 'tmap_dims.parm'
#	include "tmap_dset.parm"
	include 'xdset_info.cmn_text'
	external xdset_info_data
	include 'xtm_grid.cmn_text'
	external xgt_grid_data
	include 'gkscm2.cmn'
	include	'ferret.parm'
	include	'slash.parm'
	include 'errmsg.parm'
	include 'movies.parm'
	include 'gfdl_vms.parm'
	include 'xprog_state.cmn'
	include 'xplot_state.cmn'
	include 'xvariables.cmn'
	include 'xcontext.cmn'
	include 'xtext_info.cmn'
	include 'xcommand.cmn'
	include 'xrevision.cmn'
	include 'xinterrupt.cmn'
	include	'xfr_grid.cmn'
	include	'xgks_devices.cmn'
	include	'xalgebra.cmn'
	include	'xgrid_chg_fcns.cmn'
	include 'xrisc.cmn'          ! 12/94 SPLIT_LIST buffer
        include 'EF_Util.cmn'
#else
	INCLUDE 'TMAP_FORMAT:TMAP_DIMS.PARM'
	INCLUDE 'TMAP_FORMAT:TMAP_DSET.PARM'
	INCLUDE 'TMAP_FORMAT:XDSET_INFO.CMN'
	INCLUDE 'TMAP_FORMAT:XTM_GRID.CMN'
	INCLUDE 'TMAP_PPLV11INC:GKSCM2.CMN'
	INCLUDE	'FERRET_CMN:FERRET.PARM'
	INCLUDE	'FERRET_CMN:MOVIES.PARM'
	INCLUDE	'FERRET_CMN:SLASH.PARM'
	INCLUDE 'FERRET_CMN:ERRMSG.PARM'
	INCLUDE 'FERRET_CMN:GFDL_VMS.PARM'
	INCLUDE 'FERRET_CMN:XPROG_STATE.CMN'
	INCLUDE 'FERRET_CMN:XPLOT_STATE.CMN'
	INCLUDE 'FERRET_CMN:XVARIABLES.CMN'
	INCLUDE 'FERRET_CMN:XCONTEXT.CMN'
	INCLUDE 'FERRET_CMN:XTEXT_INFO.CMN'
	INCLUDE 'FERRET_CMN:XCOMMAND.CMN'
	INCLUDE 'FERRET_CMN:XREVISION.CMN'
	INCLUDE 'FERRET_CMN:XINTERRUPT.CMN'
	INCLUDE	'FERRET_CMN:XFR_GRID.CMN'
	INCLUDE	'FERRET_CMN:XGKS_DEVICES.CMN'
	INCLUDE	'FERRET_CMN:XALGEBRA.CMN'
	INCLUDE	'FERRET_CMN:XGRID_CHG_FCNS.CMN'
	INCLUDE 'FERRET_CMN:XINTERRUPT.CMN'

#endif  

* local parameter declarations:
	LOGICAL	    brief
	INTEGER	    len_show_str,
     .		    slash_mem_free,
     .		    slash_mem_temp,
     .		    slash_mem_perm,
     .		    slash_brief,
     .		    slash_full,
     .		    slash_vars,
     .		    slash_file,
     .		    slash_grid_dyn,
     .		    slash_var_dset,
     .		    slash_external,
     .		    slash_internal
	PARAMETER ( len_show_str = 132,		! length of output line buffer
     .		    brief	 = .TRUE.,
     .		    slash_mem_temp = 1 + 2,
     .		    slash_mem_perm = 1 + 3,
     .		    slash_mem_free = 1 + 4,
     .		    slash_brief    = 1 + 1,
     .		    slash_full     = 1 + 2,
     .		    slash_vars     = 1 + 3,
     .		    slash_file     = 1 + 4,
     .		    slash_grid_dyn = 1 + 9,
     .		    slash_var_dset = 1 + 1,
     .		    slash_external = 1 + 2,
     .		    slash_internal = 1 + 3 )

* calling argument declarations:
	REAL memory( mem_blk_size, max_mem_blks )

* local variable declarations:
	LOGICAL	MATCH4, MATCH_TEMPLATE, IS_TEMPLATE,
     .		slash_all, limited_show, nothing,
     .		only_1, done_1, sfull, svars, sfile, sbrief, first
	INTEGER TM_LENSTR1, UVAR_NUMBER, ALIAS_ID, STR_UPCASE,
     .		REGION_NUMBER, GRID_FROM_NAME, FIND_DSET_NUMBER,
     .		TM_GET_LINENUM, VIEWPORT_NUMBER, STR_CASE_BLIND_COMPARE,
     .		GCF_FIND_FCN, EFCN_MATCH_TEMPLATE,
     .		category, mods_cx, v1_blk1, v1_nblks, grid, cx,
     .		status, iqual, mode, iset, i, icmnd, isub, n,
     .		len, len0, iuvar, ivar, dset, line, ivp, i1, i2,
     .		vax_code, rqst_dset
	REAL	xtemp, ytemp
	CHARACTER EXPR_NAME*8, TM_FMT*10,
     .		default*10, show_str*255, arg*24, name*32, arg30*30

	INTEGER efcn_scan

* for string passing to C function in SHOW FUNCTION
      INTEGER      slen
      PARAMETER  ( slen = ef_max_description )
#ifdef sun
      BYTE      fhol(slen)      ! c-type Hollerith string buffer
#else
      INTEGER*1 fhol(slen)      ! c-type Hollerith string buffer
#endif


* command given with /ALL ?
	risc_buff = ' '
	slash_all = .FALSE.
	DO 10 iqual = 1, num_qualifiers
	IF ( qualifier_list( iqual ) .EQ. 1 ) slash_all = .TRUE.
 10	CONTINUE
	limited_show = subcmnd_num .NE. 1 .OR. .NOT. slash_all

* if the user entered a blank-separated list insteas of comma-separated
* then make it behave like a comma-separated list (4/96)
	CALL ARG_TO_ITEM

* select subcommand
	GOTO ( 100, 200, 300, 400, 500, 600, 700, 800, 900,1000,
     .        1100,1200,1300,1400,1500,1600,1700,1800,1900,2000,
     .	      2100,2200 )
     .								subcmnd_num

* SHOW '    ' - (no action)
* arrival at this point usually means an illegal subcommand was given
* and was assumed to be an argument by the command parser
 100	IF ( num_args .GE. 1 )
     .		CALL ERRMSG( ferr_invalid_subcmnd,status,
     .			cmnd_buff( arg_start(1):arg_end(1) ), *5000 )
	IF ( limited_show ) RETURN

* SHOW WINDOWS
 200	nothing = .TRUE.
	CALL SPLIT_LIST(pttmode_explct, ttout_lun, ' OPEN ACTIVE', 12)
	DO 210 i = 1, max_windows
	   IF ( wn_open( i ) ) THEN
	      nothing = .FALSE.
	      IF ( wn_active( i ) ) THEN
	         WRITE ( risc_buff, '(I4,T9,A)' ) i, '*'
	         CALL SPLIT_LIST(pttmode_explct, ttout_lun, risc_buff, 0)
	         risc_buff = ' '
	      ELSE
	         WRITE ( risc_buff, '(I4)' ) i
	         CALL SPLIT_LIST(pttmode_explct, ttout_lun, risc_buff, 0)
	         risc_buff = ' '
	      ENDIF
	   ENDIF
 210	CONTINUE
	IF ( nothing ) CALL SPLIT_LIST(pttmode_explct, ttout_lun,
     .		'   no windows open', 0) 
	IF ( limited_show ) RETURN

* SHOW REGION reg1,reg2,...,regn
 300	CONTINUE
* show default region
	IF ( num_args .EQ. 0 .OR. slash_all ) CALL SHOW_REGION( cx_last )
* show requested regions
	IF ( num_items .GE. 1 ) THEN
	   DO 320 i = 1, num_items
	      IF ( interrupted ) CALL ERRMSG
     .			( ferr_interrupt, status, ' ', *5000 )
	      arg = cmnd_buff( item_start(i):item_end(i) )
	      cx = REGION_NUMBER( arg(:4) )
	      IF ( cx .EQ. unspecified_int4 ) CALL ERRMSG
     .				( ferr_unknown_arg, status, arg(:4), *5000 )
	      CALL SHOW_REGION( cx )
 320	   CONTINUE
	ELSEIF ( slash_all ) THEN
* show all user-defined and pre-defined regions
	   DO 330 cx = 0, min_context, -1
	      IF ( cx_name( cx ) .NE. unspecified_name4 )
     .			CALL SHOW_REGION( cx )
 330	   CONTINUE
	ENDIF
	IF ( limited_show ) RETURN

* SHOW AXIS
* show a particular named axis
 400	IF ( num_args .EQ. 0 .AND. .NOT.slash_all ) THEN
	   CALL SPLIT_LIST(pttmode_explct, ttout_lun,
     .			'SHOW what axis ?', 0)
	   RETURN
	ENDIF

* write column headings
	WRITE ( risc_buff, 3000 )
 3000	FORMAT ( T2,'name',T13,'axis',T31,'# pts',T39,
     .           'start',T60,'end' )
	CALL SPLIT_LIST(pttmode_explct, ttout_lun, risc_buff, 0)
	risc_buff = ' '
	IF ( num_args .GT. 0 ) THEN
	   DO 420 i = 1, num_items
* show the named line(s)
	      i1 = STR_UPCASE(name,cmnd_buff(item_start(i):item_end(i)))
	      DO 410 line = 1, max_lines
	         IF ( interrupted ) CALL ERRMSG
     .			( ferr_interrupt, status, ' ', *5000 )
* kob* 3/99 - initialiaze line_name w/ 64 char string to match new length
	         IF ( line_name(line) .EQ. char_init64 ) GOTO 410
	         IF (MATCH_TEMPLATE(line_name(line),name))
     .					CALL SHOW_LINE( line, .NOT.brief )
 410	      CONTINUE
 420	   CONTINUE
	ELSE

* show all the defined axes
	   DO 450 line = 1, max_lines
	      IF ( interrupted ) CALL ERRMSG
     .			( ferr_interrupt, status, ' ', *5000 )
* kob* 3/99 - initialiaze line_name w/ 64 char string to match new length
	      IF ( line_name(line) .EQ. char_init64 ) GOTO 450
	      CALL SHOW_LINE( line, brief )
 450	   CONTINUE

	ENDIF
	IF ( limited_show ) RETURN

* SHOW EXPRESSION
 500	IF ( num_uvars_in_cmnd .EQ. cmnd_uvars_not_given ) THEN
	   WRITE ( risc_buff, '(8X,A)' ) 
     .				'SET EXPRESSION has not been given or implied'
	   CALL SPLIT_LIST(pttmode_explct, ttout_lun, risc_buff, 0)
	   risc_buff = ' '
	ELSE
	   WRITE ( risc_buff, '(8X,A)' ) 'current output expression(s):'
	   CALL SPLIT_LIST(pttmode_explct, ttout_lun, risc_buff, 0)
	   risc_buff = ' '
	   DO 510 i = 1, num_uvars_in_cmnd
	      iuvar = UVAR_NUMBER( EXPR_NAME(i) )
	      IF ( iuvar .NE. unspecified_int4 ) THEN
	         WRITE (risc_buff,'(8X,A)')
     .			 uvar_text(iuvar)(:TM_LENSTR1(uvar_text(iuvar)))
	         CALL SPLIT_LIST(pttmode_explct, ttout_lun, risc_buff, 0)
	         risc_buff = ' '
	      ENDIF
 510	   CONTINUE
	ENDIF
	IF ( limited_show ) RETURN

* SHOW LIST
 600	WRITE ( risc_buff, 3600 ) '/PRECISION =', list_digits
	CALL SPLIT_LIST(pttmode_explct, ttout_lun, risc_buff, 0)
	risc_buff = ' '
 3600	FORMAT ( 8X,'SET LIST',A,I4 )
        IF ( list_fmt_type .EQ. plist_unformatted ) THEN
	   WRITE ( risc_buff, 3605 ) 'UNFORMATTED'
	ELSEIF ( list_fmt_type .EQ. plist_epic ) THEN
	   WRITE ( risc_buff, 3605 ) 'EPIC'
	ELSEIF ( list_fmt_type .EQ. plist_gt )   THEN
	   WRITE ( risc_buff, 3605 ) 'GT'
	ELSEIF ( list_fmt_type .EQ. plist_cdf )   THEN
	   WRITE ( risc_buff, 3605 ) 'CDF'
	ELSEIF ( list_fmt_type .EQ. plist_stream )   THEN
	   WRITE ( risc_buff, 3605 ) 'STREAM'
	ELSEIF ( list_fmt_type .EQ. plist_comma_del )   THEN
	   WRITE ( risc_buff, 3605 ) 'COMMA_DELIMITED'
	ELSEIF ( list_fmt_type .EQ. plist_tab_del )   THEN
	   WRITE ( risc_buff, 3605 ) 'TAB_DELIMITED'
	ELSEIF ( list_format_given )	THEN
	   WRITE ( risc_buff, 3605 ) list_format(1:TM_LENSTR1(list_format))
	ELSE
	   WRITE ( risc_buff, 3605 ) '[Default]'
	ENDIF
	CALL SPLIT_LIST(pttmode_explct, ttout_lun, risc_buff, 0)
	risc_buff = ' '
 3605	FORMAT ( 8X,'SET LIST/FORMAT = ',A )
	IF ( list_heading ) THEN
	   WRITE ( risc_buff, 3608 ) 'EN'
	ELSE
	   WRITE ( risc_buff, 3608 ) 'DIS'
	ENDIF
	CALL SPLIT_LIST(pttmode_explct, ttout_lun, risc_buff, 0)
	risc_buff = ' '
 3608	FORMAT ( 8X,'SET LIST/HEADING is ',A,'ABLED' )
	IF ( list_append ) THEN
	   WRITE ( risc_buff, 3609 ) 'EN'
	ELSE
	   WRITE ( risc_buff, 3609 ) 'DIS'
	ENDIF
	CALL SPLIT_LIST(pttmode_explct, ttout_lun, risc_buff, 0)
	risc_buff = ' '
 3609	FORMAT ( 8X,'SET LIST/APPEND is ',A,'ABLED' )
	IF ( list_file(1:4) .EQ. 'AUTO' ) THEN
	   WRITE ( risc_buff, 3620 ) list_file(1:TM_LENSTR1(list_file))//
     .							' [created by program]'
	ELSE
	   WRITE ( risc_buff, 3620 ) list_file(1:TM_LENSTR1(list_file))
	ENDIF
	CALL SPLIT_LIST(pttmode_explct, ttout_lun, risc_buff, 0)
	risc_buff = ' '
 3620	FORMAT (8X,'SET LIST/FILE=',A,' ( when LIST/FILE is used )' )
	IF ( limited_show ) RETURN

* SHOW DATA_SET
 700 	sbrief = qual_given( slash_brief ) .GT. 0 
	sfull  = qual_given( slash_full  ) .GT. 0
	svars  = qual_given( slash_vars  ) .GT. 0   .OR. sfull
 	sfile  = qual_given( slash_file  ) .GT. 0   .OR. sfull
	only_1 = num_args .GT. 0
	IF ( only_1 ) THEN
	   name = cmnd_buff(arg_start(1):arg_end(1))
	   dset = FIND_DSET_NUMBER( name )
	   IF ( dset .EQ. unspecified_int4 ) GOTO 5071
	ENDIF

	WRITE ( risc_buff, 3700 )	! heading line
	CALL SPLIT_LIST(pttmode_explct, ttout_lun, risc_buff, 0)
	risc_buff = ' '

	DO 790 iset = 1, maxdsets
	   IF ( interrupted ) CALL ERRMSG
     .			( ferr_interrupt, status, ' ', *5000 )

	   IF ( ds_name(iset) .EQ. char_init40
     .	   .OR. (only_1 .AND. iset .NE. dset) ) GOTO 790

* ... is it the default data set?
	   IF ( iset .EQ. cx_data_set(cx_last) ) THEN
	      default = '(default)'
	      len0 = 9
	   ELSE
	      default = ' '
	      len0 = 1
	   ENDIF
	   len = TM_LENSTR1( ds_des_name( iset ) )
	   WRITE (risc_buff,3710) iset,ds_des_name(iset)(:len),default(:len0)
	   CALL SPLIT_LIST(pttmode_explct, ttout_lun, risc_buff, 0)
	   risc_buff = ' '
* ... by default only name and variables are shown
	   IF ( svars       ) CALL SHOW_DATA_SET_PARAMS ( ttout_lun, iset )
	   IF ( .NOT.sbrief ) CALL SHOW_DATA_SET_VARS( ttout_lun, iset, svars )
	   IF ( sfile       ) CALL SHOW_DATA_FILES      ( ttout_lun, iset )
 790	CONTINUE

	IF ( limited_show ) RETURN
 3700	FORMAT (T6,'currently SET data sets:')
 3710	FORMAT (T2,I4,'> ',A,2X,A)

* SHOW MODE mode
 800	WRITE ( risc_buff, 3800 )
	CALL SPLIT_LIST(pttmode_explct, ttout_lun, risc_buff, 0)
	risc_buff = ' '
 3800	FORMAT( T7,'MODE',T23,'STATE',T36,'ARGUMENT' )
	slash_all = slash_all .OR. num_items .EQ. 0
	DO 850 mode = 1, max_modes
	   IF ( mode_name ( mode ) .EQ. pmode_unused ) GOTO 850
* show this mode ?
	   IF ( .NOT. slash_all ) THEN
	      DO 810 i = 1, num_items
	         name = cmnd_buff( item_start(i):item_end(i) )
	         IF ( MATCH4( name, mode_name( mode ) )) GOTO 820
 810	      CONTINUE
	      GOTO 850	! no match
	   ENDIF
 820	   show_str = '      '//mode_name( mode )
	   IF ( mode_state( mode, 1 ) ) THEN
	      show_str(24:) = 'SET'
	   ELSE
	      show_str(21:) = 'CANCELLED'
	   ENDIF
	   IF ( mode .EQ. pmode_time_lab ) THEN
	      show_str(36:) = date_labels( ABS(ax_dec_pt(t_dim)) )
	   ELSEIF ( mode .EQ. pmode_verify ) THEN
	      show_str(36:) = verify_modes(mode_arg(mode,1))
	   ELSEIF ( mode .EQ. pmode_journal ) THEN
	      show_str(36:) = journal_file
	   ELSEIF ( mode .EQ. pmode_ppllist ) THEN
	      show_str(36:) = ppllist_file
	   ELSEIF ( mode .EQ. pmode_metafile ) THEN
	      show_str(36:) = meta_file			! from GKSCM2  1/94
	   ELSEIF ( mode_arg( mode,1 ) .NE. unspecified_int4 ) THEN
	      WRITE ( show_str(36:42), '(I7)' ) mode_arg( mode,1 )
	   ENDIF
	   CALL SPLIT_LIST(pttmode_explct, ttout_lun,
     .			show_str(:TM_LENSTR1(show_str)), 0)
 850	CONTINUE
	IF ( limited_show ) RETURN

* SHOW MOVIE
 900    IF ( frame_file .EQ. ' ' ) THEN
           WRITE ( risc_buff, 3905 ) 'There is no SET MOVIE file name'
        ELSE
           WRITE ( risc_buff, 3905 ) 'SET MOVIE/FILE = '//
     .     frame_file(1:TM_LENSTR1(frame_file))
        ENDIF
	CALL SPLIT_LIST(pttmode_explct, ttout_lun, risc_buff, 0)
	risc_buff = ' '
 	IF ( frame_file .eq. plaser_flag ) THEN
	   IF ( frame_on ) THEN
	      WRITE ( risc_buff, 3910 ) 'MOVIE', 'EN'
	   ELSE
	      WRITE ( risc_buff, 3910 ) 'MOVIE', 'DIS'
	   ENDIF
        ELSE
           IF ( frame_compress .EQ. 'RLE' ) THEN
              WRITE ( risc_buff, 3920 ) 'ON'
           ELSE
              WRITE ( risc_buff, 3920 ) 'OFF'
           ENDIF
        ENDIF
	CALL SPLIT_LIST(pttmode_explct, ttout_lun, risc_buff, 0)
	risc_buff = ' '
 3905	FORMAT (8X,A)
 3910	FORMAT (8X,'SET ',A,' is ',A,'ABLED')
 3920	FORMAT (8X,'compression is ',A)
	IF ( limited_show ) RETURN

* SHOW VARIABLES
* ... SHOW VARIABLES/DIAGNOSTIC
 1000	IF ( slash_all .OR. qual_given(slash_show_var_diag) .GT. 0 ) THEN
	   IF ( num_args .GT. 0 ) THEN
	      arg = cmnd_buff(arg_start(1):arg_end(1))
	   ELSE
	      arg = unspecified_name4
	   ENDIF
	   CALL GET_MEMORY( max_mrs, v1_blk1, v1_nblks, status )
	   IF ( status .NE. ferr_ok ) GOTO 5000
	   CALL SHOW_DIAG_VARS( arg, memory(1,v1_blk1) )
	   CALL FREE_MEMORY( v1_blk1, v1_nblks )
	ENDIF
* ... SHOW VARIABLES/USER
	IF ( slash_all .OR. qual_given(slash_show_var_diag) .EQ. 0 ) THEN
* ... get dset number if appropriate:  SHOW VAR/USER/D=dset ...
	   iqual = qual_given( slash_var_dset )
	   IF ( iqual .GT. 0 ) THEN
	      CALL EQUAL_STRING( cmnd_buff(qual_start(iqual):qual_end(iqual)),
     .			      show_str, status )  ! show_str is just a buffer
	      IF ( status .NE. ferr_ok ) RETURN
	      IF ( show_str .EQ. ' ' ) THEN
	         rqst_dset = pdset_irrelevant	! global variables
	      ELSE
	         rqst_dset = FIND_DSET_NUMBER( show_str )
	         IF ( rqst_dset .EQ. unspecified_int4) THEN
	            CALL WARN('Unknown data set: '
     .			//show_str(:TM_LENSTR1(show_str)))
	            RETURN
	         ENDIF
	      ENDIF
	   ELSE
	      rqst_dset = unspecified_int4	! all variables
	   ENDIF
* ... SHOW VAR specific_var ... various cases of how to show:
*	SHOW VAR name		- show ALL user variables of this name
*	SHOW VAR name[D=n]	- show only variable from named data set
*	SHOW VAR/D=n name	- show only variable from named data set
	   IF ( num_args .GT. 0 ) THEN
	      DO 1020 i = 1, num_items
	         i1 = STR_UPCASE(arg,cmnd_buff(item_start(i):item_end(i)))
	         IF ( INDEX(arg,'[') .GT. 0 ) THEN
* ... bug (minor): square brackets **may** not contain D=dset information
* 	a work-around: pass a context with data set info (instead of cx_last)
		    dset = pdset_irrelevant	! default (not used ...)
                    CALL PARSE_NAM_DSET(arg, cx_last, dset,
     .				     category, ivar, mods_cx, status)
	            IF (status .EQ. ferr_ok) CALL SHOW_1_UVAR
     .			(ttout_lun, ivar, show_str, ' ', .TRUE.)
	         ELSEIF (  rqst_dset .NE. unspecified_int4) THEN
		    CALL FIND_VAR_NAME(rqst_dset,arg,category,ivar)
	            IF ( ivar .EQ. munknown_var_name
     .		    .OR. category .NE. cat_user_var ) GOTO 5100
	            CALL SHOW_1_UVAR(ttout_lun,ivar,show_str,' ',.TRUE.)
	         ELSE
* ... show ALL user variables matching this name (or name template 4/96)
	            DO 1015 ivar = 1, max_uvar
	               IF (uvar_num_items(ivar).EQ.uvar_deleted) GOTO 1015
	               IF ( uvar_parent(ivar).NE.0
     .		      .AND. .NOT.mode_diagnostic   ) GOTO 1015     ! 7/97
!	               IF (MATCH_NAME(arg,uvar_name_code(ivar)))
	               IF (MATCH_TEMPLATE(uvar_name_code(ivar),arg))
     .			CALL SHOW_1_UVAR
     .				(ttout_lun, ivar, show_str,' ',.TRUE.)
 1015	            CONTINUE
	         ENDIF
 1020	      CONTINUE
	   ELSEIF (rqst_dset .NE. unspecified_int4) THEN
* ... show user variables belonging to the specified data set
	      DO 1030 i = 1, max_uvar
	         IF (uvar_num_items(i) .EQ. uvar_deleted  ) GOTO 1030
	         IF (uvar_name_code(i)(1:3) .EQ. 'EX#'    ) GOTO 1030
	         IF ( rqst_dset .NE. uvar_dset(i)	  ) GOTO 1030
	         IF ( uvar_parent(i).NE.0
     .		 .AND. .NOT.mode_diagnostic		  ) GOTO 1030  ! 7/97
	         CALL SHOW_1_UVAR(ttout_lun,i,show_str,'     ',.TRUE.)
 1030	      CONTINUE
	   ELSE
* ... show ALL user variables of every stripe
* ... 8/95 sorted by data set
	      CALL SPLIT_LIST(pttmode_explct, ttout_lun,
     .			' Created by DEFINE VARIABLE:', 0)
* ... locate the highest data set number in use
	      DO 1035 i1 = maxdsets, 1, -1
	         IF ( ds_name(i1) .NE. char_init40 ) GOTO 1040
 1035	      CONTINUE
 1040	      DO 1060 dset = 1, i1
* ... loop through the currently initialized data sets
	         DO 1050 i = 1, max_uvar
	            IF (uvar_num_items(i) .EQ. uvar_deleted  ) GOTO 1050
	            IF (uvar_name_code(i)(1:3) .EQ. 'EX#'    ) GOTO 1050
	            IF ( dset .NE. uvar_dset(i)		     ) GOTO 1050
	            IF ( uvar_parent(i).NE.0
     .		    .AND. .NOT.mode_diagnostic		     ) GOTO 1050 ! 7/97
	            CALL SHOW_1_UVAR(ttout_lun,i,show_str,'     ',.TRUE.)
 1050	         CONTINUE
 1060	      CONTINUE
* ... global variables defined without /D
	      first = .TRUE.
	      DO 1070 i = 1, max_uvar
	         IF (uvar_num_items(i) .EQ. uvar_deleted  ) GOTO 1070
	         IF (uvar_name_code(i)(1:3) .EQ. 'EX#'    ) GOTO 1070
	         IF ( uvar_dset(i) .NE. unspecified_int4  ) GOTO 1070
	         IF ( uvar_parent(i).NE.0
     .		 .AND. .NOT.mode_diagnostic		  ) GOTO 1070  ! 7/97
		 IF ( first ) THEN
	            CALL SPLIT_LIST(pttmode_explct, ttout_lun,
     .	' >>> Definitions that replace any file variable of same name:'
     .		, 0)
	            first = .FALSE.	            
	         ENDIF
	         CALL SHOW_1_UVAR(ttout_lun,i,show_str,'     ',.TRUE.)
 1070	      CONTINUE
* ... global variables defined with /D
	      first = .TRUE.
	      DO 1080 i = 1, max_uvar
	         IF (uvar_num_items(i) .EQ. uvar_deleted  ) GOTO 1080
	         IF (uvar_name_code(i)(1:3) .EQ. 'EX#'    ) GOTO 1080
	         IF ( uvar_dset(i) .NE. pdset_irrelevant  ) GOTO 1080
	         IF ( uvar_parent(i).NE.0
     .		 .AND. .NOT.mode_diagnostic		  ) GOTO 1080  ! 7/97
		 IF ( first ) THEN
	            CALL SPLIT_LIST(pttmode_explct, ttout_lun,
     .	' >>> Definitions used if no file variable of this name exists:'
     .		, 0)
	            first = .FALSE.	            
	         ENDIF
	         CALL SHOW_1_UVAR(ttout_lun,i,show_str,'     ',.TRUE.)
 1080	      CONTINUE
	   ENDIF
 4010	FORMAT (1x,A,' = ',A)
 4020	FORMAT (T20,'"',A,'"')
	ENDIF
	IF ( limited_show ) RETURN

* SHOW COMMAND [name]
 1100	only_1 = num_args .GT. 0
	done_1 = .FALSE.
	IF ( .NOT.only_1 ) THEN
	   WRITE (risc_buff, 4100)
     .				program_name(1:len_program_name),
     .				progname_mod (1:len_progname_mod),
     .				revision_level
	   CALL SPLIT_LIST(pttmode_explct, ttout_lun, risc_buff, 0)
	   risc_buff = ' '
 4100	   FORMAT(' Commands in Program ',A,1X,A,' version',F5.2,':')
	ENDIF
* ... loop through commands
	DO 1130 icmnd = 1, total_num_commands
	   IF ( commands(icmnd)(:4) .EQ. '****' ) GOTO 1130
	   IF ( interrupted ) CALL ERRMSG
     .			( ferr_interrupt, status, ' ', *5000 )
* ... if  only 1 command is of interest is this it ?
	   IF ( only_1 .AND. .NOT.MATCH4(cmnd_buff(arg_start(1):arg_end(1)),
     .				commands(icmnd)(:4)) ) GOTO 1130
	   done_1 = .TRUE.
	   len0 = TM_LENSTR1( commands(icmnd) )
* ... and for each command, its subcommands
	   DO 1120 isub = subcommand_pointer(icmnd),
     .			  subcommand_pointer(icmnd)+num_subcommands(icmnd)-1
	      IF ( subcommands(isub)(:4) .EQ. '****' ) GOTO 1120
	      show_str = commands(icmnd)(:len0) // ' ' // subcommands(isub)
	      len = TM_LENSTR1( show_str )
* ... and for each subcommand, its qualifiers
	      DO 1110 iqual =   qualifier_pointer(isub),
     .				qualifier_pointer(isub)+num_qualifs(isub)-1
	         IF ( qualifiers(iqual)(:4) .EQ. '****' ) GOTO 1110
	         show_str = show_str(1:len) // '/' // qualifiers(iqual)
	         len = len + 1 + TM_LENSTR1( qualifiers(iqual) )
	         IF ( len .GE. 70 ) THEN
	            CALL SPLIT_LIST(pttmode_explct, ttout_lun,
     .				' '//show_str, len+1)
	            show_str = ' '
	            len = len0 + 2
	         ENDIF
 1110	      CONTINUE
 1115	      IF ( show_str .NE. ' ' ) CALL SPLIT_LIST
     .			(pttmode_explct, ttout_lun, ' '//show_str, len+1)
 1120	   CONTINUE
 1130	CONTINUE
	IF ( only_1 .AND. .NOT.done_1 ) THEN
	   WRITE (risc_buff, 4150)
     .					cmnd_buff( arg_start(1):len_cmnd )
	   CALL SPLIT_LIST(pttmode_explct, ttout_lun, risc_buff, 0)
	   risc_buff = ' '
 4150	   FORMAT(' Not a recognized command: ',A)
	ENDIF
	IF ( .NOT.(only_1.AND.done_1) ) THEN
	   CALL SPLIT_LIST(pttmode_explct, ttout_lun,' ', 1)
	   CALL SPLIT_LIST(pttmode_explct, ttout_lun,
     .		' Use SHOW ALIAS to see alternative command names', 0)
	ENDIF
 1199	IF ( limited_show ) RETURN

* SHOW MEMORY
 1200	show_str = TM_FMT(max_mem_blks*mem_blk_size/1.E6,3,12,len)  
	WRITE (risc_buff, *)
     .	   'Current size of FERRET memory cache: '
     .		//show_str(:len)//' MegaWords  (1 word = 4 bytes)'
	CALL SPLIT_LIST(pttmode_explct, ttout_lun, risc_buff, 0)
	risc_buff = ' '
	IF ( qual_given( slash_mem_free ) .GT. 0 ) THEN
	   CALL CHECK_MEMORY( ttout_lun )
	ELSEIF( slash_all
     .     .OR. qual_given(slash_mem_temp) .GT. 0
     .	   .OR. qual_given(slash_mem_perm) .GT. 0 ) THEN
	   CALL SHOW_MEM_VARS
	ENDIF
	IF ( limited_show ) RETURN

* SHOW GRID grid_or_var1 grid_or_var2 ... grid_or_varN
* first examine the context info ( /X ... etc. )
 1300	CALL STACK_PTR_UP ( cx_stack_ptr, max_context, status )
	IF ( status .NE. ferr_ok ) GOTO 5000
	mods_cx = cx_stack_ptr
	CALL INIT_CONTEXT_MODS( mods_cx )
	CALL GET_CONTEXT_MODS (	cx_last,
     .				cmnd_buff,
     .				mods_cx,
     .				max_qual_list,
     .				num_qualifiers,
     .				qual_start,
     .				qual_end,
     .				unknown_qual_ok,
     .				status	)
	IF ( status .NE. ferr_ok ) GOTO 5000

	IF ( num_args .GE. 1 ) THEN

* loop through the variables named 
! bug fix: 11/91 *sh* : changed comma list to blank-separated argument list
!	    4/96: *sh* used ARG_TO_ITEM to harmonize
	   DO 1350 ivar = 1, num_items
	      i1 = STR_UPCASE(arg,cmnd_buff(item_start(ivar):item_end(ivar)))
	      IF ( IS_TEMPLATE( arg ) ) THEN
	         DO 1340 grid = 1, max_grids
	            IF ( grid_name(grid) .EQ. char_init16 ) GOTO 1340
	            IF (MATCH_TEMPLATE(grid_name(grid) ,arg)) THEN
	               WRITE ( risc_buff, 4220 ) '  '//grid_name(grid)
	               CALL SPLIT_LIST(pttmode_explct, ttout_lun,
     .				risc_buff, 0)
	               risc_buff = ' '
	            ENDIF
 1340	         CONTINUE
	      ELSE
	         grid = GRID_FROM_NAME
     .                    ( cmnd_buff(item_start(ivar):item_end(ivar)),
     .			  cx_last, status )
	         IF ( status .NE. ferr_ok ) GOTO 1350
                 IF ( grid .EQ. unspecified_int4 ) GOTO 5300
	         WRITE ( risc_buff, 4210 ) grid_name( grid )
	         CALL SPLIT_LIST(pttmode_explct, ttout_lun, risc_buff, 0)
	         risc_buff = ' '
	         CALL SHOW_GRID( grid, mods_cx )
	      ENDIF
 1350	   CONTINUE
 4210	   FORMAT ( 4X,'GRID ',A)

	ELSEIF( qual_given(slash_grid_dyn) .GT. 0 ) THEN
	   CALL SHOW_DYN_GRIDS(mods_cx)
	ELSEIF ( slash_all ) THEN
	   WRITE ( risc_buff, 4215 ) grid_name(mgrid_abstract)
	   CALL SPLIT_LIST(pttmode_explct, ttout_lun, risc_buff, 0)
	   risc_buff = ' '
 4215	   FORMAT(' Default grid for DEFINE VARIABLE is ',A)
* show 'em all the names
	   WRITE ( risc_buff, 4220 ) '** GRIDS NAMES **'
	   CALL SPLIT_LIST(pttmode_explct, ttout_lun, risc_buff, 0)
	   risc_buff = ' '
	   DO 1310 grid = 1, max_grids
	      IF ( interrupted ) CALL ERRMSG
     .			( ferr_interrupt, status, ' ', *5000 )
	      IF ( grid_name( grid ) .EQ. char_init16 ) GOTO 1310
	      WRITE ( risc_buff, 4220 ) '  '//grid_name( grid )
	      CALL SPLIT_LIST(pttmode_explct, ttout_lun, risc_buff, 0)
	      risc_buff = ' '
 4220	      FORMAT (4X,A)
 1310	   CONTINUE

	ELSE
!	   IF ( mgrid_abstract .EQ. unspecified_int4 ) THEN
!	      WRITE ( risc_buff, 4215 ) 'unknown'
!	   ELSE
	      WRITE ( risc_buff, 4215 ) grid_name(mgrid_abstract)
!	   ENDIF
	   CALL SPLIT_LIST(pttmode_explct, ttout_lun, risc_buff, 0)
	   risc_buff = ' '
* display the grid from last command
	   cx = is_cx( 1 )
	   IF ( cx.EQ.0 .OR. cx.EQ.unspecified_int4 )	GOTO 1390
	   grid = cx_grid( cx )
	   IF ( grid .EQ. unspecified_int4 ) 		GOTO 1390
	   WRITE ( risc_buff, 4200 ) grid_name(grid)
	   CALL SPLIT_LIST(pttmode_explct, ttout_lun, risc_buff, 0)
	   risc_buff = ' '
	   WRITE ( risc_buff, 4210 ) grid_name(grid)
	   CALL SPLIT_LIST(pttmode_explct, ttout_lun, risc_buff, 0)
	   risc_buff = ' '
 4200	   FORMAT (' Last successful data access was on grid ',A)
	   CALL SHOW_GRID( grid, mods_cx )
	ENDIF

* release context stack space
	CALL STACK_PTR_DN ( cx_stack_ptr, cx_stack_ptr_base, status )
	IF ( status .NE. ferr_ok ) GOTO 5000

* successful completion
 1390 	RETURN

* SHOW VIEWPORT view1,view2,...
* column headings
 1400	WRITE ( risc_buff, 3142 )
	CALL SPLIT_LIST(pttmode_explct, ttout_lun, ' ', 1)
	CALL SPLIT_LIST(pttmode_explct, ttout_lun, risc_buff, 0)
	CALL SPLIT_LIST(pttmode_explct, ttout_lun, ' ', 1)
	risc_buff = ' '
 
	IF ( num_args .GE. 1 ) THEN
* ... loop through the viewports named
	   DO 1420 i = 1, num_items
* ... identify the viewport number from the template (or name)	      
	      i1 = STR_UPCASE(arg,cmnd_buff(item_start(i):item_end(i)))
	      DO 1410 ivp = 1, max_viewport
	         IF ( vp_name(ivp).EQ.unspecified_name4 ) GOTO 1410
	         IF ( MATCH_TEMPLATE(vp_name(ivp), arg) ) THEN 
		    IF ( vp_xclip(ivp) .EQ. unspecified_val4 ) THEN
		       xtemp = 0.0
		       ytemp = 0.0
		    ELSE
		       xtemp = vp_xclip(ivp)
		       ytemp = vp_yclip(ivp)
		    ENDIF		 
	            WRITE (risc_buff,3146) vp_name(ivp), vp_size(ivp),
     .					   vp_xorg(ivp), xtemp,
     .					   vp_yorg(ivp), ytemp
	            CALL SPLIT_LIST(pttmode_explct, ttout_lun, risc_buff, 0)
	            risc_buff = ' '
	         ENDIF
 1410	      CONTINUE
 1420	   CONTINUE
	ELSE
* ... show 'em all
	   DO 1450 ivp = 1,max_viewport
	      IF ( vp_name(ivp).NE.unspecified_name4 ) THEN
		 IF ( vp_xclip(ivp) .EQ. unspecified_val4 ) THEN
		    xtemp = 0.0
		    ytemp = 0.0
		 ELSE
		    xtemp = vp_xclip(ivp)
		    ytemp = vp_yclip(ivp)
		 ENDIF		 
	         WRITE (risc_buff,3146)	vp_name(ivp), vp_size(ivp),
     .					vp_xorg(ivp), xtemp,
     .					vp_yorg(ivp), ytemp
	         CALL SPLIT_LIST(pttmode_explct, ttout_lun, risc_buff, 0)
	         risc_buff = ' '
	      ENDIF
 1450	   CONTINUE
	ENDIF
* tell what's current
	WRITE ( risc_buff, 3148 ) vp_name(vp_num)
	CALL SPLIT_LIST(pttmode_explct, ttout_lun, risc_buff, 0)
	risc_buff = ' '

 	RETURN
 3142	FORMAT(' name',T19,'text',T27,'xlimits',T39,'ylimits')
 3146	FORMAT(1X,A,T18,F5.2,T26,F4.2,',',F4.2,T38,F4.2,',',F4.2)
 3148	FORMAT( T4,'current viewport is ',A)

* SHOW TRANSFORMS
 1500   CONTINUE
	CALL SPLIT_LIST(pttmode_explct, ttout_lun,
     .		'variable transforms e.g.SST[T=1-jan:15-mar@DDC]', 47)
	CALL SHOW_TRANSFORMS
	CALL SPLIT_LIST(pttmode_explct, ttout_lun, ' ', 1)	! blank line
	CALL SPLIT_LIST(pttmode_explct, ttout_lun,
     .		'regridding transforms e.g.SST[GX=x5deg@AVE]', 43)
	CALL SHOW_REGRID_TRANSFORMS
	IF ( limited_show ) RETURN

* SHOW ALIAS
 1600   WRITE (risc_buff, 3160) 'Alias', 'Command'
	CALL SPLIT_LIST(pttmode_explct, ttout_lun, risc_buff, 0)
	risc_buff = ' '
        WRITE (risc_buff, 3160) '-----', '-------'
	CALL SPLIT_LIST(pttmode_explct, ttout_lun, risc_buff, 0)
	risc_buff = ' '
        IF ( num_args .EQ. 0 .OR. slash_all ) THEN
	   DO 1610 i = 1, total_num_alias
	      IF ( alias_name(i) .EQ. unspecified_name4 ) GOTO 1610
	      WRITE ( risc_buff, 3160 ) alias_name(i), alias(i)
	      CALL SPLIT_LIST(pttmode_explct, ttout_lun, risc_buff, 0)
	      risc_buff = ' '
 1610	   CONTINUE
        ELSE
           DO 1630 i1 = 1, num_items
	      i2 = STR_UPCASE(arg,cmnd_buff(item_start(i1):item_end(i1)))
	      DO 1620 i = 1, total_num_alias    
	         IF ( alias(i) .EQ. unspecified_name4 ) GOTO 1620
	         IF ( MATCH_TEMPLATE(alias_name(i), arg) ) THEN 
                    WRITE ( risc_buff, 3160 ) alias_name(i), alias(i)
	            CALL SPLIT_LIST(pttmode_explct, ttout_lun,
     .					risc_buff, 0)
	            risc_buff = ' '
                 ENDIF
 1620	      CONTINUE
 1630      CONTINUE
        ENDIF
 3160   FORMAT (T4,A,T16,A)
	IF ( limited_show ) RETURN

* SHOW SYMBOLS sym1 sym2 ...
*           or SHOW SYMBOL/ALL
 1700	slash_all = slash_all .OR. num_args .EQ. 0
* (this lovely piece of loop code is borrowed from the PLOT+ routine getcmd.F)
	i1=0
 1710   CALL LSTSYM(arg30,show_str,len,i1,i2)	! get next symbol
	len0 = TM_LENSTR1(arg30)
        IF(i2.EQ.0)THEN
	   IF ( num_args .GE. 1 ) THEN
* ... check the names specified by template matching 
	      DO 1720 i = 1, num_items
	         n = STR_UPCASE(arg,
     .			cmnd_buff(item_start(i):item_end(i)))
	         IF ( MATCH_TEMPLATE(arg30(:len0),arg) ) GOTO 1750
 1720	      CONTINUE
	      GOTO 1710			! no match
	   ENDIF
* ... display the symbol definition
* *kob* 10/97 added check for len = 0
 1750	   IF ( len .GT. 0 ) THEN
	       CALL SPLIT_LIST(pttmode_explct, ttout_lun,
     .			arg30(:len0)//' = "'//show_str(:len)//'"', 0)
	   ELSE 
               CALL SPLIT_LIST(pttmode_explct, ttout_lun,
     .			arg30(:len0)//' = ""', 0)
	   ENDIF
           GOTO 1710
	ENDIF

* successful completion
 	RETURN

* SHOW ****
 1800   CONTINUE
* SHOW ****
 1900   CONTINUE
* SHOW ****
 2000   CONTINUE

* SHOW FUNCTIONS
 2100   CONTINUE
* /BRIEF or full description?
 	sbrief = qual_given( slash_brief ) .GT. 0 

* show a whole list or named functions, only?
	IF ( num_args .GT. 0 ) THEN

* ... one or more functions were explicitly named
	   DO 2110 i = 1, num_items
           i1 = STR_UPCASE(arg,cmnd_buff(item_start(i):item_end(i)))
	      len0 = item_end(i) - item_start(i) + 1
* ... identify the named function
* ... first try internal, non-grid-changing functions ...
	      DO 2104 i2 = 1, num_functions
	         IF ( interrupted ) CALL ERRMSG
     .			( ferr_interrupt, status, ' ', *5000 )
	         IF (MATCH_TEMPLATE(alg_fcn(i2),arg(:len0)))
     .           CALL SHOW_1_FUNCTION(ttout_lun, .FALSE., sbrief, i2)
 2104         CONTINUE
* ... second try internal, grid-changing functions
	      DO 2106 i2 = 1, gfcn_num_internal
	         IF ( interrupted ) CALL ERRMSG
     .			( ferr_interrupt, status, ' ', *5000 )
              IF ( MATCH_TEMPLATE(gfcn_name(i2),arg(:len0)) ) CALL
     .                SHOW_1_FUNCTION(ttout_lun, .TRUE., sbrief, i2)
 2106         CONTINUE
* ... third try external functions
	      DO 2108 i2 = gfcn_num_internal+1, gfcn_num_internal+
     .             efcn_scan(gfcn_num_internal)
	         IF ( interrupted ) CALL ERRMSG
     .			( ferr_interrupt, status, ' ', *5000 )
              CALL TM_FTOC_STRNG( arg(:len0), fhol, slen )
              IF ( EFCN_MATCH_TEMPLATE(i2,fhol) .EQ. 1) CALL
     .            SHOW_1_FUNCTION(ttout_lun, .TRUE., sbrief, i2)
 2108         CONTINUE
 2110      CONTINUE


* show lists of functions
	ELSE

* ... show all the internal functions (omit for SHOW FUNC/EXTERN)
	   IF ( qual_given(slash_external) .EQ. 0 ) THEN
* ... internal non-grid-changing functions
	      CALL SPLIT_LIST(pttmode_explct, ttout_lun,
     .			'Functions internal to Ferret:', 0)
	      DO 2120 i = 1, num_functions
	         IF ( alg_fcn(i) .NE. unspecified_name4 ) CALL
     .                SHOW_1_FUNCTION(ttout_lun, .FALSE., sbrief, i)
 2120         CONTINUE
* ... internal grid-changing functions
	      DO 2130 i = 1, gfcn_num_internal
	         CALL SHOW_1_FUNCTION(ttout_lun, .TRUE., sbrief, i)
 2130         CONTINUE
	   ENDIF

* ... show all the external functions
* 3/24/99 *kob* clean up duty - change below to blank space rather than
*               null string
	   IF ( qual_given(slash_internal) .EQ. 0 ) THEN
              CALL SPLIT_LIST(pttmode_explct, ttout_lun,
     .             ' ', 0)       ! Print a blank line
              CALL SPLIT_LIST(pttmode_explct, ttout_lun,
     .           'Externally defined functions available to Ferret:',
     .             0)
	      DO 2140 i = gfcn_num_internal+1, gfcn_num_internal+
     .             efcn_scan(gfcn_num_internal)
	         CALL SHOW_1_FUNCTION(ttout_lun, .TRUE., sbrief, i)
 2140	      CONTINUE
	   ENDIF

	ENDIF
	IF ( limited_show ) RETURN

* SHOW QUERIES
 2200   CONTINUE
	DO 2210 i = 1, max_queries
	   IF ( queries(i) .NE. unspecified_name4 )CALL SPLIT_LIST
     .			(pttmode_explct, ttout_lun, queries(i), 0)
 2210	CONTINUE
	IF ( limited_show ) RETURN

* error exit
 5000	RETURN
 5071	CALL ERRMSG( ferr_unknown_data_set, status,
     .			name(:TM_LENSTR1(name)), *5000 )
 5100	CALL ERRMSG( ferr_unknown_variable, status, arg, *5000 )
 5300   CALL ERRMSG( ferr_invalid_command, status, 'grid for '//
     .              cmnd_buff(item_start(ivar):item_end(ivar))
     .              //' is not available.'//pCR//
     .              '(LOAD this variable and try SHOW GRID again.)',
     .              *5000  )
	END

