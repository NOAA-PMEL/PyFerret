    	SUBROUTINE XEQ_SHOW( memory )

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
* V500 -3/99 *jc*  - added a blank line between internal and external functions
*                   - added /EXTERNAL qualifier to SHOW FUNCTION
*                   - fixed EFCN_MATCH_TEMPLATE to behave like MATCH_TEMPLATE
*                     so that multiple functions are shown
*       4/99 *sh* - added MODE STUPID:weak_cache
*       4/99 *sh* - added SHOW AXIS/lims=lo:hi
* V510 4/00 *sh* - netCDF grids are now dynamic
*      5/00 *sh* - added SHOW FUNC/DETAILS
* V522 7/00 *sh* - SHOW SYMBOLS now shows special symbols, too
* V530 8/00 *sh* - bug fix to SHOW SYMBOLS - avoid accidental PPL$XPIXEL eval
* V540 9/01 *sh* - SHOW VIEWPORT to show state of /AXES
* v540 *acm* 10/01 increase length of ds_name
* v540 *acm*  1/02 Change SHOW FUNC so that shared object files are 
*                  not listed on the command: SHO FUNC/INTERNAL func_name
* V554 *acm* 8/03  New calls to make XML-style output for SHOW DATA, SHOW DATA/VAR
*                  SHOW AXIS, SHOW GRID
* V554 *acm* 9/03  Add output for SHOW AXIS in /DODS binary format, 
*                  see the flag do_dods
* V570: *acm* 5/04 - add MODE GRATICULE[:argument] 
* V580: *acm* 10/04 - add xml tags <datasets>  </datasets>, dataset name and title
*                     <axes>  </axes> for SHOW AXIS/XML
* V580: *acm* 10/04 - Add xml output for global and dataset variables. SHOW_1_UVAR_XML
* V600: *acm*  7/05 Fix bug 1186: SHOW FUNC with * properly matches patterns for
*                   external functions as well as internally linked functions.
*                   call MATCH_TEMPLATE for all comparisons.
* V600  *acm*  8/05 Allow more digits in xrevision number
* V600: *acm*  6/05 - Attribute control: SHOW ATT varname[d=].attname SHOW ATT/ALL varname[d=]
*                     SHOW DATA/ATTR [d=]
* 9/06 *acm* 6.01  fixing bug 1439 long symbol values; change length of show_str to 2048
* V602  2/07 *acm* Fix bug 1492, changing attributes of coordinate variables 
* v604 *acm* 7/07 increase length of ds_name
* V604  *acm* 7/07 Dataset names have been increased to 1024, but on output this runs into
*                  conflicts with Fortran INQUIRE and OPEN calls. Disallow output names
*                  longer than 256 characters.
* V606  8/07 *acm* Send informational and error messages that are returned from 
*                  commands via SPLIT_LIST to std error rather than std out, but not
*                  the outputs of SHOW commands. These go to std out.

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
        include 'netcdf.inc'

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
     .		    slash_att,
     .		    slash_grid_dyn,
     .		    slash_var_dset,
     .		    slash_external,
     .		    slash_internal,
     .		    slash_xml,
     .		    slash_attr,
     .              slash_grid_xml,
     .              slash_axis_xml,
     .              slash_axis_dods,
     .              slash_axis_file,
     .              slash_axis_append,
     .              slash_axis_clobber,
     .              slash_axis_shedges,
     .              slash_attr_dset,
     .              slash_attr_output
	PARAMETER ( len_show_str = 2048,		! length of output line buffer
     .		    brief	 = .TRUE.,
     .		    slash_mem_temp = 1 + 2,
     .		    slash_mem_perm = 1 + 3,
     .		    slash_mem_free = 1 + 4,
     .		    slash_brief    = 1 + 1,
     .		    slash_full     = 1 + 2,
     .		    slash_vars     = 1 + 3,
     .		    slash_file     = 1 + 4,
     .		    slash_att      = 1 + 5,
     .		    slash_grid_dyn = 1 + 9,
     .		    slash_var_dset = 1 + 1,
     .		    slash_external = 1 + 2,
     .		    slash_internal = 1 + 3,
     .		    slash_xml      = 1 + 5,
     .		    slash_attr     = 1 + 6,
     .		    slash_grid_xml = 1 + 10,
     .		    slash_axis_xml = 1 + 9,
     .		    slash_axis_dods = 1 + 10,
     .		    slash_axis_file = 1 + 11,
     .		    slash_axis_append = 1 + 12,
     .		    slash_axis_clobber = 1 + 13,
     .		    slash_axis_shedges = 1 + 14,
     .		    slash_attr_dset = 1 + 1,
     .		    slash_attr_output = 1 + 2 )

* calling argument declarations:
	REAL memory( mem_blk_size, max_mem_blks )

* local variable declarations:
	LOGICAL	MATCH4, MATCH_TEMPLATE, IS_TEMPLATE, TM_HAS_STRING,
     .		NC_GET_ATTRIB, slash_all, limited_show, nothing, sdetail,
     .		only_1, done_1, sfull, svars, sfile, sbrief, first,
     .          sxml, do_dods, clobber, fexist, list_at_eof, sho_edges,
     .          firstxml, got_it, coordvar, sattr, do_err, sho_out
	INTEGER TM_LENSTR1, TM_LENSTR, UVAR_NUMBER, STR_UPCASE,
     .		REGION_NUMBER, GRID_FROM_NAME, FIND_DSET_NUMBER,
     .		EFCN_MATCH_TEMPLATE, TM_GET_LINENUM,
     .		category, mods_cx, v1_blk1, v1_nblks, grid, cx,
     .		status, iqual, mode, iset, i, icmnd, isub, n,
     .		len, len0, iuvar, ivar, dset, line, ivp, i1, i2,
     .		rqst_dset, tmp_grid, num_predefined_axes, qfile, 
     .          flen, wsize, wkblk, len_test, len_mchars,
     .          j, num_indices, varid, attlen, maxlen, attype,
     .          vtype, nvdims, vdims(8), nvatts, iatt, dset_last,
     .          llen, loc, cat, var, attoutflag, all_outflag, dot, 
     .          vseq, attid, save_deleted_list_result(max_uvar)
	REAL	xtemp, ytemp, attvals(10)
	CHARACTER EXPR_NAME*8, TM_FMT*12,
     .		default*10, show_str*2048, arg*24, name*32, arg30*30
	CHARACTER	filename*2048, raw_name*2048, varatt*641,
     .			fformatted*12, varname*512, attname*128,
     .			efname*40, upname*40, aname*128

* 10/01 *kob* had to change decl of axesOrCorner for picky f95
        CHARACTER*6 axesOrCorner(2)

	INTEGER efcn_scan

* local data declarations
        DATA axesOrCorner(1)/'axes '/,
     .       axesOrCorner(2)/'edges'/

* for string passing to C function in SHOW MODE FUNCTION
      INTEGER      slen, dlen
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
	      cx = REGION_NUMBER( arg )
	      IF ( cx .EQ. unspecified_int4 ) CALL ERRMSG
     .				( ferr_unknown_arg, status, arg, *5000 )
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
	   CALL SPLIT_LIST(pttmode_explct, err_lun,
     .			'SHOW what axis ?', 0)
	   RETURN
	ENDIF
 	sxml  = qual_given( slash_axis_xml  ) .GT. 0 
 	do_dods  = qual_given( slash_axis_dods  ) .GT. 0 
        clobber = qual_given( slash_axis_clobber  ) .GT. 0
        sho_edges = qual_given( slash_axis_shedges  ) .GT. 0
        qfile = qual_given( slash_axis_file ) 
	list_at_eof  = list_append .OR. 
     .           qual_given( slash_axis_append ) .GT. 0

        IF (sxml) firstxml = .TRUE.

* decode the limit qualifiers
	CALL STACK_PTR_UP ( cx_stack_ptr, max_context, status )
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

* "allocate" (use) scratch grid
        tmp_grid =  max_grids-2  ! see docs in TM_ALLO_DYN_GRID.F
        CALL INIT_GRID( tmp_grid, char_init16, mnormal )

*  For DODS output, always write to a file

* ... output to a file
	IF (do_dods ) THEN
           fformatted = 'UNFORMATTED'
* ... get full file name - from LIST/FILE= or from SET LIST/FILE=
           raw_name = 'FERRET.DAT'
           IF ( qfile .GT. 0) THEN
#ifdef unix
* preserve case of filename for unix
              CALL EQUAL_STR_LC(
     .		cmnd_buff(qual_start(qfile):qual_end(qfile)),
     .		raw_name, status )
#else
* VMS filenames in upper case always
              CALL EQUAL_STRING(
     .			cmnd_buff(qual_start(qfile):qual_end(qfile)),
     .                  raw_name, status )
#endif
           ENDIF
           IF ( status .NE. ferr_ok ) GOTO 5000
           llen = TM_LENSTR1(raw_name)
           IF (llen .GT. 256 ) GOTO 5080
	   INQUIRE( FILE = raw_name, NAME = raw_name, EXIST = fexist )
	   flen = TM_LENSTR1( raw_name )
* ... append or not ?
	   IF ( fexist .AND. list_at_eof ) clobber = .false.

	ENDIF



* write column headings
        IF (.NOT.sxml .AND. .NOT.do_dods) THEN
	   WRITE ( risc_buff, 3000 )
 3000	   FORMAT ( T2,'name',T13,'axis',T31,'# pts',T39,
     .           'start',T60,'end' )
	   CALL SPLIT_LIST(pttmode_explct, ttout_lun, risc_buff, 0)
        ENDIF
	risc_buff = ' '
	IF ( num_args .GT. 0 ) THEN

	   DO 420 i = 1, num_items
* show the named line(s)
	      i1 = STR_UPCASE(name,cmnd_buff(item_start(i):item_end(i)))
*  ... first show the static lines
	      DO 410 line = 1, max_lines
	         IF ( interrupted ) CALL ERRMSG
     .			( ferr_interrupt, status, ' ', *5000 )
	         IF ( line_name(line) .EQ. char_init16 ) GOTO 410
	         IF ( line_name(line)(1:1) .EQ. '(' ) GOTO 410
	         IF (MATCH_TEMPLATE(line_name(line),name)) THEN
                   IF (sxml) THEN
                      IF (firstxml) THEN
                         risc_buff = '<axes>'
                         CALL SPLIT_LIST(pttmode_explct, ttout_lun,
     .                                risc_buff, 0)
                         firstxml = .FALSE.
                      ENDIF
                      CALL SHOW_LINE_XML( ttout_lun, line)
                   ELSE IF (do_dods) THEN
	              DO 402 i2 = 1, 4
	                IF (qual_given( slash_show_grid_x0+i2).GT.0
     .                 .OR. qual_given( slash_show_grid_i0+i2).GT.0)THEN
                           grid_line(i2,tmp_grid) = line


* reserve working array; need length of axis; 2* because we will 
* retain double precision of the coordinates for DODS binary output,
* add 1 for extra point in edges definitions.

	                   wsize =  2*(1+line_dim(line))  
                           CALL GET_WORK_SPC( wsize, wkblk, status )
			   IF (sho_edges) THEN
                              CALL show_line_edges_dods (
     .                              tmp_grid, mods_cx, 
     .                              i2, memory(1,wkblk), clobber, 
     .                              raw_name, status )
                           ELSE
                              CALL show_line_coords_dods (
     .                              tmp_grid, mods_cx, 
     .                              i2, memory(1,wkblk), clobber, 
     .                              raw_name, status )
                           ENDIF
 	                   CALL RELEASE_WORK_SPC
                        ENDIF
 402                  CONTINUE

                   ELSE
                      CALL SHOW_LINE( line, .NOT.brief )
	              DO 405 i2 = 1, 4
	                IF (qual_given( slash_show_grid_x0+i2).GT.0
     .                 .OR. qual_given( slash_show_grid_i0+i2).GT.0)THEN
                         grid_line(i2,tmp_grid) = line
                         CALL SHOW_LINE_COORDS( tmp_grid, mods_cx, i2 )
                        ENDIF
 405                  CONTINUE
                   ENDIF

                 ENDIF
 410	      CONTINUE
*  ... then show the dynamic lines
              line = 0
 414          CALL TM_NEXT_DYN_LINE( line, *416)
	         IF ( interrupted ) CALL ERRMSG
     .			( ferr_interrupt, status, ' ', *5000 )
	         IF (MATCH_TEMPLATE(line_name(line),name)) THEN

                   IF (sxml) THEN
                      IF (firstxml) THEN
                         risc_buff = '<axes>'
                         CALL SPLIT_LIST(pttmode_explct, ttout_lun,
     .                                risc_buff, 0)
                         firstxml = .FALSE.
                      ENDIF
                      CALL SHOW_LINE_XML( ttout_lun, line)
                   ELSE IF (do_dods) THEN

	              DO 411 i2 = 1, 4
	                IF (qual_given( slash_show_grid_x0+i2).GT.0
     .                 .OR. qual_given( slash_show_grid_i0+i2).GT.0)THEN
                           grid_line(i2,tmp_grid) = line

* reserve working array; need length of axis; 2* because we will 
* retain double precision of the coordinates for DODS binary output.
* add 1 for extra point in edges definitions.

	                   wsize =  2*(1+line_dim(line))  
                           CALL GET_WORK_SPC( wsize, wkblk, status )
			   IF (sho_edges) THEN
                              CALL show_line_edges_dods (
     .                              tmp_grid, mods_cx, 
     .                              i2, memory(1,wkblk), clobber, 
     .                              raw_name, status )
                           ELSE
                              CALL show_line_coords_dods (
     .                              tmp_grid, mods_cx, 
     .                              i2, memory(1,wkblk), clobber, 
     .                              raw_name, status )
                           ENDIF
 	                   CALL RELEASE_WORK_SPC


                        ENDIF
 411                  CONTINUE 

                   ELSE
                      CALL SHOW_LINE( line, .NOT.brief )
	              DO 412 i2 = 1, 4
	                IF (qual_given( slash_show_grid_x0+i2).GT.0
     .              .OR. qual_given( slash_show_grid_i0+i2).GT.0) THEN
                         grid_line(i2,tmp_grid) = line
                         CALL SHOW_LINE_COORDS( tmp_grid, mods_cx, i2 )
                       ENDIF
 412                 CONTINUE
                   ENDIF

                 ENDIF
	         GOTO 414
 416          CONTINUE
 420	   CONTINUE
	ELSE

* show all the defined axes
*  ... first show the static lines (for XML output skip pre-defined axes)
           num_predefined_axes = TM_GET_LINENUM('EZ')
	   DO 450 line = 1, max_lines
	      IF ( interrupted ) CALL ERRMSG
     .			( ferr_interrupt, status, ' ', *5000 )
	      IF ( line_name(line) .EQ. char_init16) GOTO 450
              IF (.NOT.sxml .AND. .NOT.do_dods) THEN
                 CALL SHOW_LINE( line, brief )
              ENDIF
              IF (sxml .AND. line .GT. num_predefined_axes) THEN
                 IF (firstxml) THEN
                    risc_buff = '<axes>'
                    CALL SPLIT_LIST(pttmode_explct, ttout_lun,
     .                              risc_buff, 0)
                    firstxml = .FALSE.
                 ENDIF
                 CALL SHOW_LINE_XML( ttout_lun, line)
              ENDIF

 450	   CONTINUE
*  ... then show the dynamic lines
           line = 0
 460       CALL TM_NEXT_DYN_LINE( line, *470)
	      IF ( interrupted ) CALL ERRMSG
     .			( ferr_interrupt, status, ' ', *5000 )
              IF (sxml) THEN
                 IF (firstxml) THEN
                    risc_buff = '<axes>'
                    CALL SPLIT_LIST(pttmode_explct, ttout_lun,
     .                              risc_buff, 0)
                    firstxml = .FALSE.
                 ENDIF
                 CALL SHOW_LINE_XML( ttout_lun, line)

              ELSE
	         CALL SHOW_LINE( line, brief )
              ENDIF
              GOTO 460
 470       CONTINUE
	ENDIF
        IF (.NOT. firstxml .AND. sxml) THEN
           risc_buff = '</axes>'
           CALL SPLIT_LIST(pttmode_explct, ttout_lun, risc_buff, 0)
        ENDIF
	IF ( limited_show ) RETURN

* SHOW EXPRESSION
 500	IF ( num_uvars_in_cmnd .EQ. cmnd_uvars_not_given ) THEN
	   WRITE ( risc_buff, '(8X,A)' ) 
     .				'SET EXPRESSION has not been given or implied'
	   CALL SPLIT_LIST(pttmode_explct, err_lun, risc_buff, 0)
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
 	sattr  = qual_given( slash_attr  ) .GT. 0  
        IF (sattr) sbrief = .TRUE.
 	sxml  = qual_given( slash_xml  ) .GT. 0   
	only_1 = num_args .GT. 0
	IF ( only_1 ) THEN
	   name = cmnd_buff(arg_start(1):arg_end(1))
	   dset = FIND_DSET_NUMBER( name )
	   IF ( dset .EQ. unspecified_int4 ) GOTO 5071
	ENDIF

        IF (.NOT. sxml) THEN
           WRITE ( risc_buff, 3700 )	! heading line
           CALL SPLIT_LIST(pttmode_explct, ttout_lun, risc_buff, 0)
        ELSE
           risc_buff = '<datasets>'
           CALL SPLIT_LIST(pttmode_explct, ttout_lun, risc_buff, 0)
        ENDIF

	risc_buff = ' '

	DO 790 iset = 1, maxdsets
	   IF ( interrupted ) CALL ERRMSG
     .			( ferr_interrupt, status, ' ', *5000 )

	   IF ( ds_name(iset) .EQ. char_init1024
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
           IF (sxml) THEN

* Write data set name
              IF (len0 .EQ. 1)
     .        WRITE (risc_buff,3711) ds_des_name(iset)(:len)
              IF (len0 .EQ. 9)
     .        WRITE (risc_buff,3712) ds_des_name(iset)(:len)
	      CALL SPLIT_LIST(pttmode_explct, ttout_lun, risc_buff, 0)
* Write data set title
	      len = TM_LENSTR1(ds_title(iset))
	      WRITE ( risc_buff, 2020 ) ds_title(iset)(1:len)
	      CALL SPLIT_LIST(pttmode_explct, ttout_lun, risc_buff, 0)

	      len = TM_LENSTR(ds_mod_title(iset))
	      IF (len .GT. 0) THEN
	         len = TM_LENSTR1(ds_mod_title(iset))
	         WRITE ( risc_buff, 2030 ) ds_mod_title(iset)(1:len)
	         CALL SPLIT_LIST(pttmode_explct, ttout_lun, risc_buff, 0)
              ENDIF

           ELSE
	      WRITE (risc_buff,3710) iset,ds_des_name(iset)(:len),default(:len0)
	      CALL SPLIT_LIST(pttmode_explct, ttout_lun, risc_buff, 0)
           ENDIF

	   risc_buff = ' '
* ... by default only name and variables are shown
           IF ( sxml) THEN
              IF (svars) THEN
                 CALL SHOW_DATA_SET_VARS_XML (ttout_lun, iset)
              ELSE
                 CALL SHOW_DATA_SET_XML (ttout_lun, iset)
              ENDIF
           ELSE
	      IF ( svars       ) CALL SHOW_DATA_SET_PARAMS 
     .                             (ttout_lun, iset)
	      IF ( .NOT.sbrief ) CALL SHOW_DATA_SET_VARS
     .                             (ttout_lun, iset, svars)
	      IF ( sfile       ) CALL SHOW_DATA_FILES      
     .                             (ttout_lun, iset)
	      IF ( sattr       ) CALL SHOW_DATA_SET_ATTRS      
     .                             (ttout_lun, iset)
           ENDIF
           IF (sxml) THEN
              risc_buff = '</dataset>'
              CALL SPLIT_LIST(pttmode_explct, ttout_lun, risc_buff, 0)
           ENDIF
 790	CONTINUE

        IF (sxml) THEN
           risc_buff = '</datasets>'
           CALL SPLIT_LIST(pttmode_explct, ttout_lun, risc_buff, 0)
        ENDIF

	IF ( limited_show ) RETURN

 2020   FORMAT( '<title>', A, '</title>' )
 2030   FORMAT( '<title_mod>', A, '</title_mod>' )
 3700	FORMAT (T6,'currently SET data sets:')
 3710	FORMAT (T2,I4,'> ',A,2X,A)
 3711	FORMAT ('<dataset name="',A,'" default="false">')
 3712	FORMAT ('<dataset name="',A,'" default="true">')

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
              len_mchars = TM_LENSTR(mode_name(mode)(:4))
	      DO 810 i = 1, num_items
	         name = cmnd_buff( item_start(i):item_end(i) )
                 len_test = TM_LENSTR(name)
	         IF ( MATCH4( name, len_test,
     .             mode_name( mode),len_mchars )) GOTO 820
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
	   ELSEIF ( mode .EQ. pmode_STUPID ) THEN
	      IF (mode_arg(mode,1).EQ.1) show_str(36:) = 'weak_cache'
	   ELSEIF ( mode .EQ. pmode_grat ) THEN
              dlen = TM_LENSTR1(mode_grat_buff)
	      show_str(36:) = mode_grat_buff(:dlen)
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

 1000   sxml  = qual_given(slash_show_var_xml) .GT. 0 
        IF (sxml) firstxml = .TRUE.
* ... SHOW VARIABLES/DIAGNOSTIC
        IF ( slash_all .OR. qual_given(slash_show_var_diag) .GT. 0 ) THEN
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
	            IF (status .EQ. ferr_ok) THEN 
                       IF (category .EQ. cat_user_var) THEN
                          IF (sxml) THEN
	                     CALL SHOW_1_UVAR_XML 
     .                         (ttout_lun,ivar,show_str,firstxml)
                          ELSE
	                    CALL SHOW_1_UVAR
     .                      (ttout_lun,ivar,show_str,' ',.TRUE.)
                          ENDIF

*  if not a user var, see if its a dataset variable from a nc file.
                       ELSE
                         len = INDEX(arg,'[') - 1
                         IF (INDEX(arg,'.') .NE. 0) GOTO 5100
                         CALL CD_GET_VAR_SEQ( dset, arg(:len), varid )
                         IF (varid .GT. 0) CALL SHOW_1_DSVAR (
     .                         ttout_lun, dset, arg, varid, show_str)
                       ENDIF

                    ENDIF
	         ELSEIF (  rqst_dset .NE. unspecified_int4) THEN
		    CALL FIND_VAR_NAME(rqst_dset,arg,category,ivar)
	            IF ( ivar .EQ. munknown_var_name
     .		        .OR. category .NE. cat_user_var ) THEN
     
*  if not a user var, see if its a dataset variable from a nc file.
                       len = TM_LENSTR1(arg)
                       IF (INDEX(arg,'.') .NE. 0) GOTO 5100
                       CALL cd_get_var_seq( rqst_dset, arg(:len), varid )
                       IF (varid .GT. 0) THEN
                          CALL SHOW_1_DSVAR( ttout_lun, 
     .                      rqst_dset, arg, varid, show_str)
                       ELSE 
                          GOTO 5100
                       ENDIF
                    ELSE
                       IF (sxml) THEN
	                  CALL SHOW_1_UVAR_XML 
     .                      (ttout_lun,ivar,show_str,firstxml)
                       ELSE
	                 CALL SHOW_1_UVAR
     .                      (ttout_lun,ivar,show_str,' ',.TRUE.)
                       ENDIF
                    ENDIF
	         ELSE
* ... show ALL user variables matching this name (or name template 4/96)
*	            DO 1015 ivar = 1, max_uvar
*	               IF (uvar_num_items(ivar).EQ.uvar_deleted) GOTO 1015
                    CALL deleted_list_get_undel(uvar_num_items_head,
     .                                         deleted_list_result(1),
     .                                         max_uvar, num_indices )
                    DO 1012, i2 = 1, max_uvar
                      save_deleted_list_result(i2) = 
     .                       deleted_list_result(i2)
 1012	            CONTINUE
                    DO 1015 j = 1, num_indices
                       ivar = save_deleted_list_result(j)
	               IF ( uvar_parent(ivar).NE.0
     .		      .AND. .NOT.mode_diagnostic   ) GOTO 1015     ! 7/97
!	               IF (MATCH_NAME(arg,uvar_name_code(ivar)))

                       IF (MATCH_TEMPLATE(uvar_name_code(ivar),arg))THEN
                         IF (sxml) THEN
     			    CALL SHOW_1_UVAR_XML
     .                         (ttout_lun,ivar,show_str,firstxml)
                         ELSE
     			    CALL SHOW_1_UVAR
     .				(ttout_lun, ivar, show_str,' ',.TRUE.)
                         ENDIF
                      ENDIF
 1015	            CONTINUE
	         ENDIF
 1020	      CONTINUE
	   ELSEIF (rqst_dset .NE. unspecified_int4) THEN
* ... show user variables belonging to the specified data set
*	      DO 1030 i = 1, max_uvar
*	         IF (uvar_num_items(i) .EQ. uvar_deleted  ) GOTO 1030
              CALL deleted_list_get_undel(uvar_num_items_head,
     .                                    deleted_list_result(1),
     .                                    max_uvar, num_indices )
              DO 1022, i2 = 1, max_uvar
                save_deleted_list_result(i2) = deleted_list_result(i2)
 1022         CONTINUE
              DO 1030 j = 1, num_indices
                 i = save_deleted_list_result(j)
	         IF (uvar_name_code(i)(1:3) .EQ. 'EX#'    ) GOTO 1030
	         IF ( rqst_dset .NE. uvar_dset(i)	  ) GOTO 1030
	         IF ( uvar_parent(i).NE.0
     .		 .AND. .NOT.mode_diagnostic		  ) GOTO 1030  ! 7/97
                 IF (sxml) THEN
                    CALL SHOW_1_UVAR_XML 
     .                      (ttout_lun,i,show_str,firstxml)
                 ELSE
                    CALL SHOW_1_UVAR
     .                 (ttout_lun,i,show_str,'     ',.TRUE.)
                 ENDIF
 1030	      CONTINUE
	   ELSE
* ... show ALL user variables of every stripe
* ... 8/95 sorted by data set
	      IF (.NOT. sxml) CALL SPLIT_LIST(pttmode_explct, ttout_lun,
     .			' Created by DEFINE VARIABLE:', 0)
* ... locate the highest data set number in use
	      DO 1035 i1 = maxdsets, 1, -1
	         IF ( ds_name(i1) .NE. char_init1024) GOTO 1040
 1035	      CONTINUE
 1040	      DO 1060 dset = 1, i1
* ... loop through the currently initialized data sets
*	         DO 1050 i = 1, max_uvar
*	            IF (uvar_num_items(i) .EQ. uvar_deleted  ) GOTO 1050
                 CALL deleted_list_get_undel(uvar_num_items_head,
     .                                      deleted_list_result(1),
     .                                      max_uvar, num_indices )
     
                 DO 1042, i2 = 1, max_uvar
                    save_deleted_list_result(i2) = 
     .		      deleted_list_result(i2)
 1042            CONTINUE
                 DO 1050 j = 1, num_indices
                    i = save_deleted_list_result(j)
	            IF (uvar_name_code(i)(1:3) .EQ. 'EX#'    ) GOTO 1050
	            IF ( dset .NE. uvar_dset(i)		     ) GOTO 1050
	            IF ( uvar_parent(i).NE.0
     .		    .AND. .NOT.mode_diagnostic		     ) GOTO 1050 ! 7/97

                    IF (sxml) THEN
                       CALL SHOW_1_UVAR_XML
     .                      (ttout_lun,i,show_str,firstxml)
                    ELSE
                       CALL SHOW_1_UVAR
     .                    (ttout_lun,i,show_str,'     ',.TRUE.)
                    ENDIF

 1050	         CONTINUE
 1060	      CONTINUE
* ... global variables defined without /D

	      first = .TRUE.

              CALL deleted_list_get_undel(uvar_num_items_head,
     .                                    deleted_list_result(1),
     .                                    max_uvar, num_indices )
              DO 1062, i2 = 1, max_uvar
                    save_deleted_list_result(i2) = 
     .		      deleted_list_result(i2)
 1062         CONTINUE
              DO 1070 j = 1, num_indices
                 i = save_deleted_list_result(j)
	         IF (uvar_name_code(i)(1:3) .EQ. 'EX#'    ) GOTO 1070
	         IF ( uvar_dset(i) .NE. unspecified_int4  ) GOTO 1070
	         IF ( uvar_parent(i).NE.0
     .		 .AND. .NOT.mode_diagnostic		  ) GOTO 1070  ! 7/97

                 IF (sxml) THEN
                    CALL SHOW_1_UVAR_XML
     .                      (ttout_lun,i,show_str,firstxml)
                 ELSE
		    IF ( first ) THEN
	               CALL SPLIT_LIST(pttmode_explct, ttout_lun,
     .	  ' >>> Definitions that replace any file variable of same name:'
     .    , 0)
	               first = .FALSE.	            
	            ENDIF
                    CALL SHOW_1_UVAR
     .                 (ttout_lun,i,show_str,'     ',.TRUE.)
                 ENDIF

 1070	      CONTINUE
* ... global variables defined with /D
	      first = .TRUE.
*	      DO 1080 i = 1, max_uvar
*	         IF (uvar_num_items(i) .EQ. uvar_deleted  ) GOTO 1080
              CALL deleted_list_get_undel(uvar_num_items_head,
     .                                    deleted_list_result(1),
     .                                    max_uvar, num_indices )
              DO 1072, i2 = 1, max_uvar
                    save_deleted_list_result(i2) = 
     .		      deleted_list_result(i2)
 1072         CONTINUE
              DO 1080 j = 1, num_indices
                 i = save_deleted_list_result(j)
	         IF (uvar_name_code(i)(1:3) .EQ. 'EX#'    ) GOTO 1080
	         IF ( uvar_dset(i) .NE. pdset_irrelevant  ) GOTO 1080
	         IF ( uvar_parent(i).NE.0
     .		 .AND. .NOT.mode_diagnostic		  ) GOTO 1080  ! 7/97

                 IF (sxml) THEN
                    CALL SHOW_1_UVAR_XML
     .                      (ttout_lun,i,show_str,firstxml)
                 ELSE
		    IF ( first ) THEN
	               CALL SPLIT_LIST(pttmode_explct, ttout_lun,
     .	' >>> Definitions used if no file variable of this name exists:'
     .    , 0)
	               first = .FALSE.	            
	            ENDIF
                    CALL SHOW_1_UVAR
     .                 (ttout_lun,i,show_str,'     ',.TRUE.)
                 ENDIF



 1080	      CONTINUE
	   ENDIF
 4010	FORMAT (1x,A,' = ',A)
 4020	FORMAT (T20,'"',A,'"')
	ENDIF
        IF (.NOT. firstxml .AND. sxml) THEN
           risc_buff = '</global>'
           CALL SPLIT_LIST(pttmode_explct, ttout_lun, risc_buff, 0)
        ENDIF

	IF ( limited_show ) RETURN

* SHOW COMMAND [name]
 1100	only_1 = num_args .GT. 0
	done_1 = .FALSE.
	IF ( .NOT.only_1 ) THEN
           show_str = TM_FMT(revision_level, 5, 12, dlen)
	   WRITE (risc_buff, 4100)
     .				program_name(1:len_program_name),
     .				progname_mod (1:len_progname_mod),
     .				show_str(1:dlen)
	   CALL SPLIT_LIST(pttmode_explct, ttout_lun, risc_buff, 0)
	   risc_buff = ' '
 4100	   FORMAT(' Commands in Program ',A,1X,A,' version',A,':')
	ENDIF
* ... loop through commands
        len_test = TM_LENSTR(cmnd_buff(arg_start(1):arg_end(1)))
	DO 1130 icmnd = 1, total_num_commands
	   IF ( commands(icmnd)(:4) .EQ. '****' ) GOTO 1130
	   IF ( interrupted ) CALL ERRMSG
     .			( ferr_interrupt, status, ' ', *5000 )
* ... if  only 1 command is of interest is this it ?
           len_mchars = TM_LENSTR(commands(icmnd)(:4))
	   IF ( only_1 .AND. .NOT.MATCH4(cmnd_buff(arg_start(1):arg_end(1)),
     .		len_test,commands(icmnd),len_mchars) ) GOTO 1130
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
 	sxml  = qual_given( slash_grid_xml  ) .GT. 0 
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
*    ... first check static grids
	         DO 1340 grid = 1, max_grids
	            IF ( grid_name(grid) .EQ. char_init16 ) GOTO 1340
	            IF (MATCH_TEMPLATE(grid_name(grid) ,arg)) THEN
	               WRITE ( risc_buff, 4220 ) '  '//grid_name(grid)
	               CALL SPLIT_LIST(pttmode_explct, ttout_lun,
     .				risc_buff, 0)
	               risc_buff = ' '
	            ENDIF
 1340	         CONTINUE
*    ... then check dynamic grids
                 grid = 0
 1344            CALL TM_NEXT_DYN_GRID( grid, *1346 )
	            IF (MATCH_TEMPLATE(grid_name(grid) ,arg)) THEN
	               WRITE ( risc_buff, 4220 ) '  '//grid_name(grid)
	               CALL SPLIT_LIST(pttmode_explct, ttout_lun,
     .				risc_buff, 0)
	               risc_buff = ' '
	            ENDIF
	            GOTO 1344
 1346               CONTINUE
	      ELSE
	         grid = GRID_FROM_NAME
     .                    ( cmnd_buff(item_start(ivar):item_end(ivar)),
     .			  cx_last, status )
	         IF ( status .NE. ferr_ok ) GOTO 1350
                 IF ( grid .EQ. unspecified_int4 ) GOTO 5300
	         IF (sxml) THEN 
                    CALL SHOW_GRID_XML(ttout_lun, grid, unspecified_int4 )
                 ELSE
	            WRITE ( risc_buff, 4210 ) grid_name( grid )
	            CALL SPLIT_LIST(pttmode_explct, ttout_lun, risc_buff, 0)
	            risc_buff = ' '
	            CALL SHOW_GRID( grid, mods_cx )
	         ENDIF
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
	   DO 1360 grid = 1, max_grids
	      IF ( interrupted ) CALL ERRMSG
     .			( ferr_interrupt, status, ' ', *5000 )
	      IF ( grid_name( grid ) .EQ. char_init16 ) GOTO 1360
	      WRITE ( risc_buff, 4220 ) '  '//grid_name( grid )
	      CALL SPLIT_LIST(pttmode_explct, ttout_lun, risc_buff, 0)
	      risc_buff = ' '
 4220	      FORMAT (4X,A)
 1360	   CONTINUE
*    ... then check dynamic grids
           grid = 0
 1370      CALL TM_NEXT_DYN_GRID( grid, *1379 )
	      IF ( grid_name(grid)(1:1) .EQ. '(' ) GOTO 1370
	      WRITE ( risc_buff, 4220 ) '  '//grid_name( grid )
	      CALL SPLIT_LIST(pttmode_explct, ttout_lun, risc_buff, 0)
	      risc_buff = ' '
	      GOTO 1370
 1379      CONTINUE

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
                    IF (vp_by_axis(ivp)) THEN
                      i2 = 1
                    ELSE
                      i2 = 2
                    ENDIF
	            WRITE (risc_buff,3146) vp_name(ivp), vp_size(ivp),
     .					   vp_xorg(ivp), xtemp,
     .					   vp_yorg(ivp), ytemp,
     .                                     axesOrCorner(i2)
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
                 IF (vp_by_axis(ivp)) THEN
                   i2 = 1
                 ELSE
                   i2 = 2
                 ENDIF
	         WRITE (risc_buff,3146)	vp_name(ivp), vp_size(ivp),
     .					vp_xorg(ivp), xtemp,
     .					vp_yorg(ivp), ytemp,
     .                                  axesOrCorner(i2)
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
 3142	FORMAT(' name',T19,'text',T27,'xlimits',T39,'ylimits',T50,'mode')
 3146	FORMAT(1X,A,T18,F5.2,T26,F4.2,',',F4.2,T38,F4.2,',',F4.2,T50,A)
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
* SPECIAL SYMBOLS
	i1=0
 1702   i1 = i1 + 1
        CALL SPECIAL_SYMBOL(i1, arg30,show_str,len) ! get next symbol namees
	len0 = TM_LENSTR1(arg30)
        IF(arg30 .NE. ' ')THEN
	   IF ( num_args .GE. 1 ) THEN
* ... check the names specified by template matching 
	      DO 1703 i = 1, num_items
	         n = STR_UPCASE(arg,
     .			cmnd_buff(item_start(i):item_end(i)))
	         IF ( MATCH_TEMPLATE(arg30(:len0),arg) ) GOTO 1705
 1703         CONTINUE
	      GOTO 1702			! no match
	   ENDIF
* ... display the symbol definition
 1705      CALL SPECIAL_SYMBOL(0, arg30,show_str,len) ! get the sym value
           CALL SPLIT_LIST(pttmode_explct, ttout_lun,
     .			arg30(:len0)//' = "'//show_str(:len)//'"', 0)
           GOTO 1702
	ENDIF

* PLOT+ SYMBOLS
 1709   i1=0
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

* SHOW ATTRIBUTE
 1800   CONTINUE

* 
* ... SHOW ATT varname.att ... various cases of how to show:
*       SHOW ATT/ALL varname    - show all attributes for var in current dset
*       SHOW ATT/ALL varname[D=n]
*	SHOW ATT varname.attrib	- show attrib on variable in current dset
*	SHOW ATT name[D=n]	- show attrib on variable from named data set
*	SHOW ATT/D=n varname.attrib - show attrib on variable from named data set
*	SHOW ATT/OUTPUT also list the output flag for the attribute

        sho_out = qual_given( slash_attr_output )  .GT. 0

        dset = cx_data_set(cx_last)
        dset_last = pdset_irrelevant

* ... get dset number if present:  SHOW ATT/D=dset ...
	iqual = qual_given( slash_attr_dset ) 
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
           show_str = cmnd_buff(item_start(1):item_end(1))

           CALL GET_NEW_CX( cx_last, cx_cmnd, .TRUE., status )
           IF ( status .NE. ferr_ok ) GOTO 5000
 
           IF (show_str(1:1) .EQ. '(') THEN
              dset = cx_data_set(cx_cmnd)   ! initial value to try; will return dset
              CALL ISIT_COORD_VAR ( show_str, dset, 
     .              varname, coordvar, status )

              IF ( status .NE. ferr_ok ) THEN 
                 dset = -2  ! a user-defined axis (coordinate variable)
                 CALL ISIT_COORD_VAR ( show_str, dset, 
     .              varname, coordvar, status )
                    IF ( status .NE. ferr_ok ) THEN
                    dset = cx_data_set(cx_cmnd)
                    GOTO 5920
                 ENDIF
              ENDIF

           ELSE

* Get the dataset if given as [d=]

	      IF ( INDEX(show_str,'[') .GT. 0 ) THEN
                 dset = pdset_irrelevant	! default (not used ...)
                 CALL PARSE_NAM_DSET(show_str, cx_last, dset, cat, var,
     .                 mods_cx, status)
              ELSE
                 cx = is_cx( 1 )
                 IF (cx .EQ. 0 .OR. cx .EQ. unspecified_int4)
     .                cx = cx_cmnd
                dset = cx_data_set(cx)
             ENDIF
	   ENDIF


* See if the variable is a user-defined variable.

           varname = show_str
           dot = INDEX(show_str,'.')
           IF (dot .GT. 0) varname = show_str(1:dot-1)
           CALL FIND_VAR_NAME(pdset_irrelevant, varname, cat, var)
           IF (  var .NE. munknown_var_name .AND.
     .            cat .EQ. cat_user_var) dset = -1

* Or use command context to get data set

           IF (dset .EQ. pdset_irrelevant .OR. 
     .        dset .EQ. unspecified_int4) dset = cx_data_set(cx_cmnd) 

	ENDIF

	IF ( num_args .EQ. 0  ) GOTO 5060
        
        IF ( slash_all ) THEN

           DO 1820 i = 1, num_items

	      varname = cmnd_buff( item_start(i):item_end(i) )
              IF(TM_HAS_STRING(varname, '.') ) THEN
                 varid = 0
                 CALL CD_GET_VAR_INFO (dset, varid, varname, vtype,
     .                              nvdims, vdims, nvatts, coordvar, 
     .                              all_outflag, status)
              ENDIF

              IF (dset .EQ. pdset_irrelevant .OR. 
     .            dset .EQ. unspecified_int4) GO TO 5072

              CALL CD_GET_VAR_ID (dset, varname, varid, status)
              IF (status .NE. ferr_ok .OR. dset .EQ. unspecified_int4 .OR.
     .           (varid .EQ. 0 .AND.
     .           .NOT.TM_HAS_STRING(varname, '.')) ) GOTO 5200

* Header line

              IF (.NOT. sxml) THEN
                 IF (dset .NE. dset_last ) THEN
                    IF (dset .EQ. -1) THEN
                       risc_buff = 
     .                   '     attributes for user-defined variables'
                    ELSE IF (dset .EQ. -2) THEN
                       risc_buff = 
     .                   '     attributes for coordinate axis'
                    ELSE
                       llen = TM_LENSTR1(ds_des_name(dset))
                       WRITE ( risc_buff, 4800 ) ds_des_name(dset)(:llen)
                    ENDIF
                    CALL SPLIT_LIST(pttmode_explct, ttout_lun, 
     .                     risc_buff, 0) 
                 ENDIF
                 dset_last = dset
              ENDIF

              CALL CD_GET_VAR_INFO (dset, varid, varname, vtype,
     .                              nvdims, vdims, nvatts, coordvar, 
     .                              all_outflag, status)

              DO 1810 iatt = 1, nvatts

                 CALL CD_GET_VAR_ATT_INFO (dset, varid, iatt, aname, 
     .               attype, attlen, attoutflag, status )

                 IF (status .NE. ferr_ok) GOTO 1810

                 maxlen = size_rbuff
                 IF (attype .NE. NCCHAR) maxlen = 10
                 got_it = NC_GET_ATTRIB ( dset, varid, aname,
     .                                   .TRUE., varname, maxlen,
     .                                   attlen, attoutflag, risc_buff, 
     .                                   attvals )

                 IF (got_it) THEN
                    IF (.NOT. sxml) 
     .                 CALL SHOW_ATTRIBUTE (varname, coordvar, aname, 
     .                              attype, attlen, risc_buff, attvals, 
     .                              attoutflag, sho_out)  

                 ENDIF
1810          CONTINUE

1820       CONTINUE 
        ELSE
	   DO 1830 i = 1, num_items
	      IF ( interrupted ) CALL ERRMSG
     .			( ferr_interrupt, status, ' ', *5000 )

	      varatt = cmnd_buff( item_start(i):item_end(i) )

              IF (INDEX(varatt,'.') .EQ. 0) GOTO 5070

              IF ( INDEX(varatt,'[') .GT. 0 ) THEN

* ... bug (minor): square brackets **may** not contain D=dset information
* 	a work-around: pass a context with data set info (instead of cx_last)

                 dset = pdset_irrelevant	! default (not used ...)
                 CALL PARSE_NAM_DSET(varatt, cx_last, dset,
     .				     category, ivar, mods_cx, status)   

                 IF (dset .EQ. pdset_irrelevant .OR. 
     .               dset .EQ. unspecified_int4) THEN
                       varname = varatt
                       GO TO 5072
                  ENDIF
              ENDIF

* do_err: issue error message if attrib does not exist on varname.attname
              do_err = .TRUE.  
              CALL BREAK_VARATTNAME ( varatt, dset, varname, 
     .                  attname, varid, do_err, status )

              IF(TM_HAS_STRING(varname, '.') ) THEN
                 varid = 0
                 CALL CD_GET_VAR_INFO (dset, varid, varname, vtype,
     .                              nvdims, vdims, nvatts, coordvar, 
     .                              all_outflag, status)
              ENDIF


              IF (status .NE. ferr_ok) GOTO 5000

* get var name in the same case as in the file, and get coordvar,
* flag for whether the var is a coordinate axis

              CALL CD_GET_VAR_INFO (dset, varid, varname, vtype,
     .                              nvdims, vdims, nvatts, coordvar, 
     .                              all_outflag, status)

* Write header line
              IF (.NOT. sxml) THEN
                 IF (dset .NE. dset_last ) THEN
                    llen = TM_LENSTR1(ds_des_name(dset))
                    WRITE (risc_buff, 4800) ds_des_name(dset)(:llen)
                    CALL SPLIT_LIST(pttmode_explct, ttout_lun, 
     .                     risc_buff, 0) 
                 ENDIF
                 dset_last = dset
              ENDIF

              CALL CD_GET_VAR_ATT_ID (dset, varid, attname, attid,
     .               status)
              IF (attid .GT. 0) CALL CD_GET_VAR_ATT_INFO (dset, varid,
     .               attid, aname, attype, attlen, attoutflag, status )

              maxlen = size_rbuff
              IF (attype .NE. NCCHAR) maxlen = 10
              got_it = NC_GET_ATTRIB ( dset, varid, aname,
     .                        .TRUE., varname, maxlen,
     .                        attlen, attoutflag, risc_buff, 
     .                        attvals )

              IF (varid .EQ. 0) THEN
              loc = INDEX(varatt, ".") - 1
                 varname = varatt(1:loc)
              ENDIF

              IF (got_it) CALL SHOW_ATTRIBUTE (varname, coordvar, aname,
     .                       attype, attlen, risc_buff, attvals, 
     .                       attoutflag, sho_out)

1830	   CONTINUE 
        ENDIF
        
4800	FORMAT (T6,'attributes for dataset: ', A)

* successful completion
 	RETURN

* SHOW ****
 1900   CONTINUE
* SHOW ****
 2000   CONTINUE

* SHOW FUNCTIONS
 2100   CONTINUE
* /BRIEF or full description?
 	sbrief  = qual_given( slash_brief ) .GT. 0 
 	sdetail = qual_given( slash_show_func_detail ) .GT. 0 

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
     .           CALL SHOW_1_FUNCTION(ttout_lun, .FALSE.,
     .                                sbrief, sdetail, i2)
 2104         CONTINUE
* ... second try internal, grid-changing functions
	      DO 2106 i2 = 1, gfcn_num_internal
	         IF ( interrupted ) CALL ERRMSG
     .			( ferr_interrupt, status, ' ', *5000 )
              IF ( MATCH_TEMPLATE(gfcn_name(i2),arg(:len0)) ) CALL
     .                SHOW_1_FUNCTION(ttout_lun, .TRUE.,
     .                                sbrief, sdetail, i2)
 2106         CONTINUE
* ... third try external functions
              IF (qual_given(slash_internal) .EQ. 0) THEN
 	         DO 2108 i2 = gfcn_num_internal+1, gfcn_num_internal+
     .                efcn_scan(gfcn_num_internal)
	            IF ( interrupted ) CALL ERRMSG
     .			   ( ferr_interrupt, status, ' ', *5000 )

                 CALL EFCN_GET_NAME (i2, fhol)
                 CALL TM_CTOF_STRNG( fhol, efname, slen )
                 i1 = STR_UPCASE(upname, efname)

                 IF ( MATCH_TEMPLATE(upname,arg(:len0) )) CALL
     .               SHOW_1_FUNCTION(ttout_lun, .TRUE.,
     .                               sbrief, sdetail, i2)

c                 CALL TM_FTOC_STRNG( arg(:len0), fhol, slen )
c                 IF ( EFCN_MATCH_TEMPLATE(i2,fhol) .EQ. 1) CALL
c     .               SHOW_1_FUNCTION(ttout_lun, .TRUE.,
c     .                               sbrief, sdetail, i2)

 2108            CONTINUE  
              ENDIF
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
     .                SHOW_1_FUNCTION(ttout_lun, .FALSE.,
     .                  sbrief, sdetail, i)
 2120         CONTINUE
* ... internal grid-changing functions
	      DO 2130 i = 1, gfcn_num_internal
	         CALL SHOW_1_FUNCTION(ttout_lun, .TRUE.,
     .                  sbrief, sdetail, i)
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
	         CALL SHOW_1_FUNCTION(ttout_lun, .TRUE.,
     .                  sbrief, sdetail, i)
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
 5010	CALL ERRMSG( ferr_erreq,status,filename(:flen),*5000 )
 5050	CALL ERRMSG( ferr_invalid_command, status,
     .			'/CLOBBER not specified and file exists: '
     .			//raw_name, *1000 )
 5071	CALL ERRMSG( ferr_unknown_data_set, status,
     .			name(:TM_LENSTR1(name)), *5000 )
 5072	CALL ERRMSG( ferr_invalid_command, status,
     .         'dataset not found for varname.attname '//
     .          varname(:TM_LENSTR1(varname)), *5000 )
 5100   CALL ERRMSG( ferr_unknown_variable, status, arg, *5000)
 5200	CALL ERRMSG( ferr_unknown_variable, status, varname, *5000 )
 5300   CALL ERRMSG( ferr_invalid_command, status, 'grid for '//
     .              cmnd_buff(item_start(ivar):item_end(ivar))
     .              //' is not available.'//pCR//
     .              '(LOAD this variable and try SHOW GRID again.)',
     .              *5000  )
 5060	CALL ERRMSG( ferr_invalid_command, status,
     .			'SHOW ATTRIBUTE given with no argument '
     .			, *5000 )
 5070	CALL ERRMSG( ferr_invalid_command, status,
     .	    'SHOW ATTRIBUTE given with no attribute: '//
     .	    'Use SHOW ATT/ALL varname, or SHOW ATT varname.attname '
     .			, *5000 )
 5080	CALL ERRMSG( ferr_prog_limit, status,
     .  'Cannot append to file w/ name longer than 256 characters: '//pCR//
     .  'Limitation of Fortran INQUIRE and OPEN calls '	
     .			, *5000 )
 5920	llen = TM_LENSTR(show_str)
	CALL ERRMSG( ferr_invalid_command, status,
     .   'variable, axis, or attribute does not exist'//
     .    show_str(:llen), *5000 )
	END
