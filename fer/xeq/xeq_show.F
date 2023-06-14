    	SUBROUTINE XEQ_SHOW

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
* V631 11/09 *acm* For XML-formatted output check for and replace the strings < > and &
*                  with &lt; etc.
* V65   3/10 *acm* Remove unused SHOW AXIS in /DODS binary format
* V65   3/10 *acm* SHOW/OUTFILE=  writes the output to a file rather than std out.
*                  Implemented so far only for SHOW AXIS, SHOW DATA, SHOW VAR, SHOW GRID,
*                  to be implemented for all SHOW subcommands in the future
* V67   *acm* 3/11 - Fix for ticket 1819, size of arg for MODE DESPERATE
* V67   *acm* 3/11 - Fix for ticket 1825, call cd_get_var_id instead of cd_get_var_seq
* v675  *acm  4/12 Fix ticket 1934, simpler SHOW ATTRIBUTE syntax.

*       *acm* 3/12 removing use of old GT and TS data formats
*       *acm* 3/12 Add E and F dimensions (use nferdims in tmap_dims.parm
* v675  *acm  4/12 Fix ticket 1934, simpler SHOW ATTRIBUTE syntax.
* V6.8  acm  6/12 Implement mode_6d_lab, set by default. If canceled, the 
*                 various SHOW commnands will not list info for E,F dimensions.
*                 SHOW with 6D variable and mode_6d_lab canceled, is an error.
* V683 *acm*  8/12 On SHOW LIST, show the output-type setting from SET LIST/OUTTYPE=  
* V685 *acm*  2/13 LET/REMOTE changes: show internally-defined climatological axes
*                  in xml output.
* V685 *acm*  6/13 Fixes for SHOW VAR/D= on file variables.
* PyFr *kms*  7/13 Add additional information to SHOW WINDOWS
* V685 *sh*  9/13 Added SHOW VARIABLE/TREE  (a.k.a. /DEPENDENCIES)
* V690 *sh* 12/13 Added SHOW TRANSFORMS of auxiliary variable regridding
* V686 *acm* 11/13 Allow symbol names to be up to 120 characters long
* V690 *sh* 1/14 Add SHOW VAR/LAYERZ (formerly /SIGMA)
* V691+*acm* 8/14 Fix ticket 2188: SHOW MEMORY should say memory words are 8 bytes.
* V691+ 8/14 *acm* Fix ticket 2194: SHOW with /OUTFILE and not /XML. 
* V69+ *acm*  9/14 Ticket 1758. Memory request uses number of blocks not words
* V693+ 11/14 *sh* renaming 'sigma' as 'layerz' throughout
*        2/15 *sh* attempted but rejected hack to deal with left-over dset=-1
* V6.95 *acm* 3/15 Changes for PyFerret fonts. New SHOW TEXT
* V6796 11/15 *acm* Ticket 2322: report dataset name or number in error msg, 
*                   on SHOW DATA when datset not open
* V697  12/15 *acm* Changes to add details on SHOW GRID uvar to indicate 
*                   compressed or subset of axes
* V697  12/15 *acm* Fix call to routine SHOW_GRID
* V698 *sh* 1/16 Removed "Ferret-defined" annotation
* V698  2/16 *acm* For ticket 2352: variables defined with LET/D are
*                  added to that dataset in the attribute structure.
*           2/16 replaced char_init with char_init16 (untested) 
*                (this is the *only* place in all the code using char_init)
* V698  3/16 *acm* Removed Philander-Seigel code including diagnostic 
*                  variables, pre-defined axes and grids, cat_calc_var.
* V698  3/16 *acm* Changes for ticket 2078: improved SHOW COMMANDS
* V698  4/16 *acm* For ticket 2399: Apply MODE CALENDAR settings to F axes too
* V7    5/16 *acm* See ticket 2352. For LET/D variables look for info under that dataset
* V701 *acm* 10/16 Ticket 2422: variables uvar_grid, uvar_data_type, uvar_aux_cat, 
*                  uvar_aux_var to be replaced with calls to LIST routines in NCF_Util.c 
* V702 1/17 *sh* for trac enhancement #2369 -- dynamic memory management
* V720 3/17 *acm* Ticket 2132, attribute of user-defined variable, do not try to show dataset info
* v745 *acm* 12/18 Issue 1909, option to write dates in ISO8601 format
*                  new mode datefmt for formatted date output
* V751 *acm* 6/19 Issue 1903: memory set and reported in terms of bytes
* v76  *acm* 11/19  Issue 1959: Change SET MEMORY/SIZE= to SET MEMORY/MAX= 
*                               (keep /SIZE for bkwds compatibility)
* v76  *acm* 11/19  Issue 1961: Extend the fix for 1916 which focused on
*                   getting uvar grid info for RETURN= Make the same changes 
*                   using cx_only setting for SHOW GRID of user-variables
* *acm* 2/2022: Issue 106. In SHOW TEXT, if listing color by percent rgb, the 
*                      Format for the numbers didn't allow listing 100% 
*  1/2023 *acm* Code cleanup: Replace the computed GOTO statement 
*                  "GOTO ( 100,200, ... )  subcmnd_num"  with IF blocks

        IMPLICIT NONE

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
	include	'xdependency_tree.cmn'
	include	'xgrid_chg_fcns.cmn'
	include 'xrisc.cmn'          ! 12/94 SPLIT_LIST buffer
        include 'EF_Util.cmn'
        include 'netcdf.inc'
        INCLUDE 'fgrdel.cmn'
        include 'pyfonts.cmn'

* local parameter declarations:
	LOGICAL	    brief
	INTEGER	    len_show_str,
     .		    slash_mem_free,
     .		    slash_mem_diag,
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
     .              slash_attr_dset,
     .              slash_attr_output
     .              
	PARAMETER ( len_show_str = 2048,		! length of output line buffer
     .		    brief	 = .TRUE.,
     .		    slash_mem_diag = 1 + 1,
     .		    slash_mem_temp = 1 + 2,
     .		    slash_mem_perm = 1 + 3,
     .		    slash_mem_free = 1 + 4,
     .		    slash_brief    = 1 + 1,
     .		    slash_full     = 1 + 2,
     .		    slash_vars     = 1 + 3,
     .		    slash_file     = 1 + 4,
     .		    slash_att      = 1 + 5,
     .		    slash_grid_dyn = 1 + 13,
     .		    slash_var_dset = 1 + 1,
     .		    slash_external = 1 + 2,
     .		    slash_internal = 1 + 3,
     .		    slash_xml      = 1 + 5,
     .		    slash_attr     = 1 + 6,
     .		    slash_grid_xml = 1 + 14,
     .		    slash_axis_xml = 1 + 13,
     .		    slash_attr_dset = 1 + 1,
     .		    slash_attr_output = 1 + 2)

* calling argument declarations:

* local variable declarations:
	LOGICAL	MATCH4, MATCH_TEMPLATE, IS_TEMPLATE, TM_HAS_STRING,
     .		NC_GET_ATTRIB, slash_all, limited_show, 
     .		nothing, sdetail, only_1, done_1, sfull, svars, sfile, 
     .		sbrief, first, sxml, clobber, append, firstxml, got_it, 
     .		coordvar, sattr, do_err, sho_out, sh_att_dset,
     .		lineok, sho_tree, uvgrid, showsub, sub_matched
	INTEGER TM_LENSTR1, TM_LENSTR, UVAR_NUMBER, STR_UPCASE,
     .		REGION_NUMBER, GRID_FROM_NAME, FIND_DSET_NUMBER,
     .		EFCN_MATCH_TEMPLATE, TM_GET_LINENUM, STR_SAME, TM_GET_GRIDNUM,
     .		category, mods_cx, grid, cx,
     .		status, iqual, mode, iset, i, icmnd, isub, n,
     .		len, len0, iuvar, ivar, dset, line, ivp, i1, i2,
     .		rqst_dset, tmp_grid, num_predefined_axes, 
     .          flen, wsize, len_test, len_mchars, outlist_file,
     .          j, num_indices, varid, attlen, maxlen, attype,
     .          vtype, nvdims, vdims(8), nvatts, iatt, dset_last,
     .          llen, loc, cat, var, attoutflag, all_outflag, dot, brkt,
     .          vseq, attid, save_deleted_list_result(max_uvar),
     .          cache_size, cache_nelems, cache_preemption,
     .          sho_file, idim, ndims, nvars, ngatts, recdim, npts,
     .          vartype, outflag, num_clim_1, num_clim_n,
     .          base_cx, base_isp, grid_blocks, lensub, len_msubs
        INTEGER saved_num_indices, saved_index_list(max_uvar),
     .                             arr_buff(max_uvar)
	INTEGER aux_cat(nferdims), aux_var(nferdims)  ! get info from LIST 
	REAL	xtemp, ytemp, attvals(100), cache_mb, cache_n, cache_p, rsize,
     .          v1, v2, v3
	CHARACTER EXPR_NAME*8, TM_FMT*48, TM_FMT_TRIM*48, VAR_CODE*128, 
     .		  ALG_TRANS_CODE*3,
     .		  default*10, show_str*2048, arg*128, name*128, argsym*120
	CHARACTER	varatt*641, varname*512, attname*128,
     .			efname*40, upname*40, aname*128, outstring*2048,
     .                  actstr*1, aastr*3, qualstr*6, dispstr*3,
     .                  dname*128, sub4*4

* 10/01 *kob* had to change decl of axesOrCorner for picky f95
        CHARACTER*6 axesOrCorner(2)

	INTEGER efcn_scan

* local data declarations
        DATA axesOrCorner(1)/'axes '/,
     .       axesOrCorner(2)/'edges'/

* for string passing to C function in SHOW MODE FUNCTION
      INTEGER      slen, dlen
      PARAMETER  ( slen = ef_max_description )
      INTEGER*1 fhol(slen)      ! c-type Hollerith string buffer

* initialize
	show_lun = ttout_lun
	sxml = .FALSE.

* command given with /ALL ?
	risc_buff = ' '
	slash_all = .FALSE.
	DO iqual = 1, num_qualifiers
	   IF ( qualifier_list( iqual ) .EQ. 1 ) slash_all = .TRUE.
	ENDDO
	limited_show = subcmnd_num .NE. 1 .OR. .NOT. slash_all

* if the user entered a blank-separated list instead of comma-separated
* then make it behave like a comma-separated list (4/96)
	CALL ARG_TO_ITEM

* If slash_all, then will show all of these

 10	CONTINUE

* execute subcommand

* SHOW '    ' - (no action)
* arrival at this point usually means an illegal subcommand was given
* and was assumed to be an argument by the command parser

      IF (subcmnd_num .EQ. 1) THEN

	IF ( num_args .GE. 1 )
     .		CALL ERRMSG( ferr_invalid_subcmnd,status,
     .			cmnd_buff( arg_start(1):arg_end(1) ), *5000 )
	IF ( limited_show ) THEN
	   RETURN
	ELSE
	   subcmnd_num = subcmnd_num + 1
	   GOTO 10
	ENDIF

      ENDIF

* SHOW WINDOWS

      IF (subcmnd_num .EQ. 2) THEN

	nothing = .TRUE.
	CALL SPLIT_LIST(pttmode_explct, show_lun, 
     .  ' OPEN  ACTIVE  ANTIALIAS  QUALITY  DISPLAY  THICKEN  TEXTPROM', 
     .  61)
 201    FORMAT(I4,6X,A1,7X,A3,5X,A6,5X,A3,6X,F4.2,5X,F4.2)
	DO i = 1, max_windows
	   IF ( wn_open(i) ) THEN
	      nothing = .FALSE.
	      IF ( wn_active(i) ) THEN
                 actstr = '*'
              ELSE
                 actstr = ' '
              ENDIF
              IF ( antialias(i) ) THEN
                 aastr = 'YES'
              ELSE
                 aastr = ' NO'
              ENDIF
              IF ( enginename(i) .EQ. 'Cairo' ) THEN
                 qualstr = '  HIGH'
                 dispstr = ' NO'
              ELSEIF ( enginename(i) .EQ. 'PipedViewerPQ' ) THEN
                 qualstr = '  HIGH'
                 dispstr = 'YES'
              ELSEIF ( enginename(i) .EQ. 'PipedImager' ) THEN
                 qualstr = ' DRAFT'
                 dispstr = 'YES'
              ELSEIF ( enginename(i) .EQ. 'NoDisplayPQ' ) THEN
                 qualstr = '  HIGH'
                 dispstr = ' NO'
              ELSE
                 qualstr = 'CUSTOM'
                 dispstr = ' ? '
              ENDIF
              risc_buff = ' '
              WRITE ( risc_buff, 201 ) i, actstr, aastr, qualstr, 
     .          dispstr, thickfactor(i), wn_textscale(i)/wn_pplscale(i)
              CALL SPLIT_LIST(pttmode_explct, show_lun, risc_buff, 0)
              risc_buff = ' '
	   ENDIF
	ENDDO
	IF ( nothing ) CALL SPLIT_LIST(pttmode_explct, show_lun,
     .		'   no windows open', 0) 
	IF ( limited_show ) THEN
	   RETURN
	ELSE
	   subcmnd_num = subcmnd_num + 1
	   GOTO 10
	ENDIF

      ENDIF

* SHOW REGION reg1,reg2,...,regn

      IF (subcmnd_num .EQ. 3) THEN

* show default region
	IF ( num_args .EQ. 0 .OR. slash_all ) CALL SHOW_REGION( cx_last )
* show requested regions
	IF ( num_items .GE. 1 ) THEN
	   DO i = 1, num_items
	      IF ( interrupted ) CALL ERRMSG
     .			( ferr_interrupt, status, ' ', *5000 )
	      arg = cmnd_buff( item_start(i):item_end(i) )
	      cx = REGION_NUMBER( arg )
	      IF ( cx .EQ. unspecified_int4 ) CALL ERRMSG
     .				( ferr_unknown_arg, status, arg, *5000 )
	      CALL SHOW_REGION( cx )
	   ENDDO
	ELSEIF ( slash_all ) THEN
* show all user-defined and pre-defined regions
	   DO cx = 0, min_context, -1
	      IF ( cx_name( cx ) .NE. unspecified_name4 )
     .			CALL SHOW_REGION( cx )
	   ENDDO
	ENDIF
	IF ( limited_show ) THEN
	   RETURN
	ELSE
	   subcmnd_num = subcmnd_num + 1
	   GOTO 10
	ENDIF

      ENDIF

* SHOW AXIS
* show a particular named axis

      IF (subcmnd_num .EQ. 4) THEN

	IF ( num_args .EQ. 0 .AND. .NOT.slash_all ) THEN
	   CALL SPLIT_LIST(pttmode_explct, err_lun,
     .			'SHOW what axis ?', 0)
	   RETURN
	ENDIF
	
	sxml = qual_given( slash_axis_xml ) .GT. 0
	IF (sxml) firstxml = .TRUE.

* 	SHOW AXIS/FILE=[/APPEND][/CLOBBER]

        sho_file = qual_given( slash_show_axis_file )
	clobber = qual_given( slash_show_axis_clobber ) .GT. 0
	append = qual_given( slash_show_axis_append ) .GT. 0

	IF (sho_file .GT. 0) THEN
	   CALL OPEN_SHOW_FILE (show_lun, sho_file,  
     .                          clobber, append, status)
	   IF (status .NE. ferr_ok ) GOTO 5000
	ENDIF

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

* write column headings
        IF (.NOT.sxml) THEN
	   WRITE ( risc_buff, 3000 )
 3000	   FORMAT ( T2,'name',T13,'axis',T31,'# pts',T39,
     .           'start',T60,'end' )
	   CALL SPLIT_LIST(pttmode_explct, show_lun, risc_buff, 0)
        ENDIF
	risc_buff = ' '
	IF ( num_args .GT. 0 ) THEN

	   DO i = 1, num_items
* show the named line(s)
	      i1 = STR_UPCASE(name,cmnd_buff(item_start(i):item_end(i)))
*  ... first show the static lines
	      DO line = 1, max_lines
	         IF ( interrupted ) CALL ERRMSG
     .			( ferr_interrupt, status, ' ', *5000 )
	         IF ( line_name(line) .EQ. char_init16 ) CYCLE
	         IF ( line_name(line)(1:1) .EQ. '(' ) CYCLE
	         IF (MATCH_TEMPLATE(line_name(line),name)) THEN
                   IF (sxml) THEN
                      IF (firstxml) THEN
                         risc_buff = '<axes>'
                         CALL SPLIT_LIST(pttmode_explct, show_lun,
     .                                risc_buff, 0)
                         firstxml = .FALSE.
                      ENDIF
                      CALL SHOW_LINE_XML( show_lun, line)

                   ELSE
                      CALL SHOW_LINE( line, .NOT.brief )
	              DO i2 = 1, nferdims
	                IF (qual_given( slash_show_grid_x0+i2).GT.0
     .                 .OR. qual_given( slash_show_grid_i0+i2).GT.0)THEN
                         grid_line(i2,tmp_grid) = line
                         CALL SHOW_LINE_COORDS( tmp_grid, mods_cx, i2 )
                        ENDIF
	              ENDDO
                   ENDIF

                 ENDIF
	      ENDDO
*  ... then show the dynamic lines
              line = 0
 414          CALL TM_NEXT_DYN_LINE( line, *416)
	         IF ( interrupted ) CALL ERRMSG
     .			( ferr_interrupt, status, ' ', *5000 )
	         IF (MATCH_TEMPLATE(line_name(line),name)) THEN

                   IF (sxml) THEN
                      IF (firstxml) THEN
                         risc_buff = '<axes>'
                         CALL SPLIT_LIST(pttmode_explct, show_lun,
     .                                risc_buff, 0)
                         firstxml = .FALSE.
                      ENDIF
                      CALL SHOW_LINE_XML( show_lun, line)

                   ELSE
                      CALL SHOW_LINE( line, .NOT.brief )
	              DO i2 = 1, nferdims
	                IF (qual_given( slash_show_grid_x0+i2).GT.0
     .              .OR. qual_given( slash_show_grid_i0+i2).GT.0) THEN
                         grid_line(i2,tmp_grid) = line
                         CALL SHOW_LINE_COORDS( tmp_grid, mods_cx, i2 )
                       ENDIF
	              ENDDO
                   ENDIF

                 ENDIF
	         GOTO 414
 416          CONTINUE
	   ENDDO
	ELSE

* show all the defined axes
*  ... first show the static lines (for XML output skip pre-defined axes)
           num_predefined_axes = TM_GET_LINENUM('EZ')
           num_clim_1 = TM_GET_LINENUM('MONTH_IRREG')
           num_clim_n = TM_GET_LINENUM('ABSTRACT') - 1
	   DO line = 1, max_lines
	      IF ( interrupted ) CALL ERRMSG
     .			( ferr_interrupt, status, ' ', *5000 )
	      IF ( line_name(line) .EQ. char_init16) CYCLE
              IF (.NOT.sxml) THEN
                 CALL SHOW_LINE( line, brief )
              ENDIF
	      lineok = ( line.GT.num_predefined_axes )
c	      lineok = ( line.GT.num_predefined_axes .OR.
c     .           (line.GE.num_clim_1 .AND. line.LE.num_clim_n) )
              IF (sxml .AND. lineok) THEN
                 IF (firstxml) THEN
                    risc_buff = '<axes>'
                    CALL SPLIT_LIST(pttmode_explct, show_lun,
     .                              risc_buff, 0)
                    firstxml = .FALSE.
                 ENDIF
                 CALL SHOW_LINE_XML( show_lun, line)
              ENDIF

	   ENDDO
*  ... then show the dynamic lines
           line = 0
 460       CALL TM_NEXT_DYN_LINE( line, *470)
	      IF ( interrupted ) CALL ERRMSG
     .			( ferr_interrupt, status, ' ', *5000 )
              IF (sxml) THEN
                 IF (firstxml) THEN
                    risc_buff = '<axes>'
                    CALL SPLIT_LIST(pttmode_explct, show_lun,
     .                              risc_buff, 0)
                    firstxml = .FALSE.
                 ENDIF
                 CALL SHOW_LINE_XML( show_lun, line)

              ELSE
	         CALL SHOW_LINE( line, brief )
              ENDIF
              GOTO 460
 470       CONTINUE

	ENDIF
        IF (.NOT. firstxml .AND. sxml) THEN
           risc_buff = '</axes>'
           CALL SPLIT_LIST(pttmode_explct, show_lun, risc_buff, 0)
        ENDIF
	
	IF (sho_file .GT. 0) CLOSE( UNIT = show_lun, ERR = 5000 )

	IF ( limited_show ) THEN
	   RETURN
	ELSE
	   subcmnd_num = subcmnd_num + 1
	   GOTO 10
	ENDIF

      ENDIF

* SHOW EXPRESSION

      IF (subcmnd_num .EQ. 5) THEN

	IF ( num_uvars_in_cmnd .EQ. cmnd_uvars_not_given ) THEN
	   WRITE ( risc_buff, '(8X,A)' ) 
     .				'SET EXPRESSION has not been given or implied'
	   CALL SPLIT_LIST(pttmode_explct, err_lun, risc_buff, 0)
	   risc_buff = ' '
	ELSE
	   WRITE ( risc_buff, '(8X,A)' ) 'current output expression(s):'
	   CALL SPLIT_LIST(pttmode_explct, show_lun, risc_buff, 0)
	   risc_buff = ' '
	   DO i = 1, num_uvars_in_cmnd
	      iuvar = UVAR_NUMBER( EXPR_NAME(i) )
	      IF ( iuvar .NE. unspecified_int4 ) THEN
	         WRITE (risc_buff,'(8X,A)')
     .			 uvar_text(iuvar)(:TM_LENSTR1(uvar_text(iuvar)))
	         CALL SPLIT_LIST(pttmode_explct, show_lun, risc_buff, 0)
	         risc_buff = ' '
	      ENDIF
	   ENDDO
	ENDIF
	IF ( limited_show ) THEN
	   RETURN
	ELSE
	   subcmnd_num = subcmnd_num + 1
	   GOTO 10
	ENDIF

      ENDIF

* SHOW LIST

      IF (subcmnd_num .EQ. 6) THEN

	WRITE ( risc_buff, 3600 ) '/PRECISION =', list_digits
	CALL SPLIT_LIST(pttmode_explct, show_lun, risc_buff, 0)
	risc_buff = ' '
 3600	FORMAT ( 8X,'SET LIST',A,I4 )
        IF ( list_fmt_type .EQ. plist_unformatted ) THEN
	   WRITE ( risc_buff, 3605 ) 'UNFORMATTED'
	ELSEIF ( list_fmt_type .EQ. plist_epic ) THEN
	   WRITE ( risc_buff, 3605 ) 'EPIC'
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
	CALL SPLIT_LIST(pttmode_explct, show_lun, risc_buff, 0)
	risc_buff = ' '
 3605	FORMAT ( 8X,'SET LIST/FORMAT = ',A )
	IF ( list_heading ) THEN
	   WRITE ( risc_buff, 3608 ) 'EN'
	ELSE
	   WRITE ( risc_buff, 3608 ) 'DIS'
	ENDIF
	CALL SPLIT_LIST(pttmode_explct, show_lun, risc_buff, 0)
	risc_buff = ' '
 3608	FORMAT ( 8X,'SET LIST/HEADING is ',A,'ABLED' )
	IF ( list_append ) THEN
	   WRITE ( risc_buff, 3609 ) 'EN'
	ELSE
	   WRITE ( risc_buff, 3609 ) 'DIS'
	ENDIF
	CALL SPLIT_LIST(pttmode_explct, show_lun, risc_buff, 0)
	risc_buff = ' '
 3609	FORMAT ( 8X,'SET LIST/APPEND is ',A,'ABLED' )
	IF ( list_file(1:4) .EQ. 'AUTO' ) THEN
	   WRITE ( risc_buff, 3620 ) list_file(1:TM_LENSTR1(list_file))//
     .							' [created by program]'
	ELSE
	   WRITE ( risc_buff, 3620 ) list_file(1:TM_LENSTR1(list_file))
	ENDIF
	CALL SPLIT_LIST(pttmode_explct, show_lun, risc_buff, 0)
	risc_buff = ' '
 3620	FORMAT (8X,'SET LIST/FILE=',A,' ( when LIST/FILE is used )' )
	IF ( list_outtype(1:3) .EQ. 'DOU') WRITE ( risc_buff, 3630 ) 'Double'
	IF ( list_outtype(1:3) .EQ. 'FLO') WRITE ( risc_buff, 3630 ) 'Float'
	IF ( list_outtype(1:3) .EQ. 'INT') WRITE ( risc_buff, 3630 ) 'Int'
	IF ( list_outtype(1:3) .EQ. 'SHO') WRITE ( risc_buff, 3630 ) 'Short'
	IF ( list_outtype(1:3) .EQ. 'BYT') WRITE ( risc_buff, 3630 ) 'Byte'
	IF ( list_outtype(1:3) .EQ. 'DFL') WRITE ( risc_buff, 3630 ) '[Default]'

	CALL SPLIT_LIST(pttmode_explct, show_lun, risc_buff, 0)
	risc_buff = ' '
 3630	FORMAT ( 8X,'SET LIST/OUTTYPE = ',A )

        CALL NETCDF4_SHOW_LIST_SETTINGS 
	IF ( limited_show ) THEN
	   RETURN
	ELSE
	   subcmnd_num = subcmnd_num + 1
	   GOTO 10
	ENDIF

      ENDIF

* SHOW DATA_SET

      IF (subcmnd_num .EQ. 7) THEN

 	CALL SHOW_DATA(dset, status) 
  
        IF (status .NE. ferr_ok) GOTO 5000
	IF ( dset .EQ. unspecified_int4 ) GOTO 5071

	IF ( limited_show ) THEN
	   RETURN
	ELSE
	   subcmnd_num = subcmnd_num + 1
	   GOTO 10
	ENDIF

      ENDIF

* SHOW MODE mode

      IF (subcmnd_num .EQ. 8) THEN

	WRITE ( risc_buff, 3800 )
	CALL SPLIT_LIST(pttmode_explct, show_lun, risc_buff, 0)
	risc_buff = ' '
 3800	FORMAT( T7,'MODE',T23,'STATE',T36,'ARGUMENT' )
	slash_all = slash_all .OR. num_items .EQ. 0
	DO 850 mode = 1, max_modes
	   IF ( mode_name ( mode ) .EQ. pmode_unused ) GOTO 850
* show this mode ?
	   IF ( .NOT. slash_all ) THEN
              len_mchars = TM_LENSTR(mode_name(mode)(:4))
	      DO i = 1, num_items
	         name = cmnd_buff( item_start(i):item_end(i) )
                 len_test = TM_LENSTR(name)
	         IF ( MATCH4( name, len_test,
     .             mode_name( mode),len_mchars )) GOTO 820
	      ENDDO
	      GOTO 850	! no match
	   ENDIF
 820	   show_str = '      '//mode_name( mode )
	   IF ( mode_state( mode, 1 ) ) THEN
	      show_str(24:) = 'SET'
	   ELSE
	      show_str(21:) = 'CANCELED'
	   ENDIF
	   IF ( mode .EQ. pmode_time_lab ) THEN
	      show_str(36:) = date_labels( ABS(ax_dec_pt(t_dim)) )
	   ELSEIF ( mode .EQ. pmode_f_lab ) THEN
	      show_str(36:) = date_labels( ABS(ax_dec_pt(f_dim)) )
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
	   ELSEIF ( mode .EQ. pmode_datefmt ) THEN
	      show_str(36:) = datefmt_modes(mode_arg(mode,1))
	   ELSEIF ( mode_arg( mode,1 ) .NE. unspecified_int4 ) THEN
	      WRITE ( show_str(36:42), '(I7)' ) mode_arg( mode,1 )
	      IF (STR_SAME(show_str(36:42), '*******') .EQ. 0) 
     .		 WRITE ( show_str(36:47), '(I11)' ) mode_arg( mode,1 )
	   ENDIF
	   CALL SPLIT_LIST(pttmode_explct, show_lun,
     .			show_str(:TM_LENSTR1(show_str)), 0)
 850	CONTINUE
	IF ( limited_show ) THEN
	   RETURN
	ELSE
	   subcmnd_num = subcmnd_num + 1
	   GOTO 10
	ENDIF

      ENDIF

* SHOW MOVIE
* Movies are discontinued with Ferret v6.6: NetCDF-4 / HDF5

      IF (subcmnd_num .EQ. 9) THEN

        CALL WARN ( 'Movies are discontinued as of Ferret v6.6' )
	IF ( limited_show ) THEN
	   RETURN
	ELSE
	   subcmnd_num = subcmnd_num + 1
	   GOTO 10
	ENDIF

      ENDIF

* SHOW VARIABLES

      IF (subcmnd_num .EQ. 10) THEN

	sxml  = qual_given(slash_show_var_xml) .GT. 0 
        IF (sxml) firstxml = .TRUE.

* get dset number if appropriate:  SHOW VAR/D=dset ...
	rqst_dset = unspecified_int4	
	iqual = qual_given( slash_var_dset )
	IF ( iqual .GT. 0 ) THEN
	   CALL EQUAL_STRING( cmnd_buff(qual_start(iqual):qual_end(iqual)),
     .			   show_str, status )  ! show_str is just a buffer
	   IF ( status .NE. ferr_ok ) RETURN
	   IF ( show_str .NE. ' ' ) THEN
	      rqst_dset = FIND_DSET_NUMBER( show_str )
	      IF ( rqst_dset .EQ. unspecified_int4) THEN
	         CALL WARN('Unknown data set: '
     .			//show_str(:TM_LENSTR1(show_str)))
	         RETURN
	      ENDIF
	   ENDIF
	ENDIF

* ... SHOW VARIABLE/LAYERZ varname [,varname2, ...]
	IF (qual_given(slash_show_var_layerz) .GT. 0) THEN
	   IF (slash_all .OR. num_args.EQ.0) THEN
*    ... get list of varIDs of all variables that have __LayerzRef_ attribute
              CALL GET_FVARS_LIST_BY_ATTNAME_AND_VAL
     .            (patnam_layerz, unspecified_name4, max_uvar,
     .             arr_buff, saved_index_list, num_indices)
              CALL GET_UVARS_LIST_BY_ATTNAME_AND_VAL
     .	          (patnam_layerz, unspecified_name4, max_uvar-num_indices,
     .             arr_buff(num_indices+1),
     .             saved_index_list(num_indices+1), n )
              num_indices = num_indices + n
              IF (num_indices.EQ.0) RETURN
           ELSE
*    ... get list of varIDs of all variables requested
              num_indices = num_items
	      DO i1 = 1, num_indices
	         i2 = STR_UPCASE(varname,
     .			         cmnd_buff(item_start(i1):item_end(i1)))
                 len = item_end(i1) - item_start(i1) + 1
	         dset = unspecified_int4    ! to be determined during parse
	         CALL PARSE_NAM_DSET( varname, cx_last, dset,
     .			              cat, var, mods_cx, status )
	         IF ( status .NE. ferr_ok ) RETURN
	         IF ( var .EQ. munknown_var_name ) THEN
                    arg = varname
                    GOTO 5100    ! unknown variable
                 ENDIF
* get the variable ID in the linked list  (user vars stored under dset=pdset_uvars?)
                 IF (cat .EQ. cat_user_var) THEN 
                    rqst_dset = pdset_uvars  

* See if it's a LET/D variable, info stored with its dataset not pdset_uvars
                    CALL CD_GET_VAR_ID (rqst_dset, varname, varid, status)
                    IF (rqst_dset.EQ.pdset_uvars .AND. status.NE.ferr_ok) THEN
                       rqst_dset = dset
                       CALL CD_GET_VAR_ID (rqst_dset, varname, varid, status)
                    ENDIF
                    IF (status.NE.ferr_ok) GOTO 5100
                 ELSE
                    rqst_dset = dset
		    CALL CD_GET_VAR_ID (rqst_dset, varname, varid, status)
                 ENDIF

                 arr_buff(i1) = varid
                 saved_index_list(i1) = rqst_dset
              ENDDO
           ENDIF
* loop through the requested variables displaying layerz info
           DO i1 = 1, num_indices
* get the attribute value of '__LayerzRef_' (patnam_layerz)
              varid = arr_buff(i1)
              dset  = saved_index_list(i1)
              maxlen = size_rbuff
              CALL CD_GET_VAR_INFO (dset, varid, varname, vtype,
     .                              nvdims, vdims, nvatts, coordvar, 
     .                              all_outflag, status)
              len = TM_LENSTR1(varname)
              got_it = NC_GET_ATTRIB ( dset, varid, patnam_layerz,
     .                                 .TRUE., varname, maxlen,
     .                                 attlen, attoutflag, risc_buff,
     .                                 attvals )
              IF (.NOT.got_it) THEN
* ... we only arrive here if request was for named vars: SH VAR/SIG a, b, c
* ... no explicit layerz reference defined for this variable
                 show_str =
     .         '  No variable holds vertical (layerz) coordinates of '
     .                // varname(:len)

* look for inherited layerz dependencies 
	         dset = unspecified_int4    ! to be determined during parse
	         CALL PARSE_NAM_DSET( varname, cx_last, dset,
     .			              cat, var, mods_cx, status )
                 IF (cat .EQ. cat_user_var) THEN
* ... trigger grid-getting, which also gets layerz associations

                    grid = GRID_FROM_NAME
     .                    (cmnd_buff(item_start(i1):item_end(i1)),
     .			  cx_last, status )

                 ENDIF   ! is a uvar
              ELSE
* ... show the explicit __layerz_ref_ attribute value
                show_str = '  ' // risc_buff(:attlen) //
     .         ' holds the designated vertical (layerz) coordinates of ' //
     .          varname(:len)
              ENDIF
	      CALL SPLIT_LIST(pttmode_explct, show_lun, show_str, 0)
           ENDDO
           RETURN
	ENDIF

* 	SHOW VARIABLES/FILE=[/APPEND][/CLOBBER]

        sho_file  = qual_given( slash_show_var_file    )
	clobber   = qual_given( slash_show_var_clobber ) .GT. 0
	append    = qual_given( slash_show_var_append  ) .GT. 0
        sho_tree  = qual_given( slash_show_var_tree    ) .GT. 0

	IF (sho_file .GT. 0) THEN
	   CALL OPEN_SHOW_FILE (show_lun, sho_file, 
     .                          clobber, append, status)
	   IF (status .NE. ferr_ok ) GOTO 5000
	ENDIF

* ... SHOW VARIABLE/TREE [expr]
        IF ( sho_tree ) THEN 
*     /TREE=USER and /TREE=FILE
	   iqual = qual_given( slash_show_var_tree )
	   CALL EQUAL_STRING(cmnd_buff(qual_start(iqual):qual_end(iqual)),
     .			   show_str, status )  ! show_str is just a buffer
	   IF ( status .NE. ferr_ok ) RETURN
	   IF     ( show_str(1:1) .EQ. 'A'
     .     .OR.     slash_all             ) THEN
             dependency_tree_show_uvars = .TRUE.  ! show all vars
             dependency_tree_show_fvars = .TRUE.
	   ELSEIF ( show_str(1:1) .EQ. ' '
     .         .OR. show_str(1:1) .EQ. 'U') THEN
             dependency_tree_show_uvars = .TRUE.  ! show user vars only
             dependency_tree_show_fvars = .FALSE.
	   ELSEIF ( show_str(1:1) .EQ. 'F' ) THEN
             dependency_tree_show_uvars = .FALSE. ! show file vars only
             dependency_tree_show_fvars = .TRUE.
           ELSE
              GOTO 5150
           ENDIF
           base_isp = 8
           base_cx = 8
           CALL INIT_DEPENDENCIES( cx_last, .TRUE., base_isp, status )
           IF (status.NE. ferr_ok) GOTO 5050
           IF (slash_all .OR. num_args.EQ.0) THEN
*       SHOW VAR/TREE  [no expression] with or without /ALL -- shows all uvars
              CALL DELETE_OLD_EXPR
              CALL deleted_list_get_undel(uvar_num_items_head, 
     .                          deleted_list_result(1),
     .                          max_uvar, num_indices )
              IF (num_indices .EQ. 0) GOTO 1010
* ... we need to save a private copy of the index list, else it gets corrupted
*       by a call to delete_list_get_undel in INIT_DEPENDENCIES
              saved_num_indices = num_indices
              DO i = 1, saved_num_indices
		 saved_index_list(i) = deleted_list_result(i)
 	      ENDDO
              DO i = 1, saved_num_indices
                 CALL CRAWL_DEPENDENCIES
     .                          ( 
     .                          uvar_name_code(saved_index_list(i)), 
     .                          base_cx, status )
                 IF (status .NE. ferr_ok) GOTO 5050
	      ENDDO
	   ELSE
*       SHOW VAR/TREE  expr1, expr2, ...
*  ... multiplicity of items is handled by BREAK_UP_EXPR in CRAWL_DEPENDENCIES
              CALL CRAWL_DEPENDENCIES
     .                   ( 
     .                     cmnd_buff(item_start(1):item_end(num_items)), 
     .                     base_cx, status )
              IF (status .NE. ferr_ok) GOTO 5050
           ENDIF
           CALL DISPLAY_DEPENDENCY( workmem(plegacy_work_buffer)%ptr )
 1010      CALL EXIT_DEPENDENCY_MODE
            IF (sho_file .GT. 0) CLOSE( UNIT = show_lun, ERR = 5000 )
           RETURN
        ENDIF

* ... SHOW VARIABLES/USER
	IF ( slash_all .OR. qual_given(slash_show_var_diag) .EQ. 0 ) THEN
* ... get dset number if appropriate:  SHOW VAR/USER/D=dset ...
	   iqual = qual_given( slash_var_dset )
	   IF ( iqual .GT. 0 ) THEN
	      CALL EQUAL_STRING( cmnd_buff(qual_start(iqual):qual_end(iqual)),
     .			      show_str, status )  ! show_str is just a buffer
	      IF ( status .NE. ferr_ok ) GOTO 4030
	      IF ( show_str .EQ. ' ' ) THEN
	         rqst_dset = pdset_irrelevant	! global variables
	      ELSE
	         rqst_dset = FIND_DSET_NUMBER( show_str )
	         IF ( rqst_dset .EQ. unspecified_int4) THEN
	            CALL WARN('Unknown data set: '
     .			//show_str(:TM_LENSTR1(show_str)))
	            GOTO 4030
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
                    IF (dset .EQ. pdset_irrelevant .AND. 
     .                  rqst_dset .NE. unspecified_int4)
     .                         dset = rqst_dset
	            IF (status .EQ. ferr_ok) THEN 
                       IF (category .EQ. cat_user_var) THEN
                          IF (sxml) THEN
	                     CALL SHOW_1_UVAR_XML 
     .                         (show_lun,ivar,show_str,firstxml)
                          ELSE
	                    CALL SHOW_1_UVAR
     .                      (show_lun,ivar,show_str,' ',.TRUE.)
                          ENDIF

*  if not a user var, see if its a dataset variable from a nc file.
                       ELSE
                         len = INDEX(arg,'[') - 1
                         IF (INDEX(arg,'.') .NE. 0) GOTO 5100
			 CALL CD_GET_VAR_ID (dset, arg(:len), varid, status)
                         IF (status .EQ. ferr_ok) THEN
			    CALL SHOW_1_DSVAR (show_lun, dset, arg, varid)
                         ENDIF
                       ENDIF

                    ENDIF
	         ELSEIF (  rqst_dset .NE. unspecified_int4) THEN
		    CALL FIND_VAR_NAME(rqst_dset,arg,category,ivar)
	            IF ( ivar .EQ. munknown_var_name
     .		        .OR. category .NE. cat_user_var ) THEN
     
*  if not a user var, see if its a dataset variable from a nc file.
                       len = TM_LENSTR1(arg)
                       IF (INDEX(arg,'.') .NE. 0) GOTO 5100
                       CALL CD_GET_VAR_ID (rqst_dset, arg(:len), varid, status)
                       IF (status .EQ. ferr_ok)  THEN
		          CALL SHOW_1_DSVAR (show_lun, rqst_dset, arg, varid)
                       ELSE 
                          GOTO 5100
                       ENDIF
                    ELSE
                       IF (sxml) THEN
	                  CALL SHOW_1_UVAR_XML 
     .                      (show_lun,ivar,show_str,firstxml)
                       ELSE
	                 CALL SHOW_1_UVAR
     .                      (show_lun,ivar,show_str,' ',.TRUE.)
                       ENDIF
                    ENDIF
	         ELSE
* ... show ALL user variables matching this name (or name template 4/96)

                    CALL deleted_list_get_undel(uvar_num_items_head,
     .                                         deleted_list_result(1),
     .                                         max_uvar, num_indices )
                    DO i2 = 1, max_uvar
                      save_deleted_list_result(i2) = 
     .                       deleted_list_result(i2)
		    ENDDO
                    DO j = 1, num_indices
                       ivar = save_deleted_list_result(j)
	               IF ( uvar_parent(ivar).NE.0
     .		      .AND. .NOT.mode_diagnostic   ) CYCLE
!	               IF (MATCH_NAME(arg,uvar_name_code(ivar)))

                       IF (MATCH_TEMPLATE(uvar_name_code(ivar),arg))THEN
                         IF (sxml) THEN
     			    CALL SHOW_1_UVAR_XML
     .                         (show_lun,ivar,show_str,firstxml)
                         ELSE
     			    CALL SHOW_1_UVAR
     .				(show_lun, ivar, show_str,' ',.TRUE.)
                         ENDIF
                      ENDIF
                    ENDDO
	         ENDIF
 1020	      CONTINUE
	   ELSEIF (rqst_dset .NE. unspecified_int4) THEN
* ... show user variables belonging to the specified data set
              CALL deleted_list_get_undel(uvar_num_items_head,
     .                                    deleted_list_result(1),
     .                                    max_uvar, num_indices )
              DO i2 = 1, max_uvar
                save_deleted_list_result(i2) = deleted_list_result(i2)
              ENDDO
              DO j = 1, num_indices
                 i = save_deleted_list_result(j)
	         IF (uvar_name_code(i)(1:3) .EQ. 'EX#'    ) CYCLE
	         IF ( rqst_dset .NE. uvar_dset(i)	  ) CYCLE
	         IF ( uvar_parent(i).NE.0
     .		 .AND. .NOT.mode_diagnostic		  ) CYCLE
                 IF (sxml) THEN
                    CALL SHOW_1_UVAR_XML 
     .                      (show_lun,i,show_str,firstxml)
                 ELSE
                    CALL SHOW_1_UVAR
     .                 (show_lun,i,show_str,'     ',.TRUE.)
                 ENDIF
	      ENDDO
	   ELSE
* ... show ALL user variables of every stripe
* ... 8/95 sorted by data set
	      IF (.NOT. sxml) CALL SPLIT_LIST(pttmode_explct, show_lun,
     .			' Created by DEFINE VARIABLE:', 0)
* ... locate the highest data set number in use
	      DO i1 = maxdsets, 1, -1
	         IF ( ds_name(i1) .NE. char_init2048) EXIT
	      ENDDO

* ... loop through the currently initialized data sets
	      DO dset = 1, i1

                 CALL deleted_list_get_undel(uvar_num_items_head,
     .                                      deleted_list_result(1),
     .                                      max_uvar, num_indices )
     
                 DO i2 = 1, max_uvar
                    save_deleted_list_result(i2) = 
     .		      deleted_list_result(i2)
		 ENDDO
                 DO j = 1, num_indices
                    i = save_deleted_list_result(j)
	            IF (uvar_name_code(i)(1:3) .EQ. 'EX#'    ) CYCLE
	            IF ( dset .NE. uvar_dset(i)		     ) CYCLE
	            IF ( uvar_parent(i).NE.0
     .		    .AND. .NOT.mode_diagnostic		     ) CYCLE

                    IF (sxml) THEN
                       CALL SHOW_1_UVAR_XML
     .                      (show_lun,i,show_str,firstxml)
                    ELSE
                       CALL SHOW_1_UVAR
     .                    (show_lun,i,show_str,'     ',.TRUE.)
                    ENDIF

                 ENDDO
	      ENDDO

* ... global variables defined without /D

	      first = .TRUE.

              CALL deleted_list_get_undel(uvar_num_items_head,
     .                                    deleted_list_result(1),
     .                                    max_uvar, num_indices )
              DO i2 = 1, max_uvar
                    save_deleted_list_result(i2) = 
     .		      deleted_list_result(i2)
	      ENDDO
              DO j = 1, num_indices
                 i = save_deleted_list_result(j)
	         IF (uvar_name_code(i)(1:3) .EQ. 'EX#'    ) CYCLE
	         IF ( uvar_dset(i) .NE. unspecified_int4  ) CYCLE
	         IF ( uvar_parent(i).NE.0
     .		 .AND. .NOT.mode_diagnostic		  ) CYCLE

                 IF (sxml) THEN
                    CALL SHOW_1_UVAR_XML
     .                      (show_lun,i,show_str,firstxml)
                 ELSE
		    IF ( first ) THEN
	               CALL SPLIT_LIST(pttmode_explct, show_lun,
     .	  ' >>> Definitions that replace any file variable of same name:'
     .    , 0)
	               first = .FALSE.	            
	            ENDIF
                    CALL SHOW_1_UVAR
     .                 (show_lun,i,show_str,'     ',.TRUE.)
                 ENDIF

	      ENDDO

* ... global variables defined with /D
	      first = .TRUE.
              CALL deleted_list_get_undel(uvar_num_items_head,
     .                                    deleted_list_result(1),
     .                                    max_uvar, num_indices )
              DO i2 = 1, max_uvar
                    save_deleted_list_result(i2) = 
     .		      deleted_list_result(i2)
	      ENDDO
              DO j = 1, num_indices
                 i = save_deleted_list_result(j)
	         IF (uvar_name_code(i)(1:3) .EQ. 'EX#'    ) CYCLE
	         IF ( uvar_dset(i) .NE. pdset_irrelevant  ) CYCLE
	         IF ( uvar_parent(i).NE.0
     .		 .AND. .NOT.mode_diagnostic		  ) CYCLE

                 IF (sxml) THEN
                    CALL SHOW_1_UVAR_XML
     .                      (show_lun,i,show_str,firstxml)
                 ELSE
		    IF ( first ) THEN
	               CALL SPLIT_LIST(pttmode_explct, show_lun,
     .	' >>> Definitions used if no file variable of this name exists:'
     .    , 0)
	               first = .FALSE.	            
	            ENDIF
                    CALL SHOW_1_UVAR
     .                 (show_lun,i,show_str,'     ',.TRUE.)
                 ENDIF

	      ENDDO

	   ENDIF

 4010	FORMAT (1x,A,' = ',A)
 4020	FORMAT (T20,'"',A,'"')
	ENDIF
        IF (.NOT. firstxml .AND. sxml) THEN
           risc_buff = '</global>'
           CALL SPLIT_LIST(pttmode_explct, show_lun, risc_buff, 0)
        ENDIF
 4030	CONTINUE
	IF (sho_file .GT. 0) CLOSE( UNIT = show_lun, ERR = 5000 )

	IF ( limited_show ) THEN
	   RETURN
	ELSE
	   subcmnd_num = subcmnd_num + 1
	   GOTO 10
	ENDIF

      ENDIF

* SHOW COMMAND/ALL /BRIEF [name] [subcommand]

      IF (subcmnd_num .EQ. 11) THEN

	only_1 = num_args .GT. 0
 
* /BRIEF or full list of subcommands and qualifiers?
 	sbrief  = qual_given( slash_brief ) .GT. 0 

	done_1 = .FALSE.
	IF ( .NOT.only_1 ) THEN
           show_str = TM_FMT(revision_level, 5, 12, dlen)
	   WRITE (risc_buff, 4100)
     .				program_name(1:len_program_name),
     .				progname_mod (1:len_progname_mod),
     .				show_str(1:dlen)
	   CALL SPLIT_LIST(pttmode_explct, show_lun, risc_buff, 0)
	   risc_buff = ' '
 4100	   FORMAT(' Commands in Program ',A,1X,A,' version',A,':')
	ENDIF
* ... loop through commands
        len_test = TM_LENSTR(cmnd_buff(arg_start(1):arg_end(1)))

* ... did they give a subcommand?
	sub_matched = .FALSE.
	showsub = (num_args .GE. 2)
	IF (showsub) THEN
	   sub4 = cmnd_buff(arg_start(2):arg_end(2))
	   IF ( INDEX(sub4,'/') .GT. 0) sub4 = sub4(:INDEX(sub4,'/')-1) 
           lensub = TM_LENSTR(sub4)
	ENDIF

	DO icmnd = 1, total_num_commands
	   IF ( commands(icmnd)(:4) .EQ. '****' ) CYCLE
	   IF (done_1) CYCLE
	   IF ( interrupted ) CALL ERRMSG
     .			( ferr_interrupt, status, ' ', *5000 )
* ... if  only 1 command is of interest is this it ?
           len_mchars = TM_LENSTR(commands(icmnd)(:4))
	   IF ( only_1 .AND. .NOT.MATCH4(cmnd_buff(arg_start(1):arg_end(1)),
     .		len_test,commands(icmnd),len_mchars) ) CYCLE
	   IF (only_1) done_1 = .TRUE.
	   len0 = TM_LENSTR1( commands(icmnd) )

* ... and for each command, its subcommands

	   DO isub = subcommand_pointer(icmnd),
     .			  subcommand_pointer(icmnd)+num_subcommands(icmnd)-1
              
	      IF ( subcommands(isub)(:4) .EQ. '****' ) CYCLE
	      IF (showsub) THEN 
	         len_msubs = TM_LENSTR(subcommands(isub)(:4)) 
	         IF (MATCH4(sub4,lensub, subcommands(isub),len_msubs)) THEN
                    IF (sub_matched) CYCLE  
		    sub_matched = .TRUE. ! return the first match, which is what Ferret will use
	         ELSE
		    CYCLE
	         ENDIF
	      ENDIF

	      show_str = commands(icmnd)(:len0) // ' ' // subcommands(isub)
	      len = TM_LENSTR1( show_str )

* ... and for each subcommand, its qualifiers (except on /BRIEF listing)
              IF (sbrief) GOTO 1115
	      DO iqual =   qualifier_pointer(isub),
     .				qualifier_pointer(isub)+num_qualifs(isub)-1
	         IF ( qualifiers(iqual)(:4) .EQ. '****' ) CYCLE
	         show_str = show_str(1:len) // '/' // qualifiers(iqual)
	         len = len + 1 + TM_LENSTR1( qualifiers(iqual) )
	         IF ( len .GE. 70 ) THEN
	            CALL SPLIT_LIST(pttmode_explct, show_lun,
     .				' '//show_str, len+1)
	            show_str = ' '
	            len = len0 + 2
	         ENDIF
	      ENDDO

 1115	      IF ( show_str .NE. ' ' ) CALL SPLIT_LIST
     .			(pttmode_explct, show_lun, ' '//show_str, len+1)

*  ... brief listing, show only the command names (if all commands)
	      IF (sbrief .AND. .NOT.only_1) EXIT
	   ENDDO

	ENDDO
	IF ( only_1 .AND. .NOT.done_1 ) THEN
	   WRITE (risc_buff, 4150)
     .					cmnd_buff( arg_start(1):len_cmnd )
	   CALL SPLIT_LIST(pttmode_explct, show_lun, risc_buff, 0)
	   risc_buff = ' '
 4150	   FORMAT(' Not a recognized command: ',A)
	ENDIF
	IF ( .NOT.(only_1.AND.done_1) ) THEN
	   CALL SPLIT_LIST(pttmode_explct, show_lun,' ', 1)
	   CALL SPLIT_LIST(pttmode_explct, show_lun,
     .		' Use SHOW ALIAS to see alternative command names', 0)
	ENDIF
 1199	IF ( limited_show ) THEN
	   RETURN
	ELSE
	   subcmnd_num = subcmnd_num + 1
	   GOTO 10
	ENDIF

      ENDIF

* SHOW MEMORY

      IF (subcmnd_num .EQ. 12) THEN

	IF ( qual_given(slash_mem_diag) .GT. 0 ) THEN
* ... SHOW MEMORY/DIAGNOSTIC
           IF (nsplits .EQ. 0) THEN
	      CALL SPLIT_LIST(pttmode_explct, show_lun,
     .             '    No split/gather occurred in the last evaluation', 0)
           ELSE
	      CALL SPLIT_LIST(pttmode_explct, show_lun,
     .             '    Last gather', 0)
	      CALL SPLIT_LIST(pttmode_explct, show_lun,
     .             '        Variable    Axis   Xform     '
     .             //'Chunk    Repeated', 0)
              DO i = 1, nsplits
                 name = VAR_CODE(split_cat(i), split_var(i))
                 WRITE (risc_buff, 4201)
     .                          name,
     .                          ww_dim_name(split_axis(i)),
     .                          ALG_TRANS_CODE(split_trans(i)),
     .                          split_frag(i),
     .                          split_repeats(i)
	         CALL SPLIT_LIST(pttmode_explct, show_lun, risc_buff, 0)

                 IF (i .EQ. max_splits) THEN
                    CALL SPLIT_LIST(pttmode_explct, show_lun,
     .                      " ... and others", 0)
                    EXIT
                 ENDIF
              ENDDO
           ENDIF
* ... run the memory diagnostic as part of /DIAGNOSTIC
	   CALL CHECK_MEMORY( show_lun )
        ENDIF
 4201   FORMAT(8x,A15,A1,A8,I10,I12)

        CALL SHOW_MEM_USAGE_LINE('SET MEMORY/MAX', max_mem_allowed, .TRUE.)
        CALL SHOW_MEM_USAGE_LINE('Peak demand', peak_essential_mem, .FALSE.)
        CALL SHOW_MEM_USAGE_LINE('Current cache', total_mem, .FALSE.)
        IF (permanent_mem .NE. 0 ) THEN
           CALL SHOW_MEM_USAGE_LINE('Permanent', permanent_mem, .FALSE.)
        ENDIF

	IF ( qual_given( slash_mem_free ) .GT. 0 ) THEN
	   CALL CHECK_MEMORY( show_lun )
	ELSEIF( slash_all
     .     .OR. qual_given(slash_mem_temp) .GT. 0
     .	   .OR. qual_given(slash_mem_perm) .GT. 0 ) THEN
	   CALL SHOW_MEM_VARS
	ENDIF
	IF ( limited_show ) THEN
	   RETURN
	ELSE
	   subcmnd_num = subcmnd_num + 1
	   GOTO 10
	ENDIF

      ENDIF

* SHOW GRID grid_or_var1 grid_or_var2 ... grid_or_varN
* first examine the context info ( /X ... etc. )

      IF (subcmnd_num .EQ. 13) THEN

	CALL STACK_PTR_UP ( cx_stack_ptr, max_context, status )
	IF ( status .NE. ferr_ok ) GOTO 5000
	mods_cx = cx_stack_ptr
 	sxml  = qual_given( slash_grid_xml  ) .GT. 0 


* 	SHOW GRID/OUTFILE=[/APPEND][/CLOBBER]
        sho_file = qual_given( slash_show_grid_file )
	clobber = qual_given( slash_show_grid_clobber ) .GT. 0
	append = qual_given( slash_show_grid_append ) .GT. 0

	IF (sho_file .GT. 0) THEN
	   CALL OPEN_SHOW_FILE (show_lun, sho_file,  
     .                          clobber, append, status)
	   IF (status .NE. ferr_ok ) GOTO 5000
	ENDIF



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

* fix for 1961
	   cx_only = .TRUE.

*  Allocate dynamic line memory for a list of uvars where uvar-grid info gets
*  set in this routine.
		    
	   CALL GET_LINE_DYNMEM(max_uvar, uvar_cx_only_lm, status)
	   uvar_cxo_len = 0

	   CALL DELETE_OLD_UGRIDS

* loop through the variables named 
! bug fix: 11/91 *sh* : changed comma list to blank-separated argument list
!	    4/96: *sh* used ARG_TO_ITEM to harmonize
	   DO ivar = 1, num_items
	      i1 = STR_UPCASE(arg,cmnd_buff(item_start(ivar):item_end(ivar)))
	      IF ( IS_TEMPLATE( arg ) ) THEN
*    ... first check static grids
	         DO grid = 1, max_grids
	            IF ( grid_name(grid) .EQ. char_init16 ) CYCLE
	            IF (MATCH_TEMPLATE(grid_name(grid) ,arg)) THEN
	               WRITE ( risc_buff, 4220 ) '  '//grid_name(grid)
	               CALL SPLIT_LIST(pttmode_explct, show_lun,
     .				risc_buff, 0)
	               risc_buff = ' '
	            ENDIF
		 ENDDO
*    ... then check dynamic grids
                 grid = 0
 1344            CALL TM_NEXT_DYN_GRID( grid, *1346 )
	            IF (MATCH_TEMPLATE(grid_name(grid) ,arg)) THEN
	               WRITE ( risc_buff, 4220 ) '  '//grid_name(grid)
	               CALL SPLIT_LIST(pttmode_explct, show_lun,
     .				risc_buff, 0)
	               risc_buff = ' '
	            ENDIF
	            GOTO 1344
 1346               CONTINUE
	      ELSE

		 grid = GRID_FROM_NAME
     .                    ( cmnd_buff(item_start(ivar):item_end(ivar)),
     .			  cx_last, status )
	         IF ( status .NE. ferr_ok ) CYCLE
                 IF ( grid .EQ. unspecified_int4 ) GOTO 5300
	         IF (sxml) THEN 
                    CALL SHOW_GRID_XML(show_lun, grid, unspecified_int4 )
                 ELSE
	            WRITE ( risc_buff, 4210 ) grid_name( grid )
		    IF (dset.NE.unspecified_int4 .AND. dset.NE.pdset_irrelevant
     .		       .AND. dset.NE.pdset_uvars .AND. dset.NE.pdset_coordvars) THEN
	            IF ( TM_HAS_STRING(ds_type(dset), 'ENS') ) THEN
		       dlen = TM_LENSTR1(grid_name(grid) )
		       WRITE ( risc_buff, 4211) grid_name(grid)(:dlen)
		    ENDIF
	            IF ( TM_HAS_STRING(ds_type(dset), 'FCT') ) THEN
		       dlen = TM_LENSTR1(grid_name(grid) )
		       WRITE ( risc_buff, 4212) grid_name(grid)(:dlen)
		    ENDIF
		    ENDIF
	            CALL SPLIT_LIST(pttmode_explct, show_lun, risc_buff, 0)

* If this was a 'show grid grid_name' do not list the grid-line context information
		    i1 = TM_GET_GRIDNUM( arg )
		    uvgrid = (i1 .EQ. unspecified_int4 )

	            risc_buff = ' '
	            CALL SHOW_GRID( grid, mods_cx, uvgrid, status )
		    IF (status .NE. ferr_ok) GOTO 5000
	         ENDIF

	      ENDIF

c Free the temporary storage for uvar-grid information 
	      IF (cx_only) THEN
	         CALL RESET_UVAR_GRIDS
	         CALL FREE_LINE_DYNMEM(uvar_cx_only_lm)

	         cx_only = .FALSE.

	      ENDIF

	   ENDDO

 4210	   FORMAT ( 4X,'GRID ',A)
 4211	   FORMAT ( 4X,'GRID ',A, '  Ensemble Aggregetion Grid')
 4212	   FORMAT ( 4X,'GRID ',A, '  Forecast Aggregation Grid')

	ELSEIF( qual_given(slash_grid_dyn) .GT. 0 ) THEN
	   CALL SHOW_DYN_GRIDS(mods_cx)
	ELSEIF ( slash_all ) THEN
	   WRITE ( risc_buff, 4215 ) grid_name(mgrid_abstract)
	   CALL SPLIT_LIST(pttmode_explct, show_lun, risc_buff, 0)
	   risc_buff = ' '
 4215	   FORMAT(' Default grid for DEFINE VARIABLE is ',A)
* show 'em all the names
	   WRITE ( risc_buff, 4220 ) '** GRIDS NAMES **'
	   CALL SPLIT_LIST(pttmode_explct, show_lun, risc_buff, 0)
	   risc_buff = ' '
	   DO grid = 1, max_grids
	      IF ( interrupted ) CALL ERRMSG
     .			( ferr_interrupt, status, ' ', *5000 )
	      IF ( grid_name( grid ) .EQ. char_init16 ) CYCLE
	      WRITE ( risc_buff, 4220 ) '  '//grid_name( grid )
	      CALL SPLIT_LIST(pttmode_explct, show_lun, risc_buff, 0)
	      risc_buff = ' '
 4220	      FORMAT (4X,A)
	   ENDDO
*    ... then check dynamic grids
           grid = 0
 1370      CALL TM_NEXT_DYN_GRID( grid, *1379 )
	      IF ( grid_name(grid)(1:1) .EQ. '(' ) GOTO 1370
	      WRITE ( risc_buff, 4220 ) '  '//grid_name( grid )
	      CALL SPLIT_LIST(pttmode_explct, show_lun, risc_buff, 0)
	      risc_buff = ' '
	      GOTO 1370
 1379      CONTINUE

	ELSE
	   WRITE ( risc_buff, 4215 ) grid_name(mgrid_abstract)
	   CALL SPLIT_LIST(pttmode_explct, show_lun, risc_buff, 0)
	   risc_buff = ' '
* display the grid from last command
	   cx = is_cx( 1 )
	   IF ( cx.EQ.0 .OR. cx.EQ.unspecified_int4 )	GOTO 1390
	   grid = cx_grid( cx )
	   IF ( grid .EQ. unspecified_int4 ) 		GOTO 1390
	   WRITE ( risc_buff, 4200 ) grid_name(grid)
	   CALL SPLIT_LIST(pttmode_explct, show_lun, risc_buff, 0)
	   risc_buff = ' '
	   WRITE ( risc_buff, 4210 ) grid_name(grid)
	   CALL SPLIT_LIST(pttmode_explct, show_lun, risc_buff, 0)
	   risc_buff = ' '
 4200	   FORMAT (' Last successful data access was on grid ',A)
	   uvgrid = (grid_name(grid) .NE. char_init16) ! show context for last command
	   CALL SHOW_GRID( grid, mods_cx, uvgrid, status )
	   IF (status .NE. ferr_ok) GOTO 5000
	ENDIF

* release context stack space
	CALL STACK_PTR_DN ( cx_stack_ptr, cx_stack_ptr_base, status )
	IF ( status .NE. ferr_ok ) GOTO 5000

* successful completion
        IF (sho_file .GT. 0) CLOSE( UNIT = show_lun, ERR = 5000 )

 1390 	IF ( limited_show ) THEN
	   RETURN
	ELSE
	   subcmnd_num = subcmnd_num + 1
	   GOTO 10
	ENDIF

      ENDIF

* SHOW VIEWPORT view1,view2,...
* column headings

      IF (subcmnd_num .EQ. 14) THEN

	WRITE ( risc_buff, 3142 )
	CALL SPLIT_LIST(pttmode_explct, show_lun, ' ', 1)
	CALL SPLIT_LIST(pttmode_explct, show_lun, risc_buff, 0)
	CALL SPLIT_LIST(pttmode_explct, show_lun, ' ', 1)
	risc_buff = ' '
 
	IF ( num_args .GE. 1 ) THEN
* ... loop through the viewports named
	   DO i = 1, num_items
* ... identify the viewport number from the template (or name)	      
	      i1 = STR_UPCASE(arg,cmnd_buff(item_start(i):item_end(i)))
	      DO ivp = 1, max_viewport
	         IF ( vp_name(ivp).EQ.unspecified_name4 ) CYCLE
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
	            CALL SPLIT_LIST(pttmode_explct, show_lun, risc_buff, 0)
	            risc_buff = ' '
	         ENDIF
	      ENDDO
	   ENDDO
	ELSE
* ... show 'em all
	   DO ivp = 1,max_viewport
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
	         CALL SPLIT_LIST(pttmode_explct, show_lun, risc_buff, 0)
	         risc_buff = ' '
	      ENDIF
	   ENDDO
	ENDIF
* tell what's current
	WRITE ( risc_buff, 3148 ) vp_name(vp_num)
	CALL SPLIT_LIST(pttmode_explct, show_lun, risc_buff, 0)
	risc_buff = ' '

 3142	FORMAT(' name',T19,'text',T27,'xlimits',T39,'ylimits',T50,'mode')
 3146	FORMAT(1X,A,T18,F5.2,T26,F4.2,',',F4.2,T38,F4.2,',',F4.2,T50,A)
 3148	FORMAT( T4,'current viewport is ',A)
 
	IF ( limited_show ) THEN
	   RETURN
	ELSE
	   subcmnd_num = subcmnd_num + 1
	   GOTO 10
	ENDIF

      ENDIF

* SHOW TRANSFORMS

      IF (subcmnd_num .EQ. 15) THEN

	CALL SPLIT_LIST(pttmode_explct, show_lun,
     .		'variable transforms e.g.SST[T=1-jan:15-mar@DDC]', 47)
	CALL SHOW_TRANSFORMS
	CALL SPLIT_LIST(pttmode_explct, show_lun, ' ', 1)	! blank line
	CALL SHOW_REGRID_TRANSFORMS

	IF ( limited_show ) THEN
	   RETURN
	ELSE
	   subcmnd_num = subcmnd_num + 1
	   GOTO 10
	ENDIF

      ENDIF

* SHOW ALIAS

      IF (subcmnd_num .EQ. 16) THEN

	WRITE (risc_buff, 3160) 'Alias', 'Command'
	CALL SPLIT_LIST(pttmode_explct, show_lun, risc_buff, 0)
	risc_buff = ' '
        WRITE (risc_buff, 3160) '-----', '-------'
	CALL SPLIT_LIST(pttmode_explct, show_lun, risc_buff, 0)
	risc_buff = ' '
        IF ( num_args .EQ. 0 .OR. slash_all ) THEN
	   DO i = 1, total_num_alias
	      IF ( alias_name(i) .EQ. unspecified_name4 ) CYCLE
	      WRITE ( risc_buff, 3160 ) alias_name(i), alias(i)
	      CALL SPLIT_LIST(pttmode_explct, show_lun, risc_buff, 0)
	      risc_buff = ' '
	   ENDDO
        ELSE
           DO i1 = 1, num_items
	      i2 = STR_UPCASE(arg,cmnd_buff(item_start(i1):item_end(i1)))
	      DO i = 1, total_num_alias    
	         IF ( alias(i) .EQ. unspecified_name4 ) CYCLE
	         IF ( MATCH_TEMPLATE(alias_name(i), arg) ) THEN 
                    WRITE ( risc_buff, 3160 ) alias_name(i), alias(i)
	            CALL SPLIT_LIST(pttmode_explct, show_lun,
     .					risc_buff, 0)
	            risc_buff = ' '
                 ENDIF
	      ENDDO
	   ENDDO
        ENDIF
 3160   FORMAT (T4,A,T16,A)
	IF ( limited_show ) THEN
	   RETURN
	ELSE
	   subcmnd_num = subcmnd_num + 1
	   GOTO 10
	ENDIF

      ENDIF

* SHOW SYMBOLS sym1 sym2 ...
*           or SHOW SYMBOL/ALL

      IF (subcmnd_num .EQ. 17) THEN

	slash_all = slash_all .OR. num_args .EQ. 0
* SPECIAL SYMBOLS
	i1=0
 1702   i1 = i1 + 1
        CALL SPECIAL_SYMBOL(i1, argsym,show_str,len) ! get next symbol namees
	len0 = TM_LENSTR1(argsym)
        IF(argsym .NE. ' ')THEN
	   IF ( num_args .GE. 1 ) THEN
* ... check the names specified by template matching 
	      DO i = 1, num_items
	         n = STR_UPCASE(arg,
     .			cmnd_buff(item_start(i):item_end(i)))
	         IF ( MATCH_TEMPLATE(argsym(:len0),arg) ) GOTO 1705
	      ENDDO
	      GOTO 1702			! no match
	   ENDIF
* ... display the symbol definition
 1705      CALL SPECIAL_SYMBOL(0, argsym,show_str,len) ! get the sym value
           CALL SPLIT_LIST(pttmode_explct, show_lun,
     .			argsym(:len0)//' = "'//show_str(:len)//'"', 0)
           GOTO 1702
	ENDIF

* PLOT+ SYMBOLS
 1709   i1=0
 1710   CALL LSTSYM(argsym,show_str,len,i1,i2)	! get next symbol
	len0 = TM_LENSTR1(argsym)
        IF(i2.EQ.0)THEN
	   IF ( num_args .GE. 1 ) THEN
* ... check the names specified by template matching 
	      DO i = 1, num_items
	         n = STR_UPCASE(arg,
     .			cmnd_buff(item_start(i):item_end(i)))
	         IF ( MATCH_TEMPLATE(argsym(:len0),arg) ) GOTO 1750
	      ENDDO
	      GOTO 1710			! no match
	   ENDIF
* ... display the symbol definition
* *kob* 10/97 added check for len = 0
 1750	   IF ( len .GT. 0 ) THEN
	       CALL SPLIT_LIST(pttmode_explct, show_lun,
     .			argsym(:len0)//' = "'//show_str(:len)//'"', 0)
	   ELSE 
               CALL SPLIT_LIST(pttmode_explct, show_lun,
     .			argsym(:len0)//' = ""', 0)
	   ENDIF
           GOTO 1710
	ENDIF

* successful completion
 	IF ( limited_show ) THEN
	   RETURN
	ELSE
	   subcmnd_num = subcmnd_num + 1
	   GOTO 10
	ENDIF

      ENDIF

* SHOW ATTRIBUTE

      IF (subcmnd_num .EQ. 18) THEN

* 
* ... SHOW ATT varname.att ... various cases of how to show:
*       SHOW ATT/ALL varname    - show all attributes for var in current dset
*       SHOW ATT/ALL varname[D=n]
*	SHOW ATT varname.attrib	- show attrib on variable in current dset
*	SHOW ATT name[D=n]	- show attrib on variable from named data set
*	SHOW ATT/D=n varname.attrib - show attrib on variable from named data set
*	SHOW ATT/OUTPUT also list the output flag for the attribute

	IF ( num_args .EQ. 0  ) THEN
	   IF (limited_show) THEN
	      GOTO 5060
	   ELSE
	      subcmnd_num = subcmnd_num + 1
	      GOTO 10
	   ENDIF
	ENDIF

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

           coordvar = .FALSE.
           IF (show_str(1:1) .EQ. '(') THEN
              dset = cx_data_set(cx_cmnd)   ! initial value to try; will return dset
              CALL ISIT_COORD_VAR ( show_str, dset, 
     .              varname, coordvar, status )

              IF ( status .NE. ferr_ok ) THEN 
                 dset = pdset_coordvars  ! a user-defined axis (coordinate variable)
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
	   brkt = INDEX(show_str,'[')
	   IF ( brkt .GT. 0 ) varname = show_str(1:brkt-1)
           IF (dot .GT. 0) varname = show_str(1:dot-1)
           CALL FIND_VAR_NAME(dset, varname, cat, var)
           IF (  var .NE. munknown_var_name .AND.
     .           cat .EQ. cat_user_var) dset = pdset_uvars

* Or use command context to get data set

           IF (dset .EQ. pdset_irrelevant .OR. 
     .        dset .EQ. unspecified_int4) dset = cx_data_set(cx_cmnd) 

	ENDIF
	      

* Is it show att dataset-name or dataset-number?

        sh_att_dset = var .EQ. munknown_var_name
	IF (coordvar) sh_att_dset = .FALSE.       ! var not set for (coordvar) syntax
	IF (dot.GT.0) sh_att_dset = .FALSE.       ! it is show att varname.attname
                                                  ! it is show att . for global atts.
	IF ((dot.GT.0) .AND. TM_LENSTR(show_str) .EQ. 1) THEN
	   sh_att_dset = .FALSE. 
	   slash_all = .TRUE.
	ENDIF

	IF (sh_att_dset) THEN                    ! Do as for SHOW DAT/ATT
	   qual_given( slash_attr  ) = 1
	   CALL SHOW_DATA (dset, status)

        ELSE  ! It is show att for variable

        IF ( slash_all .OR. dot.EQ.0) THEN  ! SHOW ATT/ALL varname (/ALL optional)

           DO i = 1, num_items

	      varname = cmnd_buff( item_start(i):item_end(i) )
              IF(TM_HAS_STRING(varname, '.') ) THEN
                 varid = 0
                 CALL CD_GET_VAR_INFO (dset, varid, varname, vtype,
     .                              nvdims, vdims, nvatts, coordvar, 
     .                              all_outflag, status)
              ENDIF

              IF (dset .EQ. pdset_irrelevant .OR. 
     .            dset .EQ. unspecified_int4) GO TO 5072

* Check for the attribute info in "dset" found above, 
* or in the default datset indicated by cx_data_set(cx_last)

              CALL CD_GET_VAR_ID (dset, varname, varid, status)
              IF (status .NE. ferr_ok) THEN
	         iset = cx_data_set(cx_last)
		 CALL CD_GET_VAR_ID (iset, varname, varid, status)
		 IF (status .EQ. ferr_ok) dset = iset
	      ENDIF
              IF (status .NE. ferr_ok .OR. dset .EQ. unspecified_int4 .OR.
     .           (varid .EQ. 0 .AND.
     .           .NOT.TM_HAS_STRING(varname, '.')) ) GOTO 5200

* Header line

              IF (.NOT. sxml) THEN
                 IF (dset .NE. dset_last ) THEN
                    IF (dset .EQ. pdset_uvars) THEN
                       risc_buff = 
     .                   '     attributes for user-defined variables'
                    ELSE IF (dset .EQ. pdset_coordvars) THEN
                       risc_buff = 
     .                   '     attributes for coordinate axis'
                    ELSE
                       llen = TM_LENSTR1(ds_des_name(dset))
                       WRITE ( risc_buff, 4800 ) ds_des_name(dset)(:llen)
                    ENDIF
                    CALL SPLIT_LIST(pttmode_explct, show_lun, 
     .                     risc_buff, 0) 
                 ENDIF
                 dset_last = dset
              ENDIF

              CALL CD_GET_VAR_INFO (dset, varid, varname, vtype,
     .                              nvdims, vdims, nvatts, coordvar, 
     .                              all_outflag, status)

              DO iatt = 1, nvatts

                 CALL CD_GET_VAR_ATT_INFO (dset, varid, iatt, aname, 
     .               attype, attlen, attoutflag, status )

                 IF (status .NE. ferr_ok) CYCLE
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
	      ENDDO

	   ENDDO

        ELSE  ! SHOW ATT varname.attname

	   DO i = 1, num_items
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
                 IF (dset.NE.dset_last .AND. dset.GT.pdset_irrelevant) THEN
                    llen = TM_LENSTR1(ds_des_name(dset))
                    WRITE (risc_buff, 4800) ds_des_name(dset)(:llen)
                    CALL SPLIT_LIST(pttmode_explct, show_lun, 
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

	   ENDDO

        ENDIF

        ENDIF  ! sh att dataset vs sho att variable

4800	FORMAT (T6,'attributes for dataset: ', A)

* successful completion
 	IF ( limited_show ) THEN
	   RETURN
	ELSE
	   subcmnd_num = subcmnd_num + 1
	   GOTO 10
	ENDIF

      ENDIF

* SHOW NCCACHE

      IF (subcmnd_num .EQ. 19) THEN

	CALL CD_GET_CHUNK_CACHE (cache_size, cache_nelems, 
     .    cache_preemption, status)
        cache_mb = cache_size/1.e6
	name = TM_FMT_TRIM(cache_mb, -2, 12, dlen)
        show_str = 'Current NCDF Chunk Cache size '//name(1:dlen)
	len = TM_LENSTR1(show_str)

        cache_n = cache_nelems
	name = TM_FMT(cache_n, 0, 12, dlen)

	show_str = show_str(1:len)//' MB, n_elems = '//name(1:dlen)
	len = TM_LENSTR1(show_str)

        cache_p = cache_preemption
	name = TM_FMT(cache_p, 0, 12, dlen)

	show_str = show_str(1:len)//', preemption = '//name(1:dlen)
	len = TM_LENSTR1(show_str)

	CALL SPLIT_LIST(pttmode_explct, ttout_lun,
     .			show_str(1:len), 0)
	IF ( limited_show ) THEN
	   RETURN
	ELSE
	   subcmnd_num = subcmnd_num + 1
	   GOTO 10
	ENDIF

      ENDIF

* SHOW GIFFILE

      IF (subcmnd_num .EQ. 20) THEN

	CALL GET_METAFILE_NAME(outstring)
        len = TM_LENSTR1(outstring)
        IF ( len .GT. 0 ) THEN
           show_str = 'Current default filename: ' //outstring(1:len)
        ELSE
           show_str = 'No current default filename'
        ENDIF
        len = TM_LENSTR1(show_str)
        CALL SPLIT_LIST(pttmode_explct, ttout_lun, show_str(1:len), 0)
        	IF ( limited_show ) THEN
	   RETURN
	ELSE
	   subcmnd_num = subcmnd_num + 1
	   GOTO 10
	ENDIF

      ENDIF

* SHOW FUNCTIONS

      IF (subcmnd_num .EQ. 21) THEN

* /BRIEF or full description?
 	sbrief  = qual_given( slash_brief ) .GT. 0 
 	sdetail = qual_given( slash_show_func_detail ) .GT. 0 

* show a whole list or named functions, only?
	IF ( num_args .GT. 0 ) THEN

* ... one or more functions were explicitly named

	   DO i = 1, num_items
           i1 = STR_UPCASE(arg,cmnd_buff(item_start(i):item_end(i)))
	      len0 = item_end(i) - item_start(i) + 1
* ... identify the named function
* ... first try internal, non-grid-changing functions ...
	      DO i2 = 1, num_functions
	         IF ( interrupted ) CALL ERRMSG
     .			( ferr_interrupt, status, ' ', *5000 )
	         IF (MATCH_TEMPLATE(alg_fcn(i2),arg(:len0)))
     .           CALL SHOW_1_FUNCTION(show_lun, .FALSE.,
     .                                sbrief, sdetail, i2)
	      ENDDO
* ... second try internal, grid-changing functions
	      DO i2 = 1, gfcn_num_internal
	         IF ( interrupted ) CALL ERRMSG
     .			( ferr_interrupt, status, ' ', *5000 )
              IF ( MATCH_TEMPLATE(gfcn_name(i2),arg(:len0)) ) CALL
     .                SHOW_1_FUNCTION(show_lun, .TRUE.,
     .                                sbrief, sdetail, i2)
	      ENDDO
* ... third try external functions
              IF (qual_given(slash_internal) .EQ. 0) THEN
 	         DO i2 = gfcn_num_internal+1, gfcn_num_internal+
     .                efcn_scan(gfcn_num_internal)
	            IF ( interrupted ) CALL ERRMSG
     .			   ( ferr_interrupt, status, ' ', *5000 )

                 CALL EFCN_GET_NAME (i2, fhol)
                 CALL TM_CTOF_STRNG( fhol, efname, slen )
                 i1 = STR_UPCASE(upname, efname)

                 IF ( MATCH_TEMPLATE(upname,arg(:len0) )) CALL
     .               SHOW_1_FUNCTION(show_lun, .TRUE.,
     .                               sbrief, sdetail, i2)

	         ENDDO
              ENDIF
	   ENDDO

* show lists of functions
	ELSE

* ... show all the internal functions (omit for SHOW FUNC/EXTERN)
	   IF ( qual_given(slash_external) .EQ. 0 ) THEN
* ... internal non-grid-changing functions
	      CALL SPLIT_LIST(pttmode_explct, show_lun,
     .			'Functions internal to Ferret:', 0)
	      DO i = 1, num_functions
	         IF ( alg_fcn(i) .NE. unspecified_name4 ) CALL
     .                SHOW_1_FUNCTION(show_lun, .FALSE.,
     .                  sbrief, sdetail, i)
	      ENDDO
* ... internal grid-changing functions
	      DO i = 1, gfcn_num_internal
	         CALL SHOW_1_FUNCTION(show_lun, .TRUE.,
     .                  sbrief, sdetail, i)
	      ENDDO
	   ENDIF

* ... show all the external functions
* 3/24/99 *kob* clean up duty - change below to blank space rather than
*               null string
	   IF ( qual_given(slash_internal) .EQ. 0 ) THEN
              CALL SPLIT_LIST(pttmode_explct, show_lun,
     .             ' ', 0)       ! Print a blank line
              CALL SPLIT_LIST(pttmode_explct, show_lun,
     .           'Externally defined functions available to Ferret:',
     .             0)
	      DO i = gfcn_num_internal+1, gfcn_num_internal+
     .             efcn_scan(gfcn_num_internal)
	         CALL SHOW_1_FUNCTION(show_lun, .TRUE.,
     .                  sbrief, sdetail, i)
	      ENDDO
	   ENDIF

	ENDIF
	IF ( limited_show ) THEN
	   RETURN
	ELSE
	   subcmnd_num = subcmnd_num + 1
	   GOTO 10
	ENDIF

      ENDIF

* SHOW TEXT 

      IF (subcmnd_num .EQ. 22) THEN

	IF (.NOT.pyfont) THEN
	   CALL SPLIT_LIST(pttmode_explct, show_lun,
     .			'Hershey fonts as in classic Ferret ', 0)

	ELSE
	CALL SPLIT_LIST(pttmode_explct, show_lun,
     .			'default ', 0)

c* ... Font
	CALL SPLIT_LIST(pttmode_explct, show_lun,
     .			'     /FONT='//pyfnt_fontname, 0)
* ... Color
	
* Convert rgb back to % values
	v1 = pyfnt_red*100.
	v2 = pyfnt_grn*100.
	v3 = pyfnt_blu*100.
	WRITE (risc_buff, 2201) v1, v2, v3
 2201	FORMAT ('     /COLOR= (', 2(f4.0,','), f4.0,') (%RGB)')
	len = TM_LENSTR1(risc_buff)
	CALL GET_COLOR_NAME (v1, v2, v3, name, len0)
	IF (len0 .GT. 0) risc_buff = '     /COLOR="'//name(:len0)//'"'
	CALL SPLIT_LIST(pttmode_explct, show_lun, risc_buff, 0)

* ... Italic, Bold
	IF (pyfnt_italic .EQ. 1) THEN
	   CALL SPLIT_LIST(pttmode_explct, show_lun,
     .			'     /ITALIC=ON', 0)

	ELSE
	   CALL SPLIT_LIST(pttmode_explct, show_lun,
     .			'     /ITALIC=OFF', 0)

	ENDIF

	IF (pyfnt_bold .EQ. 1) THEN
	   CALL SPLIT_LIST(pttmode_explct, show_lun,
     .			'     /BOLD=ON', 0)

	ELSE
	   CALL SPLIT_LIST(pttmode_explct, show_lun,
     .			'     /BOLD=OFF', 0)
        ENDIF

	WRITE (risc_buff, 2202) pyfnt_isiz
 2202	FORMAT ('     /ISIZ=',i2)
	CALL SPLIT_LIST(pttmode_explct, show_lun,
     .			risc_buff, 0)

* Text groups: TITLE LOGO, etc. List settings for any that are
* different from the overall defaults.

        DO i1 = ititl, icbar
           IF (pyf_set(i1)) CALL SHOW_PYTEXT_GROUP(i1)
	ENDDO

	ENDIF  ! classic-Hershey or not

	IF ( limited_show ) THEN
	   RETURN
	ELSE
	   subcmnd_num = subcmnd_num + 1
	   GOTO 10
	ENDIF

      ENDIF


* SHOW QUERIES

      IF (subcmnd_num .EQ. 23) THEN

	DO i = 1, max_queries
	   IF ( queries(i) .NE. unspecified_name4 )CALL SPLIT_LIST
     .			(pttmode_explct, show_lun, queries(i), 0)
	ENDDO

	RETURN

      ENDIF

* error exit
 5050   CALL RELEASE_DYN_WORK_SPACE
 5000	CONTINUE
	
c Free any temporary storage for uvar-grid information 
	IF (cx_only) THEN
	   CALL RESET_UVAR_GRIDS
	   CALL FREE_LINE_DYNMEM(uvar_cx_only_lm)
	   cx_only = .FALSE.
	ENDIF

	RETURN
 5071	name = cmnd_buff(arg_start(1):arg_end(1))
	CALL ERRMSG( ferr_unknown_data_set, status,
     .			name(:TM_LENSTR1(name)), *5000 )
 5072	CALL ERRMSG( ferr_invalid_command, status,
     .         'dataset not found for varname.attname '//
     .          varname(:TM_LENSTR1(varname)), *5000 )
 5100   CALL ERRMSG( ferr_unknown_variable, status, arg, *5000)
 5150   CALL ERRMSG( ferr_unknown_qualifier, status,
     .          'SHOW VAR/TREE= may take arguments ALL, FILE, '//
     .          'and USER (default)', *5000)
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
     .   'variable, axis, or attribute does not exist '//
     .    show_str(:llen), *5000 )


 5930	CALL ERRMSG( ferr_invalid_command, status,
     .		'SHOW/FILE= what name?', *5000 )

 5090	CALL ERRMSG( ferr_prog_limit, status,
     .  'Cannot write to file w/ name longer than 256 characters: '//pCR//
     .  'Limitation of Fortran INQUIRE and OPEN calls '	
     .			, *5000 )

	END
