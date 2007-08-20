 	SUBROUTINE XEQ_DEFINE( memory )

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
* execute the DEFINE command

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 - 9/22/87
* revision 1.0 - 7/27/88 - added DEFINE GRID and DEFINE AXIS code
* revision 1.1 - 8/31/88 - added DEFINE VIEWPORT
* revision 1.2 -11/21/88 - new TM library: FIND_LIKE_GRID-->TM_FIND_LIKE_GRID
*					   TRANSFER_GRID --> TM_COPY_GRID
*					   READ_GRIDFILE --> TM_READ_GRIDFILE
*					         UNIT_ID --> TM_UNIT_ID
*					  FIND_LIKE_AXIS -->TM_FIND_LIKE_LINE
*                                       FIND_LINE_NUMBER --> TM_GET_LINENUM
*					      BREAK_DATE --> TM_BREAK_DATE
*			using TM_FIND_LINE_SLOT and TM_FIND_GRID_SLOT
*			and include files XUNITS and XGRID are from TMAP_FORMAT
* revision 1.3 - 1/15/89 - changed default T0 for time axes from 1-JAN-1800 to
*			   15-JAN-1901 for PPLUS WHOI format limitations
* V200:  6/29/89 - reordered lo/hi_ss arguments
*		 - added DEFINE VARIABLE
*		 - DEFINE AX: allow calendar time axis using time steps
*		 - added DEFINE AXIS/FROM_VARIABLE
*		 - DEFINE AXIS&GRID check for legal name
*		 - DEFINE VARIABLE accepts infix notation
*	 3/19/90 - fixed logic bugs when redefining axes and grids
*		 - eliminated warnings for duplicated definitions
*	 3/22/90 - don't check axis units if units are unknown
*	 4/24/90 - DEFINE VAR bug for var names of >8 chars
*	  5/2/90 - DEFINE REG/DX= name tried to work from cx_last instead of
*		   previous region "name"
* Unix/RISC 3/12/91 (incidental) LET/TITLE now preserves lower case letters
*                - DEFINE GRID/FILE=name changed to case sensitive name
* V230:   6/8/92 - added DEFINE ALIAS (bug fixes 7/29)
*         7/6/92 - added DEFINE AXIS/DEPTH (units="m" used to determine up/dn)
*        7/24/92 - bug fix: DEFINE REGION was behaving like /DEFAULT always
*                - delete uvars and uvar grids when @ regions are redefined
*       10/22/92 - added units on uvars
*       11/12/92 - added qualifiers /TEXT, /XLIMITS, /YLIMITS to DEFINE VIEW
*                  and made it auto-scale the axes
*	 3/29/93 - restore the old-style functioning of DEFINE VIEW/SIZE
* V300:  4/22/93 - fix handling of enclosing quotes on DEFINE ALIAS
* V301:	11/03/93 - added DEFINE AXIS/NPOINTS
*	11/25/93 - bug fix: DEFINE AXIS/FROM/T0= expr
*	 1/19/94 - added DEFINE VARIABLE/QUIET
* V313:  ?? *kob* - IBM port shortened a couple lines and removed a couple tabs
*	11/06/94 *sh* - inserted the bug fixes that had already been made in
*		   the checked out version of this routine as follows:
* V320: 5/94 - array "memory" as a calling argument
*	8/94 - replaced calls to TM_FIND_LINE_SLOT and TM_FIND_GRID_SLOT with
*		ALLO_MANAGED_GRID/AXIS to that deleted slots would be reused
* V320: 12/29/94 - use SPLIT_LIST to redirect tty output for GUI
* V400:  6/22/95 - added DEFINE SYMBOL name = value
* V420:  7/95 - Added support for  LET/D=dset var = ...
*	 11/95 - check to see if DEFINE AXIS/FROM actually has REGULAR points
*	       - default for T0 set at start of start century (not always 1900)
*	  2/96 - allow color to separate lo/high viewport limits (bug 2/12)
*	  2/96 - fixed bug introduced 11/96. Code used a non-existent variable
*		 called "regular" which was not caught by the compiler
*      2/29/96 - bug: same var name couldn't be used in multiple dsets
*	  4/96 - bug: tabs around "=" not handled right in LET A = expr
* V430:7/10/96 - allow axes with units that contradict their orientations
*		 Issue a warning
*Linux Port 1/97 *kob* - Added a preprocessor include for tmap_dset.parm
*			  because it needed a preprocessor.
* V450:  11/96 - Using reformulated version of INIT_UVAR (RPN not called
*		 from here). Part of changes to support external function.
*	  1/97 - Issue warning if DEFINE AXIS/UNIT=MONTH is used
*       - add check on cx_calendar rather than cx_lo/hi_ww to indicate whether time
*         is date or timesteps
* V491+ - 7/98 *sh* - allow case-sensitive var name in DEFINE GRID/LIKE=varname
* V500 - 4/16/99 *sh* use DELETE_USER_VAR to delete old definition when uvar
*	name is redefined ... else child vars to not get removed
*	 4/20/99 *sh* added DEFINE AXIS/EDGES
*	 4/28/99 *sh* allow DEFINE AXIS with repeated values -> microadjust
*	 7/99 *sh* - changed def'n of "month" so change warning message
* V510 - 8/99 *sh* - allow DEFINE GRID /LIKE=user_var /X=user_var (bump use
*			counts on dynamic axes
*	12/99 *sh* - allow new syntax DEFINE AXIS name=expression
*			infer units from formatted (say) X=130e:80W
*		and use routine name_equal_string in multiple places
*	2/00 *sh* - on LET cmnd when replacing an existing variable definition
*		allow for possible child vars ==> delete old BEFORE init_uvar
*		is called on new
*	3/00 *sh* - remove backslash escape characters from DEFINE SYMBOL
*	     *sh* - allow automatic generation of T0=1-jan-0000
*	4/00 *sh* - set use counts when defining grids and axes
*	5/00 *sh* - support for LET/BAD=
* V530: *sh* 9/00 - added initial data type support in get_cmnd_data 
* v530  *acm* 2/01 - DEFINE AXIS/CALENDAR  calendar name for alternative 
*                    calendar types.
*	*sh* 4/01 - bug fix: DEFINE AXIS/FROM_DATA must check if n=1
* V531: *sh* 5/01 - changed text of "unknown axis units" message
*		  - during DEFINE GRID defer purging duplicate-name grid
*		    until after new static one is defined -- else dynamic
*		    axes that are needed can get wiped out
*	*sh* 6/01 - increase buff1 size for DEFINE AXIS/LIKE=expr
* V540:	*sh* 9/01 - added DEFINE VIEWPORT/AXES
* V540: *acm*12/01 -bug fix: consolidate arguments broken up by command parser 
*                   back into a single arg with call to all_1_arg
* V541: *acm*8/02  Changes to recover line_mem storage when irregular
*                  axis is cancelled; redefine line_mem pointer before 
*                  call to purge_mr_axis.
* V542: *sh* 7/02 - added support for sub-span modulo axes
* V550: *acm*10/02- Change reg_name to char*24, to match cx_name
* V551: *acm*  2/03 - On DEFINE AXIS/x name=expr, only reset num_args when
*                     the expression is a const array {  }  (fixes bug w/ more
*                     complex expr, for example "DEF AX/X  xax = a/2 + b/2")
* 2/03 *kob* - g77 port - g77 won't allow intrinsic functions in PARAMETER
*                         statements.  use an character constant instead 
* v552: *acm* 4/03 Bug Fix: checking length of variable name. Length now 128
* V553  *acm* 6/03 Use 'XX' or 'YY' for line_direction for non-lat/lon axes. 
* v554 *acm* 11/03 reg_name had NOT been changed to *24, to match cx_name
* v554 *acm* 11/03 Adjust definition of npoints, checking to see if the upper
*                  limit of the axis is include in the upper grid box. (bug 673)
* v554 *acm* 11/03 def axis/units=yr  was not the same as def axis/units=year
*                  Change this so yr and year mean a year in whatever calendar 
*                  we are using.
* v554 *acm* 3/4   The adjustment in definition of npoints needs fuzzy comparison
*                  not strict .GE. (under g77 linux)
* V570 *acm* 5/04  Add the calendar name 365_DAY as equivalent for NOLEAP, and add 
*                  the calendar ALL_LEAP = 366_DAY.
* V570 *acm* 5/05  Changes for CF bounds 
*                  syntax for user-specified bounds:
*                  DEFINE AXIS/X/BOUNDS xax={1,2,5}, {0.5,1.5,1.5,3,3,6}
*                  DEFINE AXIS/Z/BOUNDS zax=zcoord,zbnd
* V580 *acm* 10/05 correct misspelling in statement 5582
* V580 *acm* 11/04 Use cx_cal_id to keep track of which calendar used
*                  when cx_lo_ww and cx_hi_ww are computed.
* V581 *acm*  3/05 Catch case where errors in def axis/start:lo:del, before
*                  trying to call RESET_SECS_FROM_BC (bug1203)
* V581 *acm*  5/05 For OPeNDAP HDF files, do not check whether axis name is legal.
*                  If it was given in quotes, skip the quotes when finding it in
*                  the list of axis names.
* V600 *acm* 3/06  fixing bugs 439&1390, new arg to get_new_cx.
* V600 *acm* 3/06  Fix bug 1400: Add more convenient syntax for user-specified bounds:
*                  Bounds can be either 2*N or N+1 values, or three lists of length N:
*                  coords, lo_bound, hi_bound
*                  DEFINE AXIS/X/BOUNDS xax={1,2,5}, {0.5,1.5,3,6}
*                  DEFINE AXIS/Z/BOUNDS zax=zcoord, zbnd
*                  DEFINE AXIS/Z/BOUNDS zax={1,2,5}, {0.5,1.5,3}, {1.5,3,6}
*                  
* V600 *acm* 3/06  Fix bug 1401: See the comment: a single point is always "regular" --
*                  causes incorrect treatment of the axis if the axis has BOUNDS!!
* V600 *acm*  8/05 DEFINE ATTRIBUTE[/type=][/D=] varname[d=].attname = <numeric scalar or 
*                  vector, or string>
* V600 *acm*  9/06 Fixes to bug 1443; tm_check_bnds is changed so it works differently
*                  for a regular axis. Here we need to set line_regular to false when
*                  defining the axis from data. It may later be set to true if the data is
*                  in fact regular.
* V601 *acm* 9/06  fixing bug 1439 long symbol values; change length of err_string to 2048
* V601 *acm*11/06 Fix bug 1470: on a DEFINE AXIS command which is redefining an axis, change
*                 values in the attribute structure for the axis.
* V602  2/07 *acm* Fix bug 1492, changing attributes of coordinate variables 
* V603  5/07 *acm* Fix bug 1511: if DEFINE AXIS or DEFINE ATT change a time
*                  origin, include the origin in the units attribute as "since t0"
*                  If units attribute is changed for a time axis, get the time
*                  origin if it exists and append it to the units string.

	include	'tmap_errors.parm'
#	include	"tmap_dset.parm"
	include	'tmap_dims.parm'
	include 'xunits.cmn_text'
	external xunits_data
	include 'xtm_grid.cmn_text'
	external xgt_grid_data
	include	'ferret.parm'
	include 'errmsg.parm'
	include 'rpn.parm'
	include 'gfdl_vms.parm'	
	include 'xprog_state.cmn'
	include 'xplot_state.cmn'
	include 'xcontext.cmn'
	include 'xvariables.cmn'
	include 'xtext_info.cmn'
	include 'xcommand.cmn'
	include 'calendar.decl'
	include 'calendar.cmn'	
        include 'netcdf.inc'

* local parameter declarations:
	INTEGER	    slash_dflt,
     .		    slash_file,
     .		    slash_like,
     .		    slash_units,
     .		    slash_x0,
     .		    slash_t0,
     .		    slash_name,
     .		    slash_from,
     .		    slash_depth,
     .		    slash_modulo,
     .		    slash_npoints,
     .		    slash_edges,
     .		    slash_calendar,
     .		    slash_text,
     .		    slash_xlimits,
     .		    slash_ylimits,
     .		    slash_size,
     .		    slash_origin,
     .		    slash_clip,
     .		    slash_let_title,
     .		    slash_let_units,
     .		    slash_let_quiet,
     .		    slash_let_dset,
     .		    slash_let_bad,
     .		    slash_define_vp_by_ax,
     .              slash_bounds,
     .              slash_type,
     .              slash_dset,
     .              slash_output

	LOGICAL	    explct_defn
	PARAMETER ( slash_dflt   = 9,
     .		    slash_file   = 1 + 4,
     .		    slash_like   = 1 + 5,
     .		    slash_units  = 1 + 5,
     .		    slash_x0     = 0,	! slash_x minus 1
     .		    slash_t0     = 1 + 6,
     .		    slash_name	 = 1 + 7,
     .		    slash_from	 = 1 + 8,
     .		    slash_depth	 = 1 + 9,
     .		    slash_modulo = 1 + 10,
     .		    slash_npoints= 1 + 11,
     .		    slash_edges  = 1 + 12,
     .		    slash_calendar = 1 + 13,
     .              slash_text   = 1 + 0,
     .		    slash_xlimits= 1 + 1,
     .		    slash_ylimits= 1 + 2,
     .		    slash_size   = 1 + 3,
     .		    slash_origin = 1 + 4,
     .		    slash_clip   = 1 + 5,
     .		    slash_let_title  = 1 + 0,
     .		    slash_let_units  = 1 + 1,
     .		    slash_let_quiet  = 1 + 2,
     .		    slash_let_dset   = 1 + 3,
     .		    slash_let_bad    = 1 + 4,
     .		    slash_define_vp_by_ax = 1 + 6,
     .              slash_bounds     = 1+14,
     .              slash_type       = 1+1,
     .              slash_dset       = 1+0,
     .              slash_output     = 1+2,
     .		    explct_defn  = .FALSE. )

* calling argument declarations:
	REAL	memory( mem_blk_size, max_mem_blks )

* local variable declarations:
	LOGICAL	  TM_DIGIT, TM_LEGAL_NAME, TM_LINE_MATCH, TM_GRID_MATCH,
     .		  TM_FPEQ, TM_DFPEQ, TM_CHECK_BNDS, MATCH_NAME,
     .            NC_GET_ATTRIB, TM_LENSTR,
     .            dup_name, delta_given, create, irreg, its_reg, its_edges, 
     .            its_calendar, its_modulo, has_repeated, has_bounds,
     .            new_att, new_att_modulo, got_it
	INTEGER   REGION_NUMBER, GRID_FROM_NAME, VIEWPORT_NUMBER, STR_UPCASE,
     .		  CX_DIM_LEN, ALIAS_ID,
     .		  TM_GET_LINENUM, TM_LENSTR1,
     .		  TM_GET_GRIDNUM, TM_UNIT_ID,
     .		  ALLO_MANAGED_AXIS, ALLO_MANAGED_GRID, FIND_DSET_NUMBER,
     .            TM_GET_CALENDAR_ID, STR_SAME,
     .		  tmap_status, npoints, i, 
     .		  status, idim, vax_code, islot, iunit, iline, cat, var,
     .		  orient, mods_cx, cx, mr, n, i1, i2, i3, i4, i5, i6, pos,
     .		  grid, iqual, ax_grid, ivp, qp, iseg, uvar, dset, slen, s1,
     .		  old_line, old_grid, frst_pt, dim(4), ndim, natom,
     .		  at_type  (maxatoms), at_id  (maxatoms),
     .		  at_start (maxatoms), at_end (maxatoms), cal_id, 
     .            dflt_cal_id, nmonths, ndays, d_before_mon(12), 
     .            mon_by_d(366), d_in_mon(12), len_cal, n2, n3, s2len,
     .            prev_cal_id, j, num_indices, varid, attype_spec,
     .            mr_perm(max_intrp), attoutflag, attlen, coordvar, 
     .            dset_to_add, dir, type, attype, attid 

	REAL	  rbuff, bad_flag, yeardays, 
     .            xlovp, ylovp,  xhivp, yhivp,
     .            xoei,  yoei,   xcei,  ycei, scale, val,
     .            dummy, val_buf

	REAL*8	  SECS_FROM_BC, TM_WW_AXLEN,
     .		  bc_to_t0, start, new_ww, delta, end, secs2start,
     .            micro_adj, axwwlen, rmod_len, secsperyear, small, 
     .            new_att_modulo_len
	CHARACTER TM_FMT*16, 
     .            reg_name*24, buff1*128, buff2*128, buff3*128, 
     .            buff30*30, cal_name*32, buff*512,
     .            basic_orients(4)*2, err_string*2048, replmsg*24,
     .		  mon_names(12)*3, varname*512, attname*128, t2*2,
     .            new_att_units*128, new_att_calendar*128,
     .            new_att_t0*128, dcode*2, axdir*1, t0string*128, 
     .            ustring*128

        CHARACTER*1 tab
#ifdef NO_INTRINSIC_IN_PARAMETER
	PARAMETER     ( tab = o'011' )
#else
	PARAMETER     ( tab = CHAR(9))
#endif

	DATA	  basic_orients / 'WE','SN','UD','TI' /,
     .		  replmsg/'Replacing definition of '/

* statement function - value between 0 and 1
	REAL x
	LOGICAL OUTSIDE_0_1
	OUTSIDE_0_1( x ) = x.LT.0.0 .OR. x.GT.1.0

* select subcommand
	GOTO ( 100,200,300,400,500,600,700,800,900 ) subcmnd_num

* DEFINE  '    ' - (no action)
* arrival at this point usually means an illegal subcommand was given
* and was assumed to be an argument by the command parser
 100	IF ( num_args .GE. 1 ) THEN
	   CALL ERRMSG( ferr_invalid_subcmnd, status,
     .			cmnd_buff( arg_start(1):arg_end(1) ), *5000 )
	ELSE
	   CALL ERRMSG( ferr_invalid_command, status,
     .			'DEFINE what ?', *5000 )
	ENDIF

**************************************************************************
* DEFINE REGION
* ... command syntax ok ?
 200	IF ( num_args .EQ. 0 ) THEN
	   CALL ERRMSG( ferr_invalid_command, status,
     .			'DEFINE what region ?', *5000 )
	ENDIF

* get the name of the region to be defined
	reg_name = cmnd_buff(arg_start(1):arg_end(1))

* is this region already defined ?
	islot = REGION_NUMBER( reg_name )
        create = islot .EQ. unspecified_int4

* delta context specifiers (DX=,DY=, etc), if any
	CALL GET_DELTA_CONTEXT( 10, delta_given, status )
	IF ( status .NE. ferr_ok ) GOTO 5000

	IF ( delta_given .AND. .NOT.create ) THEN
* modify a previously defined region
* minor bug: normal non-delta location qualifiers are ignored in this case
	   CALL TRANSFER_CONTEXT( islot, cx_buff )
	   CALL APPLY_DELTA_CONTEXT( cx_buff, reg_name, status )
	   IF ( status .NE. ferr_ok ) GOTO 5000

	ELSE
* assemble the context based on the last command context
	   CALL GET_NEW_CX( cx_last, cx_buff, .TRUE., status )
	   IF ( status .NE. ferr_ok ) GOTO 5000

* ... apply delta limits to the default from last command
	   IF ( delta_given ) THEN
	      CALL APPLY_DELTA_CONTEXT( cx_buff, 'DEFAULT', status )
	      IF ( status .NE. ferr_ok ) GOTO 5000
	   ENDIF

* ... if this region was not previously defined find a slot
	   IF ( create ) THEN
	      DO 240 islot = 0, min_context, -1
	         IF ( cx_name(islot) .EQ. unspecified_name4 ) GOTO 245!fix 7/92
 240	      CONTINUE
* ... no free slots
	      CALL ERRMSG( ferr_prog_limit,status,
     .			   'Cancel or redefine a region',*5000 )
	   ENDIF

* ... if "/DEFAULT" was implied or given then flag for saving all dimensions
*     that are defined
 245       IF ( num_qualifiers .EQ. 0 
     .     .OR. qual_given( slash_dflt ) .GT. 0 ) THEN
	      DO 210 idim = 1, 4
	         IF (cx_by_ss(idim,cx_buff)
     .		   .AND. cx_lo_ss(cx_buff,idim) .NE. unspecified_int4
     .	        .OR..NOT.cx_by_ss(idim,cx_buff)
     .		   .AND. cx_lo_ww(idim,cx_buff) .NE. unspecified_val8)
     .				cx_given( idim, cx_buff ) = .TRUE.
 210	      CONTINUE
* ... else flag only the dimensions named
	   ENDIF
	ENDIF

* all unmodified axes must be set to unspecified
	DO 230 idim = 1, 4
 230	IF ( .NOT.cx_given(idim,cx_buff) ) CALL DEL_CX_DIM( idim, cx_buff )

* LET-defined variables may involve now-redefined @regions so wipe out
* previously computed values and grids that may be obsolete(7/92)
        IF ( .NOT.create ) THEN
           CALL PURGE_ALL_UVARS
           CALL DELETE_ALL_UVAR_GRIDS
        ENDIF

* save it for posterity
 	CALL TRANSFER_CONTEXT( cx_buff, islot )
	cx_name( islot ) = reg_name
	RETURN

**************************************************************************
* DEFINE GRID
* ... DEFINE GRID/FILE=fname
 300	iqual = qual_given( slash_file )
	IF ( iqual .GT. 0 ) THEN
* ... note "err_string" is used to save space - this is not an error
	   CALL EQUAL_STR_LC( cmnd_buff(qual_start(iqual):qual_end(iqual)),
     .			      err_string, status )
	   IF ( status .NE. ferr_ok ) RETURN
	   CALL TM_READ_GRIDFILE( err_string, tmap_status )
	   IF ( tmap_status .NE. merr_ok ) GOTO 5305
	   RETURN
	ENDIF

* DEFINE GRID gridname - what is "gridname" ?
	IF ( num_args .NE. 1 ) GOTO 5300

* must be a unique name
	vax_code = STR_UPCASE( buff3, cmnd_buff(arg_start(1):arg_end(1)) ) 
	IF ( .NOT.TM_LEGAL_NAME(buff3) ) GOTO 5100
	old_grid = TM_GET_GRIDNUM( buff3 ) 
	dup_name = old_grid .NE. unspecified_int4
	IF (dup_name .AND. old_grid .GT. max_grids ) GOTO 5310

* was a /LIKE= given ? ... set up a grid template
	iqual = qual_given( slash_like )
	IF ( iqual .GT. 0 ) THEN
	   CALL EQUAL_STR_LC( cmnd_buff(qual_start(iqual):qual_end(iqual)),
     .			      buff1, status )
	   IF ( status .NE. ferr_ok ) RETURN
	   grid = GRID_FROM_NAME( buff1, cx_last, status )
	   IF ( status .NE. ferr_ok ) RETURN
	   CALL TM_COPY_GRID( grid, mgrid_buff )
	ELSE
	   DO 310 idim = 1, 4
	      grid_line    (idim, mgrid_buff) = mnormal
              grid_out_prod(idim, mgrid_buff) = .TRUE.
 310	   CONTINUE	
	   grid_rotation( mgrid_buff ) = 0.0
	ENDIF

* axes named with /X=line_grid_or_var, /Y= ...
	DO 320 idim = 1, 4
	   iqual = qual_given( idim )
	   IF ( iqual .GT. 0 ) THEN
	      CALL EQUAL_STRING( cmnd_buff(qual_start(iqual):qual_end(iqual)),
     .				 buff1, status )
	      IF ( status .NE. ferr_ok ) RETURN
	      iline = TM_GET_LINENUM( buff1 )
	      IF ( iline .NE. unspecified_int4 ) THEN
* ... line given by name - don't allow inappropriate units
	         iunit = line_unit_code( iline )
 	         IF ( iunit.NE.0 ) THEN
 	            IF ( (  idim.EQ.t_dim .AND.
     .		     (iunit.LT.pun_1st_time .OR. iunit.GT.pun_last_time) )
     .	            .OR. (  idim.NE.t_dim .AND.
     .		     (iunit.LT.pun_1st_len  .OR. iunit.GT.pun_last_len ) )
     .	            .OR. (  idim.EQ.z_dim .AND.
     .	              iunit.EQ.pun_degrees )     ) GOTO 5330
	         ENDIF
	         grid_line(idim, mgrid_buff) = iline
	      ELSE
* ... line implied through grid or variable name
	         ax_grid = GRID_FROM_NAME( buff1, cx_last, status )
	         IF ( status .NE. ferr_ok ) RETURN
	         IF ( ax_grid .EQ. unspecified_int4 ) GOTO 5320
	         grid_line(idim, mgrid_buff) = grid_line( idim, ax_grid )
	      ENDIF
	   ENDIF
 320	CONTINUE

* check for repeated definitions and illegal redefinitions
	IF ( dup_name ) THEN
	   IF ( TM_GRID_MATCH( old_grid,mgrid_buff ) ) THEN
	      RETURN					! dup name/dup def
	   ENDIF
	ENDIF

* find a slot to store the new grid
	status = ALLO_MANAGED_GRID( grid )
	IF ( status .NE. merr_ok ) GOTO 5000

* save the grid and name it (8/99 - also bump use counts of dynamic axes)
 340	CALL TM_COPY_GRID_W_LINE_USE( mgrid_buff, grid )
	grid_name( grid ) = buff3

* purge old duplicate-name grid.  We deferred until after new static grid
* was defined so new dynamic axes weren't wiped out
	IF ( dup_name ) THEN
* ... redefinition makes all past assumptions incorrect
	   CALL SPLIT_LIST(pttmode_ops, ttout_lun,
     .			replmsg//'grid '//grid_name(old_grid), 0)
	   CALL PURGE_MR_GRID( old_grid, status )
	   IF ( status .NE. merr_ok ) GOTO 5000
* ... deallocate dynamic axes used in previous definition of this grid
	   DO 350 idim = 1,4
	      CALL TM_DEALLO_DYN_LINE( grid_line(idim,old_grid) )
 350	   CONTINUE
	   grid_name( old_grid ) = char_init16
	ENDIF

	RETURN

**************************************************************************
* DEFINE VARIABLE
* find name to define
* 2 syntaxes are allowed: "DEFINE VAR NAME=TEXT", "DEFINE VAR NAME TEXT"
* in the latter be careful that "TEXT" may contain "="
 400	IF ( num_args .LT. 1 ) GOTO 5300
	CALL NAME_EQUAL_STRING( buff1, pos, status )
	IF ( status .NE. ferr_ok )  GOTO 5000
	IF ( pos .EQ. 0 ) GOTO 5400
	IF ( .NOT.TM_LEGAL_NAME(buff1) ) GOTO 5410
        IF ( TM_LENSTR1(buff1) .GT. 128 ) GOTO 5410
	CALL FIND_VAR_NAME ( pdset_irrelevant, buff1, cat, var )
	IF (  var .NE. munknown_var_name
     .	.AND. cat .EQ. cat_pseudo_var    ) GOTO 5420

* /D=xxx: has the user requested a data-set-specific definition?
	iqual = qual_given( slash_let_dset )
	IF ( iqual .GT. 0 ) THEN
	   CALL EQUAL_STRING( cmnd_buff(qual_start(iqual):qual_end(iqual)),
     .			      buff3, status )
	   IF ( status .NE. ferr_ok ) RETURN
	   IF ( buff3 .EQ. ' ' ) THEN
	      dset = pdset_irrelevant	! "/D", alone for default-to-dset
	      buff3 = 'No current data set'	! ... for error reporting
	      IF ( dset .EQ. unspecified_int4) GOTO 5470
	   ELSE
	      dset = FIND_DSET_NUMBER( buff3 )
	      IF ( dset .EQ. unspecified_int4) GOTO 5470
	   ENDIF
	ELSE
	   dset = unspecified_int4
	ENDIF

* title given ?  (preserves lower case via EQUAL_STR_LC 3/91)
	buff3 = ' '
	iqual = qual_given( slash_let_title )
	IF ( iqual .GT. 0 ) THEN
	   CALL EQUAL_STR_LC( cmnd_buff(qual_start(iqual):qual_end(iqual)),
     .			      buff3, status )
	   IF ( status .NE. ferr_ok ) RETURN
	ENDIF

* units given ?
	buff2 = ' '
	iqual = qual_given( slash_let_units )
	IF ( iqual .GT. 0 ) THEN
	   CALL EQUAL_STR_LC( cmnd_buff(qual_start(iqual):qual_end(iqual)),
     .			      buff2, status )
	   IF ( status .NE. ferr_ok ) RETURN
	ENDIF

* bad flag given ?
	iqual = qual_given( slash_let_bad )
	IF ( iqual .GT. 0 ) THEN
	   CALL EQUAL_VAL( cmnd_buff(qual_start(iqual):qual_end(iqual)),
     .                     bad_flag, status )
	   IF ( status .NE. ferr_ok ) RETURN
	ELSE
	   bad_flag = bad_val4
	ENDIF

* break the expression into atoms and identify them

        t2 = cmnd_buff(pos:pos+1)
        IF (t2 .EQ. '..') cmnd_buff(pos:pos+1) = 'aa'
	CALL ALG_BREAK_UP( cmnd_buff(pos:len_cmnd),at_start,at_end,natom,
     .			   err_string,status )
        cmnd_buff(pos:pos+1) = t2

	IF ( status .NE. ferr_ok ) GOTO 5440
	CALL ALG_ID( cmnd_buff(pos:len_cmnd), natom, at_start, at_end,
     .		     at_type, at_id, status )
	IF ( status .NE. ferr_ok ) GOTO 5000

* if this definition replaces a variable of the same name in this data set
* then purge cached calculations (8/95)
	IF ( dset .GT. pdset_irrelevant ) THEN
	   CALL FIND_VAR_NAME ( dset, buff1, cat, var )
	   IF ( cat .EQ. cat_file_var
     .	  .AND. var .NE. munknown_var_name ) CALL PURGE_ALL_UVARS
	ENDIF

* if this is a global forground definition that replaces any variable
* then purge cached calculations (8/95)
	IF ( dset .EQ. unspecified_int4 ) THEN
	   CALL FIND_VAR_NAME ( dset, buff1, cat, var )
	   IF ( var .NE. munknown_var_name ) CALL PURGE_ALL_UVARS
	ENDIF

* do we need to delete a previous definition of this var ?
* Note: need to purge old defs ALSO if the same name is now redefined in a 
*	new data set -- expressions using this name chg interpretation  
*	DO 410 i1 = 1, max_uvar
*	   IF ( uvar_num_items(i1) .EQ. uvar_deleted ) GOTO 410
*	   vax_code = STR_CASE_BLIND_COMPARE( uvar_name_code(i1), buff1 )
*	   IF ( vax_code .EQ. vms_str_success ) THEN
        CALL string_array_find_caseblind(uvar_name_code_head,
     .                                   buff1,
     .                                   LEN(buff1),
     .                                   string_array_result(1),
     .                                   max_uvar,
     .                                   num_indices)        
        DO 410 j = 1, num_indices
              i1 = string_array_result(j)
              IF ( uvar_num_items(i1) .EQ. uvar_deleted ) GOTO 410
* ... delete only if same name in same data set
	      IF ( uvar_dset(i1) .EQ. dset ) THEN
!	         IF ( qual_given(slash_let_quiet) .EQ. 0 ) THEN
!	            err_string = FULL_UVAR_NAME(i1,i2)	! buffer
!	            CALL SPLIT_LIST(pttmode_ops, ttout_lun,
!     .			replmsg// 'variable '//err_string(:i2), 0)
!	         ENDIF
	         CALL DELETE_USER_VAR(i1)
!  pre-4/99	         uvar_num_items(i1) = uvar_deleted
	      ENDIF
* ... reuse of name  makes all past assumptions incorrect
	      CALL PURGE_ALL_UVARS	! not done above for pdset_irrelevant
*	   ENDIF
 410	CONTINUE

* now file the expression in the user-defined variable area
	CALL INIT_UVAR ( buff1, cmnd_buff(pos:len_cmnd), buff3, buff2, dset,
     .			 bad_flag, explct_defn, uvar,
     .			 at_type, at_id, at_start, at_end,
     .			 natom, 1, len_cmnd-pos+1, status )
	IF ( status .NE. ferr_ok ) RETURN

* is there at least one slot blank ?
	DO 430 i1 = 1, max_uvar
 430	IF ( uvar_num_items(i1) .EQ. uvar_deleted ) RETURN

* no - not allowed to fill the last slot !
*	uvar_num_items(uvar) = uvar_deleted
        CALL deleted_list_modify(uvar_num_items_head, uvar,
     .                       uvar_deleted )
	CALL ERRMSG( ferr_prog_limit, status,
     .		'too many user-defined variables'//pCR//
     .		'cancel or redefine some variables', *5000 )

**************************************************************************
* DEFINE AXIS
* ... /FILE=
 500	IF ( qual_given( slash_file ) .GT. 0 ) GOTO 300 ! like DEFINE GRID/FILE

* DEFINE AXIS - what is "axisname" ?
	iqual = qual_given( slash_name )
	IF ( iqual .GT. 0 ) THEN
	   CALL EQUAL_STRING( cmnd_buff(qual_start(iqual):qual_end(iqual)),
     .			      buff3, status )
	   IF ( status .NE. ferr_ok ) RETURN
	   pos = arg_start(1)	! cmnd arg is the expression to eval
	ELSE	   
	   IF ( num_args .LT. 1 ) GOTO 5300
	   CALL NAME_EQUAL_STRING( buff3, pos, status )
	   IF ( status .NE. ferr_ok )  GOTO 5000

	   IF ( pos .GT. 0 ) THEN

* consolidate arguments broken up by command parser back into a single arg

              IF (cmnd_buff(pos:pos) .EQ. '{') THEN
                 CALL all_1_arg
	         num_args = 1	 ! used by GET_CMND_DATA
              ENDIF
	      arg_start(1) = pos ! skip over "name ="
	   ELSE
	      CONTINUE ! pos=0 signals that no expression was given
	   ENDIF
	ENDIF

* DEFINE AXIS - modulo ?
	iqual = qual_given( slash_modulo )
        new_att_modulo = .FALSE.
        new_att_modulo_len = -1.
	its_modulo = iqual .GT. 0 
	IF ( its_modulo ) THEN
           new_att_modulo = .TRUE.
	   CALL EQUAL_VAL( cmnd_buff(qual_start(iqual):qual_end(iqual)),
     .			      rbuff, status )
	   IF ( status .NE. ferr_ok ) RETURN
	   IF (rbuff .EQ.  unspecified_val4) THEN
	      rmod_len = 0.0D0
	   ELSE
	      rmod_len = ABS(rbuff)  ! ABS to tolerate negatives
              new_att_modulo_len = rmod_len
	   ENDIF
	ELSE
	   rmod_len = 0.0D0
	ENDIF

* is this to be a regular or irregular definition?
	irreg = qual_given(slash_name) .GT. 0
     .	  .OR. (qual_given(slash_name) .EQ. 0
     .		.AND. pos .GT. 0)	! /NAME=XXX or "NAME= expression"

* must be a unique name
c	IF ( .NOT.TM_LEGAL_NAME( buff3 ) ) GOTO 5100
        slen = TM_LENSTR1( buff3)
        s1 = 1
        IF (buff3(1:1) .EQ. "'" .AND. buff3(slen:slen) .EQ. "'") THEN
           s1 = 2
           slen = slen - 1
        ENDIF
	old_line = TM_GET_LINENUM( buff3(s1:slen) )
	dup_name = old_line .NE. unspecified_int4

* get the X,Y,Z,T regions
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

* orientation must be uniquely defined
	orient = unspecified_int4
	DO 510 idim = 1, 4
	   IF ( qual_given(slash_x0+idim) .GT. 0 ) THEN
	      IF ( orient .NE. unspecified_int4 ) GOTO 5510
	      orient = idim
	   ENDIF
 510	CONTINUE
	IF ( irreg ) THEN
* ... orientation may be inferred from clues if irregular coord syntax is used
	    IF ( qual_given(slash_depth) .GT. 0 ) orient = z_dim
	    IF ( qual_given(slash_T0)    .GT. 0 ) orient = t_dim
	ENDIF
        IF ( qual_given(slash_depth) .GT. 0
     . .AND. orient .NE. z_dim            ) GOTO 5505

* if orientation is unclear then assume it is X
	IF ( orient .EQ. unspecified_int4 ) THEN
	   CALL WARN (
     .	     "Orientation not specified via /X,/Y,/Z, or /T. X assumed.")
	   orient = x_dim
	ENDIF

* release context stack space
	CALL STACK_PTR_DN ( cx_stack_ptr, cx_stack_ptr_base, status )
	IF ( status .NE. ferr_ok ) GOTO 5000

* /EDGES?
	its_edges =  qual_given( slash_edges ) .GT. 0

* /BOUNDS?
	has_bounds =  qual_given( slash_bounds ) .GT. 0
        IF (has_bounds .AND. its_edges) GOTO 5730

* /UNITS = DEGREES,METERS, ... SECONDS (or supply defaults)
        buff1 = ' '
        new_att_units = ' ' 
	iqual = qual_given( slash_units )
	IF ( iqual .GT. 0 ) THEN
	   CALL EQUAL_STRING( cmnd_buff(qual_start(iqual):qual_end(iqual)),
     .			      buff1, status )
	   IF ( status .NE. ferr_ok ) RETURN
           new_att_units = buff1

	ELSEIF ( cx_formatted(orient)
     .	   .AND. (orient.EQ.x_dim .OR. orient.EQ.y_dim) ) THEN
	   buff1 = 'degrees'	! infer from input like "X=130E:80W"
	ELSEIF ( orient .EQ. t_dim
* 8/97 *kob* use cx_calender to check if time is date rather than time step
     .  .AND.  (     qual_given(slash_T0) .GT. 0
     .		.OR. cx_calendar(mods_cx))) THEN
	   buff1 = 'HOURS'
	ELSE
	   buff1 = 'NONE'
	ENDIF

* is it a calendar axis definition?
	its_calendar =  orient.EQ.t_dim .AND. cx_calendar(mods_cx)

* *acm* other calendar names* /CALENDAR = GREGORIAN, NOLEAP, JULIAN, 360_DAY, ALL_LEAP
*       Gregorian is the default, also may be called STANDARD.
*       NOLEAP may also be called 365_DAY.
*       ALL_LEAP may also be called 366_DAY

        cal_name = 'GREGORIAN'		! default
        new_att_calendar = " "
        dflt_cal_id = TM_GET_CALENDAR_ID (cal_name)
	iqual = qual_given( slash_calendar )
	IF ( iqual .GT. 0 ) THEN
	   CALL EQUAL_STRING( cmnd_buff(qual_start(iqual):qual_end(iqual)),
     .			      cal_name, status )
	   IF ( status .NE. ferr_ok ) RETURN
           new_att_calendar = cal_name
	ELSEIF ( orient .EQ. t_dim  .AND.  
     .         ( qual_given(slash_T0) .GT. 0  .OR. 
     .           cx_calendar(mods_cx)) ) THEN
	   cal_name = 'GREGORIAN'
	ENDIF

* 9/01 CF conventions have adopted the name 360_day for a 360-day year.

        IF (cal_name .EQ. '360') cal_name = '360_DAY' 
        IF (cal_name .EQ. 'D360') cal_name = '360_DAY' 
        IF (cal_name .EQ. 'STANDARD') cal_name = 'GREGORIAN' 
        IF (cal_name(1:6) .EQ. 'COMMON') cal_name = 'NOLEAP' 
        IF (cal_name(1:6) .EQ. '365_DA') cal_name = 'NOLEAP' 
        IF (cal_name(1:6) .EQ. '366_DA') cal_name = 'ALL_LEAP' 
        cal_id = TM_GET_CALENDAR_ID (cal_name)
	IF (cal_id .LE. 0  .OR. cal_id .GT. max_calendars ) GO TO 5720
        CALL TM_GET_CALENDAR_ATTRIBUTES (cal_id, nmonths, 
     .                 ndays, yeardays, mon_names, d_in_mon, 
     .                 d_before_mon, mon_by_d)
        IF (new_att_calendar .NE. " ") new_att_calendar = cal_name

* interpret the line units
	IF ( buff1 .EQ. 'NONE' .OR. buff1 .EQ. ' ' ) THEN
	   buff1 = ' '
	   iunit = 0
	ELSE
	   iunit = TM_UNIT_ID( buff1 )
	   IF ( iunit .EQ. 0 ) THEN
	       CALL WARN('(fyi) units are not subject to auto-conversion: '
     .			//buff1)

C /calendar=/unit=year gets length of year in that calendar.
           ELSEIF (iunit .EQ. -6 .OR. iunit .EQ. -10) THEN  
	      IF (cal_id .EQ. d360) iunit = -9
	      IF (cal_id .EQ. julian) iunit = -12
	      IF (cal_id .EQ. noleap) iunit = -11
	      IF (cal_id .EQ. all_leap) iunit = -8

C /units=months  is 1/12 of the length of the year
	   ELSEIF (iunit .EQ. -5) THEN    ! 1/97: "-5" is "MONTHS"
              IF (cal_id .eq. GREGORIAN) THEN
                 WRITE (err_string,3544) yeardays
              ELSE
                 WRITE (err_string,3540) yeardays
                 IF (cal_id .eq. 2) THEN	! noleap
                    iunit = -13
                 ELSE IF (cal_id .eq. 3) THEN	! Julian
		    iunit = -15
		 ELSE IF (cal_id .eq. 4) THEN	! 360_DAY
		    iunit = -14
		 ELSE IF (cal_id .eq. 5) THEN	! ALL_LEAP
		    iunit = -16
	         ENDIF
*  Append calendar name to units string.
		 slen = TM_LENSTR1 (buff1)
		 len_cal = TM_LENSTR1 (cal_name)
		 buff1 = buff1(:slen)//'('//cal_name(:len_cal)//')'
              ENDIF
	      
 3540         FORMAT 
     .        ('/UNIT=MONTHS  ... using 1/12 of ',F6.2,' days')
 3544         FORMAT 
     .        ('/UNIT=MONTHS is ambiguous ... using 1/12 of ',F8.4,' days')
	         CALL WARN (err_string) 
	   ELSE
* ... Issue warning for inappropriate units like HOURS on X ax
*     and render the units unconvertible
 	      IF ( (  orient.EQ.t_dim .AND.
     .		   ( iunit.LT.pun_1st_time .OR. iunit.GT.pun_last_time ) )
     .	      .OR. (  orient.NE.t_dim .AND.
     .		   ( iunit.LT.pun_1st_len  .OR. iunit.GT.pun_last_len  ) )
     .	      .OR. (  orient.EQ.z_dim .AND.
     .	             iunit.EQ.pun_degrees )  ) THEN	! 7/96
	         CALL WARN
     .		   ('Units appear to conflict with orientation:'
     .			//' no conversions possible')
	         iunit = 0
	      ENDIF	      
	   ENDIF
 	ENDIF

* find a space to catalog it
	status = ALLO_MANAGED_AXIS( iline )
	IF ( status .NE. merr_ok ) GOTO 5000

* was /NPOINTS given ?
	iqual = qual_given( slash_npoints )
	IF ( iqual .GT. 0 ) THEN
	   CALL EQUAL_val( cmnd_buff(qual_start(iqual):qual_end(iqual)),
     .			      rbuff, status )
	   IF ( status .NE. ferr_ok ) RETURN
	   IF ( rbuff .LE. 0.0 .OR. rbuff .GT. 1E6 ) GOTO 5582
	   npoints = rbuff
	ELSE
	   npoints = unspecified_int4
	ENDIF

* GET THE AXIS POINTS: IRREGULAR (from_variable) or regular
	IF ( irreg ) THEN
           line_regular( iline ) = .FALSE.  ! this may be reset to true later.
* ... get axis points from a memory variable
	   CALL GET_CMND_DATA ( memory, cx_last, ptype_float, status )
	   IF ( status .NE. ferr_ok ) RETURN
           IF ( num_uvars_in_cmnd .NE. 1 .AND.
     .         (.NOT. has_bounds) )GOTO 5580
           cx = is_cx(1)
* ... make sure it's an line (not a point,plane,etc.) of data
	   CALL GET_CX_DIMS( cx, ndim, dim )
	   IF ( ndim .GT. 1 ) GOTO 5580
	   idim = dim(1)
	   n    = CX_DIM_LEN( idim, cx )

* ... Checks on the bounds specification
           IF (has_bounds) THEN
              its_reg = .FALSE.
	      IF ( num_uvars_in_cmnd .LT. 2 .OR. 
     .             num_uvars_in_cmnd .GT. 3) GOTO 5580
              cx = is_cx(2)
	      mr = is_mr(2)
* ... make sure it's an line (not a point,plane,etc.) of data
	      CALL GET_CX_DIMS( cx, ndim, dim )
	      IF ( ndim .GT. 1 ) GOTO 5580
* ... check dimension if given as coords, lo_bnds, hi_bnds
              cx = is_cx(3)
	      mr = is_mr(3)
	      CALL GET_CX_DIMS( cx, ndim, dim )
	      IF ( ndim .GT. 1 ) GOTO 5580
* ... make sure the bounds have dimension 2*N or N+1
              idim = dim(1)
              n2   = CX_DIM_LEN( idim, is_cx(2) )
              IF (num_uvars_in_cmnd .EQ. 3) 
     .           n3 = CX_DIM_LEN( idim, is_cx(3) )
              IF (n2 .EQ. 2*n) THEN
                 n2 = 2*n
              ELSEIF (n2 .EQ. n+1) THEN
                 n2 = n+1
              ELSEIF (num_uvars_in_cmnd .EQ. 3) THEN
                 IF (n2 .NE. n3) GOTO 5745
                 IF (n2 .NE. n) GOTO 5745
              ELSE
                 GOTO 5740
              ENDIF
	   ENDIF

* ... reset cx to read the coordinates first
           cx = is_cx(1)
	   mr = is_mr(1)
	   CALL GET_CX_DIMS( cx, ndim, dim )
	   idim = dim(1)
	   n    = CX_DIM_LEN( idim, cx )

* ... original start of free line memory
	   frst_pt = next_line_mem_pos
* ... allocate some storage for the coordinates
	   IF (frst_pt+3*n .GT. maxlinestore) GOTO 5590

* ... store coordinate positions (error status already checked)
           IF (has_bounds) THEN
              CALL NON_ARRAY_SUBSC( is_mr, 2 )
           ELSE
              CALL NON_ARRAY_SUBSC( mr, 1 )
           ENDIF

	   IF (its_edges) THEN
* ... "/edges" definition - 4/99
	      n = n - 1  ! edges array is one longer than points
	      IF (n .LE. 0) GOTO 5520
	      CALL EXTRACT_DP_LINE( cx, memory(1, mr_blk1(mr)),
     .				 line_mem(frst_pt+n), idim, n+1 )
	      npoints = n
	      micro_adj = 1E-6 *
     .		(line_mem(frst_pt+n+n) - line_mem(frst_pt+n))
* ... are there repeated values?
	      has_repeated = .FALSE.
	      DO 515 i1 = frst_pt+1, frst_pt+n
	         IF      (line_mem(i1+n-1) .GT. line_mem(i1+n)) THEN
	           GOTO 5570
	         ELSEIF  (line_mem(i1+n-1) .EQ. line_mem(i1+n)) THEN
	           has_repeated = .TRUE.
		 ENDIF
 515	      CONTINUE
* ... make micro-adjustments if axis has repeated values (4/99)
	      IF ( has_repeated ) THEN
	         CALL WARN('Axis has repeated values -- micro-adjusting ...')
	         IF (micro_adj .EQ. 0.D0) GOTO 5560
	         DO 526  i1 = frst_pt+n+1, frst_pt+n+n
	             IF  (line_mem(i1-1) .EQ. line_mem(i1)) THEN
*    ... locate end of repeating block
	                DO 520 i2 = i1, frst_pt+n+n
	                   IF (line_mem(i2) .GT. line_mem(i1)) GOTO 521
 520		        CONTINUE
	                i2 =  frst_pt+n+n+1    ! one beyond last value
*    ... micro-adjust repeating values and edges from i1 to i2-1
 521		        DO 522 i3 = i1, i2-1
	                   line_mem(i3) = line_mem(i3) +
     .                         (i3-i1+1)*micro_adj
 522		        CONTINUE
	                IF ( i2 .LE. frst_pt+n+n
     .		       .AND. (line_mem(i2-1) .GE. line_mem(i2))) GOTO 5560
	             ENDIF
 526	          CONTINUE
	      ENDIF
* ... store points: midway between box boundaries ...
	      has_repeated = .FALSE.
	      DO 528 i1 = frst_pt+1, frst_pt+n
	         line_mem(i1-1) = (line_mem(i1+n-1)+line_mem(i1+n)) * 0.5
 528	      CONTINUE
	      its_reg = .TRUE.	! until proven otherwise
	      start = line_mem(frst_pt+n)
              delta = line_mem(frst_pt+n+1) - start
              DO 530 i1 = frst_pt+n+2,frst_pt+n+n+1
                 IF (.NOT.TM_FPEQ( SNGL(delta),
     .               SNGL(line_mem(i1)-line_mem(i1-1))) )
     .						its_reg = .FALSE.
 530	      CONTINUE

	   ELSE

* NOT EDGES -- POINTS

	      CALL EXTRACT_DP_LINE( cx, memory(1, mr_blk1(mr)),
     .				 line_mem(frst_pt), idim, n )
	      IF ( npoints .EQ. unspecified_int4 ) npoints = n
* ... a single point is always "regular" (unless definition has BOUNDS!!)
	      IF (n .EQ. 1 .AND. .NOT.has_bounds) THEN
	         start = line_mem(frst_pt)
	         delta = 1
	         its_reg = .TRUE.
	         GOTO 560
	      ENDIF
	      micro_adj = 1E-6 *
     .		(line_mem(frst_pt+n-1) - line_mem(frst_pt))
* ... are there repeated values?
	      has_repeated = .FALSE.
	      DO 532 i1 = frst_pt+1, frst_pt+npoints-1
	         IF      (line_mem(i1-1) .GT. line_mem(i1)) THEN
	            GOTO 5570
	         ELSEIF  (line_mem(i1-1) .EQ. line_mem(i1)) THEN
	            has_repeated = .TRUE.
	         ENDIF
 532	      CONTINUE
* ... make micro-adjustments if axis has repeated values (4/99)
	      IF ( has_repeated ) THEN
	         CALL WARN('Axis has repeated values -- micro-adjusting ...')
	         IF (micro_adj .EQ. 0.D0) GOTO 5560
	         DO 538  i1 = frst_pt+1, frst_pt+npoints-1
	            IF  (line_mem(i1-1) .EQ. line_mem(i1)) THEN
*    ... locate end of repeating block
	               DO 534 i2 = i1, frst_pt+npoints-1
	                  IF (line_mem(i2) .GT. line_mem(i1)) GOTO 535
 534		       CONTINUE
	                i2 =  frst_pt+npoints    ! one beyond last value
*    ... micro-adjust repeating values from i1 to i2-1
 535		        DO 536 i3 = i1, i2-1
	                   line_mem(i3) = line_mem(i3) +
     .	                     (i3-i1+1)*micro_adj
 536		        CONTINUE
	                IF ( i2 .LT. frst_pt+npoints .AND. 
     .		        (line_mem(i2-1) .GE. line_mem(i2))) GOTO 5560
	             ENDIF
 538	          CONTINUE
	      ENDIF

* Bounds specified by user

              IF (has_bounds) THEN

                 cx = is_cx(2)
                 mr = is_mr(2)

	         CALL EXTRACT_DP_LINE2( cx, memory(1, mr_blk1(mr)),
     .	  	    line_mem(next_line_mem_pos + n), idim, n2 )

* If given as coords, lo_bounds, hi_bounds, then
* 1) check that lo_bounds{i+1} = hi_bounds{i}
* 2) If ok, add the upper high bound to the list and treat 
*    this as the case where we give the bounds as N+1 values

                 IF (num_uvars_in_cmnd .EQ. 3) THEN
                    cx = is_cx(3)
                    mr = is_mr(3)
                    CALL EXTRACT_DP_LINE2( cx, memory(1, mr_blk1(mr)),
     .	 	       line_mem(next_line_mem_pos + n+n2), idim, n3 )
                    DO i = 1, n3-1
                       IF (line_mem(next_line_mem_pos + n+i) .NE.
     .                      line_mem(next_line_mem_pos + 2*n+i-1) ) 
     .                     GOTO 5750
                    ENDDO
                    line_mem(next_line_mem_pos + 2*n) = 
     .                    line_mem(next_line_mem_pos + 3*n-1)
                    n2 = n+1
                 ENDIF

* Check that bounds enclose points, and that they are contiguous
	         line_subsc1 ( iline ) = frst_pt
                 line_dim    ( iline ) = npoints
                 IF (.NOT. TM_CHECK_BNDS (iline, n2, buff3) ) GO TO 5750

              ELSE

*... create bounds equidistant between grid points

*    ... store initial box lower bound (start of axis)
	         line_mem(frst_pt+npoints) = line_mem(frst_pt) -
     .			0.5 * ( line_mem(frst_pt+1) - line_mem(frst_pt) )
*    ... mid-axis box boundaries ...
	         DO 539 i1 = frst_pt+1, frst_pt+npoints-1
	            line_mem(i1+npoints) = (line_mem(i1-1)+line_mem(i1)) * 0.5
 539	         CONTINUE
*    ... upper-most box bound (end of axis)
	         i2 = frst_pt+npoints-1	! last box point
	         line_mem(i2+npoints+1) = line_mem(i2) +
     .			0.5 * ( line_mem(i2) - line_mem(i2-1) )
	         its_reg = .TRUE.	! until proven otherwise
	         start = line_mem(frst_pt)
                 delta = line_mem(frst_pt+1) - start
                 DO 550 i1 = frst_pt+2,frst_pt+npoints-1
                    IF (.NOT.TM_FPEQ( SNGL(delta),
     .                  SNGL(line_mem(i1)-line_mem(i1-1))) )
     .                       its_reg = .FALSE.
 550	         CONTINUE
              ENDIF

	   ENDIF

* ... check to see if it was regularly spaced after all
 560	   IF ( its_reg ) THEN
	      line_start    ( iline ) = start
	      line_delta    ( iline ) = delta
	      line_regular  ( iline ) = .TRUE.
	      line_subsc1   ( iline ) = unspecified_int4
	   ELSE	      
* ... save irregular line data
	      line_subsc1 ( iline ) = frst_pt
	      line_start  ( iline ) = line_mem(frst_pt)
	      line_delta  ( iline ) = unspecified_val8
              line_dim    ( iline ) = npoints   ! need this here so TM_CHECK_BNDS works.
	      line_regular( iline ) = .FALSE.

	   ENDIF
	ELSE

* ... start,end,delta specification for a REGULAR axis.  
* acm These are properties of the line, so reset according to the calendar.
*     Need to do this only when it came in as a date string; these are flagged
*     as negative in TRANSLATE_LIMIT

	   start = cx_lo_ww(orient,mods_cx)
	   end   = cx_hi_ww(orient,mods_cx)
           prev_cal_id = cx_cal_id(mods_cx)
           IF (prev_cal_id .EQ. unspecified_int4) prev_cal_id = 1

	   IF ( orient .EQ. t_dim  .AND.  cal_id .NE. prev_cal_id .AND.
     .          start .LT. 0.  .AND.  start .NE. unspecified_val8.AND.
     .          end   .LT. 0.  .AND.  end   .NE. unspecified_val8) THEN
	      CALL RESET_SECS_FROM_BC (-start, new_ww, prev_cal_id, 
     .                                 cal_id, status)
	      start = -1. * new_ww

	      CALL RESET_SECS_FROM_BC (-end, new_ww, prev_cal_id, 
     .                                 cal_id, status)
	      IF ( status .NE. ferr_ok ) RETURN

	      end = -1. * new_ww
              cx_cal_id(mods_cx) = cal_id
	   ENDIF

	   IF ( its_edges ) THEN
	     IF ( npoints .EQ. unspecified_int4 ) THEN
	       delta = cx_delta (orient,mods_cx)
	       IF ( delta.EQ.unspecified_val8 .OR. delta.LE.0.D0) GOTO 5520
	       IF ( .NOT.its_calendar ) THEN
	          npoints = ( (end - start) / delta ) + 0.9999
	       ENDIF
	     ELSE
	       delta = (end - start) / npoints
	       IF (its_calendar) delta = -1 * delta/un_convert(iunit)
	       IF ( delta .LE. 0.0D0 ) GOTO 5520
	     ENDIF
	     IF ( its_calendar ) THEN
	       start = start - delta/2.0*un_convert(iunit)
	       end   = end   + delta/2.0*un_convert(iunit)
	     ELSE
	       start = start + delta/2.0
	       end   = end   - delta/2.0
	     ENDIF
	   ELSE
	     IF ( npoints .EQ. unspecified_int4 ) THEN
	       delta = cx_delta (orient,mods_cx)
	       IF ( delta.EQ.unspecified_val8 .OR. delta.LE.0.D0) GOTO 5520
	       IF ( .NOT.its_calendar ) THEN
	          npoints = INT(( (end - start) / delta ) + 1.9999)
                  small = MIN(1.e-9, delta)
		  IF (start + delta* (npoints-1) - 0.5*delta .GT. end+small) 
     .              npoints = npoints-1
	       ENDIF
	     ELSEIF ( npoints .EQ. 1 ) THEN
	       IF ( end .NE. start ) GOTO 5530
	       delta = 1.D0
	     ELSE
	       delta = (end - start) / (npoints-1)
	       IF (its_calendar) delta = -1 * delta/un_convert(iunit)
	       IF ( delta .LE. 0.0D0 ) GOTO 5520
	     ENDIF
	   ENDIF

	   line_start    ( iline ) = start
	   line_delta    ( iline ) = delta
	   line_regular  ( iline ) = .TRUE.
	   line_subsc1   ( iline ) = unspecified_int4
	ENDIF

* /T0 = date (or supply default)
	iqual = qual_given( slash_T0 )
        new_att_t0 = " "
	IF ( iqual .GT. 0 ) THEN
	   CALL EQUAL_STRING( cmnd_buff(qual_start(iqual):qual_end(iqual)),
     .			      buff2, status )
	   IF ( status .NE. ferr_ok ) RETURN
	   CALL TM_BREAK_DATE( buff2, cal_id, i1,i2,i3,i4,i5,i6, 
     .                         status )
	   IF ( buff2 .EQ. ' ' .OR. status .NE. ferr_ok ) GOTO 5540
           new_att_t0 = buff2
	ELSE
	   WRITE (buff2,3550) mon_names(1)
 3550      FORMAT ('15-', A3, '-1901')		! default
	   IF (its_calendar ) THEN
	      i1 = 20
 580	      bc_to_t0 = SECS_FROM_BC( buff2, cal_id, status )      
	      IF ( status .NE. ferr_ok ) GOTO 5540
	      IF ( bc_to_t0 + start .GT. 0 ) THEN	! start is before T0?
	         IF ( i1 .LT. 0 ) GOTO 5540
	         i1 = i1 - 1
	         WRITE (buff2,3560) mon_names(1), i1
 3560	         FORMAT('01-',A3,'-',I2.2,'00')
	         GOTO 580
	      ENDIF
	   ENDIF
	ENDIF

* Combine new attribute units and/or time origin to express
* the units attribute as units since time-origin

        IF (orient .EQ. t_dim) THEN
           IF (new_att_units .NE. ' ') THEN
              i1 = TM_LENSTR1(new_att_units)
              new_att_units = new_att_units(1:i1)//' since ' 
     .            // new_att_t0
              IF (new_att_t0 .EQ. ' ') 
     .            new_att_units = new_att_units(1:i1)//' since ' 
     .            // line_t0(iline)
           ELSE
              IF (new_att_t0 .NE. ' ') THEN
                 i1 = TM_LENSTR1(line_units(iline))
                 new_att_units = line_units(iline)(1:i1)//' since ' 
     .            // new_att_t0
              ENDIF
           ENDIF
        ENDIF

* save the line in memory - but don't catalog its name
	line_dim      ( iline ) = npoints
	line_units    ( iline ) = buff1	   
	line_t0       ( iline ) = buff2
	line_cal_name ( iline ) = cal_name
	line_unit_code( iline ) = iunit
	line_direction( iline ) = 'NA'
	IF ( orient .EQ. t_dim ) line_tunit(iline) = un_convert( iunit )

* is it a formattable axis ?
	IF ( orient .EQ. x_dim ) THEN
	   line_direction( iline ) = 'XX'
           IF (iunit .EQ. pun_degrees) line_direction( iline ) = 'WE'

	ELSEIF ( orient .EQ. y_dim  ) THEN
           line_direction( iline ) = 'YY'
	   IF (iunit .EQ. pun_degrees) line_direction( iline ) = 'SN'

	ELSEIF ( orient .EQ. z_dim ) THEN
           IF ( qual_given( slash_depth ) .GT. 0 ) THEN
	      line_direction( iline ) = 'UD'    ! GEOG_LABEL requires "meters"
           ELSE
	      line_direction( iline ) = 'DU'
           ENDIF

* ... time axis specified as date strings  (ELSEIF mod 11/93 *sh*)
	ELSEIF ( .NOT.irreg  .AND. its_calendar ) THEN
	   line_direction(iline) = 'TI'
	   bc_to_t0 = SECS_FROM_BC( line_t0(iline), cal_id, status)
	   line_dim  (iline) = (start-end) / (line_tunit(iline)*delta) + 1.9999
	   line_start(iline) = (-line_start(iline)-bc_to_t0)/line_tunit(iline)

* ... time axis specified as time step values with T0 given
	ELSEIF ( orient.EQ.t_dim .AND. qual_given(slash_T0).GT.0 ) THEN
	   line_direction(iline) = 'TI'
	ELSEIF ( orient.EQ.t_dim .AND. line_direction(iline) .EQ. 'NA') THEN
	   line_direction(iline) = 'TT'
	ENDIF

* ... coordinate pointer for next line to save
* Bounds are stored as edges, so increase by N coordinates and N+1 edges

* acm move this to before purge_mr_axis, which also resets next_line_mem_pos
*     after it deletes old_line.

	IF (irreg .AND. .NOT.line_regular(iline))
     .			next_line_mem_pos = frst_pt + 2*n+1

* automatic longitude modulo
	axwwlen = TM_WW_AXLEN(iline)
	IF (line_direction(iline) .EQ. 'WE' ) THEN
	  IF (axwwlen .LE. 360.D0) THEN
	     its_modulo = .TRUE.
	     IF (rmod_len .EQ. 0.0D0)  rmod_len = 360.D0
	     new_att_modulo = .TRUE.
	     new_att_modulo_len = rmod_len

	  ENDIF
*  automatic calendar modulo
	ELSEIF (line_direction(iline) .EQ. 'TI' ) THEN
	  secsperyear = cals_yeardays(cal_id) * (24 * 60 * 60)
	  bc_to_t0 = SECS_FROM_BC( line_t0(iline), cal_id, status)
	  CALL TM_WW_AX_1_N(iline, start, end)
	  secs2start = bc_to_t0 + start*line_tunit(iline)
	  IF ( (secs2start .LE. secsperyear)
     .	 .AND. (axwwlen*line_tunit(iline) .LE. secsperyear) ) THEN
	     its_modulo = .TRUE.
	     IF (rmod_len .EQ. 0.0D0)  rmod_len =
     .				       secsperyear/line_tunit(iline)
	     new_att_modulo = .TRUE.
	     new_att_modulo_len = rmod_len
	  ENDIF
	ENDIF
* set modulo and sanity check it
	line_modulo( iline) = its_modulo
	line_modulo_len( iline ) = rmod_len
	IF (rmod_len .NE. 0.0D0) THEN
	   IF (axwwlen .GT. rmod_len) THEN
             IF (TM_DFPEQ(axwwlen, rmod_len)) THEN
	       line_modulo_len( iline ) = 0.0D0
             ELSE
               GOTO 5525
	     ENDIF
           ENDIF
	ENDIF


* check for redefinitions
	IF ( dup_name ) THEN
	   IF ( TM_LINE_MATCH( old_line, iline ) ) THEN
	      RETURN					! dup name/dup def
	   ELSE
* ... redefinition makes all past assumptions incorrect
	      CALL SPLIT_LIST(pttmode_ops, ttout_lun,
     .			replmsg//'axis '//line_name( old_line ), 0)

* ... Check the attribute structure for this axis name and change 
*     attribute values as needed.

	      CALL REDEFINE_AX_ATT (line_name(old_line), new_att_units, 
     .           new_att_t0, new_att_calendar, new_att_modulo,
     .           new_att_modulo_len, line_direction(iline), 
     .           line_regular(iline) )

	      CALL PURGE_MR_AXIS( old_line, iline, status )
	      IF ( status .NE. ferr_ok ) GOTO 5000
	      CALL TM_DEALLO_DYN_LINE( old_line )

	   ENDIF

        ELSE

* add newly defined axis to the user-variable dset of the linked-list strucure
           
           dset_to_add = -2  ! user-defined coordinate variable dataset.

           err_string = buff3
           type = NCDOUBLE
           coordvar = 1
           varid = 0        ! Will be returned with variable ID
           bad_flag = bad_val4   ! coord vars dont have missing, but set to something
           CALL CD_ADD_COORD_VAR( dset_to_add, buff3, varid,  
     .         line_units(iline), type, coordvar, bad_flag, status)
           IF ( status .NE. ferr_ok ) GOTO 5600
         
* Get direction (also used to set axis direction, below). Set the
* point_spacing attribute. If the axis is a time axis, by default this
* is not written to the file as the axis may be the record axis.

           dcode = line_direction(iline)
           IF (dcode .EQ. 'XX' .OR. dcode .EQ. 'WE') dir = 1
           IF (dcode .EQ. 'YY' .OR. dcode .EQ. 'SN') dir = 2
           IF (dcode .EQ. 'ZZ' .OR. dcode .EQ. 'UD' .OR. dcode .EQ. 'DU') dir = 3
           IF (dcode .EQ. 'TT' .OR. dcode .EQ. 'TI') dir = 4

           attname = 'point_spacing'
           attype = NCCHAR
           attoutflag = 1
           IF ( line_regular(iline) ) THEN
              attlen = 4
              CALL CD_PUT_NEW_ATTR (dset_to_add, varid, attname, 
     .        attype, attlen, attoutflag, 'even', dummy, status) 
           ELSE
              attlen = 6
              CALL CD_PUT_NEW_ATTR (dset_to_add, varid, attname, 
     .        attype, attlen, attoutflag, 'uneven', dummy, status) 
           ENDIF

* Axis attribute, with coordinate direction

           attname = 'axis'
           attype = NCCHAR
           attoutflag = 1

           dcode = line_direction(iline)
           IF (dcode .EQ. 'XX' .OR. dcode .EQ. 'WE') axdir = 'X'
           IF (dcode .EQ. 'YY' .OR. dcode .EQ. 'SN') axdir = 'Y'
           IF (dcode .EQ. 'ZZ' .OR. dcode .EQ. 'UD' .OR. 
     .                              dcode .EQ. 'DU') axdir = 'Z'
           IF (dcode .EQ. 'TT' .OR. dcode .EQ. 'TI') axdir = 'T'
           attlen = 1
           CALL CD_PUT_NEW_ATTR (dset_to_add, varid, attname, 
     .           attype, attlen, attoutflag, axdir, dummy, status) 

* Set the axis direction

           dir = 0
           IF (axdir .EQ. 'X') dir = 1
           IF (axdir .EQ. 'Y') dir = 2
           IF (axdir .EQ. 'Z') dir = 3
           IF (axdir .EQ. 'T') dir = 4
           CALL CD_SET_ATT_AXDIR (dset_to_add, varid, dir, status)

* Bounds on irregular axis output by default.

           IF (irreg) THEN
              attname = 'bounds'
              attype = NCCHAR
              attoutflag = 1

              attlen = TM_LENSTR1(buff3)
              attlen = attlen + 5
              buff2 = buff3(:slen)//'_bnds'
              CALL CD_PUT_NEW_ATTR (dset_to_add, varid, attname, 
     .           attype, attlen, attoutflag, buff2, dummy, status) 
           ENDIF

* Modulo attribute

           IF ( line_modulo(iline) ) THEN
              attname = 'modulo'
              attype = NCDOUBLE
              attoutflag = 1
              attlen = 1
              CALL CD_PUT_NEW_ATTR_DP (dset_to_add, varid, attname, 
     .           attype, attlen, attoutflag, buff1, line_modulo_len(iline), status) 
           ENDIF

* Positive down. 

           IF (line_direction(iline) .EQ. 'UD') THEN
              attname = 'positive'
              attype = NCCHAR
              attoutflag = 1
              attlen = 4
              CALL CD_PUT_NEW_ATTR (dset_to_add, varid, attname, 
     .           attype, attlen, attoutflag, 'down', dummy, status) 
           ENDIF

* Do not write positive=up. This is the default Ferret behavior
           IF (line_direction(iline) .EQ. 'DU') THEN
              attname = 'positive'
              attype = NCCHAR
              attoutflag = 0
              attlen = 2
              CALL CD_PUT_NEW_ATTR (dset_to_add, varid, attname, 
     .           attype, attlen, attoutflag, 'up', dummy, status) 
           ENDIF

           IF (line_direction(iline)(1:1) .EQ. 'T') THEN
              IF (its_calendar) THEN
                 attname = 'calendar'
                 attype = NCCHAR
                 attoutflag = 1
                 attlen = TM_LENSTR1(cal_name)

* By default if an axis is on the standard Gregorian calendar we do
* not write out the calendar attribute
                 IF (cal_id .eq. GREGORIAN) attoutflag = 0
                 CALL CD_PUT_NEW_ATTR (dset_to_add, varid, attname, 
     .             attype, attlen, attoutflag, cal_name, 
     .             dummy, status) 

                 attname = 'time_origin'
                 attype = NCCHAR
                 attoutflag = 1
                 attlen = TM_LENSTR1(line_t0(iline))
                 CALL CD_PUT_NEW_ATTR (dset_to_add, varid, attname, 
     .               attype, attlen, attoutflag, line_t0(iline), 
     .               dummy, status) 

              ENDIF
 
           ENDIF

	ENDIF

* now catalog it in memory by saving its name
	line_name( iline ) = buff3
	RETURN

**************************************************************************
* DEFINE VIEWPORT
 600	IF ( num_args .NE. 1 ) GOTO 5300
* check out viewport name
	buff1 = cmnd_buff( arg_start(1):arg_end(1) )
	ivp = VIEWPORT_NUMBER( buff1 )
	IF ( ivp .EQ. unspecified_int4 ) THEN
* ... new name - find an unused slot
	   DO 610 ivp = 1, max_viewport
 610	   IF ( vp_name(ivp) .EQ. unspecified_name4 ) GOTO 620
	   GOTO 5610	! no free slots
	ELSE
* ... name in use - flag segments in this viewport as deleted
	   IF ( ivp .EQ. mvp_dflt ) GOTO 5640
	   CALL SPLIT_LIST(pttmode_ops, ttout_lun,
     .			'Re-defining viewport '//vp_name(ivp), 0)
	   DO 614 iseg = vp_seg0(ivp)+1 , vp_segn(ivp)
 614	   CALL GDSG( iseg ) 
	   vp_seg0 ( ivp ) = 0
	   vp_segn ( ivp ) = 0
	ENDIF

* pre-set all values to defaults
 620	vp_size ( ivp ) = 1.0
	vp_xorg ( ivp ) = 0.0
	vp_yorg ( ivp ) = 0.0
	vp_xclip( ivp ) = unspecified_val4   ! old style  3/29/93
	vp_yclip( ivp ) = unspecified_val4   ! old style  3/29/93
	vp_seg0 ( ivp ) = 0
	vp_segn ( ivp ) = 0

* /AXES
* 10/01 *kob* - add ".GT. 0" in below logical check
	vp_by_axis(ivp) = qual_given( slash_define_vp_by_ax ) .GT. 0

***************** OLD CODE (retained for greater upwards compatibility)
* /ORIGIN = x,y
	qp = qual_given( slash_origin )
	IF ( qp .GT. 0 ) THEN
	   IF (vp_by_axis(ivp)) GOTO 5660
	   CALL EQUAL_STRING( cmnd_buff(qual_start(qp):qual_end(qp)),
     .			   buff2, status )
	   IF ( status .NE. ferr_ok ) RETURN
	   IF ( .NOT.TM_DIGIT(buff2) .OR. INDEX(buff2,',').EQ.0 ) GOTO 5630
	   READ ( buff2, *, ERR=5630 ) vp_xorg(ivp), vp_yorg(ivp)
	   IF ( OUTSIDE_0_1(vp_xorg(ivp)) .OR. OUTSIDE_0_1(vp_yorg(ivp)) )
     .	      							   GOTO 5620
	ENDIF

* /SIZE=q
	qp = qual_given( slash_size )
	IF ( qp .GT. 0 ) THEN
	   IF (vp_by_axis(ivp)) GOTO 5660
	   CALL EQUAL_VAL( cmnd_buff(qual_start(qp):qual_end(qp)),
     .			   vp_size(ivp), status )
	   IF ( status .NE. ferr_ok ) RETURN
	   IF ( vp_size(ivp) .LT. 0.0 ) GOTO 5620
           vp_size(ivp) = -1. * vp_size(ivp)    ! flag for old-style viewports
           CALL WARN (
     .          'The /SIZE qualifier will force Ver. 2.2 behavior'
     .	        //' for DEFINE VIEWPORT')
	ENDIF

* /CLIP = x,y
	qp = qual_given( slash_clip )
	IF ( qp .GT. 0 ) THEN
	   IF (vp_by_axis(ivp)) GOTO 5660
	   CALL EQUAL_STRING( cmnd_buff(qual_start(qp):qual_end(qp)),
     .			   buff2, status )
	   IF ( status .NE. ferr_ok ) RETURN
	   IF ( .NOT.TM_DIGIT(buff2) .OR. INDEX(buff2,',').EQ.0 ) GOTO 5630
	   READ ( buff2, *, ERR=5630 ) vp_xclip(ivp), vp_yclip(ivp)
	   IF ( vp_xclip(ivp).LE.vp_xorg(ivp) .OR. vp_xclip(ivp).GT.1.0
     .	   .OR. vp_yclip(ivp).LE.vp_yorg(ivp) .OR. vp_yclip(ivp).GT.1.0 )
     .	      							   GOTO 5620
	ENDIF

***************** END OF OLD CODE (retained for greater upwards compatibility)

* default clipping at 1,1 in new style viewports  (immaterial, I think ...)
	IF ( qual_given(slash_size) .EQ. 0 ) THEN
	   IF ( vp_xclip(ivp).EQ.unspecified_val4) THEN
	      vp_xclip(ivp) = 1.0
	      vp_yclip(ivp) = 1.0
	   ENDIF
	ENDIF

* /XLIMITS = xlow,xhigh 
	qp = qual_given( slash_xlimits )
	IF ( qp .GT. 0 ) THEN
	   CALL EQUAL_STRING( cmnd_buff(qual_start(qp):qual_end(qp)),
     .			   buff2, status )
	   IF ( status .NE. ferr_ok ) RETURN
	   i1 = TM_LENSTR1(buff2)
	   pos = MAX( INDEX(buff2,','), INDEX(buff2,':') )
	   IF ( pos.LE.1 .OR. pos.EQ.i1 ) GOTO 5630
	   IF (.NOT.TM_DIGIT(buff2) )  GOTO 5630
	   READ ( buff2(1:pos-1), *, ERR=5630 ) vp_xorg(ivp)
	   IF (.NOT.TM_DIGIT(buff2(pos+1:i1))  ) GOTO 5630
	   READ ( buff2(pos+1:i1), *, ERR=5630 ) vp_xclip(ivp)
	   IF ( OUTSIDE_0_1(vp_xorg(ivp))
     .     .OR. OUTSIDE_0_1(vp_xclip(ivp))
     .     .OR. vp_xorg(ivp) .GE. vp_xclip(ivp) )  GOTO 5620
	ENDIF

* /YLIMITS = ylow,yhigh 
	qp = qual_given( slash_ylimits )
	IF ( qp .GT. 0 ) THEN
	   CALL EQUAL_STRING( cmnd_buff(qual_start(qp):qual_end(qp)),
     .			   buff2, status )
	   IF ( status .NE. ferr_ok ) RETURN
	   pos = MAX( INDEX(buff2,','), INDEX(buff2,':') )
	   i1 = TM_LENSTR1(buff2)
	   IF ( pos.LE.1 .OR. pos.EQ.i1 ) GOTO 5630
	   IF (.NOT.TM_DIGIT(buff2) )  GOTO 5630
	   READ ( buff2(:pos-1), *, ERR=5630 ) vp_yorg(ivp)
	   IF (.NOT.TM_DIGIT(buff2(pos+1:i1))  ) GOTO 5630
	   READ ( buff2(pos+1:i1), *, ERR=5630 ) vp_yclip(ivp)
	   IF ( OUTSIDE_0_1(vp_yorg(ivp))
     .     .OR. OUTSIDE_0_1(vp_yclip(ivp))
     .     .OR. vp_yorg(ivp) .GE. vp_yclip(ivp) )  GOTO 5620
	ENDIF

* /TEXT=q 
	qp = qual_given( slash_text )
	IF ( qp .GT. 0 ) THEN
	   IF ( qual_given(slash_size) .GT. 0 ) GOTO 5650
	   CALL EQUAL_VAL( cmnd_buff(qual_start(qp):qual_end(qp)),
     .			   vp_size(ivp), status )
	   IF ( status .NE. ferr_ok ) RETURN
	   IF ( vp_size(ivp) .LT. 0.001 ) GOTO 5620
	ELSEIF ( qual_given(slash_size) .EQ. 0 ) THEN     ! auto-sizing
           vp_size(ivp) = SQRT( (vp_xclip(ivp)-vp_xorg(ivp))
     .                        * (vp_yclip(ivp)-vp_yorg(ivp)) )
	   IF (vp_by_axis(ivp)) THEN
* ... vieport size is inferred from users axis limits and depends upon scale
* ... iterate a few times to approximate the correct scale factor
	      DO i1 = 1, 3
	        scale = 1.0 / SQRT( ABS(vp_size(ivp)) )
	        CALL VP_AXLIM2LIM( scale,
     .				   vp_xorg(ivp),  vp_yorg(ivp),
     .				   vp_xclip(ivp), vp_yclip(ivp),
     .                             xlovp, ylovp,  xhivp, yhivp,
     .                             xoei,  yoei,   xcei,  ycei )
	        vp_size(ivp) = SQRT((xhivp-xlovp) * (yhivp-ylovp))
	     ENDDO
	   ENDIF
	ENDIF

* successfully defined viewport
	vax_code = STR_UPCASE( vp_name(ivp), buff1 )

* if re-defining the current viewport then it takes immediate effect
	IF ( ivp .EQ. vp_num ) CALL SET_VIEWPORT( ivp )
	RETURN

**************************************************************************
* DEFINE ALIAS
 700	IF ( num_args .LT. 1 ) GOTO 5300
	CALL NAME_EQUAL_STRING( buff1, pos, status )
	IF ( status .NE. ferr_ok )  GOTO 5000
	IF ( pos .EQ. 0 ) GOTO 5400
	IF ( .NOT.TM_LEGAL_NAME(buff1) ) GOTO 5410

* find a slot to save this alias
        i1 = ALIAS_ID( buff1 )
        IF ( i1 .EQ. unspecified_int4 ) THEN
	   DO 710 i1 = num_predefined_alias+1, total_num_alias
	      IF ( alias_name(i1) .EQ. unspecified_name4 ) GOTO 720
 710	   CONTINUE
           GOTO 5710
        ENDIF

* save the alias
 720    vax_code = STR_UPCASE( alias_name(i1), buff1 ) 
        alias(i1) = cmnd_buff(pos:arg_end(num_args))
        alias_len(i1) = TM_LENSTR1( alias(i1) )
	RETURN


**************************************************************************
* DEFINE SYMBOL
 800	IF ( num_args .LT. 1 ) GOTO 5300
	CALL NAME_EQUAL_STRING( buff1, pos, status )
	IF ( status .NE. ferr_ok )  GOTO 5000
	IF ( pos .EQ. 0 ) GOTO 5400
	IF ( .NOT.TM_LEGAL_NAME(buff1) ) GOTO 5410

* if this symbol already exists then delete it
        buff30 = buff1	! must be exactly 30 characters for PLOT+ compat
	CALL GETSYM( buff30, err_string, i2, i1 )
	IF ( i1 .EQ. 0 ) CALL DELSYM(buff30,i1)

* remove backslash escape characters 
! OLD COMMENT: exactly 255 for PPL; no longer the case as of Ferret v6.01
	err_string = cmnd_buff(pos:arg_end(num_args))   
	CALL DE_ESCAPE_STRING( err_string, slen )

* define the symbol
	CALL PUTSYM( buff30, err_string, slen, i2)
! ... no check on returned error code (could be "internal error")
	RETURN

**************************************************************************
* DEFINE ATTRIBUTE /D= /TYPE= /OUTPUT
 900	IF ( num_args .LT. 1 ) GOTO 5900
	CALL NAME_EQUAL_STRING_VATT ( buff1, pos, status )
	IF ( status .NE. ferr_ok )  GOTO 5000
	IF ( pos .EQ. 0 ) GOTO 5400
	IF ( .NOT.TM_LEGAL_NAME(buff1) ) GOTO 5100

* ... get dset number if present:  DEF ATT/D=dset ...
	iqual = qual_given( slash_dset ) 
        IF ( iqual .GT. 0 ) THEN
	   CALL EQUAL_STRING( cmnd_buff(qual_start(iqual):qual_end(iqual)),
     .			      buff2, status )  
	   IF ( status .NE. ferr_ok ) RETURN
	   IF ( buff2 .NE. ' ' ) THEN
	      dset = FIND_DSET_NUMBER( buff2 )
	      IF ( dset .EQ. unspecified_int4) THEN
	         CALL WARN('Unknown data set: '
     .		 //buff2(:TM_LENSTR1(buff2)))
	         RETURN
	      ENDIF
	   ENDIF
	ENDIF

*...DEF ATT/TYPE=string or float  get attribute type if present 
*   (??? or other types on output?)
	iqual = qual_given( slash_type) 
        attype_spec = ptype_unknown
        IF ( iqual .GT. 0 ) THEN
	   CALL EQUAL_STRING( cmnd_buff(qual_start(iqual):qual_end(iqual)),
     .			      buff2, status )  
	   IF ( status .NE. ferr_ok ) RETURN
	   IF ( STR_SAME (buff2, 'string') .EQ. 0) THEN
	      attype_spec = ptype_string
           ELSE IF (STR_SAME (buff2, 'float') .EQ. 0) THEN
              attype_spec = ptype_float
           ELSE 
              GO TO 5940
	   ENDIF
	ENDIF

* ... DEFINE ATT/OUTPUT varname.attname
      attoutflag = 0
      iqual = qual_given( slash_output ) 
      IF ( iqual .GT. 0 ) attoutflag = 1 

* get the requested data somehow

        IF (arg_start(1) .LT. pos) arg_start(1) = pos
	CALL GET_PROT_CMND_DATA ( memory, cx_last, ptype_native, status )
        mr = is_mr(1)

        IF ( status .NE. ferr_ok ) THEN      ! 8/6/92
           list_fmt_type = plist_default ! restore for next command
	   list_format_given = .FALSE.
           RETURN
        ENDIF

!! ?? get dataset from varname.attname
* ... evaluate the context without computing the expression to get the dataset

        buff = cmnd_buff(item_start(1):item_end(1)) ! same as buff1?

	CALL GET_NEW_CX( cx_last, cx_cmnd, .TRUE., status )
        IF ( status .NE. ferr_ok ) GOTO 5000
 
        IF (buff(1:1) .EQ. '(') THEN
           dset = cx_data_set(cx_cmnd)   ! initial value to try; will return dset
           CALL ISIT_COORD_VAR ( buff, dset, varname, coordvar, status )
           IF ( status .NE. ferr_ok ) THEN 
              dset = -2  ! a user-defined axis (coordinate variable)
              CALL ISIT_COORD_VAR ( buff, dset, varname, 
     .              coordvar, status )
              IF ( status .NE. ferr_ok ) THEN
                 dset = cx_data_set(cx_cmnd)
                 GOTO 5920
              ENDIF
           ENDIF
        ELSE

* Get the dataset if given as [d=]
            IF (iqual .EQ. 0) iqual = 1
	    IF ( INDEX(buff,'[') .GT. 0 ) THEN
               dset = pdset_irrelevant	! default (not used ...)
               CALL PARSE_NAM_DSET(buff,
     .			 cx_last, dset, cat, var, mods_cx, status)
            ELSE
              cx = is_cx( 1 )
              IF (cx .EQ. 0 .OR. cx .EQ. unspecified_int4)
     .             cx = cx_cmnd
             dset = cx_data_set(cx)
          ENDIF
	ENDIF

* See if the variable is a user-defined variable.
        IF (dset .EQ. pdset_irrelevant .OR. 
     .     dset .EQ. unspecified_int4) THEN

           varname = buff1(1:INDEX(buff1,".")-1)
           CALL FIND_VAR_NAME(pdset_irrelevant, varname, cat, var)
           IF (  var .NE. munknown_var_name .AND.
     .           cat .EQ. cat_user_var) THEN
              dset = -1
           ELSE
              dset = cx_data_set(cx_cmnd) 
           ENDIF
        ENDIF

*  break up varname.attname.  

        CALL BREAK_VARNAME( buff1, dset, varname, attname, 
     .                               varid, status )
        IF (status .NE. ferr_ok) goto 5920

        CALL ADD_ATTRIBUTE(  memory(1, mr_blk1(mr)), mr, 
     .         buff1, dset, attype_spec, attoutflag, new_att, status )

* Change attribute value
        IF (.NOT. new_att) THEN
           CALL EDIT_ATTRIBUTE(  memory(1, mr_blk1(mr)), mr, 
     .               buff1, dset, attype_spec, status )
        ENDIF


* If the time origin was changed or added, also change units to say
* "units since time origin" using the new time origin info.
        
        iline = TM_GET_LINENUM( varname )

        slen = TM_LENSTR1(attname)
        IF (MATCH_NAME (attname,  slen, 'TIME_ORIGIN', 11)  .AND.
     .         ( line_direction(iline) .EQ. 'TI' .OR.
     .          line_direction(iline) .EQ. 'TT' ) ) THEN
           CALL CD_GET_VAR_ATT_ID (dset, varid, 'UNITS', attid, status)
           IF (status .NE. ferr_ok) GOTO 5960
           got_it = NC_GET_ATTRIB ( dset, varid, 'units',
     .                              .TRUE., varname, 128,
     .                              attlen, attoutflag, ustring, 
     .                              val_buf )
           got_it = NC_GET_ATTRIB ( dset, varid, 'time_origin',
     .                              .TRUE., varname, 128,
     .                              attlen, attoutflag, t0string, 
     .                              val_buf )
           vax_code = STR_UPCASE (buff2, ustring)
           slen = INDEX(buff2, 'SINCE')
           IF (slen .GT. 0) THEN
              slen = slen - 2
           ELSE
              slen = TM_LENSTR1(ustring)
           ENDIF
           buff3 = ustring(1:slen) // ' since ' // t0string
           attlen = TM_LENSTR1(buff3)

           ustring = 'units'
           CALL CD_REPLACE_ATTR (dset, varid, ustring, NCCHAR, 
     .         attlen, buff3, val_buf, status)
        ENDIF

* If the units were changed on a time axis, add "since time origin"
           
        slen = TM_LENSTR1(attname)
        IF (MATCH_NAME (attname,  slen, 'UNITS', 5)  .AND.
     .     ( line_direction(iline) .EQ. 'TI' .OR.
     .       line_direction(iline) .EQ. 'TT' ) ) THEN
           CALL CD_GET_VAR_ATT_ID (dset, varid, 'TIME_ORIGIN', 
     .              attid, status)
           IF (status .EQ. ferr_ok) THEN 
              got_it = NC_GET_ATTRIB ( dset, varid, 'time_origin',
     .                                .TRUE., varname, 128,
     .                                attlen, attoutflag, t0string, 
     .                                val_buf )
              got_it = NC_GET_ATTRIB ( dset, varid, 'units',
     .                                .TRUE., varname, 128,
     .                                attlen, attoutflag, ustring, 
     .                                val_buf )

              vax_code =  STR_UPCASE (buff2, ustring)
              slen = INDEX(buff2, 'SINCE')
              IF (slen .GT. 0) THEN
                 slen = slen - 2
              ELSE
                 slen = TM_LENSTR1(ustring)
              ENDIF
              buff3 = ustring(1:attlen) // ' since ' // t0string
              attlen = TM_LENSTR1(buff3)

              CALL CD_REPLACE_ATTR (dset, varid, attname, NCCHAR, 
     .            attlen, buff3, val_buf, status)
           ENDIF
        ENDIF


* Update attribute information used by plotting etc (ds_units, ...)
            CALL FIND_VAR_NAME (dset, varname, cat, uvar)
            IF (cat .EQ. cat_user_var .OR. cat .EQ. cat_file_var)
     .           CALL UPDATE_ATTRIBUTES (dset, uvar)


	RETURN

* error exit
 5000	RETURN
 5100 	CALL ERRMSG( ferr_syntax,status,'illegal name: '//buff3,*5000 )
 5300	CALL ERRMSG( ferr_invalid_command, status,
     .		'DEFINE what name?', *5000 )
 5305	CALL ERRMSG( ferr_TMAP_error, status, ' ', *5000 )
 5310	CALL ERRMSG( ferr_grid_definition, status,
     .		'name already in use:'//buff3(:TM_LENSTR1(buff3)), *5000 )
 5320	CALL ERRMSG( ferr_grid_definition, status,
     .		'unknown grid or axis:'//buff1(:TM_LENSTR1(buff1)), *5000 )
 5330	CALL ERRMSG( ferr_grid_definition, status,
     .		'inappropriate orientation for '//ww_dim_name(idim)//
     .		' axis:'//buff1(:TM_LENSTR1(buff1)), *5000 )
 5400 	CALL ERRMSG( ferr_syntax, status,
     .			buff1(:TM_LENSTR1(buff1))//' = ???', *5000 )
 5410   CALL ERRMSG( ferr_syntax,status,'illegal name: '//buff1,*5000 )
 5420   CALL ERRMSG( ferr_syntax,status,'cannot define: '//buff1,*5000 )
 5440	CALL ERRMSG( ferr_syntax, status,
     .			cmnd_buff(pos:len_cmnd)//pCR//err_string, *5000 )
 5470	CALL ERRMSG( ferr_unknown_data_set, status,
     .			buff3(:TM_LENSTR1(buff3)), *5000 )
 5505	CALL ERRMSG( ferr_invalid_command, status,
     .		'DEFINE AXIS/DEPTH needs /Z=lo:hi:del', *5000 )
 5510	CALL ERRMSG( ferr_invalid_command, status,
     .		'Conflicting /X,/Y,/Z or /T axis orientations given',
     .								*5000 )
 5520	CALL ERRMSG( ferr_grid_definition, status,
     .			'error in start,end,delta', *5000 )
 5525	CALL ERRMSG( ferr_grid_definition, status,
     .			'Axis length exceeds modulo length', *5000 )
 5530	CALL ERRMSG( ferr_grid_definition, status,
     .			'NPOINTS=1 incompatible with limits given', *5000 )
 5540	CALL ERRMSG( ferr_grid_definition, status,
     .		'/T0='//buff2(:TM_LENSTR1(buff2)), *5000 )
 5550	CALL ERRMSG( ferr_grid_definition, status,
     .		'inappropriate units:'//buff1(:TM_LENSTR1(buff1)), *5000 )
 5560	CALL ERRMSG( ferr_grid_definition, status,
     .		'unrepairable repeated axis coords', *5000 )
 5570	CALL ERRMSG( ferr_grid_definition, status,
     .		'data for DEFINE AXIS/FROM_VARIABLE is not monotonic', *5000 )
 5580	CALL ERRMSG( ferr_grid_definition, status,
     .		'data for DEFINE AXIS/FROM_VARIABLE is ambiguous', *5000 )
 5582	CALL ERRMSG( ferr_grid_definition, status,
     .		'illegal argument for DEFINE AXIS/NPOINTS', *5000 )
 5590	CALL ERRMSG( ferr_prog_limit, status,
     .		'axis coordinate storage exhausted - restart FERRET', *5000 )
 5600   slen = TM_LENSTR1(buff3)
        CALL ERRMSG( ferr_syntax, status,
     .               'error initializing '//buff3(:slen), *5000 )
 5610	CALL ERRMSG( ferr_prog_limit, status,
     .		'redefine or cancel an existing viewport', *5000 )
 5620 	CALL ERRMSG( ferr_out_of_range, status,
     .			cmnd_buff(qual_start(qp):qual_end(qp)), *5000 )
 5630 	CALL ERRMSG( ferr_syntax, status,
     .			cmnd_buff(qual_start(qp):qual_end(qp)), *5000 )
 5640 	CALL ERRMSG( ferr_invalid_command, status,
     .			'cannot re-define FULL viewport', *5000 )
 5650 	CALL ERRMSG( ferr_syntax, status,
     .	'DEFINE VIEWPORT/SIZE and /TEXT are mutually exclusive', *5000 )

 5660 	CALL ERRMSG( ferr_syntax, status,
     .	'DEFINE VIEWPORT/AXES cannot be used with obsolete /ORIGIN'
     .  //', /CLIP/ or /SIZE', *5000 )

 5710	CALL ERRMSG( ferr_prog_limit, status,
     .		'use CANCEL ALIAS to reclaim space', *5000 )
 5720	err_string = 'Invalid calendar name. Names are'
        DO 5721 i1=1, max_calendars
           len_cal = TM_LENSTR1(err_string)
           slen = TM_LENSTR1(allowed_calendars(i1))
           err_string = err_string(:len_cal)//', '//
     .                   allowed_calendars(i1)(:slen)
           len_cal = len_cal + slen
 5721   CONTINUE
        CALL ERRMSG( ferr_syntax, status,err_string, *5000 )

5730	CALL ERRMSG( ferr_syntax,status,
     .     'cannot have /BOUNDS and /EDGES together ',*5000 )

5740	buff1 = TM_FMT(FLOAT( n), 14, 16, slen)
        buff2 = TM_FMT(FLOAT(n2), 14, 16, s2len)
        CALL ERRMSG( ferr_grid_definition, status,
     .     'dimension of bounds ('// buff2(:s2len)//
     .     ') must be 2*N or N+1, N=number of axis coordinates ('//
     .     buff1(:slen) //')', *5000 )
5745	buff1 = TM_FMT(FLOAT( n), 14, 16, slen)
        buff2 = TM_FMT(FLOAT(n2), 14, 16, s2len)
        CALL ERRMSG( ferr_grid_definition, status,
     .     'dimension of upper&lower bounds ('// buff2(:s2len)//
     .     ') must be N, N=number of axis coordinates ('//
     .     buff1(:slen) //')', *5000 )

5750	CALL ERRMSG( ferr_grid_definition, status,
     .     'BOUNDS specified do not enclose coordinate points', *5000 )

5900	CALL ERRMSG( ferr_invalid_command, status,
     .		'DEFINE what name.attribute?', *5000 )
5920    slen = TM_LENSTR1(varname)

        IF (dset .GT. 0) THEN
           buff1 = TM_FMT(FLOAT(dset), 14, 16, s2len)
           CALL ERRMSG( ferr_invalid_command, status,
     .		'variable or axis does not exist in dset'//
     .           buff1(:s2len)//varname(:slen), *5000 )
        ELSE
           CALL ERRMSG( ferr_invalid_command, status,
     .		'variable or axis does not exist '//
     .           varname(:slen), *5000 )
        ENDIF
           
5930    buff1 = 'string'
        IF (at_type(1) .EQ.ptype_float) buff1 = 'float'
        slen = 6
        buff2 = 'string'
        if (attype_spec .EQ. ptype_float) buff2 = 'float'
        CALL ERRMSG( ferr_invalid_command, status,
     .		'TYPE specified does not match type of values'//
     .           cmnd_buff(pos:len_cmnd)//' is '// buff1(:slen)//
     .           'TYPE given is '// buff2(:slen), *5000 )

5940    slen = TM_LENSTR1(buff2)
        CALL ERRMSG( ferr_invalid_command, status,
     .		'attribute type must be string or float'// buff2(:slen), *5000 )

5960  slen = TM_LENSTR1(buff1)
      CALL ERRMSG( ferr_invalid_command, status,
     .   'cannot set T0 unless Units are also set'//
     .    buff1(:slen), *5000 )

	END
