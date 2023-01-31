 	SUBROUTINE XEQ_DEFINE

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
* V606  8/07 *acm* DEFINE AXIS/QUIET
* V606  8/07 *acm* Send informational and error messages that are returned from 
*                  commands via SPLIT_LIST to std error rather than std out.
* V62   2/09 *acm* Pass the original upper/lowercase spelling of user-defined variable
*                  names to init_uvar; name to be saved in the attribute structure, for 
*                  use when writing out user-defined variables when MODE UPCASE_OUTPUT 
*                  is cancelled.
* V62 *acm*  2/09 - Save original upper/lowercase spelling of axis names 
*                   in line_name_orig for CANCEL MODE UPCASE
* V62 *acm*  3/09  Fix bug 1645: Define an axis with a very large number of NPOINTS.
*                  This works if the value of NPOINTS can be specified as an integer.
* V64 *acm* 11/09  Fix to bug 1511 slightly wrong: dont use line_t0 from
*                  a previous definition of a time axis.
* V65 *acm*  2/10  Minor change to error message
* V664  8/10 *kms* Add DEFINE PYFUNC /NAME=<alias> python.module.name
* V67   1/11 *acm* User-defined attributes are set to /OUTPUT=1  by default.
*                  Write the calendar attribute, even if its the default Gregorian
* V68 *acm* 1/12   Use the same micro-adjusting scheme as we are implementing for 
*                  reading netCDF files (ticket 1910). Reset micro-adjust size if needed.
*       *acm* 3/12 Add E and F dimensions (use nferdims in tmap_dims.parm)
*       *acm* 4/12 6D Ferret: time axis may be in t or f direction. 
* V68  *acm* 5/12  Add DEFINE DATA/AGGREGATE
* V683 *acm*11/12  If DEFINE AXIS data not monotonic, add the index location 
*                  to the error message
* V685 *acm* 1/13  Fix ticket 2026: dont automatically mark a 1-point longitude
*                  axis as modulo.
* V685 *adm*  1/13 DEFINE VARIABLE/REMOTE. Require /D= to accompany /REMOTE.
* V685 *acm*  4/13 Add /HIDDEN on DEF DATA/AGG
* V685 *acm* 10/13 Fix ticket 2096, DEFINE GRID with check on units
* V686 *acm* 11/13 Allow symbol names to be up to 120 characters long
* V687 *acm*  1/14 Fixing ticket 2135; grid_use_cnt on DEFINE DATA/AGG datasets
* V690 *sh*   1/14 Formatting improvement to DEFINE ATTRIB error message
* V693+ 11/14 *sh* renaming 'sigma' as 'layerz' throughout
*                  prevent IF, ELIF, ELSE, and ENDIF from being aliased
* V695  *acm* 2/15 cx_calendar stores info about calendar formatting of T and F axes
* V695+  *sh* 4/15 error msgs from INIT_AGGREGATE_SET already given.  Remove.
* V695  *acm* 6/15 Change the error message when coordinates are decreasing.
* V695  *acm* 6/16 New subcommand DEFINE ANNOTATION
* V695 *acm*  9/15 Fix #2317: length of name sent to FIND_DSET_NUMBER
* V695 *sh*  10/15 add DEFINE DATA/AGG/T
*                  rename routine INIT_AGGREGATE_DSET to INIT_EF_AGGREGATE_DSET
* V697 12/15 *acm* Fix ticket 2336: consistent application of CAN MODE UPCASE
* V698 1/16 *sh*   Allow DEFINE DATA/AGG w/out a name -- auto-name from script
* V698 2/16 *acm*  For ticket 1786: if mode_upcase is canceled write axis name
*                  using original upper/lowercase spelling
* V698  2/16 *acm* For ticket 2352: variables defined with LET/D are
*                  added to that dataset in the attribute structure.
*       3/16 *sh*  bug fix use GO_FILE rather than LAST_GO_FILE as AGG name
* V698  4/16 *acm* Fix ticket 2380: use /T0 to indicate a T axis, only if 
*                  direction is otherwise undefined.
* V698  4/16 *acm* Allow DEFINE VAR/BAD=nan
* V698  4/16 *acm* Fix ticket 2400: DEFINE AXIS/EDGES had bugs in testing for
*                  regular axis, and in setting the axis start when regular.
* V698  5/16 *acm* for ticket 1432, could write a note when axis definition is a 
*                  supbspan modolo axis with length almost the modulo length.
*                  (commented out for now.)
* V698  5/16 *acm* Ticket 2416 allow all output types for DEFINE ATT.
* V7    5/16 *acm* See ticket 2352. For LET/D variables look for info under that dataset
* V7   *acm*  6/12 Add Union aggregations: DEFINE DATA/AGG/U
* V710 12/16 *acm* Ticket 2158: working with true monthly time axes.
*                  Add DEFINE AXIS /MONTHLY /LIKE=
* V710  4/16 *acm* new TM_UNITS_CAL to account for calendar in units id
* V702 12/16 *acm* handling modulo true-month axes added after merge back to trunk.
* V7022 1/17 *acm* ticket 2497, Back off the auto-detection of monthly axes.
* 1/13/2017 *acm* Do the repeated-coordinates checking in the new routine
*                 TM_CHECK_COORDS, also called by cd_get_1_axis, to consistently
*                 handle almost-equal coordinates, differing by machine precision.
* 1/24/2017 *acm* Ticket 2504: If Ferret is computuing axis bounds, set them such 
*                 that the axis lies within the given modulo length.
* 12/2017   *acm* Small tweak to error handling; any errors already noted in add-attribute
* V740 *acm* 2/18 Issue 966: LET/LIKE= Get attributes from an existing variable
* V740 *acm* 2/18 Issue 966: DEFINE AXIS/LIKE= Get attributes from an existing axis
* V740 *acm* 3/18 More on DEF AX/LIKE, getting T0, modulo axis characteristics set correctly.
* v744 11/2018 *acm* Issue 1906: If PROLEPTIC_GREGORIAN is given then 
*                  override mixed-gregorian calendar handling.
* v7441*acm* 12/18 Issue 1910: Allow dates in commands to be in ISO 8601 form, yyyy-mm-dd
* V7.6 *acm*  6/20 For DSG data if there's a feature-mask applied to the dataset and the
*                  uer redefines the mask variable, verify and set the new mask.
* v767 6/22 *acm* See issue 112: Detect whether axis length is too large for 
*                 4-byte integer indexing
* 12/2022 *acm* Code cleanup: Ensure consistent data types in subroutine calls 
*  1/2023 *acm* Code cleanup: Replace the computed GOTO statement 
*                  "GOTO ( 100,200, ... )  subcmnd_num"  with IF blocks

        IMPLICIT NONE
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
        include 'xgrid_chg_fcns.cmn'
	include 'xinterrupt.cmn'
	include 'calendar.decl'
	include 'calendar.cmn'	
        include 'netcdf.inc'
        include 'gkscm1_inc.decl'
        include 'GKSCM1.INC' ! wsid
        include 'xrisc.cmn'


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
     .		    slash_monthly,
     .		    slash_def_ax_like,
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
     .		    slash_let_remote,
     .		    slash_let_like,
     .		    slash_define_vp_by_ax,
     .              slash_bounds,
     .              slash_def_att_type,
     .              slash_def_att_dset,
     .              slash_def_att_output,
     .              slash_def_att_quiet,
     .              slash_def_ax_quiet,
     .              slash_pyfunc_name,
     .              slash_def_aggregate,
     .              slash_def_agg_title,
     .              slash_def_agg_quiet,
     .              slash_def_agg_t,
     .              slash_def_agg_e,
     .              slash_def_agg_f,
     .              slash_def_agg_hide,
     .              slash_def_agg_u,
     .              slash_def_agg_taxis,
     .              slash_def_agg_faxis,
     .              slash_def_agg_toverlap,
     .              slash_def_annot_nlab

	LOGICAL	    explct_defn
	PARAMETER ( slash_dflt   = 1 + 12,
     .		    slash_file   = 1 + 6,
     .		    slash_like   = 1 + 7,
     .		    slash_units  = 1 + 7,
     .		    slash_x0     = 0,	! slash_x minus 1
     .		    slash_t0     = 1 + 8,
     .		    slash_name	 = 1 + 9,
     .		    slash_from	 = 1 + 10,
     .		    slash_depth	 = 1 + 11,
     .		    slash_modulo = 1 + 12,
     .		    slash_npoints= 1 + 13,
     .		    slash_edges  = 1 + 14,
     .		    slash_calendar = 1 + 15,
     .		    slash_bounds = 1+16,
     .		    slash_def_ax_quiet = 1+17,
     .		    slash_monthly= 1 + 18,
     .		    slash_def_ax_like  = 1+19,
     .		    slash_text   = 1 + 0,
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
     .		    slash_let_remote = 1 + 5,
     .		    slash_let_like   = 1 + 6,
     .		    slash_define_vp_by_ax = 1 + 6,
     .              slash_def_att_type    = 1 + 1,
     .              slash_def_att_dset    = 1 + 0,
     .              slash_def_att_output  = 1 + 2,
     .              slash_def_att_quiet   = 1 + 3,
     .              slash_pyfunc_name = 1,
     .              slash_def_aggregate = 1 + 0,
     .              slash_def_agg_t     = 1 + 4,
     .              slash_def_agg_e     = 1 + 5,
     .              slash_def_agg_f     = 1 + 6,
     .              slash_def_agg_title = 1 + 7,
     .              slash_def_agg_quiet = 1 + 8,
     .              slash_def_agg_hide  = 1 + 9,
     .              slash_def_agg_u     = 1 + 10,
     .              slash_def_agg_taxis = 1 + 11,
     .              slash_def_agg_faxis = 1 + 12,
     .              slash_def_agg_toverlap = 1 + 13,
     .              slash_def_annot_nlab = 10,		! must be slash_annotate_siz+1
     .		    explct_defn  = .FALSE. )

* calling argument declarations:

* local variable declarations:

	LOGICAL	  TM_DIGIT, TM_LEGAL_NAME, TM_LINE_MATCH, TM_GRID_MATCH,
     .		  TM_FPEQ, TM_DFPEQ, TM_CHECK_BNDS, MATCH_NAME,
     .            NC_GET_ATTRIB, TM_LEGAL_NAME_OP, TM_HAS_STRING,
     .            ITSA_1LINEIF, MATCH4, GO_FILE_INPUT, IS_AGG_MEMBER, 
     .            MATCH_TEMPLATE, TM_ITS_SUBSPAN_MODULO, TM_CHECK_BOX_LOC, 
     .            TM_DATE_OK, is_ok,
     .            dup_name, delta_given, create, irreg, its_reg, its_edges, 
     .            its_calendar, its_modulo, has_repeated, has_bounds,
     .            new_att, new_att_modulo, got_it, orient_t, agg_quiet,
     .            letdset, its_remote, agg_hide, def_att_quiet, purge_all,
     .            user, norm, nouser, use_strict, have_expr, 
     .            original, true_month, line_alloc, truemonth_noted,
     .            is_double, misordered, bnds_or_edges, okmod, is_like, 
     .            is_fmask, is_smask
	INTEGER   REGION_NUMBER, GRID_FROM_NAME, VIEWPORT_NUMBER, STR_UPCASE,
     .		  CX_DIM_LEN, ALIAS_ID, EFCN_SCAN,
     .		  TM_GET_LINENUM, TM_LENSTR1, 
     .		  TM_GET_GRIDNUM, TM_UNIT_ID, TM_UNITS_CAL, 
     .		  ALLO_MANAGED_AXIS, ALLO_MANAGED_GRID, FIND_DSET_NUMBER,
     .            TM_GET_CALENDAR_ID, STR_SAME,
     .		  tmap_status, npoints, i, 
     .		  status, idim, vax_code, islot, iunit, iline, cat, var,
     .		  orient, mods_cx, cx, mr, n, i1, i2, i3, i4, i5, i6, pos,
     .		  grid, iqual, ax_grid, ivp, qp, iseg, uvar, dset, slen, s1,
     .		  old_line, old_grid, dim(nferdims), ndim, natom,
     .		  at_type  (maxatoms), at_id  (maxatoms),
     .		  at_start (maxatoms), at_end (maxatoms), cal_id, 
     .            dflt_cal_id, nmonths, ndays, d_before_mon(12), 
     .            mon_by_d(366), d_in_mon(12), len_cal, n2, n3, s2len,
     .            prev_cal_id, j, num_indices, varid, attype_spec,
     .            attoutflag, attlen, coordvar, 
     .            dset_to_add, dir, type, attype, attid, ibuff, i0,  
     .            agg_dim, nagfiles, ivar, item, nlab, ierr, nlen,  
     .            llen, line, dstart, units, npts, ipt1, ipte, iline_b,
     .            tline, fline, idx, vaxcode, toverlap, mr1(1)

	REAL	  rbuff, bad_flag, yeardays, 
     .            val, dummy, val_buf, xloc, yloc, halign, angle, size,
     .            epsilon

	REAL*4	  r4_unspec, scale, xlovp, ylovp,  xhivp, yhivp,
     .            xoei,  yoei,   xcei,  ycei
	REAL*8	  SECS_FROM_BC, TM_WW_AXLEN, GET_LINE_COORD,
     .		  bc_to_t0, start, new_ww, delta, end, secs2start,
     .            micro_adj, axwwlen, rmod_len, secsperyear, small, 
     .            new_att_modulo_len, madj, firs_coord, last_coord, timefac, 
     .            vlo, vhi, idel, steps_total
	CHARACTER TM_FMT*48, TM_CLEAN_FILENAME*512,SECS_TO_DATE_OUT*11,
     .            reg_name*24, buff1*128, buff2*512, buff3*512, 
     .            buffsym*120, cal_name*32, buff*512,
     .            basic_orients(nferdims)*2, err_string*2048, replmsg*24,
     .		  mon_names(12)*3, varname*512, attname*128, t2*2,
     .            new_att_units*128, new_att_calendar*128,
     .            new_att_t0*128, dcode*2, axdir*1, t0string*128, 
     .            ustring*128, reserved_flow_name(3)*4, is_nan*3,
     .            buff4*16, buff5*16, datestr*11, likename*64,
     .            tmatch*32, fmatch*32

        CHARACTER*1 tab
#ifdef NO_INTRINSIC_IN_PARAMETER
	PARAMETER     ( tab = o'011' )
#else
	PARAMETER     ( tab = CHAR(9))
#endif

	DATA	  basic_orients / 'WE','SN','UD','TI','E','F' /,
     .		  replmsg/'Replacing definition of '/
        DATA      reserved_flow_name / 'ELIF','ELSE','ENDI' /

* statement function - value between 0 and 1
	REAL x, vx, vy
	LOGICAL OUTSIDE_0_1
	OUTSIDE_0_1( x ) = x.LT.0.0 .OR. x.GT.1.0

* execute subcommand

* DEFINE  '    ' - (no action)
* arrival at this point usually means an illegal subcommand was given
* and was assumed to be an argument by the command parser

      IF (subcmnd_num .EQ. 1) THEN

	IF ( num_args .GE. 1 ) THEN
	   CALL ERRMSG( ferr_invalid_subcmnd, status,
     .			cmnd_buff( arg_start(1):arg_end(1) ), *5000 )
	ELSE
	   CALL ERRMSG( ferr_invalid_command, status,
     .			'DEFINE what ?', *5000 )
	ENDIF

      ENDIF

**************************************************************************
* DEFINE REGION
* ... command syntax ok ?

      IF (subcmnd_num .EQ. 2) THEN

	IF ( num_args .EQ. 0 ) THEN
	   CALL ERRMSG( ferr_invalid_command, status,
     .			'DEFINE what region ?', *5000 )
	ENDIF

* get the name of the region to be defined
	reg_name = cmnd_buff(arg_start(1):arg_end(1))

* is this region already defined ?
	islot = REGION_NUMBER( reg_name )
        create = islot .EQ. unspecified_int4

* delta context specifiers (DX=,DY=, etc), if any
	CALL GET_DELTA_CONTEXT( 14, delta_given, status )
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
	      DO islot = 0, min_context, -1
	         IF ( cx_name(islot) .EQ. unspecified_name4 ) GOTO 245
	      ENDDO
* ... no free slots
	      CALL ERRMSG( ferr_prog_limit,status,
     .			   'Cancel or redefine a region',*5000 )
	   ENDIF

* ... if "/DEFAULT" was implied or given then flag for saving all dimensions
*     that are defined
 245       IF ( num_qualifiers .EQ. 0 
     .     .OR. qual_given( slash_dflt ) .GT. 0 ) THEN
	      DO idim = 1, nferdims
	         IF (cx_by_ss(idim,cx_buff)
     .		   .AND. cx_lo_ss(cx_buff,idim) .NE. unspecified_int4
     .	        .OR..NOT.cx_by_ss(idim,cx_buff)
     .		   .AND. cx_lo_ww(idim,cx_buff) .NE. unspecified_val8)
     .				cx_given( idim, cx_buff ) = .TRUE.
	      ENDDO
* ... else flag only the dimensions named
	   ENDIF
	ENDIF

* all unmodified axes must be set to unspecified
	DO idim = 1, nferdims
	   IF ( .NOT.cx_given(idim,cx_buff) ) CALL DEL_CX_DIM( idim, cx_buff )
 	ENDDO

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

      ENDIF

**************************************************************************
* DEFINE GRID
* ... DEFINE GRID/X/Y/Z/T/E/F /FILE /LIKE

 300  continue
      IF (subcmnd_num .EQ. 3) THEN

	iqual = qual_given( slash_file )
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
	   DO idim = 1, nferdims
	      grid_line    (idim, mgrid_buff) = mnormal
              grid_out_prod(idim, mgrid_buff) = .TRUE.
	   ENDDO	
	   grid_rotation( mgrid_buff ) = 0.0
	ENDIF

* axes named with /X=line_grid_or_var, /Y= ...
	DO idim = 1, nferdims
	   iqual = qual_given( idim )
	   IF ( iqual .GT. 0 ) THEN
	      CALL EQUAL_STRING( cmnd_buff(qual_start(iqual):qual_end(iqual)),
     .				 buff1, status )
	      IF ( status .NE. ferr_ok ) RETURN
	      iline = TM_GET_LINENUM( buff1 )
	      IF ( iline .NE. unspecified_int4 ) THEN
* ... line given by name - don't allow inappropriate units
 	         orient_t = idim.EQ.t_dim .OR. idim.EQ.f_dim
		 iunit = line_unit_code ( iline )
 	         IF ( iunit.NE.0 ) THEN
 	            IF ( (  orient_t .AND.
     .		     (iunit.LT.pun_1st_time .OR. iunit.GT.pun_last_time) )
     .	            .OR. (  (.NOT.orient_t) .AND.
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
	ENDDO

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
	   CALL SPLIT_LIST(pttmode_ops, err_lun,
     .			replmsg//'grid '//grid_name(old_grid), 0)
	   CALL PURGE_MR_GRID( old_grid, status )
	   IF ( status .NE. merr_ok ) GOTO 5000
* ... deallocate dynamic axes used in previous definition of this grid
	   DO idim = 1,nferdims
	      CALL TM_DEALLO_DYN_LINE( grid_line(idim,old_grid) )
	   ENDDO
	   grid_name( old_grid ) = char_init16
	ENDIF

	RETURN

      ENDIF

**************************************************************************
* DEFINE VARIABLE
* find name to define
* 2 syntaxes are allowed: "DEFINE VAR NAME=TEXT", "DEFINE VAR NAME TEXT"
* in the latter be careful that "TEXT" may contain "="

      IF (subcmnd_num .EQ. 4) THEN

	purge_all = .FALSE.  ! initialize
        IF ( num_args .LT. 1 ) GOTO 5300
	CALL NAME_EQUAL_STRING( buff1, pos, status )
	CALL NAME_EQUAL_STRING_LC( varname, pos, status )
	IF ( status .NE. ferr_ok )  GOTO 5000
	IF ( pos .EQ. 0 ) GOTO 5400
	IF ( .NOT.TM_LEGAL_NAME(buff1) ) GOTO 5410
	IF ( .NOT.TM_LEGAL_NAME_OP(buff1) ) GOTO 5430
        IF ( TM_LENSTR1(buff1) .GT. 128 ) GOTO 5410
	CALL FIND_VAR_NAME ( pdset_irrelevant, buff1, cat, var )
	IF (  var .NE. munknown_var_name
     .	.AND. cat .EQ. cat_pseudo_var    ) GOTO 5420

* /D=xxx: has the user requested a data-set-specific definition?
	iqual = qual_given( slash_let_dset )
	letdset = ( iqual .GT. 0 ) 
	IF (letdset) THEN
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

* acm 4/16 - allow /bad=NaN
	IF ( iqual .GT. 0 ) THEN

           CALL EQUAL_STRING (cmnd_buff(qual_start(iqual):qual_end(iqual)),
     .                         is_nan, status)
           IF ( status .NE. ferr_ok ) RETURN     ! *sh* 1/14 added status check
           IF (is_nan .EQ. ' ') GOTO 5400
           IF (TM_HAS_STRING(is_nan, 'NaN')) THEN
              CALL SET_NAN(bad_flag)
           ELSE
              CALL EQUAL_VAL(  cmnd_buff(qual_start(iqual):qual_end(iqual)),
     .                     bad_flag, status )
              IF ( status .NE. ferr_ok ) RETURN
           ENDIF

	   IF ( status .NE. ferr_ok ) RETURN
	ELSE
	   bad_flag = bad_val4
	ENDIF

* LET/REMOTE ?

	its_remote = .FALSE.
	iqual = qual_given( slash_let_remote )
	IF ( iqual .GT. 0 ) THEN
	   IF (.NOT. letdset) GOTO 5480
	   its_remote = .TRUE.
	ENDIF

* break the expression into atoms and identify them

        t2 = cmnd_buff(pos:pos+1)
        IF (t2 .EQ. '..') cmnd_buff(pos:pos+1) = 'aa'
	CALL ALG_BREAK_UP( cmnd_buff(pos:len_cmnd),at_start,at_end,natom,
     .			   err_string,status )
        cmnd_buff(pos:pos+1) = t2

	IF ( status .NE. ferr_ok ) GOTO 5440

* cannot use [d= in the remote var-- the expression is evaluated on the
* remote server; same dsets not open there.
	IF (its_remote.AND. TM_HAS_STRING(cmnd_buff(pos:len_cmnd), 'd=') )
     .                                                         GOTO 5490

	CALL ALG_ID( cmnd_buff(pos:len_cmnd), natom, at_start, at_end,
     .		     at_type, at_id, status )
	IF ( status .NE. ferr_ok ) GOTO 5000

* if this definition replaces a variable of the same name in this data set
	IF ( dset .GT. pdset_irrelevant ) THEN
	   CALL FIND_VAR_NAME ( dset, buff1, cat, var )
* ... purge cached calculations (8/95)
	   IF ( ( (cat .EQ. cat_file_var) .OR.
     .            (cat .EQ. cat_pystat_var) ) .AND.
     .          (var .NE. munknown_var_name) ) CALL PURGE_ALL_UVARS
* ... purge any aggregations that utilize this dataset as a member
*     Note: this is a little stricter than need be, as there is a
*     chance that the specific variable may not be a member of the agg, even
*     though the dataset is.  But this will be a rare situation ...
*     ... arguments i and got_it are ignored
           IF (IS_AGG_MEMBER(dset, 1, i, got_it)) CALL PURGE_DSET(dset)
	ENDIF

* Was this a feature-mask for a DSG dataset(s)?  If so cancel that masking.

	slen = TM_LENSTR1(buff1) 
	CALL UNSET_FEATURE_MASKS (buff1, slen, is_fmask, is_smask)

* if this is a global foreground definition that replaces any variable
* then purge cached calculations (8/95)
	IF ( dset .EQ. unspecified_int4 ) THEN
	   CALL FIND_VAR_NAME ( dset, buff1, cat, var )
	   IF ( var .NE. munknown_var_name ) CALL PURGE_ALL_UVARS
	ENDIF

* do we need to delete a previous definition of this same-named var ?
* Note: need to purge old defs ALSO if the same name is now redefined in a 
*	new data set -- expressions using this name chg interpretation  

        CALL string_array_find_caseblind(uvar_name_code_head,
     .                                   buff1,
     .                                   LEN(buff1),
     .                                   string_array_result(1),
     .                                   max_uvar,
     .                                   num_indices)        
        DO j = 1, num_indices
              i1 = string_array_result(j)
              IF ( uvar_num_items(i1) .EQ. uvar_deleted ) CYCLE
* ... delete only if same name in same data set
	      IF ( uvar_dset(i1) .EQ. dset ) THEN
	         CALL DELETE_USER_VAR(i1, dset)
	      ENDIF
* ... reuse of name  makes all stored uvars possibly invalid
*     Also invalidates any memory-cached results that used aux of this name
	      purge_all = .TRUE.
*	   ENDIF
	ENDDO


* if this name is used as the target of an aux (curvilinear) coordinate var
* (re)defining it may invalidate cached results
        CALL GET_FVARS_LIST_BY_ATTNAME_AND_VAL
     .	  (patnam_layerz, buff1, max_uvar,
     .     string_array_result, deleted_list_result, num_indices)
        purge_all = purge_all .OR. (num_indices .GT. 0)
        IF (.NOT.purge_all) THEN
           CALL GET_UVARS_LIST_BY_ATTNAME_AND_VAL
     .	        (patnam_layerz, buff1, max_uvar,
     .           string_array_result, deleted_list_result, num_indices)
           purge_all = purge_all .OR. (num_indices .GT. 0)
        ENDIF

* purge user variables and aux var dependencies if needed
        IF (purge_all) CALL PURGE_ALL_UVARS !not done above for pdset_irrelevan


* now file the expression in the user-defined variable area
	CALL INIT_UVAR ( buff1, cmnd_buff(pos:len_cmnd), buff3, buff2, dset,
     .			 bad_flag, explct_defn, uvar,
     .			 at_type, at_id, at_start, at_end,
     .			 natom, 1, len_cmnd-pos+1, varname, 
     .			 its_remote, status )
	IF ( status .NE. ferr_ok ) RETURN


* Was the variable a feature-mask?  If so then set the mask to the new definition.

	IF (is_fmask) THEN
	   slen = TM_LENSTR1(buff1) 
	   CALL RELOAD_DSG_MASKVAR (buff1, slen)
	ENDIF
	IF (is_smask) THEN
	   slen = TM_LENSTR1(buff1) 
	   CALL RELOAD_DSG_FTRMASKVAR (buff1, slen)
	ENDIF

*     Inherit all attributes from another variable. If /TITLE, /UNITS etc were
*     given on the LET command use them. INHERIT_ATTRS does not inherit the
*     missing flags or scale and offset attributes.

* ... LET/LIKE=var1 var

	iqual = qual_given( slash_let_like )

      IF ( iqual .GT. 0 ) THEN
	 CALL EQUAL_STR_LC ( cmnd_buff(qual_start(iqual):qual_end(iqual)),
     .			      buff1, status )
         CALL INHERIT_ATTRS (buff1, varname, buff3, buff2, status)
      ENDIF

* is there at least one slot blank ?
	DO i1 = 1, max_uvar
	   IF ( uvar_num_items(i1) .EQ. uvar_deleted ) RETURN
 	ENDDO

* no - not allowed to fill the last slot !
*	uvar_num_items(uvar) = uvar_deleted
        CALL deleted_list_modify(uvar_num_items_head, uvar,
     .                       uvar_deleted )
	CALL ERRMSG( ferr_prog_limit, status,
     .		'too many user-defined variables'//pCR//
     .		'cancel or redefine some variables', *5000 )


      ENDIF

**************************************************************************
* DEFINE AXIS
* ... /FILE=

      IF (subcmnd_num .EQ. 5) THEN

	IF ( qual_given( slash_file ) .GT. 0 ) THEN
	   subcmnd_num = 3
	   GOTO 300 ! like DEFINE GRID/FILE
	ENDIF

* initialize	
	buff2 = ' '

* DEFINE AXIS - what is "axisname" ?
	iqual = qual_given( slash_name )
	IF ( iqual .GT. 0 ) THEN
	   CALL EQUAL_STRING( cmnd_buff(qual_start(iqual):qual_end(iqual)),
     .			      buff3, status )
	   CALL EQUAL_STR_LC( cmnd_buff(qual_start(iqual):qual_end(iqual)),
     .			      varname, status )
	   IF ( status .NE. ferr_ok ) RETURN
	   pos = arg_start(1)	! cmnd arg is the expression to eval
	ELSE	   
	   IF ( num_args .LT. 1 ) GOTO 5300
	   CALL NAME_EQUAL_STRING( buff3, pos, status )
	   CALL NAME_EQUAL_STRING_LC( varname, pos, status )

	   IF ( status .NE. ferr_ok )  GOTO 5000

	   IF ( pos .GT. 0 ) THEN

* consolidate arguments broken up by command parser back into a single arg
              IF (cmnd_buff(pos:pos) .EQ. '{') THEN
                 CALL ALL_1_ARG
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

* get the X,Y,Z,T,E,F regions
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

* /LIKE Get the units and other attributes from an existing axis.

	line_alloc = .FALSE.
	is_like = .FALSE.  

	iqual = qual_given( slash_def_ax_like )
	IF ( iqual .GT. 0 ) THEN

* Find the axis to mimic

	   CALL EQUAL_STRING( cmnd_buff(qual_start(iqual):qual_end(iqual)),
     .			      likename, status )
*  ... first look in the static lines
	   DO line = 1, max_lines
	      IF ( interrupted ) CALL ERRMSG
     .			( ferr_interrupt, status, ' ', *5000 )
	      IF ( line_name(line) .EQ. char_init16 ) CYCLE 
	      IF ( line_name(line)(1:1) .EQ. '(' ) CYCLE 
	      IF (MATCH_TEMPLATE(line_name(line),likename)) GOTO 450
 	   ENDDO 

*  ... then the dynamic lines
              line = 0
 435          CALL TM_NEXT_DYN_LINE( line, *440)
                 IF ( interrupted ) CALL ERRMSG
     .			( ferr_interrupt, status, ' ', *5000 )
                 IF (MATCH_TEMPLATE(line_name(line),likename)) GOTO 450
	         GOTO 435
 440          CONTINUE
           GOTO 5760
 450	   CONTINUE

* find a space to catalog it
	   status = ALLO_MANAGED_AXIS( iline )
	   IF ( status .NE. merr_ok ) GOTO 5000
	   line_alloc = .TRUE.

* copy the line definition
	   CALL TM_COPY_LINE ( line, iline )

* But don't claim the line until everything has been worked through.
	   line_name(iline) = char_init16

* Below, if /units, /depth, /calendar, /t0, /modulo  are given,
* will override what we have got from TM_COPY_LINE

	   is_like = .TRUE.  

	   rmod_len = line_modulo_len( line )
	   new_att_modulo_len = rmod_len

	   IF ( line_modulo(iline) ) its_modulo = .TRUE.

	ENDIF ! end of /LIKE=

* orientation must be uniquely defined

	orient = unspecified_int4
	DO idim = 1, nferdims
	   IF ( qual_given(slash_x0+idim) .GT. 0 ) THEN   !!!!!!!!
	      IF ( orient .NE. unspecified_int4 ) GOTO 5510
	      orient = idim
	   ENDIF
	ENDDO

	IF ( irreg  .AND. orient.EQ.unspecified_int4) THEN
* ... orientation may be inferred from clues if irregular coord syntax is used
*     (Tkt 2380: only if direction undefined - may already know its a T or F axis)
	    IF ( qual_given(slash_depth) .GT. 0 ) orient = z_dim
	    IF ( qual_given(slash_T0)    .GT. 0 ) orient = t_dim
	ENDIF
        IF ( qual_given(slash_depth) .GT. 0
     . .AND. orient .NE. z_dim            ) GOTO 5505

* if orientation is unclear then assume it is X

	IF ( orient .EQ. unspecified_int4) THEN 
	   CALL WARN (
     .	        "Orientation not specified via /X,/Y,/Z,.... X assumed.")
             orient = x_dim
	ENDIF
	orient_t = orient.EQ.t_dim .OR. orient.EQ.f_dim

* release context stack space
	CALL STACK_PTR_DN ( cx_stack_ptr, cx_stack_ptr_base, status )
	IF ( status .NE. ferr_ok ) GOTO 5000

* /EDGES?
	its_edges =  qual_given( slash_edges ) .GT. 0

* /BOUNDS?
	has_bounds =  qual_given( slash_bounds ) .GT. 0
        IF (has_bounds .AND. its_edges) GOTO 5730

	bnds_or_edges = has_bounds .OR. its_edges

* is it a calendar axis definition?
	its_calendar =  orient_t .AND. cx_calendar(orient,mods_cx)

* /MONTHLY - set the true_month flag, time axes only. (what about F axes?)
	true_month = qual_given( slash_monthly ) .GT. 0

	IF (true_month .AND. orient.NE.t_dim) GOTO 5820

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
	ELSEIF ( orient_t
* 8/97 *kob* use cx_calender to check if time is date rather than time step
     .  .AND.  (  qual_given( slash_t0 ) .GT. 0
     .		.OR. cx_calendar(orient,mods_cx))) THEN
	    IF ( .NOT. true_month ) buff1 = 'HOURS'
	ELSE
	   IF (.NOT. is_like) buff1 = 'NONE'
	ENDIF

* Initialize with units if inherited from another line
	IF (is_like .AND. iqual.EQ.0) buff1 = line_units( iline )


* /MONTHLY

	timefac = 1.
	IF ( true_month ) THEN
	   IF (qual_given(slash_units) .GT. 0) THEN
	      IF (STR_SAME(buff1,      ' '   ) .EQ. 0 .OR.
     .		  STR_SAME(buff1(1:4), 'NONE') .EQ. 0 .OR.
     .		  STR_SAME(buff1(1:3), 'MON' ) .EQ. 0) THEN
	         buff1 = 'TRUMONTH'
	      ELSE
	         GOTO 5770
	      ENDIF
	   
	   ENDIF

* Do not allow /BOUNDS, /NPOINTS, irregular, 

	   IF (has_bounds) GOTO 5780 
	   IF (irreg) GOTO 5790

* maybe later allow def axis/monthly/npoints=120/t0=1-jan-2001.  Not allowed now.
	   IF (qual_given(slash_npoints) .GT. 0) GOTO 5800

* Check that any delta given in /t=LO:HI:DELTA has value of 1
* Should be able to handle that...

	   delta = cx_delta (orient,mods_cx)
	   IF ( delta.EQ.unspecified_val8 .OR. delta.LE.0.D0) GOTO 5810

	   buff1 = 'TRUMONTH'

* Edges is implied.  Issue a note if they didn't give /EDGES
	   
	   truemonth_noted = .FALSE.
	   IF (.NOT. its_edges) THEN
	      CALL WARN ('True monthly axis has an implicit /EDGES qualifier')
	      truemonth_noted = .TRUE.
	   ENDIF
	   its_edges = .TRUE.
	   
* Will use a default if t0 not given

	   IF (qual_given( slash_T0 ) .EQ. 0) THEN
	      IF (its_calendar) THEN 
	         CALL WARN (
     .          'True monthly axis, T0 not specified. Using start of first year of axis')
	      ELSE
	         CALL WARN (
     .          'True monthly axis, T0 not specified. Using 01-jan-0000')
	      ENDIF
	      truemonth_noted = .TRUE.
	   ENDIF

	ENDIF  ! end of /MONTHLY

* *acm* other calendar names* /CALENDAR = GREGORIAN, PROLEPTIC_GREGORIAN, 
*        NOLEAP, JULIAN, 360_DAY, ALL_LEAP
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
	   IF (is_like) line_cal_name ( iline ) = cal_name
	ELSEIF ( orient_t  .AND.  
     .         ( qual_given(slash_T0) .GT. 0  .OR. 
     .           cx_calendar(orient,mods_cx)) ) THEN
	   cal_name = 'GREGORIAN'
	ENDIF

	IF (is_like) cal_name = line_cal_name ( iline )

* 9/01 CF conventions have adopted the name 360_day for a 360-day year.

        IF (cal_name .EQ. '360') cal_name = '360_DAY' 
        IF (cal_name .EQ. 'D360') cal_name = '360_DAY' 
        IF (cal_name .EQ. 'STANDARD') cal_name = 'GREGORIAN' 
        IF (cal_name(1:6) .EQ. 'PROLEP') cal_name = 'PROLEPTIC_GREGORIAN' 
        IF (cal_name(1:6) .EQ. 'COMMON') cal_name = 'NOLEAP' 
        IF (cal_name(1:6) .EQ. '365_DA') cal_name = 'NOLEAP' 
        IF (cal_name(1:6) .EQ. '366_DA') cal_name = 'ALL_LEAP' 

        cal_id = TM_GET_CALENDAR_ID (cal_name)
	IF (cal_id .LE. 0  .OR. cal_id .GT. max_calendars ) GO TO 5720
        CALL TM_GET_CALENDAR_ATTRIBUTES (cal_id, nmonths, 
     .                 ndays, yeardays, mon_names, d_in_mon, 
     .                 d_before_mon, mon_by_d)
        IF (new_att_calendar .NE. " ") new_att_calendar = cal_name

* True-month calendar:
* If no time origin given, by default the monthly axis origin will be 
* the beginning of the first month of the axis, if calendar date range 
* given, or 1-jan-0000

	IF (true_month) THEN
	   IF (qual_given( slash_T0 ) .EQ. 0) THEN
	      IF (its_calendar) THEN
	         start = ABS(cx_lo_ww(orient,mods_cx))
	         datestr = SECS_TO_DATE_OUT (start, cal_id, .FALSE., 1)
                 buff2 = '01-JAN-'//datestr(1:4)
	      ELSE
	         buff2 = '01-JAN-0000'
	      ENDIF
	   ENDIF

	ENDIF 

* interpret the line units
	IF ( .NOT.is_like .AND. (buff1 .EQ. 'NONE' .OR. buff1 .EQ. ' ') ) THEN
	   buff1 = ' '
	   iunit = 0
	   
	ELSE
	   iunit = TM_UNIT_ID( buff1 )
	   IF ( iunit .EQ. 0 ) THEN
	      IF (.NOT.is_like) CALL WARN
     .		 ('(fyi) units are not subject to auto-conversion: '//buff1)

C /calendar=/unit=year gets length of year in that calendar 
C                      (true_month flag unused for yr axis)
           ELSEIF (iunit .EQ. -6 .OR. iunit .EQ. -10) THEN
	      iunit = TM_UNITS_CAL( iunit, cal_id, true_month )

C /units=months  is 1/12 of the length of the year
	   ELSEIF (iunit .EQ. -5) THEN    ! 1/97: "-5" is "MONTHS"
              WRITE (err_string,3540) yeardays
	      IF (cal_id .EQ. GREGORIAN) 
     .            WRITE (err_string,3544) yeardays
 
* Named calendar with unit=month or year gets length according to calendar's year length.

	      iunit = TM_UNITS_CAL( iunit, cal_id, true_month )

*  Append calendar name to units string.
              IF (cal_id .NE. GREGORIAN) THEN
		 slen = TM_LENSTR1 (buff1)
		 len_cal = TM_LENSTR1 (cal_name)
		 buff1 = buff1(:slen)//'('//cal_name(:len_cal)//')'
              ENDIF
	      
* issue a note about monthly axes

	      IF (true_month) THEN
	         WRITE (err_string,3546) 
		 IF (.NOT. truemonth_noted) CALL WARN (err_string) 
	      ELSE
	         CALL WARN (err_string)
	      ENDIF
	      
 3540         FORMAT 
     .        ('/UNIT=MONTHS  ... using 1/12 of ',F6.2,' days')
 3544         FORMAT 
     .        ('/UNIT=MONTHS is ambiguous ... using 1/12 of '
     .                                           ,F8.4,' days')
 3546         FORMAT ('True monthly axis')

	   ELSE

* ... Issue warning for inappropriate units like HOURS on X ax
*     and render the units unconvertible
 	      IF ( (  orient_t .AND.
     .		   ( iunit.LT.pun_1st_time .OR. iunit.GT.pun_last_time ) )
     .	      .OR. (  .NOT.orient_t .AND.
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

* find a space to catalog it (may already be done above with DEF AX/LIKE=)
	IF (.NOT. line_alloc) THEN
	   status = ALLO_MANAGED_AXIS( iline )
	   IF ( status .NE. merr_ok ) GOTO 5000
	ENDIF

* was /NPOINTS given ?
* Read as an INTEGER then it can be really large and retain accuracy.
	iqual = qual_given( slash_npoints )
	IF ( iqual .GT. 0 ) THEN
	   CALL EQUAL_val_int( cmnd_buff(qual_start(iqual):qual_end(iqual)),
     .			      ibuff, status )
	   IF ( status .NE. ferr_ok ) RETURN
	   IF ( ibuff .LE. 0 ) GOTO 5582
	   npoints = ibuff
	ELSE
	   npoints = unspecified_int4
	ENDIF

* GET THE AXIS POINTS: IRREGULAR (from_variable) or regular
	IF ( irreg ) THEN
           line_regular( iline ) = .FALSE.  ! this may be reset to true later.
* ... get axis points from a memory variable
	   CALL GET_CMND_DATA (  cx_last, ptype_float, status )
	   IF ( status .NE. ferr_ok ) RETURN
           IF ( num_uvars_in_cmnd .NE. 1 .AND.
     .         (.NOT. has_bounds) ) GOTO 5580
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

* Allocate line coordinate storage, put the pointer in array linemem(iline)
* and n+1 edges storage in lineedg(iline)

	   CALL GET_LINE_DYNMEM (n, iline, status) 
	   IF (status .NE. ferr_ok) GOTO 5000
	   
* ... If they gave upper- and lower-bounds we need to store all that,
*     temporarily. Store it in scratch line storage, line_0.
*     IF the allocation fails with bounds suggest edges.

	   IF (has_bounds) THEN

* If bounds are in a single extra variable they are of size 2*n
* If they are in two extra arrays, allocate another arrays of size n

              IF (num_uvars_in_cmnd .EQ. 3) THEN
	         CALL GET_LINE_DYNMEM (n, line_0, status) 
	         IF (status .NE. ferr_ok) GOTO 5592
              ELSEIF (n2 .EQ. 2*n) THEN
	         CALL GET_LINE_DYNMEM (n2, line_0, status) 
	         IF (status .NE. ferr_ok) GOTO 5592
              ENDIF
	   ENDIF

* ... store edge positions (error status already checked)
           IF (has_bounds) THEN
              CALL NON_ARRAY_SUBSC( is_mr, 2 )
           ELSE
              mr1(1) = mr
	      CALL NON_ARRAY_SUBSC( mr1, 1 )
           ENDIF

	   IF (its_edges) THEN
* ... "/edges" definition - 4/99
	      n = n - 1  ! edges array is one longer than points
	      IF (n .LE. 0) GOTO 5520
	      CALL EXTRACT_DP_LINE( cx, memry(mr)%ptr,
     .				 lineedg(iline)%ptr, idim, n+1 )
	      npoints = n


* ... monotonic?
	      DO i1 = 2, n
	         vhi  = GET_LINE_COORD (lineedg(iline)%ptr, i1)
	         vlo = GET_LINE_COORD (lineedg(iline)%ptr, i1-1)
	         IF (.NOT. TM_FPEQ( vlo, vhi ) .AND. vhi .LT. vlo) GOTO 5570
	      ENDDO

* ... are there repeated values?

	      has_repeated = .FALSE.
	      is_double = .TRUE.
	      misordered = .FALSE. 

	      use_strict = .FALSE.  ! unused for DEFINE AXIS
	      micro_adj = 1E-6 *
     .		(GET_LINE_COORD (lineedg(iline)%ptr, n+1) - 
     .		 GET_LINE_COORD (lineedg(iline)%ptr, 1))

	      CALL TM_CHECK_COORDS (lineedg(iline)%ptr, n+1, is_double, 
     .                              use_strict, has_repeated, misordered, 
     .                              micro_adj, epsilon, epsilon)
	      IF (misordered) GOTO 5560
	      IF (has_repeated) 
     .		CALL WARN('Axis has repeated values -- micro-adjusting ...')

* ... store points: midway between box boundaries ...

	      has_repeated = .FALSE.
	      DO i1 = 2, n+1
	         vhi = GET_LINE_COORD (lineedg(iline)%ptr,i1)
	         vlo = GET_LINE_COORD (lineedg(iline)%ptr,i1-1)
	         CALL PUT_LINE_COORD(linemem(iline)%ptr, i1-1, (vlo+vhi)* 0.5)
	      ENDDO

* Fix ticket 2400: Def of start was pointing to the lower grid cell edge not the
* lower coordinate. The loop start and stop were one off, so most axes detected
* as irregular as the last comparison took us beyond the upper edge of the axis.

	      its_reg = .TRUE.	! until proven otherwise
	      start = GET_LINE_COORD (linemem(iline)%ptr, 1)
              delta = GET_LINE_COORD (linemem(iline)%ptr, 2) - start
              DO i1 = 2, n+1
	         idel = GET_LINE_COORD (lineedg(iline)%ptr, i1) - GET_LINE_COORD (lineedg(iline)%ptr, i1-1)
		 IF (.NOT.TM_FPEQ( (delta),idel) ) THEN
		    its_reg = .FALSE.
		    EXIT ! loop
		 ENDIF
	      ENDDO

	   ELSE

* NOT EDGES -- POINTS

	      CALL EXTRACT_DP_LINE( cx, memry(mr)%ptr,
     .				 linemem(iline)%ptr, idim, n )
	      IF ( npoints .EQ. unspecified_int4 ) npoints = n

* ... a single point is always "regular" (unless definition has BOUNDS!!)
	      IF (n .EQ. 1 .AND. .NOT.has_bounds) THEN
	         start = GET_LINE_COORD (linemem(iline)%ptr, 1)
	         delta = 1
	         its_reg = .TRUE.
	         GOTO 560
	      ENDIF
	      
* ... monotonic?	      

	      DO i1 = 2, npoints  
	         vhi  = GET_LINE_COORD (linemem(iline)%ptr, i1)
	         vlo = GET_LINE_COORD (linemem(iline)%ptr, i1-1)
	         IF (.NOT. TM_FPEQ( vlo, vhi ) .AND. vhi .LT. vlo) GOTO 5570
	      ENDDO

* ... are there repeated values?

	      has_repeated = .FALSE.
	      is_double = .TRUE.
	      misordered = .FALSE. 

	      use_strict = .FALSE.  ! unused for DEFINE AXIS
	      micro_adj = 1E-6 *
     .		(GET_LINE_COORD (linemem(iline)%ptr, n) - 
     .		 GET_LINE_COORD (linemem(iline)%ptr, 1))

	      CALL TM_CHECK_COORDS (linemem(iline)%ptr, npoints, is_double, 
     .                              use_strict, has_repeated, misordered, 
     .                              micro_adj, epsilon, epsilon)
	      IF (misordered) GOTO 5560
	      IF (has_repeated) 
     .		CALL WARN('Axis has repeated values -- micro-adjusting ...')

* Bounds specified by user

              IF (has_bounds) THEN

                 cx = is_cx(2)
                 mr = is_mr(2)

		 IF (n2 .EQ. n+1) THEN 

                    IF (num_uvars_in_cmnd .EQ. 2) THEN
                       CALL EXTRACT_DP_LINE2( cx, memry(mr)%ptr,
     .	  	       lineedg(iline)%ptr, idim, n2 )
		    ENDIF

* If given as coords, lo_bounds, hi_bounds, then
* 1) check that lo_bounds{i+1} = hi_bounds{i}
* 2) If ok, add the upper high bound to the list and treat 
*    this as the case where we give the bounds as N+1 values

                 ELSE IF (n2 .EQ. 2*n) THEN
                    CALL EXTRACT_DP_LINE2( cx, memry(mr)%ptr,
     .	  	       lineedg(line_0)%ptr, idim, n2 )
		 ENDIF

		 IF (num_uvars_in_cmnd .EQ. 3) THEN
		    
                    CALL EXTRACT_DP_LINE2( cx, memry(mr)%ptr,
     .	  	    lineedg(iline)%ptr, idim, n2 )

                    cx = is_cx(3)
                    mr = is_mr(3)
                    CALL EXTRACT_DP_LINE2( cx, memry(mr)%ptr,
     .	 	       lineedg(line_0)%ptr, idim, n3 )
                    DO i = 1, n-1
		       vlo = GET_LINE_COORD (lineedg(iline)%ptr, i+1)
		       vhi = GET_LINE_COORD (lineedg(line_0)%ptr, i)
                       IF (vlo .NE. vhi  ) GOTO 5750
                    ENDDO
		    vhi = GET_LINE_COORD (lineedg(line_0)%ptr, n)
                    CALL PUT_LINE_COORD (lineedg(iline)%ptr, n+1, vhi)
		    n2 = n + 1

*  deallocate the extra n storage used for bounds.
                    CALL FREE_LINE_DYNMEM(line_0)

                 ENDIF

* Check that bounds enclose points, and that they are contiguous

		 line_dim    ( iline ) = npoints
		 IF (n2 .EQ. n+1) THEN
  		     IF ( .NOT. TM_CHECK_BOX_LOC (linemem(iline)%ptr, 
     .                     lineedg(iline)%ptr, n2) ) GOTO 5750
                 
		 ELSE
                    IF (.NOT. TM_CHECK_BNDS (linemem(iline)%ptr, 
     .                     lineedg(line_0)%ptr, n, 
     .                     iline, .TRUE., buff3) ) GO TO 5750
		    
* Change from N*2 storage of bounds to N+1 edges.
                    CALL TM_CONVERT_BOUNDS_EDGES (lineedg(line_0)%ptr, 
     .                  lineedg(iline)%ptr, n)

*  deallocate the temporary 2*n  storage used for bounds.
                    CALL FREE_LINE_DYNMEM(line_0)
		 ENDIF

              ELSE

*... create bounds equidistant between grid points

*    ... store initial box lower bound (start of axis)

                vlo = GET_LINE_COORD (linemem(iline)%ptr, 1)
                vhi = GET_LINE_COORD (linemem(iline)%ptr, 2)
                val = vlo - (vhi-vlo)/2.
		CALL PUT_LINE_COORD (lineedg(iline)%ptr, 1, val)
*    ... mid-axis box boundaries ...
	         DO i1 = 2, npoints
                    vlo = GET_LINE_COORD (linemem(iline)%ptr, i1-1)
                    vhi = GET_LINE_COORD (linemem(iline)%ptr, i1)
                    val = vlo + (vhi-vlo)/2.
                    CALL PUT_LINE_COORD (lineedg(iline)%ptr, i1, val)
		 ENDDO

*    ... upper-most box bound (end of axis)
	         val = vhi + (vhi-vlo)/2.
		 CALL PUT_LINE_COORD (lineedg(iline)%ptr, npoints+1, val)

	         its_reg = .TRUE.	! until proven otherwise

	         its_reg = .TRUE.	! until proven otherwise
	         start = GET_LINE_COORD (linemem(iline)%ptr, 1)
                 delta = GET_LINE_COORD (linemem(iline)%ptr, 2) - start
                 DO i1 = 2, npoints
	            idel = GET_LINE_COORD (lineedg(iline)%ptr, i1) - GET_LINE_COORD (lineedg(iline)%ptr, i1-1)
		    IF (.NOT.TM_FPEQ( (delta),idel) ) THEN
		       its_reg = .FALSE.
		       EXIT ! from loop
		    ENDIF
		 ENDDO
              ENDIF

	   ENDIF  ! end of irreg

* ... check to see if it was regularly spaced after all
 560	   IF ( its_reg ) THEN
	      line_start    ( iline ) = start
	      line_delta    ( iline ) = delta
	      line_regular  ( iline ) = .TRUE.
* ... save irregular line data
	   ELSE
	      line_start  ( iline ) = GET_LINE_COORD (linemem(iline)%ptr, 1)
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

	   IF ( orient_t  .AND.  cal_id .NE. prev_cal_id .AND.
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
	       
* Check for delta of 1 on true-month axis (for now)
	       IF (true_month .AND. delta .NE. 1) GOTO 5810 

	       IF ( .NOT.its_calendar ) THEN
	          npoints = ( (end - start) / delta ) + 0.9999
	       ENDIF
	     ELSE
	       delta = (end - start) / npoints
	       IF (its_calendar) delta = -1 * delta/un_convert(iunit)
	       IF ( delta .LE. 0.0D0 ) GOTO 5520
	     ENDIF
	     IF ( its_calendar ) THEN
	       start = start - (delta/2.0)*un_convert(iunit)
	       end   = end   + (delta/2.0)*un_convert(iunit)
	     ELSE
	       start = start + delta/2.0
	       end   = end   - delta/2.0
	     ENDIF

	     steps_total = (end - start) / delta
	     IF (steps_total .GT. arbitrary_large_int4) GOTO 5850

	   ELSE
	     IF ( npoints .EQ. unspecified_int4 ) THEN
	       delta = cx_delta (orient,mods_cx)
	       IF ( delta.EQ.unspecified_val8 .OR. delta.LE.0.D0) GOTO 5520
	       IF ( .NOT.its_calendar ) THEN
	       
	          steps_total = 1.9999 + (end - start) / delta
	          IF (steps_total .GT. arbitrary_large_int4) GOTO 5850

	          npoints = INT(steps_total)
                  small = MIN(1.e-9, delta)
		  IF (start + delta* (npoints-1) - 0.5*delta .GT. end+small) 
     .              npoints = npoints-1
	       ENDIF
	     ELSEIF ( npoints .EQ. 1 ) THEN
	       IF ( end .NE. start ) GOTO 5530
	       delta = 1.D0
	     ELSE
	       delta = (end - start) / (npoints-1)
	       steps_total = (end - start) / delta
	       IF (steps_total .GT. arbitrary_large_int4) GOTO 5850

	       IF (its_calendar) delta = -1 * delta/un_convert(iunit)
	       IF ( delta .LE. 0.0D0 ) GOTO 5520
	     ENDIF
	   ENDIF

	   line_start    ( iline ) = start
	   line_delta    ( iline ) = delta
	   line_regular  ( iline ) = .TRUE.
	ENDIF

	IF (irreg .AND. line_regular(iline)) CALL FREE_LINE_DYNMEM (iline)

* /T0 = date (or supply default)
	iqual = qual_given( slash_T0 )
        new_att_t0 = ' '
	
	IF (true_month .AND. buff2 .NE. ' ') iqual = 1
	IF (is_like) buff2 = line_t0 ( iline )

	IF ( iqual .GT. 0 ) THEN
	   IF (is_like) 
     .       CALL EQUAL_STRING( cmnd_buff(qual_start(iqual):qual_end(iqual)),
     .			      buff2, status )
	   IF (buff2 .EQ. ' ') 
     .       CALL EQUAL_STRING( cmnd_buff(qual_start(iqual):qual_end(iqual)),
     .			      buff2, status )
	   IF ( status .NE. ferr_ok ) RETURN

	   status = ferr_grid_definition       ! status setting if tm_date_ok false
	   is_ok = TM_DATE_OK ( buff2 , cal_id) 

	   IF ( is_ok  ) CALL TM_BREAK_DATE( buff2, cal_id, 
     .                        i1,i2,i3,i4,i5,i6,  status )
	   IF ( buff2 .EQ. ' ' .OR. status .NE. ferr_ok ) GOTO 5540
           new_att_t0 = buff2
	ELSEIF (is_like .AND. line_t0 ( iline ).NE.char_init20) THEN
	   i1 = 1  ! line-t0 is all set
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

        IF (orient_t) THEN
           IF (new_att_units .NE. ' ') THEN
              i1 = TM_LENSTR1(new_att_units)
              IF (new_att_t0 .NE. ' ') 
     .            new_att_units = new_att_units(1:i1)//' since ' 
     .            // new_att_t0
           ENDIF
        ENDIF

* save the line in memory - but don't catalog its name
	line_dim      ( iline ) = npoints
	line_units    ( iline ) = buff1	   
	line_t0       ( iline ) = buff2
	line_cal_name ( iline ) = cal_name
	line_unit_code( iline ) = iunit
	IF ( orient_t ) line_tunit(iline) = un_convert( iunit )

* for /LIKE=, Direction must match that of the original axis

	IF (is_like) THEN
	   IF (orient .EQ. x_dim) THEN
	      IF (line_direction(iline).NE.'XX' .AND. line_direction(iline).NE.'WE') GOTO 5765
	   ENDIF
	   IF (orient .EQ. y_dim) THEN
	      IF (line_direction(iline).NE.'YY' .AND. line_direction(iline).NE.'SN') GOTO 5765
	   ENDIF
	   IF (orient .EQ. z_dim) THEN
	      IF (line_direction(iline).NE.'ZZ' .AND. line_direction(iline).NE.'DU' 
     .             .AND. line_direction(iline).NE.'UD'   ) GOTO 5765
	   ENDIF
	   IF (orient .EQ. t_dim) THEN
	      IF (line_direction(iline).NE.'TT' .AND. line_direction(iline).NE.'TI') GOTO 5765
	   ENDIF
	   IF (orient .EQ. e_dim) THEN
	      IF (line_direction(iline).NE.'EE') GOTO 5765
	   ENDIF
	   IF (orient .EQ. f_dim) THEN
	      IF (line_direction(iline).NE.'FF' .AND. line_direction(iline).NE.'FI') GOTO 5765
	   ENDIF

	ELSE
	   line_direction( iline ) = 'NA'
	ENDIF

* is it a formattable axis ?
	IF ( orient .EQ. x_dim ) THEN
	   line_direction( iline ) = 'XX'
           IF (iunit .EQ. pun_degrees) THEN
	      line_direction( iline ) = 'WE'
	      line_units( iline ) = 'degrees_east'
	   ENDIF

	ELSEIF ( orient .EQ. y_dim  ) THEN
           line_direction( iline ) = 'YY'
	   IF (iunit .EQ. pun_degrees) THEN
	      line_direction( iline ) = 'SN'
	      line_units( iline ) = 'degrees_north'
	   ENDIF

	ELSEIF ( orient .EQ. z_dim ) THEN
           IF ( qual_given( slash_depth ) .GT. 0 ) THEN
	      line_direction( iline ) = 'UD'    ! GEOG_LABEL requires "meters"
           ELSE
	      IF (.NOT.is_like) line_direction( iline ) = 'DU'
           ENDIF

	ELSEIF ( orient .EQ. e_dim ) THEN
           line_direction( iline ) = 'EE'

* TODO: let  F axes have time units
	ELSEIF ( orient .EQ. f_dim ) THEN
           line_direction( iline ) = 'FF'
	   IF (.NOT.irreg  .AND. its_calendar ) THEN
	      line_direction(iline) = 'FI'
	      bc_to_t0 = SECS_FROM_BC( line_t0(iline), cal_id, status)
	      steps_total = (start-end) / (line_tunit(iline)*delta) + 1.9999
	      IF (steps_total .GT. arbitrary_large_int4) GOTO 5850
	      line_dim  (iline) = steps_total
	      line_start(iline) = (-line_start(iline)-bc_to_t0)/line_tunit(iline)

	   ENDIF
	   IF ( qual_given(slash_T0).GT.0 ) line_direction(iline) = 'FI'

* ... time axis specified as date strings  (ELSEIF mod 11/93 *sh*)
	ELSEIF (.NOT.irreg  .AND. its_calendar ) THEN
	   line_direction(iline) = 'TI'
	   bc_to_t0 = SECS_FROM_BC( line_t0(iline), cal_id, status)
	   steps_total = (start-end) / (line_tunit(iline)*delta) + 1.9999
	   IF (steps_total .GT. arbitrary_large_int4) GOTO 5850
	   line_dim  (iline) = steps_total

	   line_start(iline) = (-line_start(iline)-bc_to_t0)/line_tunit(iline)

* ... time axis specified as time step values with T0 given
	ELSEIF ( orient.EQ.t_dim .AND. qual_given(slash_T0).GT.0 ) THEN
	   line_direction(iline) = 'TI'
	ELSEIF ( orient.EQ.t_dim .AND. line_direction(iline) .EQ. 'NA') THEN
	   line_direction(iline) = 'TT'
	ENDIF

* ... true-month calendar given without calendar date range, fill in
*     start, count, direction.

	IF (true_month .AND. .NOT.its_calendar) THEN
	    line_start(iline) = cx_lo_ww(orient,mods_cx) - delta/2
	    line_direction(iline) = 'TI'
	    steps_total = 1 + 
     .       ( cx_hi_ww(orient,mods_cx) - cx_lo_ww(orient,mods_cx) )/ delta
	    IF (steps_total .GT. arbitrary_large_int4) GOTO 5850
	    line_dim(iline) = steps_total
	ENDIF

* automatic longitude modulo
	axwwlen = TM_WW_AXLEN(iline)
	IF (line_direction(iline) .EQ. 'WE' ) THEN
	  IF (axwwlen .LE. 360.D0 .AND. (npoints.GT.1)) THEN
	     its_modulo = .TRUE.
	     IF (rmod_len .EQ. 0.0D0)  rmod_len = 360.D0
	     new_att_modulo = .TRUE.
	     new_att_modulo_len = rmod_len
	  ELSE IF (.NOT.has_bounds .AND. .NOT.line_regular(iline)) THEN 
	    CALL TM_ADJUST_BOUNDS(iline, line_dim(iline), 
     .       360., axwwlen, its_modulo)

	     IF (its_modulo) THEN
	        rmod_len = 360.D0
	        new_att_modulo = .TRUE.
	        new_att_modulo_len = rmod_len
	     ENDIF   
	  ENDIF

*  Calendar modulo
	ELSEIF (.NOT.its_modulo .AND. 
     .          (line_direction(iline) .EQ. 'TI' .OR.
     .           line_direction(iline) .EQ. 'FI') ) THEN
	  secsperyear = DBLE(cals_yeardays(cal_id)) * (24.D0 * 60.D0 * 60.D0)
	  bc_to_t0 = SECS_FROM_BC( line_t0(iline), cal_id, status)
	  CALL TM_WW_AX_1_N(iline, start, end)
	  secs2start = bc_to_t0 + start*line_tunit(iline)
	  
* Set the conversion factor for true-month, months to days
	   grid_line(t_dim, dgrid_buff) = iline
	   CALL TM_MONTH_TIME_FACTOR (dgrid_buff, true_month, timefac)

	  IF (true_month) secs2start = secs2start/ timefac

	  IF ( (secs2start .LE. secsperyear)
     .	 .AND. (axwwlen*line_tunit(iline) .LE. secsperyear) ) THEN
	     its_modulo = .TRUE.
	     IF (rmod_len .EQ. 0.0D0)  rmod_len =
     .				       timefac* secsperyear/line_tunit(iline)
	     new_att_modulo = .TRUE.
	     new_att_modulo_len = rmod_len

* Is this an axis where we added bounds?  If so they might be adjustable
* to make the axis modulo

           ELSEIF (.NOT.line_regular(iline) .AND. .NOT.bnds_or_edges) THEN  
	      delta = end - start
              IF ( (secs2start .LE.  2*secsperyear)
     .	           .AND. (delta*line_tunit(iline) .LE. secsperyear) ) THEN
	         rmod_len = secsperyear/line_tunit(iline)
		 CALL TM_ADJUST_BOUNDS( iline, line_dim(iline), 
     .              rmod_len, axwwlen, okmod)

	         IF (okmod) THEN
		    its_modulo = .TRUE.
	            rmod_len = axwwlen
	            new_att_modulo = .TRUE.
	            new_att_modulo_len = rmod_len
		 ENDIF
              ENDIF
	  ENDIF

	ENDIF

* Ticket 2504: If the coordinates on an irregular axis lie within the modulo 
* length but bounds that were automatically computed from coordinate midpoints 
* make the axis too long, reset the bounds so the axis is the modulo length.
* (may have been done for special cases above)

	IF (irreg .AND. its_modulo .AND. .NOT.bnds_or_edges .AND. rmod_len.GT.0.) THEN
	   CALL TM_ADJUST_BOUNDS( iline, line_dim(iline),  
     .              rmod_len, axwwlen, okmod)

	   IF (.NOT.okmod) its_modulo = .FALSE.
	ENDIF

* set modulo and sanity check it

	line_modulo( iline) = its_modulo
	line_modulo_len( iline ) = rmod_len
	IF (rmod_len .NE. 0.0D0) THEN
	   IF (axwwlen .GT. rmod_len) THEN
             IF (TM_DFPEQ(axwwlen, rmod_len)) THEN
	       line_modulo_len( iline ) = 0.0
             ELSE
               GOTO 5525
	     ENDIF
           ENDIF
	ENDIF


* If the user said /monthly and /modulo, store the modulo length in months

	IF (true_month .AND. its_modulo) THEN
	   axwwlen = TM_WW_AXLEN(iline)  ! months, make it an integer
	   IF (INT(axwwlen+0.5) .LE. 12.) THEN
	      rmod_len = DFLOAT(INT(axwwlen+0.5))
	      axwwlen = rmod_len
	   ENDIF
	   line_modulo_len(iline) = rmod_len
	ENDIF

* Write a note when subspan modulo axis length within a grid cell of modulo length
c	CALL TM_WARN_SUBSPAN (iline)


* ... Was it a true-month axis (defined in the classic way without the /MONTHLY qualifier)?
* 1/13/2017 ticket 2497, Back off this auto-detection of monthly axes for now.
*     (If this is implemented, needs to be changed from line_mem() references to dynamic
*     coordinate methods.)
c	IF (.NOT. true_month .AND. irreg .AND. .NOT.line_modulo( iline)) THEN
c           units = TM_UNIT_ID( line_units(iline) )
c	   CALL TM_CHECK_MONTHLY_AXIS (linemem(iline)%ptr,, line_dim(iline), 
c     .              cal_id, line_t0(iline), units, line_units(iline),
c     .              start, delta, line_tunit(iline), true_month)
c	   IF (true_month) THEN
c	      line_regular(iline) = .TRUE.
c	      line_start  (iline) = start !* line_tunit(iline)/ un_convert(pun_day)
c	      line_delta  (iline) = delta
c	      last_coord = line_start(iline) + DBLE(npts-1)*line_delta(iline)
c	      firs_coord = start
c	      line_unit_code (iline) = units
c	      line_tunit  (iline) = un_convert(units)
c	      its_edges = .FALSE.
c	      irreg = .FALSE.
c	   ENDIF
c	ENDIF


* ... coordinate pointer for next line to save
* Bounds are stored as edges, so increase by N coordinates and N+1 edges

* check for redefinitions
	IF ( dup_name ) THEN
	   IF ( .NOT.is_like .AND. TM_LINE_MATCH( old_line, iline ) ) THEN
	      RETURN					! dup name/dup def
	   ELSE
* ... redefinition makes all past assumptions incorrect

* Dont write message if DEF AX/QUIET
	      IF (qual_given( slash_def_ax_quiet ) .EQ. 0 ) THEN
	         original = .TRUE.
	         CALL CHOOSE_LINE_NAME(old_line, original, buff)
	         CALL SPLIT_LIST(pttmode_ops, err_lun,
     .			replmsg//'axis '//buff, 0)
	      ENDIF

* ... Check the attribute structure for this axis name and change 
*     attribute values as needed.  

	      IF (.NOT. is_like) 
     .           CALL REDEFINE_AX_ATT (line_name(old_line), new_att_units, 
     .           new_att_t0, new_att_calendar, new_att_modulo,
     .           new_att_modulo_len, line_direction(iline), 
     .           line_regular(iline) )

	      CALL PURGE_MR_AXIS( old_line, iline, status )
	      IF ( status .NE. ferr_ok ) GOTO 5000
	      CALL TM_DEALLO_DYN_LINE( old_line )

	   ENDIF

        ELSE

* Add newly defined axis to the user-variable dset of the linked-list structure.
* Save the original upper-or lower-case spelling
           
           dset_to_add = pdset_coordvars  ! user-defined coordinate variable dataset.

           err_string = varname
           type = NCDOUBLE
           coordvar = 1
           varid = 0		 ! Will be returned with variable ID
           bad_flag = bad_val4   ! coord vars dont have missing, but set to something
           CALL CD_ADD_COORD_VAR( dset_to_add, varname, varid,  
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
           IF (dcode .EQ. 'EE') dir = 5
           IF (dcode .EQ. 'FF' .OR. dcode .EQ. 'FI') dir = 6

           attname = 'point_spacing'
           attype = NCCHAR
           attoutflag = 1
	   its_reg = .NOT.irreg .AND. .NOT.true_month
	   IF (true_month .AND. cal_name .EQ. '360_DAY') its_reg = .TRUE.
           IF ( its_reg ) THEN
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
           IF (dcode .EQ. 'EE') axdir = 'E'
           IF (dcode .EQ. 'FF' .OR. dcode .EQ. 'FI') axdir = 'F'
           attlen = 1
           CALL CD_PUT_NEW_ATTR (dset_to_add, varid, attname, 
     .           attype, attlen, attoutflag, axdir, dummy, status) 

* Set the axis direction

           dir = 0
           IF (axdir .EQ. 'X') dir = 1
           IF (axdir .EQ. 'Y') dir = 2
           IF (axdir .EQ. 'Z') dir = 3
           IF (axdir .EQ. 'T') dir = 4
           IF (axdir .EQ. 'E') dir = 5
           IF (axdir .EQ. 'F') dir = 6
           CALL CD_SET_ATT_AXDIR (dset_to_add, varid, dir, status)

* Bounds on irregular axis output by default.

           IF (.NOT.its_reg) THEN
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

           IF (line_direction(iline)(1:1) .EQ. 'T' .OR.
     .         line_direction(iline)(1:1) .EQ. 'F') THEN
              IF (its_calendar) THEN
                 attname = 'calendar'
                 attype = NCCHAR
                 attoutflag = 1
                 attlen = TM_LENSTR1(cal_name)

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
	line_name_orig( iline ) = varname
	RETURN

      ENDIF


**************************************************************************
* DEFINE VIEWPORT 

      IF (subcmnd_num .EQ. 6) THEN

	IF ( num_args .NE. 1 ) GOTO 5300
 
	r4_unspec = unspecified_val4
* check out viewport name
	buff1 = cmnd_buff( arg_start(1):arg_end(1) )
	ivp = VIEWPORT_NUMBER( buff1 )
	IF ( ivp .EQ. unspecified_int4 ) THEN
* ... new name - find an unused slot
	   DO ivp = 1, max_viewport
	      IF ( vp_name(ivp) .EQ. unspecified_name4 ) GOTO 620
 	   ENDDO
	   GOTO 5610	! no free slots
	ELSE
* ... name in use - flag segments in this viewport as deleted
	   IF ( ivp .EQ. mvp_dflt ) GOTO 5640
	   CALL SPLIT_LIST(pttmode_ops, err_lun,
     .			'Re-defining viewport '//vp_name(ivp), 0)
	   DO iseg = vp_seg0(ivp)+1 , vp_segn(ivp)
	      CALL FGD_GDSG( iseg ) 
 	   ENDDO
	   vp_seg0 ( ivp ) = 0
	   vp_segn ( ivp ) = 0
*          make sure the change is immediately seen
           CALL FGD_GUWK(wsid, 1)
	ENDIF

* pre-set all values to defaults
 620	vp_size ( ivp ) = 1.0
	vp_xorg ( ivp ) = 0.0
	vp_yorg ( ivp ) = 0.0
	vp_xclip( ivp ) = r4_unspec   ! old style  3/29/93
	vp_yclip( ivp ) = r4_unspec   ! old style  3/29/93
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
	   vx = vp_xorg(ivp)
	   vy = vp_yorg(ivp)
	   IF ( OUTSIDE_0_1(vx) .OR. OUTSIDE_0_1(vy) )
     .	      							   GOTO 5620
	ENDIF

* /SIZE=q
	qp = qual_given( slash_size )
	IF ( qp .GT. 0 ) THEN
	   IF (vp_by_axis(ivp)) GOTO 5660
	   CALL EQUAL_VAL( cmnd_buff(qual_start(qp):qual_end(qp)),
     .			   val_buf, status )
	   vp_size(ivp) = val_buf
	   IF ( status .NE. ferr_ok ) RETURN
	   IF ( val_buf .LT. 0.0 ) GOTO 5620
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
	   IF ( vp_xclip(ivp).EQ.r4_unspec) THEN
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
	   vx = vp_xorg(ivp)
	   vy = vp_xclip(ivp)
	   IF ( OUTSIDE_0_1(vx) .OR. OUTSIDE_0_1(vy)
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
	   vx = vp_yorg(ivp)
	   vy = vp_yclip(ivp)
	   IF ( OUTSIDE_0_1(vx) .OR. OUTSIDE_0_1(vy)
     .     .OR. vp_yorg(ivp) .GE. vp_yclip(ivp) )  GOTO 5620
	ENDIF

* /TEXT=q 
	qp = qual_given( slash_text )
	IF ( qp .GT. 0 ) THEN
	   IF ( qual_given(slash_size) .GT. 0 ) GOTO 5650
	   CALL EQUAL_VAL( cmnd_buff(qual_start(qp):qual_end(qp)),
     .			   val_buf, status )
	   IF ( status .NE. ferr_ok ) RETURN
	   vp_size(ivp) = val_buf
	   IF ( val_buf .LT. 0.001 ) GOTO 5620
	ELSEIF ( qual_given(slash_size) .EQ. 0 ) THEN     ! auto-sizing
           vp_size(ivp) = SQRT( (vp_xclip(ivp)-vp_xorg(ivp))
     .                        * (vp_yclip(ivp)-vp_yorg(ivp)) )
	   IF (vp_by_axis(ivp)) THEN
* ... vieport size is inferred from users axis limits and depends upon scale
* ... iterate a few times to approximate the correct scale factor
	      DO i1 = 1, 3
	        scale = 1.0 / SQRT( ABS(vp_size(ivp)) )
                xoei = wn_xwhitelo(wsid)  ! ppl xorigin
                yoei = wn_ywhitelo(wsid)  ! ppl yorigin
                xcei = wn_xwhitehi(wsid)  ! white space to right of X axis
                ycei = wn_ywhitehi(wsid)  ! white space for docmtn at top of plot
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

      ENDIF


**************************************************************************
* DEFINE ALIAS

      IF (subcmnd_num .EQ. 7) THEN

	IF ( num_args .LT. 1 ) GOTO 5300
	CALL NAME_EQUAL_STRING( buff1, pos, status )
	IF ( status .NE. ferr_ok )  GOTO 5000
	IF ( pos .EQ. 0 ) GOTO 5400
	IF ( .NOT.TM_LEGAL_NAME(buff1) ) GOTO 5410

* determine the name of the command being aliased
        DO i1 =  pos, arg_end(num_args)
           IF (cmnd_buff(i1:i1) .EQ. ' '
     .    .OR. cmnd_buff(i1:i1) .EQ. tab ) THEN
              i2 = i1 - 1   ! last char of command being aliased
              GOTO 705
           ENDIF
        ENDDO
        i2 = arg_end(num_args)

* prevent IF, ELIF, ELSE and ENDIF from being aliased, as this would mess
* up flow control in PROCESS_IF and elsewhere
* ... special test for IF command, cuz 1-line IFs are OK to alias 
 705    IF ( MATCH4(cmnd_buff(pos:i2),i2-pos+1,'IF',2) ) THEN
           i1 = STR_UPCASE (risc_buff, cmnd_buff)
           IF (.NOT.ITSA_1LINEIF(risc_buff(pos:arg_end(num_args)),
     .                           arg_end(num_args)-pos+1)) GOTO 5200
        ENDIF
        DO i1 = 1, 3
           IF (MATCH4(cmnd_buff(pos:i2),i2-pos+1,
     .                reserved_flow_name(i1),4   )) GOTO 5200
        ENDDO

* find a slot to save this alias
        i1 = ALIAS_ID( buff1 )
        IF ( i1 .EQ. unspecified_int4 ) THEN
	   DO i1 = num_predefined_alias+1, total_num_alias
	      IF ( alias_name(i1) .EQ. unspecified_name4 ) GOTO 720
	   ENDDO
           GOTO 5710
        ENDIF

* save the alias
 720    vax_code = STR_UPCASE( alias_name(i1), buff1 ) 
        alias(i1) = cmnd_buff(pos:arg_end(num_args))
        alias_len(i1) = TM_LENSTR1( alias(i1) )

	RETURN

      ENDIF


**************************************************************************
* DEFINE SYMBOL

      IF (subcmnd_num .EQ. 8) THEN

	IF ( num_args .LT. 1 ) GOTO 5300
	CALL NAME_EQUAL_STRING( buff1, pos, status )
	IF ( status .NE. ferr_ok )  GOTO 5000
	IF ( pos .EQ. 0 ) GOTO 5400
	IF ( .NOT.TM_LEGAL_NAME(buff1) ) GOTO 5410

* if this symbol already exists then delete it
        buffsym = buff1
	CALL GETSYM( buffsym, err_string, i2, i1 )
	IF ( i1 .EQ. 0 ) CALL DELSYM(buffsym,i1)

* remove backslash escape characters 
! OLD COMMENT: exactly 255 for PPL; no longer the case as of Ferret v6.01
	err_string = cmnd_buff(pos:arg_end(num_args))   
	CALL DE_ESCAPE_STRING( err_string, slen )

* define the symbol
	CALL PUTSYM( buffsym, err_string, slen, i2)
! ... no check on returned error code (could be "internal error")
	RETURN

      ENDIF


**************************************************************************
* DEFINE ATTRIBUTE /D= /TYPE= /OUTPUT /QUIET

      IF (subcmnd_num .EQ. 9) THEN

	IF ( num_args .LT. 1 ) GOTO 5900
	CALL NAME_EQUAL_STRING_VATT ( buff1, pos, status )
	IF ( status .NE. ferr_ok )  GOTO 5000
	IF ( pos .EQ. 0 ) GOTO 5400
	IF ( INDEX(buff1,'..') .EQ. 1 ) THEN
           ivar = 0  ! global attr.
           varname = '.'
	ELSE
           IF ( .NOT.TM_LEGAL_NAME(buff1) ) GOTO 5100
        ENDIF

* ... get dset number if present:  DEF ATT/D=dset ...
	iqual = qual_given( slash_def_att_dset ) 
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
*   5/16 allow other types on output.
	iqual = qual_given( slash_def_att_type) 
        attype_spec = ptype_unknown
        IF ( iqual .GT. 0 ) THEN
	   CALL EQUAL_STRING( cmnd_buff(qual_start(iqual):qual_end(iqual)),
     .			      buff2, status )  
	   IF ( status .NE. ferr_ok ) RETURN
	   IF ( STR_SAME (buff2(1:3), 'str') .EQ. 0) THEN
	      attype_spec =  2   ! NCCHAR
           ELSE IF (STR_SAME (buff2(1:3), 'flo') .EQ. 0) THEN
              attype_spec = 5   ! NCFLOAT
           ELSE IF (STR_SAME (buff2(1:3), 'byt') .EQ. 0) THEN
              attype_spec = 1   ! NCBYTE
           ELSE IF (STR_SAME (buff2(1:3), 'sho') .EQ. 0) THEN
              attype_spec = 3   ! NCSHORT
           ELSE IF (STR_SAME (buff2(1:3), 'int') .EQ. 0) THEN
              attype_spec = 4   ! NCLONG, or in c, NC_INT (4-byte integer)
           ELSE IF (STR_SAME (buff2(1:3), 'dou') .EQ. 0) THEN
              attype_spec = 6   ! NCDOUBLE
           ELSE 
              GO TO 5940
	   ENDIF
	ENDIF

* ... DEFINE ATT/OUTPUT varname.attname.  By default output flag is set to 1
      attoutflag = 1
      iqual = qual_given( slash_def_att_output ) 
      IF ( iqual .GT. 0 ) attoutflag = 1 

* ... DEFINE ATT/QUIET varname.attname
      attoutflag = 1
      def_att_quiet = qual_given( slash_def_att_quiet ) .GT. 0

* get the requested data somehow

        IF (arg_start(1) .LT. pos) arg_start(1) = pos
	CALL GET_PROT_CMND_DATA (  cx_last, ptype_native, status )
        mr = is_mr(1)

        IF ( status .NE. ferr_ok ) THEN      ! 8/6/92
           list_fmt_type = plist_default ! restore for next command
	   list_format_given = .FALSE.
           RETURN
        ENDIF

!! ?? get dataset from varname.attname
* ... evaluate the context without computing the expression to get the dataset

	item = 1
        buff = cmnd_buff(item_start(item):item_end(item)) ! same as buff1?

	CALL GET_NEW_CX( cx_last, cx_cmnd, .TRUE., status )
        IF ( status .NE. ferr_ok ) GOTO 5000
 
        IF (buff(1:1) .EQ. '(') THEN
           dset = cx_data_set(cx_cmnd)   ! initial value to try; will return dset
           CALL ISIT_COORD_VAR ( buff, dset, varname, coordvar, status )
           IF ( status .NE. ferr_ok ) THEN 
              dset = pdset_coordvars  ! a user-defined axis (coordinate variable)
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

*          Appending space just to get rid of a nonsense compiler warning
           varname = buff1(1:INDEX(buff1,".")-1)//' '
           CALL FIND_VAR_NAME(pdset_irrelevant, varname, cat, var)
           IF (  var .NE. munknown_var_name .AND.
     .           cat .EQ. cat_user_var) THEN
              dset = pdset_uvars

* Check whether its a LET/D variable
              CALL CD_GET_VAR_ID (dset, varname, varid, status)
              IF (dset.EQ.pdset_uvars .AND. status.NE.ferr_ok) THEN
                 dset = cx_data_set(cx_cmnd)
                 CALL CD_GET_VAR_ID (dset, varname, varid, status)
		 IF (status.NE.ferr_ok) dset = pdset_uvars
              ENDIF
              
           ELSE
              dset = cx_data_set(cx_cmnd) 
           ENDIF
        ENDIF

*  break up varname.attname.  

        CALL BREAK_VARNAME( buff1, dset, varname, attname, 
     .                               varid, status )
        IF (status .NE. ferr_ok) goto 5920

        CALL ADD_ATTRIBUTE(  memry(mr)%ptr, mr, 
     .         buff1, dset, attype_spec, attoutflag, new_att, status )
        IF (status .NE. ferr_ok) GOTO 5000

* Change attribute value
        IF (.NOT. new_att) THEN
           CALL EDIT_ATTRIBUTE(  memry(mr)%ptr, mr, 
     .               buff1, dset, attype_spec, def_att_quiet, status )
           IF (status .NE. ferr_ok) GOTO 6500
        ENDIF


* If the time origin was changed or added, also change units to say
* "units since time origin" using the new time origin info.
        
        iline = TM_GET_LINENUM( varname )

        IF (iline .NE. unspecified_int4) THEN
           slen = TM_LENSTR1(attname)
           IF (MATCH_NAME (attname,  slen, 'TIME_ORIGIN', 11)  .AND.
     .            ( line_direction(iline)(1:1) .EQ. 'T' .OR.
     .              line_direction(iline)(1:1) .EQ. 'F' ) ) THEN
              CALL CD_GET_VAR_ATT_ID (dset, varid, 'UNITS', attid, status)
              IF (status .NE. ferr_ok) GOTO 5960
              got_it = NC_GET_ATTRIB ( dset, varid, 'units',
     .                                 .TRUE., varname, 128,
     .                                 attlen, attoutflag, ustring, 
     .                                 val_buf )
              got_it = NC_GET_ATTRIB ( dset, varid, 'time_origin',
     .                                 .TRUE., varname, 128,
     .                                 attlen, attoutflag, t0string, 
     .                                 val_buf )
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
     .            attlen, buff3, val_buf, status)
              IF (status .NE. ferr_ok) GOTO 6500
           ENDIF

* If the units were changed on a time axis, add "since time origin"
           
           slen = TM_LENSTR1(attname)
           IF (MATCH_NAME (attname,  slen, 'UNITS', 5)  .AND.
     .        ( line_direction(iline)(1:1) .EQ. 'T' .OR.
     .          line_direction(iline)(1:1) .EQ. 'F' ) ) THEN
              CALL CD_GET_VAR_ATT_ID (dset, varid, 'TIME_ORIGIN', 
     .                 attid, status)
              IF (status .EQ. ferr_ok) THEN 
                 got_it = NC_GET_ATTRIB ( dset, varid, 'time_origin',
     .                                   .TRUE., varname, 128,
     .                                   attlen, attoutflag, t0string, 
     .                                   val_buf )
                 got_it = NC_GET_ATTRIB ( dset, varid, 'units',
     .                                   .TRUE., varname, 128,
     .                                   attlen, attoutflag, ustring, 
     .                                   val_buf )
   
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
     .               attlen, buff3, val_buf, status)
                 IF (status .NE. ferr_ok) GOTO 6500
           ENDIF
        ENDIF
        ENDIF ! (axis .NE. unspecified_int4)

* Update attribute information used by plotting etc (ds_units, ...)
            CALL FIND_VAR_NAME (dset, varname, cat, uvar)
            IF ( (cat .EQ. cat_user_var) .OR.
     .           (cat .EQ. cat_file_var) .OR.
     .           (cat .EQ. cat_pystat_var) )
     .           CALL UPDATE_ATTRIBUTES (dset, varname, uvar, status)
            IF (status .NE. ferr_ok) GOTO 6500
	RETURN

      ENDIF


**************************************************************************
* DEFINE PYFUNC /NAME=<alias> python.module.name

      IF (subcmnd_num .EQ. 10) THEN

	IF ( num_args .LT. 1 ) GOTO 6000
        IF ( num_args .GT. 1 ) GOTO 6010

        iqual = qual_given(slash_pyfunc_name)
        IF ( iqual .GT. 0 ) THEN
*           get the function name from /name=<alias>
            CALL EQUAL_STRING(cmnd_buff(qual_start(iqual):qual_end(iqual)),
     .                        buff1, status)
            IF ( status .NE. ferr_ok ) GOTO 5000
            IF ( .NOT. TM_LEGAL_NAME(buff1) ) GOTO 6020
        ELSE
*           create the function name from the last component of python.module.name
            i1 = 0
 1010       i2 = INDEX(cmnd_buff(arg_start(1)+i1:arg_end(1)), '.')
            IF ( i2 .GT. 0 ) THEN
               i1 = i1 + i2
               GOTO 1010
            ENDIF
            vax_code = STR_UPCASE(buff1, 
     .                       cmnd_buff(arg_start(1)+i1:arg_end(1)))
        ENDIF

*       Make sure the scan for standard external functions has been performed
        IF ( EFCN_SCAN(gfcn_num_internal) .LT. 0 ) GOTO 6040

        i1 = TM_LENSTR1(buff1)
        buff = cmnd_buff(arg_start(1):arg_end(1))
        i2 = arg_end(1) - arg_start(1) + 1
        CALL CREATE_PYEFCN(buff1, i1, buff, i2, err_string, slen)
        IF ( SLEN .GT. 0 ) GOTO 6030

        RETURN

      ENDIF


	
**************************************************************************
* DEFINE DATA /AGGREGATE /TITLE= /QUIET
* find name to define
* DEFINE DATA/AGGREGATE/TITLE ENSNAME=D1,D2,D3
* where D1,D2 can be already-open Ferret dataset numbers, or dataset specs
* that would allow Ferret to open the datset.


      IF (subcmnd_num .EQ. 11) THEN

	IF ( num_args .NE. 1 ) GOTO 5350


* define data valid only with /AGGREGATE
	iqual = qual_given( slash_def_aggregate )
        IF ( iqual .EQ. 0 ) GOTO 6100

* Did they say /QUIET

	iqual = qual_given( slash_def_agg_quiet )
        agg_quiet = ( iqual .GT. 0 )

* Is there a /TITLE ?
	iqual = qual_given( slash_def_agg_title )
	buff2 = char_init
	IF ( iqual .GT. 0 ) THEN
	   CALL EQUAL_STR_LC( cmnd_buff(qual_start(iqual):qual_end(iqual)),
     .			      buff2, status )
	   IF ( status .NE. ferr_ok ) RETURN
	ENDIF

* Is there a /T ?
	agg_dim = unspecified_int4
	IF (qual_given( slash_def_agg_t ) .GT. 0) agg_dim = t_dim

* Is there a /E ?
        IF (qual_given( slash_def_agg_e ) .GT. 0) THEN
           IF (agg_dim .EQ. unspecified_int4) THEN
              agg_dim = e_dim
           ELSE
              GOTO 5515
           ENDIF
        ENDIF

* Is there a /F 
        IF (qual_given( slash_def_agg_f ) .GT. 0) THEN
           IF (agg_dim .EQ. unspecified_int4) THEN
              agg_dim = f_dim
           ELSE
              GOTO 5515
           ENDIF
        ENDIF

* Is there a /U  (agg_dim will be reset to unspecified_int4 just before aggregating)

        IF (qual_given( slash_def_agg_u ) .GT. 0) THEN
           IF (agg_dim .EQ. unspecified_int4) THEN
              agg_dim = f_dim + 1
           ELSE
              GOTO 5515
           ENDIF
        ENDIF

* First check for /FAXIS= (in case /F not given) 
* as both /FAXIS= and /TAXIS= can be given for /F
        fline = unspecified_int4
        fmatch = ' '
        iqual = qual_given(slash_def_agg_faxis) 
        IF ( iqual .GT. 0 ) THEN
           IF ( agg_dim .EQ. unspecified_int4 ) THEN
              agg_dim = f_dim
           ELSE IF ( agg_dim .NE. f_dim ) THEN
              GOTO 5515
           ENDIF
           CALL EQUAL_STRING(cmnd_buff(qual_start(iqual):qual_end(iqual)), buff1, status)
           IF ( status .NE. ferr_ok ) RETURN
           IF ( buff1 .EQ. ' ' )
     .        CALL ERRMSG(ferr_syntax, status,
     .                    '/FAXIS must have a value', *5000)
* ... check if a method of matching, and/or @LAG, was specified
           idx = INDEX(buff1, '@')
           IF ( idx .GT. 0 ) THEN
              vaxcode = STR_UPCASE(fmatch, buff1(idx:))
              buff1(idx:) = ' '
           ENDIF
* ... could have just been /FAXIS=@LAG
           IF ( buff1 .NE. ' ' ) THEN
              fline = TM_GET_LINENUM(buff1)
              IF ( fline .NE. unspecified_int4 ) THEN
* ... line given by name - don't allow inappropriate units
                 iunit = line_unit_code(fline)
                 IF ( (iunit .NE. 0) .AND.
     .                ( (iunit .LT. pun_1st_time) .OR.
     .                  (iunit .GT. pun_last_time) ) )
     .              CALL ERRMSG(ferr_invalid_command, status,
     .                          '/FAXIS is not a time axis', *5000)
              ELSE
* ... line implied through grid or variable name
                 ax_grid = GRID_FROM_NAME(buff1, cx_last, status)
                 IF ( status .NE. ferr_ok ) RETURN
                 IF ( ax_grid .EQ. unspecified_int4 ) GOTO 5320
                 fline = grid_line(f_dim, ax_grid)
              ENDIF
           ELSE
* ... no axis given in /FAXIS value
              IF ( fmatch .NE. '@LAG' )
     .           CALL ERRMSG(ferr_syntax, status,
     .                 'no axis specified in /FAXIS (only @LAG '//
     .                 'can be specified without an axis)', *5000)
           ENDIF
        ENDIF

* /TAXIS=
        tline = unspecified_int4
        tmatch = ' '
        iqual = qual_given(slash_def_agg_taxis) 
        IF ( iqual .GT. 0 ) THEN
           IF ( agg_dim .EQ. unspecified_int4 ) THEN
              agg_dim = t_dim
           ELSE IF ( (agg_dim .NE. t_dim) .AND. (agg_dim .NE. f_dim) ) THEN
              GOTO 5515
           ENDIF
           CALL EQUAL_STRING(cmnd_buff(qual_start(iqual):qual_end(iqual)), buff1, status)
           IF ( status .NE. ferr_ok ) RETURN
* ... check if a method of matching, and/or @LAG, was specified
           idx = INDEX(buff1, '@')
           IF ( idx .GT. 0 ) THEN
              vaxcode = STR_UPCASE(tmatch, buff1(idx:))
              buff1(idx:) = ' '
           ENDIF
* ... could have just been /FAXIS=@LAG
           IF ( buff1 .NE. ' ' ) THEN
              tline = TM_GET_LINENUM(buff1)
              IF ( tline .NE. unspecified_int4 ) THEN
* ... line given by name - don't allow inappropriate units
                 iunit = line_unit_code(tline)
                 IF ( (iunit .NE. 0) .AND. 
     .                ( (iunit .LT. pun_1st_time) .OR.
     .                  (iunit .GT. pun_last_time) ) )
     .              CALL ERRMSG(ferr_invalid_command, status,
     .                          '/TAXIS is not a time axis', *5000)
              ELSE
* ... line implied through grid or variable name
                 ax_grid = GRID_FROM_NAME(buff1, cx_last, status)
                 IF ( status .NE. ferr_ok ) RETURN
                 IF ( ax_grid .EQ. unspecified_int4 ) GOTO 5320
                 tline = grid_line(t_dim, ax_grid)
              ENDIF
           ELSE
* ... no axis given in /FAXIS value
              IF ( tmatch .NE. '@LAG' )
     .           CALL ERRMSG(ferr_syntax, status,
     .                 'no axis specified in /TAXIS (only @LAG '//
     .                 'can be specified without an axis)', *5000)
           ENDIF
        ENDIF

        IF ( (INDEX(fmatch, '@LAG') .GT. 0) .AND.
     .       (INDEX(tmatch, '@LAG') .GT. 0) )
     .     CALL ERRMSG(ferr_syntax, status, '@LAG cannot '//
     .           'be specified in both /FAXIS and /TAXIS', *5000)

* /TOVERLAP
        toverlap = 0
        iqual = qual_given(slash_def_agg_toverlap) 
        IF ( iqual .GT. 0 ) THEN
           IF ( agg_dim .NE. t_dim )
     .        CALL ERRMSG(ferr_syntax, status, 
     .               '/TOVERLAP only valid for /T aggregation', *5000 )
           IF ( tline .EQ. unspecified_int4 )
     .        CALL ERRMSG(ferr_syntax, status, 
     .               '/TOVERLAP requires /TAXIS', *5000 )
           CALL EQUAL_VAL_INT(cmnd_buff(qual_start(iqual):qual_end(iqual)), toverlap, status)
           IF ( status .NE. ferr_ok ) RETURN
        ENDIF

* default aggregation orientation
	IF (agg_dim .EQ. unspecified_int4) agg_dim = t_dim

* define data/hide
	agg_hide = qual_given( slash_def_agg_hide ) .GT. 0

* Get the name of the dataset we are creating
* get it case-sensitive, cuz other dataset names are (they are filenames)
!!!! ******** *sh* NOTE:  CODE SHOULD BE IMPROVED HERE *******
! The DEFINE DATA command has introduced a new and ambiguous set of parsing
! rules, because the argument to the right of the equals may be either a list
! of names (optionally quoted) intermixed with numbers **OR** an expression
! that evaluates to a 1D array of text strings.  A crude heuristic is used a
! few lines below (see have_expr) to determine which of these.  We should
! instead write a new parser to do this task reliebly
! ************************************************************
	CALL NAME_EQUAL_STRING_LC( buff1, pos, status )
	IF ( status .NE. ferr_ok )  GOTO 5000
! hack: do not accept name unless it was followed by an equals sign
! to do so results in a corrupted item list ... could be fixed if important
        IF (pos .GT. 0) THEN
           i = INDEX(cmnd_buff(:pos), "=")
           IF (i.eq.0) pos = 0
        ENDIF

* was a name given?  Or implied through its absence?
	IF ( pos .EQ. 0 ) THEN
* ... "name =" was not supplied
           pos = arg_start(1)    ! point to start of argument list
           IF ( GO_FILE_INPUT() ) THEN
*     ... take name of the GO file exectuting this command
              buffsym = 'GO_FILE'
              CALL GETSYM( buffsym, buff3, slen, ierr )  ! get .agg file path
	         IF (slen.EQ.0 .OR. ierr.EQ.1) CALL ERRMSG
     .                  ( ferr_internal, status, 'xeq_define_go_file', *5000)
              buff1 = TM_CLEAN_FILENAME(buff3 )     ! get dname from dpath
           ELSE
*     ... assign a default name
              IF (agg_dim.EQ.t_dim) THEN
                 buff1 = 'TSERIES'
              ELSEIF (agg_dim.EQ.e_dim) THEN
                 buff1 = 'ENSEMBLE'
              ELSEIF (agg_dim.EQ.f_dim) THEN       
                 buff1 = 'FMRC'
              ELSEIF (agg_dim.EQ.f_dim+1) THEN       
                 buff1 = 'UNION'
              ENDIF
              buff3 = buff1    ! use dname as dpath, too
           ENDIF
        ELSE
* ... "name =" was supplied
	   IF ( .NOT.TM_LEGAL_NAME(buff1) ) GOTO 5410
	   IF ( .NOT.TM_LEGAL_NAME_OP(buff1) ) GOTO 5430
           IF ( TM_LENSTR1(buff1) .GT. 128 ) GOTO 5410
! *sh* 11/15 - As-is the parsing of the "name=item1, item2, ..."
* The routine NAME_EQUAL_STRING parses out the "name" as the first item of
* a list and returns the start position of the next item in the list
           item_start(1) = pos
           buff3 = buff1    ! use dname as dpath, too
        ENDIF

* which syntax has been used?
*      DEFINE DATA/AGG agname = data1, data2, data3, ...
*  or
*      DEFINE DATA/AGG agname = <string_array_expression>
* we will determine this through heuristics
        have_expr = INDEX(cmnd_buff(pos:),"{") .GT. 0
     .         .OR. INDEX(cmnd_buff(pos:),"(") .GT. 0
        IF (.NOT.have_expr) THEN
           IF ( num_items .EQ. 1
     .     .AND.INDEX(cmnd_buff(item_start(1):item_end(1)),'.').EQ.0
     .     .AND.INDEX(cmnd_buff(item_start(1):item_end(1)),'/').EQ.0
     .                                                            ) THEN 
              CALL FIND_VAR_NAME(pdset_irrelevant,
     .                           cmnd_buff(item_start(1):item_end(1)),
     .                           cat, var)      
              have_expr = var .NE. munknown_var_name
           ENDIF
        ENDIF

        IF (have_expr ) THEN
* ... get array of string names from a memory variable
           CALL ALL_1_ARG
	   num_args = 1	 ! used by GET_CMND_DATA
	   arg_start(1) = pos ! skip over "name ="
	   CALL GET_CMND_DATA (  cx_last, ptype_string, status )
	   IF ( status .NE. ferr_ok ) RETURN
           IF ( num_uvars_in_cmnd .NE. 1 ) GOTO 6400
           cx = is_cx(1)
* ... make sure it's an line (not a point,plane,etc.) of data
	   CALL GET_CX_DIMS( cx, ndim, dim )
	   IF ( ndim .GT. 1 ) GOTO 6400
	   idim = dim(1)
	   nagfiles = CX_DIM_LEN( idim, cx )
           mr = is_mr(1)
        ELSE
           nagfiles = num_items
           mr = dummy_mr
        ENDIF

*   buff1=dname, buff2=dtitle, buff3=dpath
        IF (agg_dim .EQ. t_dim) THEN
           CALL INIT_T_AGGREGATE_DSET(memry(mr)%ptr,
     .              nagfiles, have_expr, buff1, buff3, buff2, dset,
     .              tline, tmatch, toverlap, status)
        ELSEIF ( agg_dim .EQ. e_dim ) THEN
           CALL INIT_E_AGGREGATE_DSET (memry(mr)%ptr,
     .              nagfiles, have_expr, buff1, buff3, buff2, dset,
     .              agg_quiet, agg_hide, status)
        ELSEIF ( agg_dim .EQ. f_dim ) THEN
           CALL INIT_F_AGGREGATE_DSET (memry(mr)%ptr,
     .              nagfiles, have_expr, buff1, buff3, buff2, dset,
     .              agg_quiet, agg_hide, tline, tmatch, 
     .              fline, fmatch, status)
        ELSE
	   agg_dim = unspecified_int4
	   CALL INIT_U_AGGREGATE_DSET (memry(mr)%ptr,
     .              nagfiles, have_expr, buff1, buff3, buff2, dset,
     .              agg_quiet, agg_hide, agg_dim, status)
        ENDIF
	IF (status .NE. ferr_ok) GOTO 5000

* Make this the current dataset
	cx_data_set( cx_last ) =  dset

	RETURN

      ENDIF


**************************************************************************
* DEFINE ANNOTATION /USER/NORM/NOUSER/XPOS/YPOS/HALIGN/VALIGN/ANGLE/SIZE/NLAB label_text
* qualifiers in the same order as ANNOTATE/USER/...
*
* Find label text


      IF (subcmnd_num .EQ. 12) THEN

	IF ( num_args .LT. 1 ) GOTO 6200

	item = 1
	buff1 = cmnd_buff(item_start(item):item_end(item))

* ...DEFINE ANNOTATION/NLAB 
c ?? if nlab=0, use next avail slot?
	iqual = qual_given( slash_def_annot_nlab )
	IF ( iqual .GT. 0  ) THEN
	   CALL EQUAL_VAL( cmnd_buff(qual_start(iqual):qual_end(iqual)),
     .			      val, status )
	   buff2 = 'NLAB'
	   IF ( status .NE. ferr_ok ) GOTO 6300
	   IF ( val.EQ.unspecified_val4 .OR. val.LT.0) GOTO 6300
	   nlab = INT(val)
	ELSE
	   nlab = 0
	ENDIF

* initialize 

	size = 0.12

* Get settings

	CALL SET_ANNOTATION (user, nouser, norm, xloc, yloc, halign, 
     .         angle, size, status)
        IF (status .NE. ferr_ok) GOTO 5000 

	CALL PUT_ANNOTATION (user, nouser, norm, xloc, yloc, halign, 
     .         angle, size, nlab, buff1, status)
        IF (status .NE. ferr_ok) GOTO 5000 

	RETURN

      ENDIF
**************************************************************************

* error exit
 5000	RETURN
 5100 	CALL ERRMSG( ferr_syntax,status,'illegal name: '//buff3,*5000 )
 5200 	CALL ERRMSG( ferr_invalid_command ,status,
     .          'cannot alias IF, ELIF, ELSE and ENDIF',*5000 )
 5300	CALL ERRMSG( ferr_invalid_command, status,
     .		'DEFINE what name?', *5000 )
 5350	CALL ERRMSG( ferr_syntax, status,
     .		'Should be "NAME = dset1, dset2, dset3, ..."', *5000 )
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
 5430   CALL ERRMSG( ferr_syntax,status,
     .          'illegal name, matches an operator: '//buff1,*5000 )
 5440	CALL ERRMSG( ferr_syntax, status,
     .			cmnd_buff(pos:len_cmnd)//pCR//err_string, *5000 )
 5470	CALL ERRMSG( ferr_unknown_data_set, status,
     .			buff3(:TM_LENSTR1(buff3)), *5000 )
 5480   CALL ERRMSG( ferr_syntax,status,
     .	  '/REMOTE variable definition must also specify /D=',*5000 )
 5490   CALL ERRMSG( ferr_syntax,status,
     .	  '/REMOTE variable expression may not contain [d= ',*5000 )
 5505	CALL ERRMSG( ferr_invalid_command, status,
     .    'DEFINE AXIS/DEPTH needs /Z=lo:hi:del', *5000 )
 5510	CALL ERRMSG( ferr_invalid_command, status,
     .    'Conflicting /X/Y/Z/T/E/F axis orientations given', *5000 )
 5515	CALL ERRMSG( ferr_invalid_command, status,
     .    'Conflicting /U/T/E/F/TAXIS/FAXIS given', *5000 )
 5520	CALL FREE_LINE_DYNMEM (iline)
	CALL ERRMSG( ferr_grid_definition, status,
     .			'error in start,end,delta', *5000 )
 5525	CALL ERRMSG( ferr_grid_definition, status,
     .			'Axis length exceeds modulo length', *5000 )
 5530	CALL ERRMSG( ferr_grid_definition, status,
     .			'NPOINTS=1 incompatible with limits given', *5000 )
 5540	CALL ERRMSG( ferr_grid_definition, status,
     .		'/T0='//buff2(:TM_LENSTR1(buff2)), *5000 )
 5550	CALL ERRMSG( ferr_grid_definition, status,
     .		'inappropriate units:'//buff1(:TM_LENSTR1(buff1)), *5000 )
 5560	CALL FREE_LINE_DYNMEM (iline)
	CALL ERRMSG( ferr_grid_definition, status,
     .		'unrepairable repeated axis coords', *5000 )
 5570	CALL FREE_LINE_DYNMEM (iline)
	buff1 = TM_FMT(DBLE(i1), 0, 16, slen)
	CALL ERRMSG( ferr_grid_definition, status,
     .		'data for DEFINE AXIS/FROM_VARIABLE is not monotonically '//
     .		'increasing at index '//buff1(:slen), *5000 )
 5580	CALL ERRMSG( ferr_grid_definition, status,
     .		'data for DEFINE AXIS/FROM_VARIABLE is ambiguous', *5000 )
 5582	CALL ERRMSG( ferr_grid_definition, status,
     .		'illegal argument for DEFINE AXIS/NPOINTS', *5000 )
 5592	CALL ERRMSG( ferr_prog_limit, status,
     .		'Unable to allocate system memory for coordinate storage -  '//
     .		'use /EDGES not /BOUNDS to save space', *5000 )
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
        t2 = ' '
	DO i1=1, max_calendars
           len_cal = TM_LENSTR1(err_string)
           slen = TM_LENSTR1(allowed_calendars(i1))
           err_string = err_string(:len_cal)//t2//
     .                   allowed_calendars(i1)(:slen)
           t2 = ', '
           len_cal = len_cal + slen
	ENDDO
        CALL ERRMSG( ferr_syntax, status,err_string, *5000 )

5730	CALL ERRMSG( ferr_syntax,status,
     .     'cannot have /BOUNDS and /EDGES together ',*5000 )

5740	buff1 = TM_FMT(DBLE( n), 0, 16, slen)
        buff2 = TM_FMT(DBLE(n2), 0, 16, s2len)
        CALL ERRMSG( ferr_grid_definition, status,
     .     'dimension of bounds ('// buff2(:s2len)//
     .     ') must be 2*N or N+1, N=number of axis coordinates ('//
     .     buff1(:slen) //')', *5000 )
5745	buff1 = TM_FMT(DBLE( n), 0, 16, slen)
        buff2 = TM_FMT(DBLE(n2), 0, 16, s2len)
        CALL ERRMSG( ferr_grid_definition, status,
     .     'dimension of upper&lower bounds ('// buff2(:s2len)//
     .     ') must be N, N=number of axis coordinates ('//
     .     buff1(:slen) //')', *5000 )

5750	CALL FREE_LINE_DYNMEM (iline)
	CALL FREE_LINE_DYNMEM (line_0)
	CALL ERRMSG( ferr_grid_definition, status,
     .   'BOUNDS specified do not correctly enclose coordinate points', 
     .   *5000 )
5760	CALL ERRMSG( ferr_grid_definition, status,
     .   'DEFINE AXIS/LIKE= Axis not found '//likename(:TM_LENSTR1(likename)),
     .   *5000 )
5765	CALL ERRMSG( ferr_grid_definition, status,
     .   'DEFINE AXIS/LIKE= Direction must match original axis ',
     .   *5000 )
5770	CALL ERRMSG( ferr_grid_definition, status,
     .   'DEFINE AXIS/MONTHLY specify /UNITS=month or omit /UNITS (MONTH is default)', 
     .   *5000 )
5780	CALL ERRMSG( ferr_grid_definition, status,
     .   'DEFINE AXIS/MONTHLY True-month axis: /BOUNDS not allowed. Use /T=lo:hi:del', *5000 )
5790	CALL ERRMSG( ferr_grid_definition, status,
     .    'DEFINE AXIS/MONTHLY True-month axis: Definition /FROMDATA not allowed. '//
     .    'Use /T=LO:HI:DEL', *5000 )
5800	CALL ERRMSG( ferr_grid_definition, status,
     .   'DEFINE AXIS/MONTHLY True-month axis: /NPOINTS not allowed. Use /T=lo:hi:del', *5000 )
5810	CALL ERRMSG( ferr_grid_definition, status,
     .   'DEFINE AXIS/MONTHLY True-month axis delta-t must be 1', *5000 )
5820	CALL ERRMSG( ferr_grid_definition, status,
     .   'DEFINE AXIS/MONTHLY True-month axis: needs /T=lo:hi:del', *5000 )
5830	CALL ERRMSG( ferr_grid_definition, status,
     .   'DEFINE AXIS/MONTHLY month boundaries cannot be day '//datestr(1:2)//
     .   '. Day does not exist in all months.', *5000 )
5840	CALL ERRMSG( ferr_grid_definition, status,
     .   'DEFINE AXIS/MONTHLY must use calendar-time definition', *5000 )
5850   CALL ERRMSG(ferr_grid_definition, status,
     .     'Axis too long for 4-byte integer indexing: '
     .		//'Long dimensions not implemented', *5000)

5900	CALL ERRMSG( ferr_invalid_command, status,
     .		'DEFINE what name.attribute?', *5000 )
5920    slen = TM_LENSTR1(varname)
        IF (dset .GT. 0) THEN
           CALL ERRMSG( ferr_invalid_command, status,
     .		'variable or axis undefined or not in dataset '//
     .            varname(:slen), *5000 )
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

5940  slen = TM_LENSTR1(buff2)
      CALL ERRMSG( ferr_invalid_command, status,
     .	buff2(:slen)//' not allowed as argument of DEF ATT/TYPE='//pCR//
     .  'Must be STRING or FLOAT ',*5000 )

5960  slen = TM_LENSTR1(buff1)
      CALL ERRMSG( ferr_invalid_command, status,
     .   'cannot set T0 unless Units are also set'//
     .    buff1(:slen), *5000 )

 6000 CALL ERRMSG(ferr_invalid_command, status,
     .  'No Python module named in DEFINE PYFUNC command', *5000)

 6010 CALL ERRMSG(ferr_invalid_command, status,
     .  'More than one Python module named in DEFINE PYFUNC command',
     .   *5000)

 6020 CALL ERRMSG(ferr_syntax, status, 'illegal name: '//buff1, *5000)

 6030 CALL ERRMSG(ferr_invalid_command,status,err_string(1:slen),*5000)

 6040 CALL ERRMSG(ferr_invalid_command,status,'EFCN_SCAN failed',*5000)

 6100 CALL ERRMSG(ferr_syntax, status, 
     .    'DEFINE DATA valid only with /AGGREGATE: ', *5000)

 6200	CALL ERRMSG( ferr_invalid_command, status,
     .		'No text given for DEFINE ANNOTATION', *5000 )

 6300	CALL ERRMSG( ferr_invalid_command, status,
     .		'/NLAB missing or invalid value', *5000 )
 6400   CALL ERRMSG( ferr_grid_definition, status,
     .	    '"DEFINE DATA/AGGREGATE agname =" requires a list of names',
     .          *5000 )
 6500  slen = TM_LENSTR1(varname)
       CALL ERRMSG (ferr_internal, status,
     . 'Error defining attribute on variable '//varname(:slen), *5000)

	END

