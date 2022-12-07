	SUBROUTINE XEQ_SET

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
* V542: *sh* 7/02 - added support for sub-span modulo axes
* V550: *acm*10/02- Change arg to char*24, to match longer region names (cx_name)
* v553: *kob* 8/03 - need to increase size of buff1 for long numeric dataset names
*       *kob* 9/03 - add ich for character indicies for the internal_read_format_bug
* V570: *acm* 5/04 - add MODE GRATICULE[:argument] 
*       *acm* 6/05 - add SET AXIS/CALENDAR /T0 /UNITS
* V581: *acm*/3/05 - increase length of strings for datset title to 1024 (leave settings alone for MOVIE)
* V600 *acm*  8/05 - Allow more digits in xrevision number
* V600 *acm* 3/06  fixing bugs 439&1390, new arg to get_new_cx.
* V600 *acm* 5/2006 MODE LINECOLORS User can change the number of line colors.
*      *jli* 7/2006 added 'implicit.parm' for pline_class_stride
*      *jli* added SET AXIS/STRIDE/OFFSET/
* V601 *acm*11/06 Fix bug 1470: on a SET AXIS command which redefines attributes of an axis, change
*                 values in the attribute structure for the axis.
* V601 *acm*11/06 Fix bug 1438: A request for rediculously large memory results in an
*                 overflow in setting reconfig, so it looks like a negative number of words 
*                 requested. This should result in an error message.
* V602: *acm* 1/07 Changes for metafile-batch mode: When in gif mode or batch mode, 
*                  ignore CANCEL MODE META
* V602  2/07 *acm* Fix bug 1492, changing attributes of coordinate variables
*                  including changes to an attributes values and settings at the
*                  same time, e.g. set att/output var.modulo = 300
* V603  5/07 *acm* Fix bug 1511: if DEFINE AXIS or DEFINE ATT change a time
*                  origin, include the origin in the units attribute as "since t0"
*                  If units attribute is changed for a time axis, get the time
*                  origin if it exists and append it to the units string.
* v604 *acm* 7/07 increase length of ds_name
* V606  8/07 *acm* Send informational and error messages that are returned from 
*                  commands via SPLIT_LIST to std error rather than std out.
* V610  *acm* 3/08 - add new MODE NLEVELS
* V610  *acm* 3/08 Fixes for metafile batch mode
* V613  *acm* 9/08 Correct for bug 1598: in SET ATTRIBUTE, initialize dataset number so a 
*                  SET ATT command always computes the dataset number.
* V63   *acm* 9/09 Fix bug 1689: definition of the offset in SET AXIS/STRIDE=/OFFSET
*                  changes so that /OFF=0 implies 1st index=1; /OFF=1 implies 
*                  1st index=2. Also check for and give err msg for illegal 
*                  values STRIDE < 1 and OFFSET < 0.
* V65   *acm* 2/10 Fix bug 1724: in metafile batch mode only, if some command
*                  has opened the metafile but nothing plotted yet, and if
*                  a SET MODE META:newname.plt is issued, just rename the metafile
*                  that was opened.
* V66   *acm* 3/10 Read qualifiers for netcdf4 SET LIST/NCFORMAT etc in call to
*                  netcdf4_set_list_settings
*       *kms* 5/10 add SET AXIS/REGULAR to force interpreting the axis as regular
*       *kms* 5/10 add SET REDIRECT /TEE /JOURNAL /FILE /APPEND /CLOBBER
*                  for stdout/stderr redirection
* V66   *kms* 5/10 LIB_GET_LUN is a subroutine, not a function
*       *kms* 2/11 return "mem_blk_size" instead of "mem_size" in reconfig
* V68   *acm* 1/12 Implement micro-adjusting of repeated coordinates (ticket 1910).
*                  To keep the previous behavior the user may specify USE/STRICT.
* PyFr  *kms* 1/12 Add SET GRAPHICS /ENGINE /ANTIALIA /NOANTIAL
*       *acm* 4/12 6D Ferret: time axis may be in t or f direction. 
*       *acm* 4/12 Add new MODE FOURDEE to allow E,F,M,N,etc as user-vars
*       *acm* 4/12 Remove MODE FOURDEE, using _E, _F, _M, _N instead, throughout.
* V680 5/12 *acm* 6D Ferret, changes for DEFINE DATA/AGGREGATE
* V6.8 *acm* 7/9/2012 For batch graphics on RHEL6, change ITS_BATCH_GRAPHICS
*                     from logical to integer, then batch = ITS_BATCH_GRAPHICS().NE.0
* V683 *acm*  8/12 add new SET LIST/OUTTYPE=  
* PyFr  *kms* 1/13 Remove SET GRAPHICS /ENGINE /ANTIALIA /NOANTIAL;
*                  Add /QUALITY= /AAlias /NoAAlias /ENGINE= to SET WINDOW
* V685 *acm*  3/13 For Ensemble aggregations, call the ds_type 'ENS'
* V685 *acm*  7/13 Added FILE/FORMAT=descriptor
* V686 *acm*  1/14 Fix ticket 2126: set axis was messing up line_name storage
* V687 *acm*  3/14 New SET AXIS/NAME=
* V694 11/14 *acm* Ticket 2217: permuted ASCII reading in 6D (fix to an error message)
* V694 *acm*  2/15 ticket 2050: preserve case on var names for SET ATT/LIKE=var1 var2
* V6.95 *acm* 3/15 Changes for PyFerret fonts. New SET TEXT/
* V698+ *sh*  11/15 to support TSERIES
*	      ok_re_use_existing into TM_INIT_DSET call
*	      SET DATA tseries_cmnds.agg gets passed to the GO command (a hack)
* V698  2/16 *acm* For ticket 2352: variables defined with LET/D are
*                  added to that dataset in the attribute structure.
* V698  4/16 *acm* For ticket 2399: Apply MODE CALENDAR settings to F axes too
* V7    5/16 *acm* See ticket 2352. For LET/D variables look for info under that dataset
* V7   *acm*  6/12 Add Union aggregations: DEFINE DATA/AGG/U, ds_type 'UNI'
* V701 *acm*  7/12 Ticket 2455: On a SET AXIS/NAME=, save the original spelling and 
*                  an up-cased spelling of the axis name, as is done with DEFINE AXIS.
* V710  4/16 *acm* new TM_UNITS_CAL to account for calendar in units id
* V702  3/17 *sh* added MODE FRUGAL
*		  Fixed bug in error message for invalid SET LIST/OUT
* V720 3/17 *acm* Fix ticket 2521; do not call TM_LEGAL_UNIX_NAME for path and 
*                file name on SET REDIRECT.
* V722 *kms* 10/17 Remove threddsBrowser (too big, outdated, and unused)
* V740 *acm* 2/18 Issue 966: LET/LIKE= requires changes to the call to INHERIT_ATTRS
* V74+ *acm* 5/18 Don't upcase the title given on a SET DATA/TITLE
* v741 5/18 *acm* Issue 1651 - "use my" will look for my.agg and run it as a script to
*                 initialize an aggregation. If given w/o .agg, a hack will look for it.
* v743  7/18 *acm Issue 1887, numeric dataset names w/o extension, better test 
*                 for whether it's just a digit, "use n".
* V74+ 6/18 *acm* Changes to the handling of USE/FEATURE=  
* v743  12/18 *acm* Issue 1803, New command-line switch -verify for set mode ver:always behavior
* v745 *acm* 12/18 Issue 1909, option to write dates in ISO8601 format
*                  new mode datefmt for formatted date output
* v7441*acm* 12/18 Issue 1910: Allow dates in commands to be in ISO 8601 form, yyyy-mm-dd-
* v751 *acm*  6/19 cleanup: remove code related to SET MOVIE 
* V751 *acm* 6/19 Issue 1903: memory set and reported in terms of bytes
* v7545*acm*  4/19 Check for invalid text-group name in SET TEXT/FONT=xxx text-group
* v751 *acm*  6/19 Issue 1942: reinitialize varname and attname for SET ATTRIBUTE
* v751 *acm*  6/19 cleanup: remove code related to SET MOVIE 
* V751 *acm* 6/19 Issue 1903: memory set and reported in terms of bytes
* V751 *acm* 6/19  When setting a new DSG fmask, clear variables from memory
* v76  *acm* 11/19  Issue 1959: Change SET MEMORY/SIZE= to SET MEMORY/MAX= 
*                               (keep /SIZE for bkwds compatibility)
* V745 *acm*  5/20  added mode_dsg to control turning off all DSG ops
* V761 *acm*  9/20  Set an /SMASK for stations on a timeseriesProfile dataset or
*                  trajectories on a trajectoryProfile one.
** 10/2022 *acm* Code cleanup: remove references to obsolete gui interface

        IMPLICIT NONE
	include 'tmap_dims.parm'
	include 'tmap_errors.parm'
        include 'implicit.parm'
#	include "tmap_dset.parm"
	include 'xdset_info.cmn_text'
	external xdset_info_data
        include 'xdiag_ctrl.cmn_text'
        include 'xtm_grid.cmn_text'
        external xgt_grid_data
	include	'ferret.parm'
	include	'slash.parm'
	include 'errmsg.parm'
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
	include	'PLTCOM.DAT'		! with gksopn
	include 'gkscm2.cmn'		! with first_meta
	include	'switch_inc.decl'	! with deflts
	include	'SWITCH.INC'	! with deflts
        include 'LUNITS.INC'          ! with lttout
        include 'lunits_inc.decl'
	include 'xunits.cmn_text'
        include 'calendar.decl'
        include 'calendar.cmn'
        include 'netcdf.inc'    ! with NCCHAR
        include 'pyfonts.cmn'

      
* calling argument declarations:
!	INTEGER	reconfig   ! removed for dynamic memory management 2/2017

* local variable declarations:
	LOGICAL		TM_DATE_OK, is_ok, set_dsg, reload
	INTEGER		REGION_NUMBER, VIEWPORT_NUMBER, GRID_FROM_NAME,
     .                  TM_GET_LINENUM, TM_LENSTR, TM_LENSTR1, 
     .                  TM_UNIT_ID, TM_UNITS_CAL, TM_GET_CALENDAR_ID,
     .			FIND_DSET_NUMBER, STR_UPCASE, STR_SAME, 
     .                  equal_pos, colon_pos, dset_num, ich, status,
     .			tmap_status, mode, idim, iarg, item, lp, iv, 
     .			cx, i, grid, axis, loc, perm(nferdims), buff_len, 
     .                  cal_id, day, month, year, hour, minute, second,
     .                  units, since_t0, len_test, len_mchars, slen, 
     .                  len, attype_spec, pos, mr, varid, uvar, cat, 
     .                  iflag, isetout, ivar, dot, strided_axis, igrd, 
     .                  offset, attid, attype, attoutflag, attlen, i1, 
     .                  dset, istat, cache_size, cache_nelems, 
     .                  cache_preemption, numpts, redir_file_lun, 
     .                  iline, iset, n, igrp, nopen
	INTEGER*8	i8_val
        REAL*8          TM_WW_AXLEN, GET_LINE_COORD, axwwlen, new_att_modulo_len
        REAL*8          delta, dlo, dhi, firstval, lastval
	REAL		val_buf, cache_size_mb
	CHARACTER	TM_FMT_TRIM*48, title*1024, subtitle*64, arg*24,
     .			buff1*512, buff2*512, data_form*3, show_str*12,
     .                  varname*512, attname*128, new_att_calendar*10, 
     .                  new_att_t0*128, new_att_units*128, buff3*512,
     .                  t0string*128, ustring*128, faccess*16, 
     .                  fposition*16, fstatus*16, c1*1, typflag*10, 
     .                  fmask_var*128
	LOGICAL		TM_LEGAL_UNIX_NAME, MATCH4, TM_DIGIT, 
     .                  TM_DIGIT_ONLY,
     .                  TM_HAS_STRING, TM_LEGAL_NAME, MATCH_NAME, TM_FPEQ,
     .                  NC_GET_ATTRIB, set_last, any_delta, dset_cdf,
     .                  do_ez, tregular, its_changed, ok_cal,
     .                  coordvar, new_att_modulo,
     .                  got_it, redir_stdout, redir_stderr, redir_tee,
     .                  redir_journal, redir_append, redir_clobber, 
     .                  fexist, use_strict, is_nc, have_outtype, set_att_quiet


* local parameter declarations:
	INTEGER		slash_set_last
	PARAMETER     ( slash_set_last = 1 )
	LOGICAL		ok_re_use_existing
	PARAMETER     ( ok_re_use_existing = .TRUE.)

* functions
	LOGICAL is_secure

* initialize
!	reconfig = 0   	! zero signals NO memory reconfigure

* select subcommand
	GOTO ( 100,200,300,400,500,600,700,800,900,1000,
     .        1100,1200,1300,1400,1500,1600,1700,1800 ) subcmnd_num

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
	CALL GET_NEW_CX( cx_last, cx_buff, .TRUE., status )
	IF ( status .NE. ferr_ok ) GOTO 5000

* apply delta context specifiers (DX=,DY=, etc) if any
	CALL GET_DELTA_CONTEXT( 14, any_delta, status )
	IF ( status .NE. ferr_ok ) GOTO 5000
	IF ( any_delta ) THEN
	   CALL APPLY_DELTA_CONTEXT( cx_buff, 'DEFAULT', status )
	   IF ( status .NE. ferr_ok ) GOTO 5000
	ENDIF

* illegal delta given ?
	DO idim = 1, nferdims
	   IF ( cx_given(idim,cx_buff).AND. 
     .          cx_delta(idim,cx_buff) .NE. unspecified_val8 ) 
     .            GOTO 5320
     	ENDDO

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

*  ... SET LIST/OUTTYPE=(netcdf type), 'DOUBLE', 'FLOAT', 'INT', 'SHORT', 'BYTE'
	lp = qual_given( slash_set_list_outtype )
	IF ( lp .GT. 0 ) THEN
	   CALL EQUAL_STRING(
     .                  cmnd_buff(qual_start(lp):qual_end(lp)),
     .                  list_outtype, status )
	   IF ( status .NE. ferr_ok ) GOTO 5000
	   have_outtype = ( INDEX(list_outtype,"DOU") .GT. 0 .OR.
     .                    INDEX(list_outtype,"FLO") .GT. 0 .OR. 
     .                    INDEX(list_outtype,"INT") .GT. 0 .OR.
     .                    INDEX(list_outtype,"SHO") .GT. 0 .OR.
     .                    INDEX(list_outtype,"BYT") .GT. 0 )
	   IF (.NOT. have_outtype) GOTO 6810
	ENDIF

* Get and save any settings for netcdf-4 output

        CALL NETCDF4_SET_LIST_SETTINGS (status)
        IF ( status .NE. ferr_ok ) GOTO 5000

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

* (hack) if the argument has a ".agg" extension, then pass it to the GO
* command, allowing "USE myAgg.agg" to initialize an aggregation

	IF (num_items .GT. 0 ) THEN
*   ... make sure that only a single .agg file is specified
           n = 0  ! number of arguments with .agg extension
	   DO item = 1, num_items
	      IF ((item_end(item)-item_start(item)) .GT. 3) THEN
	         IF (cmnd_buff(item_end(item)-3:item_end(item))
     .			.EQ. '.agg')  n = n + 1
	      ENDIF
	   ENDDO
	ENDIF
	IF (n .GT. 0) THEN		! at least 1 ".agg" file was given
	   IF (n.GT.1 .OR. num_items.GT.1) GOTO 5130
	   CALL XEQ_GO
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
	     DO item = 1, num_items
	        CALL INIT_EZ_DSET( cmnd_buff(item_start(item):item_end(item)), 
     .				 dset_num, status )
	        IF ( status .NE. ferr_ok ) RETURN
* ... dummy calls to make EZ sets act like other sets
	        CALL GET_AXIS_SPEED( dset_num )
	     ENDDO	
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

* interpret the /STRICT qualifier (do not micro-adjust axes with repeated 
* coordinates) for netCDF data sets
	use_strict = .FALSE.
        loc = qual_given(slash_set_data_strict )
	use_strict = loc .GT. 0


* SET DATA/FEATURE=  Set the DSG feature type
* Options are 
*   /feature=trajectory  must have lon, lat in the obs direction
*   /feature=profile     must have depth/height in the obs direction
*   /feature=timeseries  must have time in the obs direction
*   /feature=none        to ignore feature attributes and treat it as a regular gridded file.
* Code checks to see if it can be handled with the requested feature type.

* Initialize- set this flag only if this SET DATA is a SET DATA/FEATURE (If the set's already
* open, leave the setting as-is, we're just making this set the default.)

	dsg_user_direction = int4_init

	set_dsg = qual_given(slash_set_data_feature) .GT. 0
	IF (set_dsg) THEN
	   loc = qual_given(slash_set_data_feature)
	   CALL EQUAL_STRING(
     .          cmnd_buff(qual_start(loc):qual_end(loc)),
     .                    ustring, status )
	   IF (STR_SAME(ustring(1:3), 'tra') .EQ. 0) THEN
	      dsg_user_direction = x_dim ! arbitrarily
	   ELSE IF (STR_SAME(ustring(1:3), 'pro') .EQ. 0) THEN
	      dsg_user_direction = z_dim
	   ELSE IF (STR_SAME(ustring(1:3), 'tim') .EQ. 0) THEN
	      dsg_user_direction = t_dim
	   ELSE IF  (STR_SAME(ustring(1:3), 'non') .EQ. 0) THEN
	      dsg_user_direction = no_dim
	   ELSE
	      GOTO 6910
	   ENDIF
	ENDIF

	IF (set_dsg .AND. .NOT.mode_dsg) THEN
	   CALL WARN ('MODE DSG is turned off. Ingoring SET DATA/FEATURE=') 
	ENDIF

* interpret the /ORDER permutation qualifier for a non-EZ data set
	DO idim = 1, nferdims
	   perm(idim) = unspecified_int4 ! set default
 	ENDDO
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
* Just check the first character for a digit (ticket 1974)
	   c1 = cmnd_buff(item_start(item):item_start(item))
	   buff1 = cmnd_buff(item_start(item):item_end(item))
	   IF ( c1 .GE. '0' .AND. c1 .LE. '9' ) THEN
	      buff1 = cmnd_buff(item_start(item):item_end(item))
* *kob* 4/97
#ifdef INTERNAL_READ_FORMAT_BUG
* *kob* 9/03 kludge due to g77 internal read bug which does not correctly identify
*            a character and signal an error when doing an internal read into an
*            integer variable - e.g. 11111112_3.nc was being read as an integer so Ferret
*            was trying to find dataset 11111112 and reporting an error
* *acm* 7/18 Previous test using TM_DIGIT on each individual char, which decided
*            that 1979-2010 is a number, causing a crash just below on the READ. 

	      buff_len = TM_LENSTR(buff1)
              DO ich = 1, buff_len
		c1 = buff1(ich:ich)
		IF ( ( c1.GE.'0' .AND. c1.LE.'9' )) THEN
		   CONTINUE
		ELSE
		   GOTO 740
		ENDIF
		ENDDO
	      READ (buff1,*) dset_num
#elif MANDATORY_FORMAT_WIDTHS
	      READ (buff1,*,ERR=740) dset_num
#else
	      READ (buff1,'(I)',ERR=740) dset_num
#endif
* ... check for valid number given 

	      IF   ( dset_num .LT. 1
     .	      .OR.   dset_num .GT. maxdsets
     .	      .OR.   dset_num .GT. max_gfdl_dsets )	        GOTO 5110
	      IF ( ds_name( dset_num ) .EQ. char_init2048) 	GOTO 5110
	      IF (dsg_user_direction .NE. int4_init) THEN
	         buff1 = ds_des_name(dset_num)
	         GOTO 740
	      ENDIF
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
              IF ( dset_cdf ) THEN
                 ds_dset_type = 'CDF'   ! used and cleared in tm_init_dset
              ELSEIF (data_form .EQ. 'DES') THEN
	         ds_dset_type = ' MC'
	      ELSE
	         GOTO 5710
	      ENDIF
           ENDIF

* initialize a TMAP-supported data set
           CALL TM_INIT_DSET( buff1,
     .			      ok_re_use_existing,
     .			      perm, dset_num, tregular, use_strict, 
     .			      mode_dsg, tmap_status )

* (hack, continued) If the argument is a file which has a ".agg" extension,
* merr_use_agg indicates we have found it even though they gave the 
* "use my_agg_script" without the .agg.

	   IF (tmap_status .EQ. merr_use_agg) THEN  ! .agg file
	      IF (num_items.GT.1) GOTO 5130

	      cmnd_buff(item_end(item)+1:) = '.agg'
	      item_end(item) = item_end(item) + 4
	      arg_end(item) = arg_end(item) + 4
	      CALL XEQ_GO
	      RETURN
	   ENDIF

	   IF ( tmap_status .EQ. merr_no_action ) THEN
	      GOTO 745					! already init'ed
	   ELSEIF ( tmap_status .NE. merr_ok )	  THEN
	      status = tmap_status
	      GOTO 5090					! error in .DES file
	   ENDIF

* determine relative access speeds along each axis
	   CALL GET_AXIS_SPEED( dset_num )

* check for negative time step values (warning, only)
! 3/00	   CALL CHECK_NEG_TAXIS( dset_num )

* make this data set current
 745	   cx_data_set( cx_last )	=  dset_num

* 10/01 allow /TITLE for NetCDF or DODS datasets.
*  5/18 Don't upcase the title given.
           IF (qual_given(slash_set_data_title) .GT. 0) THEN
              loc = qual_given(slash_set_data_title )
              CALL EQUAL_STR_LC(
     .                   cmnd_buff(qual_start(loc):qual_end(loc)),
     .                   title, status )
              IF ( status .NE. ferr_ok ) GOTO 5000
              ds_title(dset_num) = title
           ENDIF

* SET DATA/FMASK= for a dataset that's already open  
           IF (qual_given(slash_set_data_fmask) .GT. 0) THEN
	      loc = qual_given(slash_set_data_fmask)
	      CALL EQUAL_STRING (cmnd_buff(qual_start(loc):qual_end(loc)),
     .                        fmask_var, status)
	      IF (status .NE. ferr_ok) GOTO 5000

* ... eliminate all memory-resident user-defined vars and their uvar_grid 's
	      CALL PURGE_ALL_UVARS

	      DO mr = 1,max_mr_avail
	         IF ( mr_protected( mr ) .EQ. mr_deleted ) CYCLE
	         IF (mr_category(mr) .EQ. cat_user_var) CALL PURGE_ALL_UVARS_SUB( mr_variable(mr) )
	         CALL DELETE_VARIABLE( mr )
	      ENDDO

	      reload = .FALSE.
	      CALL LOAD_DSG_MASKVAR (dset_num, fmask_var, reload, status)
	      

           ENDIF

* SET DATA/SMASK= for a dataset that's already open  
           IF (qual_given(slash_set_data_smask) .GT. 0) THEN
	      loc = qual_given(slash_set_data_smask)
	      CALL EQUAL_STRING (cmnd_buff(qual_start(loc):qual_end(loc)),
     .                        fmask_var, status)
	      IF (status .NE. ferr_ok) GOTO 5000

* ... eliminate all memory-resident user-defined vars and their uvar_grid 's
	      CALL PURGE_ALL_UVARS

	      DO mr = 1,max_mr_avail
	         IF ( mr_protected( mr ) .EQ. mr_deleted ) CYCLE
	         IF (mr_category(mr) .EQ. cat_user_var) CALL PURGE_ALL_UVARS_SUB( mr_variable(mr) )
	         CALL DELETE_VARIABLE( mr )
	      ENDDO

	      reload = .FALSE.
	      CALL LOAD_DSG_MASK_FTRSET_VAR (dset_num, fmask_var, reload, status)

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
     .	.AND. loc .GT.2 ) THEN
           buff2(loc:) = ' '
           buff2 = buff2(2:)
        ENDIF
* ... identify the mode name
        len_test = TM_LENSTR(buff1)
	DO 810 mode = 1, max_modes
           len_mchars = TM_LENSTR(mode_name(mode)(:4))
	   IF ( MATCH4( buff1, len_test,
     .               mode_name( mode ),len_mchars )) THEN
	      IF ( mode.EQ.pmode_GKS .AND. gksopn ) GOTO 5820
* ... ignore MODE VERIFY from a GO file if MODE VERIFY:ALWAYS is set
	      IF ( mode.EQ.pmode_verify
     .	     .AND. mode_arg(mode,1).EQ.mode_set_always
     .	     .AND. csp .GT. 0 ) GOTO 820
	      iarg = unspecified_int4
	      CALL GET_MODE_ARG( mode, buff2, iarg, status )
	      IF ( status .NE. ferr_ok ) GOTO 5000
* ... MODE DESPERATE is ignored as of V7.2
	      IF ( mode .EQ. pmode_desperate ) THEN	      
		 CALL WARN ('MODE DESPERATE is deprecated. Use MODE FRUGAL')
		 GOTO 820
	      ENDIF
	      IF ( set_last ) THEN
* ... pop the mode stack
	         DO i = 2, mode_stack_size
	            mode_state(mode,i-1) = mode_state(mode,i) 
		    mode_arg  (mode,i-1) = mode_arg  (mode,i)
		    IF (mode .EQ. pmode_time_lab) 
     .		      mode_state(pmode_f_lab,i-1) = mode_state(pmode_time_lab,i)
	         ENDDO
	      ELSE
* ... push the mode stack adding on a TRUE (because it is SET)
	         DO i = mode_stack_size, 2, -1
	            mode_state(mode,i) = mode_state(mode,i-1) 
		    mode_arg  (mode,i) = mode_arg  (mode,i-1)
		    IF (mode .EQ. pmode_time_lab) 
     .		      mode_state(pmode_f_lab,i-1) = mode_state(pmode_time_lab,i)
	         ENDDO
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
                 IF ( .NOT. pplus_started ) CALL START_PPLUS(.FALSE.)
                 CALL PPLCMD(from, line, 0, 'PLTNME '//buff2, 1, 1)
                 CALL WARN('the use of "SET MODE METAFILE" is deprecated;')
                 CALL WARN('instead use "FRAME /FILE=... /FORMAT=..."')
                 mode_metafile = .TRUE.
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
	         IF ( .NOT.pplus_started ) CALL START_PPLUS(.FALSE.)
                 IF ( lttout .NE. 6 )
     .              CLOSE( UNIT=lttout, ERR = 5830 )
                 CALL LIB_GET_LUN( lttout )
                 OPEN ( FILE=ppllist_file,
     .                  UNIT=lttout,
     .                  STATUS='UNKNOWN',
     .                  ERR = 5850 )
* ... mode grat sets graticules on, and saves argument
              ELSEIF ( mode .EQ. pmode_grat ) THEN
	         grat_on = .TRUE.
                 mode_grat_buff = buff2

* ... mode linecolors sets the number of line colors on the next new window
* If there is no plot drawn yet, go ahead reset the number of colors now.
              ELSEIF ( mode .EQ. pmode_linecolors ) THEN

                 WRITE (show_str, 2000) mode_arg(pmode_linecolors, 1)
 2000            FORMAT (I3)
                 IF (no_plot_yet) THEN
                    CALL START_PPLUS(.FALSE.)
                    CALL PPLCMD (from, line, 0, 
     .                'LINECOLORS '//show_str(1:3)//' 1', 1, 1)
                    CALL DISP_RESET
                 ELSE
                    CALL PPLCMD (from, line, 0, 
     .                'LINECOLORS '//show_str(1:3)//' 0', 1, 1)
                    CALL WARN ('New # of line colors will take '//
     .                'effect on next new window: SET WIN/NEW then '//
     .                'define pen colors')
                 ENDIF

* ... mode nlevels sets the default number of contour intervals.
              ELSEIF ( mode .EQ. pmode_nlevels ) THEN
                 CALL SET_DEFAULT_NLEVELS( mode_arg(pmode_nlevels, 1) )

* ... mode shrink_ylab allows the y-axis label to automatically shrink
*     so it doesn't run off the page.
              ELSEIF ( mode .EQ. pmode_shrink_ylab ) THEN
	         CALL PPLCMD ( from, line, 0, 'SHRINKY 1', 1, 1 )

* ... Changes to mode calendar apply to F-direction as well as T.
              ELSEIF ( mode .EQ. pmode_time_lab ) THEN
	         ax_dec_pt(f_dim) = ax_dec_pt(t_dim)  
	         ax_fmt(f_dim) = ax_fmt(t_dim) 

* ... mode mode_datefmt sets the default output format for dates
              ELSEIF ( mode .EQ. pmode_datefmt ) THEN
                 date_fmt_out = mode_arg(pmode_datefmt, 1)

* ... mode dsg toggles use of DSG-specific enhancements
*     If mode_dsg was false, and datasets are open, all datasets must be
*     reinitialized.

              ELSEIF ( mode.EQ.pmode_dsg .AND. .NOT.mode_state(mode,2) ) THEN

* ... are any datasets open? They'll need to close them and reinitialize.
		 nopen = 0
		 DO iset = 1, maxdsets
		    IF ( ds_name(iset) .NE. char_init2048) nopen = nopen+1
		 ENDDO
		 IF (nopen.GT.0) GOTO 6920

		 mode_dsg = .TRUE.

	      ENDIF
	      GOTO 820
	   ENDIF
 810	CONTINUE
	   GOTO 5020
 820	RETURN

* SET MOVIE[/FILE=filename] [title]
* Movies are discontinued with Ferret v6.6: NetCDF-4 / HDF5

 900	GOTO 5086

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

* SET AXIS  /MODULO /DEPTH /CALENDAR /REGULAR /T0 /UNITS /OUTTYPE
 1200	axis = TM_GET_LINENUM( cmnd_buff(item_start(1):item_end(1)) )
        IF ( axis .EQ. unspecified_int4 ) GOTO 6210
* /MODULO
        new_att_modulo = .FALSE.
        new_att_modulo_len = 0.D0
	lp = qual_given(slash_set_axis_modulo)
	IF ( lp .GT. 0 ) THEN
	   CALL EQUAL_VAL( cmnd_buff(qual_start(lp):qual_end(lp)),
     .			      val_buf, status )
	   IF ( status .NE. ferr_ok ) RETURN
	   IF (val_buf .EQ. unspecified_val4) THEN
	      val_buf = 0.0
	   ELSE
	      val_buf = ABS(val_buf)  ! ABS to tolerate negatives
	   ENDIF
	   
* Don't allow the modulo length be set longer than the axis length.

	   IF (val_buf .NE. 0.0) THEN
	      axwwlen = TM_WW_AXLEN(axis)

	      IF (.NOT.TM_FPEQ(axwwlen, val_buf) .AND. 
     .			   axwwlen .GT. val_buf) THEN
                 arg = TM_FMT_TRIM(val_buf, -2, 12, len)
                 show_str = TM_FMT_TRIM(axwwlen, -2, 12, slen)
                 buff1 = 'illegal modulo value '//arg(:len) //
     .             ' is less than axis length '//show_str(:slen)
		 buff_len = TM_LENSTR1(buff1)
	         GOTO 6220
	      ENDIF
	   ENDIF

	   its_changed = .NOT.line_modulo(axis)
     .		    .OR.  val_buf .NE. line_modulo_len(axis)
           line_modulo( axis ) = .TRUE.
	   line_modulo_len( axis ) = val_buf

           new_att_modulo = .TRUE.
           new_att_modulo_len = val_buf

* ... wipe memory clear of stored variables - this could change all definitions
*      ( could be more selective for efficiency by checking to see where this
*      axis gets used )
	   IF ( its_changed ) THEN
              DO i = 1,max_mr_avail
		 IF ( mr_protected( i ) .NE. mr_deleted ) 
     .                                CALL DELETE_VARIABLE( i )
     	      ENDDO
           ENDIF
        ENDIF
* /DEPTH
	IF ( qual_given(slash_set_axis_depth) .GT. 0 ) THEN
           IF ( line_direction(axis) .EQ. 'DU' ) THEN
	      line_direction(axis) = 'UD'
           ENDIF
        ENDIF

* /OUTTYPE= INTYPE, DOUBLE, FLOAT, INT, SHORT, or NONE
	loc = qual_given(slash_set_axis_outtype )
	IF ( loc .GT. 0 ) THEN
           CALL EQUAL_STR_LC( cmnd_buff(qual_start(loc):qual_end(loc)),
     .			      buff1, status )
           IF ( status .NE. ferr_ok ) RETURN
           i = STR_UPCASE (typflag, buff1)

           IF (typflag(1:5) .EQ. 'INPUT'  ) THEN   ! input type
              line_dattype(axis) = 99  ! do not change from incoming type
           ELSE IF (typflag(1:3) .EQ. 'BYT' .OR. typflag(1:4) .EQ. 'INT1') THEN
              line_dattype(axis) = 1   ! NC_BYTE
           ELSE IF (typflag(1:3) .EQ. 'SHO' .OR. typflag(1:4) .EQ. 'INT2') THEN
              line_dattype(axis) = 3   ! NC_SHORT
           ELSE IF (typflag(1:3) .EQ. 'INT' .OR. typflag(1:4) .EQ. 'INT4') THEN
              line_dattype(axis) = 4   ! NC_INT (4-byte integer)
           ELSE IF (typflag(1:3) .EQ. 'FLO' .OR. typflag(1:3) .EQ. 'REA') THEN
              line_dattype(axis) = 5   ! NC_FLOAT
           ELSE IF (typflag(1:3) .EQ. 'DOU') THEN
              line_dattype(axis) = 6   ! NC_DOUBLE
           ELSE IF (typflag(1:3) .EQ. 'NON') THEN
              line_dattype(axis) = -1  ! Flag for no-write
           ELSE
              GOTO 6900
           ENDIF
	ENDIF

*/CALENDAR 
        new_att_calendar = ' '
	loc = qual_given(slash_set_axis_calendar) 
        IF (loc .GT. 0) THEN
           IF ( line_direction(axis) .EQ. 'TI' .OR. 
     .          line_direction(axis) .EQ. 'FI' ) THEN

	      CALL EQUAL_STRING(
     .                   cmnd_buff(qual_start(loc):qual_end(loc)),
     .                   buff1, status )
              IF ( status .NE. ferr_ok ) GOTO 6310

              ok_cal = .FALSE.

              IF (buff1(1:3) .EQ. '360') buff1 = '360_DAY' 
              IF (buff1(1:3) .EQ. 'STA') buff1 = 'GREGORIAN' 
              IF (buff1(1:3) .EQ. 'PRO') buff1 = 'GREGORIAN' 
              IF (buff1(1:3) .EQ. 'COM') buff1 = 'NOLEAP' 
              IF (buff1(1:3) .EQ. '365') buff1 = 'NOLEAP' 
              IF (buff1(1:3) .EQ. '366') buff1 = 'ALL_LEAP' 

              DO i = 1, mxcals
                 IF (buff1(1:3) .EQ. allowed_calendars(i)(1:3) ) THEN
                    ok_cal = .TRUE.
                    line_cal_name(axis) = allowed_calendars(i)
                 ENDIF
              ENDDO

              IF (.NOT. ok_cal) GOTO 6320
              new_att_calendar = line_cal_name(axis)

           ENDIF

        ENDIF

*  /REGULAR
        IF ( qual_given(slash_set_axis_regular) .GT. 0 ) THEN
           IF ( .NOT. line_regular(axis) ) THEN
*             ** Get first and last axis values (assumed to be valid)
*                and number of points **
              numpts = line_dim(axis)
              firstval = GET_LINE_COORD (linemem(axis)%ptr, 1)
              lastval = GET_LINE_COORD (linemem(axis)%ptr, numpts)
*             ** Free up memory used by the array of axis values **
              CALL FREE_DYN_MEM(linemem(axis)%ptr)
              CALL FREE_DYN_MEM(lineedg(axis)%ptr)
*             ** Assign values for a regular axis **
              line_start(axis) = firstval
              line_delta(axis) = (lastval - firstval) / (numpts - 1.0)
              line_regular(axis) = .TRUE.
              CALL WARN('Axis reset to be regular (evenly-spaced): '
     .                  //line_name(axis))
           ELSE
              CALL WARN('Axis is already regular: '//line_name(axis))
           ENDIF
        ENDIF

*  /T0 
        new_att_t0 = ' '
	loc = qual_given(slash_set_axis_t0) 
        IF (loc .GT. 0 ) THEN
           IF ( line_direction(axis)(1:1) .EQ. 'T' .OR.
     .          line_direction(axis)(1:1) .EQ. 'F' ) THEN

	      CALL EQUAL_STRING(
     .                cmnd_buff(qual_start(loc):qual_end(loc)),
     .                buff1, status )
              IF ( status .NE. ferr_ok ) GOTO 6410

              status = ferr_grid_definition       ! status setting if tm_date_ok false
              is_ok = TM_DATE_OK ( buff1 , 1) 
 
              IF ( is_ok  ) CALL TM_BREAK_DATE (buff1, 1, year, month, day,
     .				  hour, minute, second, status)

              IF ( status .NE. ferr_ok ) GOTO 6420
              line_t0(axis) = buff1
              new_att_t0 = buff1
              IF (line_direction(axis)(1:1) .EQ. 'T') 
     .          line_direction(axis) = 'TI'
              IF (line_direction(axis)(1:1) .EQ. 'F') 
     .          line_direction(axis) = 'FI'

           ENDIF

        ENDIF

* /NAME

	loc = qual_given(slash_set_axis_name) 
	IF (loc .GT. 0 ) THEN
	   IF ( num_args .EQ. 0 ) GOTO 6680

	   CALL EQUAL_STR_LC(
     .                   cmnd_buff(qual_start(loc):qual_end(loc)),
     .                   buff1, status )
	   IF ( status .NE. ferr_ok ) GOTO 6680

* check that the name isnt an already-defined axis

	   iline = TM_GET_LINENUM (buff1)
	   IF (iline .NE. unspecified_int4) GOTO 6690

	   IF ( .NOT.TM_LEGAL_NAME(buff1) ) GOTO 6692

* Replace the name in the linked-list structure.

          iset = pdset_coordvars  ! stored info on user-defined axes
          CALL CD_GET_VAR_ID (iset, cmnd_buff(item_start(1):item_end(1)), varid, status )
	  IF (varid .GT. 0) CALL CD_RENAME_VAR(iset, varid, buff1, status)

	   DO iset = 1, maxdsets
	      IF ( ds_name(iset) .EQ. char_init2048) CYCLE
	      CALL CD_GET_VAR_ID (iset, line_name(axis), varid, status )
              IF (varid .GT. 0) CALL CD_RENAME_VAR(iset, varid, buff1, status)
	   ENDDO

* Replace the name in Ferret's line storage.
	   line_name_orig(axis) = buff1
	   istat =  STR_UPCASE (buff1, buff1)
	   line_name(axis) = buff1

        ENDIF

*  /UNITS
* Note we do not change line direction with a resetting of units; line
* direction UD, DU, WE, SN stay as they were when they were input.

        new_att_units = ' '
	loc = qual_given(slash_set_axis_units) 
        IF (loc .GT. 0 ) THEN
           IF ( num_args .EQ. 0 ) GOTO 6410

	   CALL EQUAL_STR_LC(
     .                   cmnd_buff(qual_start(loc):qual_end(loc)),
     .                   buff1, status )
           IF ( status .NE. ferr_ok ) GOTO 6510

           since_T0 = MAX( INDEX(buff1,'since'), INDEX(buff1,'SINCE') )
           IF (since_T0 .GT. 0) GOTO 6520

           line_units(axis) = buff1
           new_att_units = buff1

* ... decode the units (code lifted from cd_get_1_axis)

           units = TM_UNIT_ID( line_units(axis) )

* Named calendar with unit=month or year gets length according to calendar's year length.

           cal_id = TM_GET_CALENDAR_ID (line_cal_name(axis))
           units = TM_UNITS_CAL( units, cal_id, .FALSE. )

           line_unit_code(axis) = units
           IF  ( units.EQ.0 .AND. line_units(axis) .NE. ' ' ) THEN
              CALL WARN ('Units  not recognized: '//line_units(axis))
              CALL WARN ('They will not be convertible:')
           ENDIF
           IF ( units .GE. 0 ) THEN
              line_tunit(axis) = real4_init
           ELSE
              line_tunit(axis) = un_convert(line_unit_code(axis))
           ENDIF

* See if we are resetting from a generic XX or YY axis to a geographic one.

           IF (line_direction(axis) .EQ. 'XX') THEN
              IF ( units .EQ. 4 .AND. 
     .              TM_HAS_STRING(line_units(axis), 'deg')
     .             .OR. TM_HAS_STRING(line_units(axis), 'lon') ) THEN 
                 line_direction(axis) = 'WE'

* Should we set axis to modulo?
                 axwwlen = TM_WW_AXLEN(axis)
                 IF ( axwwlen .LE. 360.D0 ) THEN
                    line_modulo(axis) = .TRUE.
                    IF (line_modulo_len(axis) .EQ. 0.0D0)
     .                    line_modulo_len(axis) = 360.D0
                 ENDIF

              ENDIF
           ENDIF

           IF (line_direction(axis) .EQ. 'YY') THEN
              IF ( units .EQ. 4 .AND. 
     .                  TM_HAS_STRING(line_units(axis), 'deg')
     .             .OR. TM_HAS_STRING(line_units(axis), 'lat') ) THEN 
                    line_direction(axis) = 'SN'
              ENDIF
           ENDIF

* recheck units and see if we are resetting from a geographic axis to a 
* non-geographic one or a calendar to non-calendar.

           IF (line_direction(axis) .EQ. 'WE' .OR. 
     .         line_direction(axis) .EQ. 'XX' ) THEN
              IF ( units .NE. 4 ) THEN 
                 line_direction(axis) = 'XX'
                 IF (units .LT. 0) THEN
                    CALL WARN ('Units  not recognized for X axis: '
     .                 //line_units(axis))
                    CALL WARN ('They will not be convertible')
                 ENDIF
              ENDIF
           ENDIF

           IF (line_direction(axis) .EQ. 'SN' .OR. 
     .         line_direction(axis) .EQ. 'YY' ) THEN
              IF ( units .NE. 4 ) THEN 
                 line_direction(axis) = 'YY'
                 IF (units .LT. 0) THEN
                    CALL WARN ('Units  not recognized for Y axis: '
     .                 //line_units(axis))
                    CALL WARN ('They will not be convertible')
                 ENDIF
              ENDIF
           ENDIF

           IF (line_direction(axis) .EQ. 'DU' .OR. 
     .         line_direction(axis) .EQ. 'UD' ) THEN
               IF (units .LT. 0) THEN
                 CALL WARN ('Units  not recognized for Z axis: '
     .                 //line_units(axis))
                 CALL WARN ('They will not be convertible')
              ENDIF
           ENDIF

           IF (line_direction(axis) .EQ. 'TI' .OR. 
     .         line_direction(axis) .EQ. 'TT' ) THEN
              IF ( units .GE. 0 ) THEN 
                 line_direction(axis) = 'TT'
                 CALL WARN ('Units  not recognized for T axis: '
     .                 //line_units(axis))
                 CALL WARN ('They will not be convertible')
              ENDIF
           ENDIF
           IF (line_direction(axis) .EQ. 'FI' .OR. 
     .         line_direction(axis) .EQ. 'FF' ) THEN
              IF ( units .GE. 0 ) THEN 
                 line_direction(axis) = 'FF'
                 CALL WARN ('Units  not recognized for F axis: '
     .                 //line_units(axis))
                 CALL WARN ('They will not be convertible')
              ENDIF
           ENDIF

        ENDIF  ! set axis/units

* Combine new attribute units and/or time origin to express the units
* attribute as units since time-origin

           IF (line_direction(axis)(1:1) .EQ. 'T' .OR. 
     .         line_direction(axis)(1:1) .EQ. 'F' ) THEN
              IF (new_att_units .NE. ' ') THEN
              i1 = TM_LENSTR1(new_att_units)
              new_att_units = new_att_units(1:i1)//' since ' 
     .            // new_att_t0
              IF (new_att_t0 .EQ. ' ') 
     .            new_att_units = new_att_units(1:i1)//' since ' 
     .            // line_t0(axis)
           ELSE
              IF (new_att_t0 .NE. ' ') THEN
                 i1 = TM_LENSTR1(line_units(axis))
                 new_att_units = line_units(axis)(1:i1)//' since ' 
     .            // new_att_t0
              ENDIF
           ENDIF
        ENDIF

*  /STRIDE and /OFFSET

        loc = qual_given(slash_set_axis_stride)
        IF (loc .GT. 0 ) THEN

*       Get the stride value
           IF ( loc .GT. 0 ) THEN
              CALL EQUAL_VAL( cmnd_buff(qual_start(loc):qual_end(loc)),
     .                     val_buf, status )
              IF ( status .NE. ferr_ok ) RETURN
           ELSE
              val_buf = 0
           ENDIF

* ...   STRIDE value must be positive
           delta = val_buf
           IF (delta .LT. 0) GOTO 6670  

* ...   If STRIDE value is 1 -- ignore the command.                    (II.2.c)
           IF( delta .EQ. 1) RETURN 

* ...   Elimate all cached memory                                     (II.1)
           CALL PURGE_ALL_MEMORY

* ...   Eliminate all memory-resident user-defined vars and their 
*       uvar_grid 's (II.1)
           CALL PURGE_ALL_UVARS
 
* ...   Check that it's legal to issue SET AXIS/STRIDE for this axis  (II.2.a)

*       not legal if this is a non-dynamic axis
           IF(axis .LE. max_lines) GOTO 6620           

*       not legal if it is a child axis already
           IF(line_parent(axis) .NE. 0) GOTO 6630

* ...   Check that the /stride and /offset values given are legal      (II.2.b)
           loc = qual_given(slash_set_axis_offset)
           IF (loc .GT. 0) THEN
              CALL EQUAL_VAL( cmnd_buff(qual_start(loc):qual_end(loc)),
     .                     val_buf, status )
              IF ( status .NE. ferr_ok ) RETURN
              offset = INT(val_buf)
           ELSE
              offset = 0
           ENDIF

* ...   offset must be less than the stride value
           IF(offset .GE. delta) GOTO 6640 
           IF(offset .LT. 0) GOTO 6660 

* ... Add 1 to the offset given by the user: SET AXIS/STRIDE= /OFFSET=0
* ... now means start with index=1; etc. Different from orig. implementaion
* ... which had /OFF=1 implies start with index=1.
           offset = offset + 1

* jli: is there a function to get the line orientation idim ?

           igrd=0
	   idim = 0
 6300      CALL TM_NEXT_DYN_GRID( igrd, *6399)
              DO i=1, nferdims
                 IF ( grid_line(i, igrd) .EQ. axis ) THEN
                    idim=i
                 ENDIF
	      ENDDO
           GOTO 6300
 6399      CONTINUE

           IF (idim .EQ. 0) THEN
	      CALL WARN(
     .       'Axis not found in defined grids. Stride not applied')
              RETURN
           ENDIF

* ...   Check that this native axis is used only for variables from 
*       netCDF datasets                                                (II.3)  

*       loop through all dynamic grids

           igrd = 0
 6000      CALL TM_NEXT_DYN_GRID( igrd, *6099)
              IF ( grid_line(idim, igrd) .EQ. axis ) THEN
*       loop through all of the file variables
                 DO i = 1, maxvars
                    IF (ds_grid_number(i) .EQ. igrd) THEN 
		       is_nc = (ds_type(ds_var_setnum(i)).EQ.'CDF') .OR.
     .                         (ds_type(ds_var_setnum(i)).EQ.'ENS') .OR.
     .                         (ds_type(ds_var_setnum(i)).EQ.'FCT') .OR.
     .                         (ds_type(ds_var_setnum(i)).EQ.'UNI')
                       IF (.NOT. is_nc)  
     .                   GOTO 6088
                    ENDIF
                 ENDDO
              ENDIF
           GOTO 6000
 6099      CONTINUE

* ...   Define the new strided axis                                    (II.4)

           dlo = offset      
           dhi = line_dim(axis)
 
           CALL TM_GET_LIKE_DYN_LINE( idim,
     .                                 dlo,
     .                                 dhi,
     .                                 delta,
     .                                 axis,   ! original axis
     .                                 pline_class_stride,
     .                                 strided_axis,  ! your new native-stride axis 
     .                                 status)
            IF ( status .NE. ferr_ok ) GOTO 6650

        line_offset(strided_axis) = offset

        line_reversed(strided_axis) = .FALSE.

*       Check if the native axis is reversed
        IF( (perm(idim) .LT. 0) .AND. 
     .      (perm(idim) .NE. unspecified_int4) ) THEN
            line_reversed(strided_axis) = .TRUE. 
        ENDIF

*       set modulo property
        IF(line_modulo(axis)) THEN
           IF(MOD(dhi,delta) .EQ. 0) THEN
              line_modulo(strided_axis) = .TRUE.
              line_modulo_len(strided_axis) = line_modulo_len(axis) 
           ELSE

              CALL WARN(
     . 'Stride value not an integer factor of axis length: '//
     . 'Axis loses modulo property')
           ENDIF
        ENDIF

* ...   Replace the original axis with the new dynamic axis in 
*       all grids where it is used                                      (II.5)

*       loop through all dynamic grids  

           igrd = 0

 6001      CALL TM_NEXT_DYN_GRID( igrd, *6199)
               IF ( grid_line(idim, igrd) .EQ. axis ) THEN
                  grid_line(idim,igrd) = strided_axis
                  CALL TM_USE_LINE(strided_axis)
               ENDIF
           GOTO 6001        
 6199      CONTINUE
       ENDIF

* ... Check the attribute structure for this axis name and change 
*     attribute values as needed.

       buff2 = line_name(axis)
       CALL RESET_AX_ATT (buff2, new_att_units, 
     .           new_att_t0, new_att_calendar, new_att_modulo,
     .           new_att_modulo_len, line_direction(axis), 
     .           line_regular(axis) )

	RETURN

* SET MEMORY/MAX=megawords (Or equivalently the older SET MEMORY/SIZE=megawords)
* Optional units given as /MAX=xxKb /MAX=xxMb /MAX=xxGb, looking only at the first letter.

 1300	IF ( num_args .GT. 0 ) GOTO 5800
	if (is_secure()) then
	   CALL SPLIT_LIST(pttmode_help, err_lun,
     1	        'This command is not allowed in secure mode.', 0)
	   RETURN
	ENDIF

        lp = qual_given( slash_set_memory_max )
        IF (lp.EQ.0) lp = qual_given( slash_set_memory_words )
	IF ( lp .GT. 0 ) THEN
           CALL EQUAL_STRING( cmnd_buff(qual_start(lp):qual_end(lp)),
     .			      buff2, status )

           IF ( status .NE. ferr_ok ) RETURN
	ELSE
	   val_buf = max_mem_allowed / 1E6    ! retain current max
	   RETURN
	ENDIF

* Max size may be given in Kb, Mb, Gb, or if no units, in double=words

	IF (TM_LENSTR(buff2) .EQ. 0) GOTO 5803
	
	IF (INDEX(buff2, 'K') .GT. 0) THEN
	   slen = INDEX(buff2, 'K')
	   READ (buff2(:slen-1), *, err=5803) val_buf
	   val_buf = val_buf* 1.E3/ 8
	   i8_val = (val_buf*1.0000000001D0) 

	ELSE IF (INDEX(buff2, 'M') .GT. 0) THEN
	   slen = INDEX(buff2, 'M')
	   READ (buff2(:slen-1), *, err=5803) val_buf
	   val_buf = val_buf* 1.E6/ 8
	   i8_val = (val_buf*1.0000000001D0) 

	ELSE IF (INDEX(buff2, 'G') .GT. 0) THEN
	   slen = INDEX(buff2, 'G')
	   READ (buff2(:slen-1), *, err=5803) val_buf
	   val_buf = val_buf* 1.E9/ 8
	   i8_val = (val_buf*1.0000000001D0) 

	ELSE
	   READ (buff2, *, err=5803) val_buf
	   i8_val = (val_buf*1.0000000001D0) * 1000000  !  int tuncation issues ...
	ENDIF

	IF ( val_buf .LE. 0.0 ) GOTO 5805

* reset the memory max

	max_mem_allowed = i8_val 
	peak_mem = 0
	peak_essential_mem = 0

        RETURN

* SET ATTRIBUTE/TYPE= /DATASET= /OUTPUT[=] /LIKE= /QUIET
 1400 IF ( num_args .LT. 1 ) GOTO 5900
      CALL NAME_EQUAL_STRING_VATT ( buff1, pos, status )
      IF ( status .NE. ferr_ok )  GOTO 5000

* Initialize
      dset_num = unspecified_int4
      varname = ' '
      attname = ' '

      loc = qual_given( slash_set_attr_like ) 

*     Inherit all attributes from another variable (except missing flags and
*     scale and offset attributes).
*     Send dummy blank strings to inherit_attrs (needed for LET/LIKE= but not here).
* ... SET ATT/LIKE=var1 var2

      loc = qual_given( slash_set_attr_like ) 
      IF ( loc .GT. 0 ) THEN
	 CALL EQUAL_STR_LC ( cmnd_buff(qual_start(loc):qual_end(loc)),
     .			      buff2, status )
         new_att_units = ' '
         new_att_units = ' '
         CALL INHERIT_ATTRS (buff2, buff1, new_att_units, new_att_units, status)

         RETURN
      ENDIF


* ... SET ATT/OUTPUT varname.attname
* ... SET ATT/OUTPUT=all varname
* ... SET ATT/OUTPUT=default varname
* ... SET ATT/OUTPUT=none varname

      buff2 = ' '
      loc = qual_given( slash_set_attr_output ) 
      IF ( loc .GT. 0 ) THEN
	 CALL EQUAL_STRING( cmnd_buff(qual_start(loc):qual_end(loc)),
     .			      buff2, status )
         GOTO 1410
      ENDIF

      IF ( pos.EQ.0) GOTO 5100
      IF ( INDEX(buff1,'..') .EQ. 1 ) THEN
         ivar = 0  ! global attr.
	 varname = '.'
      ELSE
         IF ( .NOT.TM_LEGAL_NAME(buff1) ) GOTO 5100
      ENDIF

      set_att_quiet = qual_given( slash_set_attr_quiet ) .GT. 0

* SET ATRIBUTE/TYPE=  varname.attname
* types are: BYTE, CHAR or STRING, SHORT or INT2, INT or INT4, FLOAT, DOUBLE
* From netcdf.inc:
*   typedef enum {
*   	NC_NAT =	0,	/* NAT = 'Not A Type' (c.f. NaN) */
*   	NC_BYTE =	1,	/* signed 1 byte integer */
*   	NC_CHAR =	2,	/* ISO/ASCII character */
*   	NC_SHORT =	3,	/* signed 2 byte integer */
*   	NC_INT =	4,	/* signed 4 byte integer */
*   	NC_FLOAT =	5,	/* single precision floating point number */
*   	NC_DOUBLE =	6	/* double precision floating point number */
*   } nc_type;

      attype_spec = ptype_unknown
      loc = qual_given(slash_set_attr_type)
      IF ( loc .GT. 0 ) THEN
         CALL EQUAL_STRING(
     .      cmnd_buff(qual_start(loc):qual_end(loc)),
     .             buff2, status )
         IF ( status .NE. ferr_ok ) RETURN

         IF (buff2(1:4) .EQ. 'BYTE'  .OR.  
     .       buff2(1:4) .EQ. 'INT1') THEN
            attype_spec = ptype_int1
         ELSE IF (buff2(1:4) .EQ. 'SHOR' .OR. 
     .            buff2(1:4) .EQ. 'INT2') THEN
            attype_spec = ptype_int2
         ELSE IF (buff2(1:4) .EQ. 'INT4' .OR.
     .       (buff2(1:3) .EQ. 'INT' .AND. 
     .        TM_LENSTR(buff2) .EQ. 3) ) THEN
            attype_spec = ptype_int4
         ELSE IF (buff2(1:4) .EQ. 'FLOA') THEN
            attype_spec = ptype_float
         ELSE IF (buff2(1:4) .EQ. 'DOUB') THEN
            attype_spec = ptype_double
         ELSE IF (buff2(1:4) .EQ. 'STRI' .OR.  
     .            buff2(1:4) .EQ. 'CHAR') THEN
             attype_spec = ptype_string
         ENDIF
      ENDIF

* ... get dset number if present:  SET ATT/D=dset varname.attname
        dset_num = pdset_irrelevant
	loc = qual_given( slash_set_attr_dset ) 
        IF ( loc .GT. 0 ) THEN
	   CALL EQUAL_STRING( cmnd_buff(qual_start(loc):qual_end(loc)),
     .			      buff2, status )  
	   IF ( status .NE. ferr_ok ) RETURN
	   IF ( buff2 .NE. ' ' ) THEN
	      dset_num = FIND_DSET_NUMBER( buff2 )
	      IF ( dset_num .EQ. unspecified_int4) THEN
	         CALL WARN('Unknown data set: '
     .		 //buff2(:TM_LENSTR(buff2)))
	         RETURN
	      ENDIF
	   ENDIF
	ENDIF

* get dataset from varname.attname
* evaluate the context without computing the expression to get the dataset

 1410   CALL GET_NEW_CX( cx_last, cx_cmnd, .TRUE., status )
        IF ( status .NE. ferr_ok ) GOTO 5000
 
        IF (buff1(1:1) .EQ. '(') THEN
           dset_num = cx_data_set(cx_cmnd)   ! initial value to try; will return dset
           CALL ISIT_COORD_VAR ( buff1, dset_num, 
     .            varname, coordvar, status )
     
           IF ( status .NE. ferr_ok ) THEN 
              dset_num = pdset_coordvars  ! a user-defined axis (coordinate variable)
              CALL ISIT_COORD_VAR ( buff1, dset_num, varname, 
     .              coordvar, status )
              IF ( status .NE. ferr_ok ) THEN
                 dset = cx_data_set(cx_cmnd)
                 GOTO 5910
              ENDIF
           ENDIF

        ELSEIF (buff1(1:1) .EQ. '.') THEN
           dset_num = cx_data_set(cx_cmnd)   ! initial value to try; will return dset

        ELSE
* See if the variable is a user-defined variable.

           varname = buff1
           dot = INDEX(buff1,'.')
           IF (dot .GT. 0) varname = buff1(1:dot-1)
           CALL FIND_VAR_NAME(pdset_irrelevant, varname, cat, ivar)
           IF (  ivar .NE. munknown_var_name .AND.
     .           cat .EQ. cat_user_var) THEN
              dset_num = pdset_uvars

* It may be a let/d variable
              CALL CD_GET_VAR_ID (dset_num, varname, varid, status)
              IF (status.NE.ferr_ok) dset_num = unspecified_int4
           ENDIF
    
           IF (dset_num .EQ. pdset_irrelevant .OR. 
     .         dset_num .EQ. unspecified_int4) THEN
              CALL EVAL_CONTEXT ( cx_cmnd, buff1, status )
              IF ( status .NE. ferr_ok ) GOTO 5000
              cx = is_cx( 1 )
              dset_num = cx_data_set(cx) 
           ENDIF

	ENDIF

        IF (buff1(1:1) .EQ. '.') dset_num = cx_data_set(cx_cmnd)

        IF (dset_num .EQ. pdset_irrelevant) THEN
           dset_num = cx_data_set(cx_last) 
           cx = cx_last
        ENDIF

        IF (dset_num .EQ. unspecified_int4) dset_num = pdset_uvars ! user variables

* change an existing attribute output flag and/or value of the attr

        iflag = 0
        isetout = 0
        IF (qual_given( slash_set_attr_output ) .GT. 0) THEN
           iflag = 1
           isetout = 1
        ENDIF

* get the data for the new value
        
        IF (arg_start(1) .LT. pos) arg_start(1) = pos
        IF ( (iflag .EQ. 1 .AND. arg_end(1) .LT. len_cmnd) .OR. 
     .       (isetout .EQ. 0) ) THEN
           CALL GET_PROT_CMND_DATA (  cx_last, ptype_native, 
     .              status )
           IF ( status .NE. ferr_ok ) THEN      ! 8/6/92
              list_fmt_type = plist_default ! restore for next command
              list_format_given = .FALSE.
              RETURN
           ENDIF
        ENDIF

* Process SET ATT/OUTPUT=all varname, SET ATT/OUTPUT=default varname, SET ATT/OUTPUT=none varname
* flag = 0 output no attributes, 
*        1 check individual attribute output flags (default)
*        2 write all attributes
*        3 reset attr flags to Ferret netcdf defaults

        slen = TM_LENSTR(buff2)
        IF (slen .GT. 0) THEN
           iflag = 1
           IF (buff2(1:3) .EQ. 'NON') iflag = 0
           IF (buff2(1:3) .EQ. 'ALL') iflag = 2
           IF (buff2(1:3) .EQ. 'DEF') iflag = 3 
           IF (INDEX(varname,'.') .NE. 0) THEN 
              IF (iflag.EQ.0 .OR.iflag.EQ.2) GOTO 5930
           ENDIF
        ELSE
           iflag = 1  
        ENDIF

* break up varname.attname.  Check that varname is in the dset

        varid = 0

        CALL BREAK_VARNAME( buff1, dset_num, varname, attname, 
     .                         varid,  status )
        IF (status .NE. ferr_ok) goto 5910

        IF (varid .EQ. 0) CALL CD_GET_VAR_ID (dset_num, varname, varid,
     .       status )
        IF (status .NE. ferr_ok) goto 5910

* Changing the attribute output flag

        IF (qual_given( slash_set_attr_output ) .GT. 0) THEN

           CALL CD_SET_ATT_FLAG(dset_num, varid, attname, iflag, status)

* If it is the offset or scale attribute, set the output flag for both

           slen = TM_LENSTR1(attname)
           IF (MATCH_NAME (attname,  slen, 'ADD_OFFSET', 10) ) THEN
              CALL CD_SET_ATT_FLAG(dset_num, varid, 'ADD_OFFSET', 
     .               iflag, status)
              CALL CD_SET_ATT_FLAG(dset_num, varid, 'SCALE_FACTOR', 
     .               iflag, status)
           ENDIF

           IF (MATCH_NAME (attname,  slen, 'SCALE_FACTOR', 10) ) THEN
              CALL CD_SET_ATT_FLAG(dset_num, varid, 'ADD_OFFSET', 
     .               iflag, status)
              CALL CD_SET_ATT_FLAG(dset_num, varid, 'SCALE_FACTOR', 
     .               iflag, status)
           ENDIF

        ENDIF

* Change attribute value

        IF ( (iflag .EQ. 1 .AND. arg_end(1) .LT. len_cmnd) .OR.
     .       (isetout .EQ. 0) ) THEN 
           mr = is_mr( 1 )
           CALL EDIT_ATTRIBUTE(  memry(mr)%ptr, mr, 
     .               buff1, dset_num, attype_spec, set_att_quiet, status )
  
! errmsg already called in edit_attribute so just exit
           IF (status .NE. ferr_ok) GOTO 5000

* If the time origin was changed, also change units to say
* "units since time origin" using the new time origin info.
           
           axis = TM_GET_LINENUM( varname )
           IF (axis .NE. unspecified_int4) THEN
              slen = TM_LENSTR1(attname)
              IF ( MATCH_NAME (attname,  slen, 'TIME_ORIGIN', 11) .AND.
     .            ( line_direction(axis)(1:1) .EQ. 'T' .OR.
     .              line_direction(axis)(1:1) .EQ. 'F' ) ) THEN
                 CALL CD_GET_VAR_ATT_ID (dset_num, varid, 'UNITS', attid, status)
                 IF (status .NE. ferr_ok) GOTO 5950
   
                  got_it = NC_GET_ATTRIB ( dset_num, varid, 'time_origin',
     .                                      .TRUE., varname, 128,
     .                                         attlen, attoutflag, t0string, 
     .                                   val_buf )
                  got_it = NC_GET_ATTRIB ( dset_num, varid, 'units',
     .                                      .TRUE., varname, 128,
     .                                      attlen, attoutflag, ustring, 
     .                                      val_buf )
                  istat =  STR_UPCASE (buff2, ustring)
                  slen = INDEX(buff2, 'SINCE')
                  IF (slen .GT. 0) THEN
                     slen = slen - 2
                  ELSE
                     slen = TM_LENSTR1(ustring)
                  ENDIF
                  buff3 = ustring(1:slen) // ' since ' // t0string
                  attlen = TM_LENSTR1(buff3)
   
                  ustring = 'units'
                  CALL CD_REPLACE_ATTR (dset_num, varid, ustring, 
     .               NCCHAR, attlen, buff3, val_buf, status)
                  IF (status .NE. ferr_ok) GOTO 5000
               ENDIF

* If the units were changed on a time axis, add "since time origin"
           
              slen = TM_LENSTR1(attname)
              IF (MATCH_NAME (attname,  slen, 'units', 5)  .AND.
     .            ( line_direction(axis)(1:1) .EQ. 'T' .OR.
     .              line_direction(axis)(1:1) .EQ. 'F' ) ) THEN
                 CALL CD_GET_VAR_ATT_ID (dset_num, varid, 'time_origin', 
     .                 attid, status)
                 IF (status .EQ. ferr_ok) THEN 

                    got_it = NC_GET_ATTRIB ( dset_num, varid, 'time_origin',
     .                                      .TRUE., varname, 128,
     .                                      attlen, attoutflag, t0string, 
     .                                      val_buf )
                    got_it = NC_GET_ATTRIB ( dset_num, varid, 'units',
     .                                      .TRUE., varname, 128,
     .                                      attlen, attoutflag, ustring, 
     .                                      val_buf )
                    istat = STR_UPCASE (buff2, ustring)
                    slen = INDEX(buff2, 'SINCE')
                    IF (slen .GT. 0) THEN
                       slen = slen - 2
                    ELSE
                       slen = TM_LENSTR1(ustring)
                    ENDIF
                    buff3 = ustring(1:slen) // ' since ' // t0string
                    attlen = TM_LENSTR1(buff3)
   
                    CALL CD_REPLACE_ATTR (dset_num, varid, attname, 
     .                  NCCHAR, attlen, buff3, val_buf, status)
                    IF (status .NE. ferr_ok) GOTO 5000
                  ENDIF
               ENDIF
            ENDIF ! (axis .NE. unspecified_int4)

* Update attribute information used by plotting etc (ds_units, ...)
            CALL FIND_VAR_NAME (dset_num, varname, cat, uvar)
            IF ( (cat .EQ. cat_user_var) .OR.
     .           (cat .EQ. cat_file_var) .OR.
     .           (cat .EQ. cat_pystat_var) )
     .           CALL UPDATE_ATTRIBUTES (dset_num, varname, uvar, status)
            IF (status .NE. ferr_ok) GOTO 5940

        ENDIF
	RETURN

* SET NCCACHE/SIZE=bytes/cache_nelems/cache_preemption
 1500	IF ( num_args .GT. 0 ) GOTO 5200
        cache_size = -1
        cache_nelems = -1
        cache_preemption = -1

        lp = qual_given( slash_set_nccache_size )
	IF ( lp .GT. 0 ) THEN
           CALL EQUAL_VAL( cmnd_buff(qual_start(lp):qual_end(lp)),
     .			   cache_size_mb, status )
           IF ( status .NE. ferr_ok ) RETURN 
	   IF (cache_size_mb .LT. 0) GOTO 6700
	   cache_size = INT(cache_size_mb* 1.e6)  ! Mbytes to bytes

	ENDIF

        lp = qual_given( slash_set_nccache_nelem )
	IF ( lp .GT. 0 ) THEN
           CALL EQUAL_VAL_INT( cmnd_buff(qual_start(lp):qual_end(lp)),
     .			   cache_nelems, status )
           IF ( status .NE. ferr_ok ) RETURN 
	   IF (cache_nelems .LT. 0) GOTO 6710
	ENDIF

        lp = qual_given( slash_set_nccache_preemp )
	IF ( lp .GT. 0 ) THEN
           CALL EQUAL_VAL_INT( cmnd_buff(qual_start(lp):qual_end(lp)),
     .			   cache_preemption, status )
           IF ( status .NE. ferr_ok ) RETURN 
	   IF (cache_preemption.LT.0 .OR. 
     .         cache_preemption.GT.1) GOTO 6720
	ENDIF

        CALL CD_SET_CHUNK_CACHE (cache_size, cache_nelems, 
     .     cache_preemption, status)

	RETURN

* SET REDIRECT /TEE /JOURNAL /FILE= /APPEND /CLOBBER [stdout] [stderr]
 1600   redir_stdout = .FALSE.
        redir_stderr = .FALSE.
*
*       Determine all the sources to redirect
        IF ( num_args .GT. 0 ) THEN
           DO i = 1,num_args
              IF ( MATCH4( cmnd_buff(arg_start(i):arg_end(i) ), 
     .              arg_end(i) - arg_start(i) + 1, 'STDOUT', 6) ) THEN
*                Redirect stdout with these settings
                 redir_stdout = .TRUE.
              ELSE IF ( MATCH4( cmnd_buff(arg_start(i):arg_end(i) ), 
     .              arg_end(i) - arg_start(i) + 1, 'STDERR', 6) ) THEN
*                Redirect stderr with these settings
                 redir_stderr = .TRUE.
              ELSE
*                Unknown source to redirect
                 CALL ERRMSG( ferr_invalid_command, status, 
     .                   'unknown redirect source: ' //
     .                   cmnd_buff( arg_start(i):arg_end(i) ), *5000 )
              ENDIF
	   ENDDO
        ELSE
*          Default: redirect stdout, but not stderr, with these settings
           redir_stdout = .TRUE.
        ENDIF
*
*       Make sure any the specified redirect sources are not already redirected
        IF ( redir_stdout .AND. 
     .       (redirect_stdout_flags .NE. redirect_none) ) THEN
           CALL ERRMSG( ferr_invalid_command, status, 
     .                  'stdout is already redirected', *5000)
        ENDIF
        IF ( redir_stderr .AND. 
     .       (redirect_stderr_flags .NE. redirect_none) ) THEN
           CALL ERRMSG( ferr_invalid_command, status, 
     .                  'stderr is already redirected', *5000)
        ENDIF
*
*       Process the qualifiers
        redir_tee = .FALSE.
        redir_journal = .FALSE.
        redir_append = .FALSE.
        redir_clobber = .FALSE.
*
*       /TEE
        lp = qual_given( slash_set_redirect_tee )
        IF ( lp .GT. 0 ) redir_tee = .TRUE.
*
*       /JOURNAL
        lp = qual_given( slash_set_redirect_journal )
        IF ( lp .GT. 0 ) redir_journal = .TRUE.
*
*       /APPEND
        lp = qual_given( slash_set_redirect_append )
        IF ( lp .GT. 0 ) redir_append = .TRUE.
*
*       /CLOBBER
        lp = qual_given( slash_set_redirect_clobber )
        IF ( lp .GT. 0 ) redir_clobber = .TRUE.
        IF ( redir_journal .AND. redir_clobber ) THEN
           CALL ERRMSG( ferr_invalid_command, status, 
     .             '/CLOBBER cannot be used with /JOURNAL', *5000 )
        ENDIF
*
*       /FILE=
        lp = qual_given( slash_set_redirect_file )
        IF ( lp .GT. 0 ) THEN
           IF ( redir_journal ) THEN
              CALL ERRMSG( ferr_invalid_command, status,
     .           'Only one of /JOURNAL or /FILE= can be given', *5000 )
           ENDIF
*
*          /FILE= was given; get the filename
           CALL EQUAL_STR_LC( cmnd_buff(qual_start(lp):qual_end(lp)),
     .                        title, status )
           IF ( status .NE. ferr_ok ) RETURN
           IF ( title .EQ. ' ' ) THEN
              CALL ERRMSG( ferr_invalid_command, status,
     .                     'SET REDIRECT /FILE= what name?', *5000 )
           ENDIF

c This check is not made for other file specifications e.g. writing data or graphics
c files, or opening datasets. (ticket 2521).
c           IF ( .NOT. TM_LEGAL_UNIX_NAME(title) ) THEN
c              CALL ERRMSG( ferr_invalid_command, status, 
c     .            'Illegal file name: ' // title, *5000 )
c           ENDIF
*
*          Open or create the file
           INQUIRE( FILE = title, EXIST = fexist )
           IF ( fexist ) THEN    
              IF ( redir_append ) THEN
#ifdef NO_ACCESS_APPEND
                 faccess = 'SEQUENTIAL'
#else
                 faccess = 'APPEND'
                 fposition = 'APPEND'
#endif
                 fstatus = 'OLD'
              ELSE IF ( redir_clobber ) THEN
                 faccess = 'SEQUENTIAL'
                 fstatus = 'REPLACE'
#ifdef F90_OPEN_FILE_CLOBBER
                 fposition = 'ASIS'
#endif
              ELSE
                 CALL ERRMSG( ferr_invalid_command, status,
     .                   'neither /CLOBBER nor /APPEND specified ' //
     .                   'and file exists: ' // title, *5000 )
              ENDIF
           ELSE
*             new file
              faccess = 'SEQUENTIAL'
              fstatus = 'NEW'
              fposition = 'ASIS'
           ENDIF
           CALL LIB_GET_LUN( redir_file_lun )
           OPEN( UNIT = redir_file_lun,
     .           FILE = title,
     .           FORM = 'FORMATTED',
#ifdef F90_OPEN_FILE_APPEND
     .           POSITION = fposition,
#else
     .           ACCESS = faccess,
#endif
     .           STATUS = fstatus,
     .           ERR    = 6800 )
*
*          Assign the file redirection variables
           IF ( redir_stdout ) THEN
              IF ( redir_tee ) THEN
                 redirect_stdout_flags = redirect_file_tee
              ELSE
                 redirect_stdout_flags = redirect_file
              ENDIF
              redirect_stdout_lun = redir_file_lun
           ENDIF
           IF ( redir_stderr ) THEN
              IF ( redir_tee ) THEN
                 redirect_stderr_flags = redirect_file_tee
              ELSE
                 redirect_stderr_flags = redirect_file
              ENDIF
              redirect_stderr_lun = redir_file_lun
           ENDIF
*
        ELSE IF ( redir_journal ) THEN
*
*          /JOURNAL was given - redirect to the journal
           IF ( redir_stdout ) THEN
              IF ( redir_tee ) THEN
                 redirect_stdout_flags = redirect_journal_tee
              ELSE
                 redirect_stdout_flags = redirect_journal
              ENDIF
*             Use the current jrnl_lun (in case it changes) when writing
              redirect_stdout_lun = -1
           ENDIF
           IF ( redir_stderr ) THEN
              IF ( redir_tee ) THEN
                 redirect_stderr_flags = redirect_journal_tee
              ELSE
                 redirect_stderr_flags = redirect_journal
              ENDIF
*             Use the current jrnl_lun (in case it changes) when writing
              redirect_stderr_lun = -1
           ENDIF
*
        ELSE
*
*          Neither /JOURNAL nor /FILE was given
           CALL ERRMSG( ferr_invalid_command, status, 
     .             'Either /JOURNAL or /FILE= must be given', *5000 )
        ENDIF
        RETURN

* SET GIFFILE
 1700   IF ( num_args .EQ. 0 ) GOTO 5100

* *kms* with the graphics delegate, treat it the same as set mode meta:"filename"
        buff2 = cmnd_buff(item_start(1):item_end(1))
        IF ( .NOT. pplus_started ) CALL START_PPLUS(.FALSE.)
        CALL PPLCMD(from, line, 0, 'PLTNME ' // buff2, 1, 1)
        CALL WARN('the use of "SET GIFFILE" is deprecated;')
        CALL WARN('instead use "FRAME /FILE=... /FORMAT=..."')
        mode_metafile = .TRUE.
        RETURN

	
* SET TEXT /FONT /COLOR /ITALIC /BOLD /ISIZ  GROUP

 1800   CONTINUE

	IF (num_items .EQ. 0) THEN 
	   igroup = fnt_all
	   CALL SET_PYTEXT_GROUP (igroup, buff1, status)

	ELSE

	   iset = 0
	   DO item = 1, num_items

	      buff2 = cmnd_buff(item_start(item):item_end(item))

	      len_test = TM_LENSTR(buff2)

	      DO igrp = 1, ngp
                 len_mchars = TM_LENSTR(pyf_group_names(igrp)(:4))
	         IF ( MATCH4( buff2, len_test,
     .               pyf_group_names( igrp ),len_mchars) ) THEN
	            CALL SET_PYTEXT_GROUP (igrp, buff1, status)
		    iset = iset + 1
		 ELSEIF ( STR_SAME( buff2(1:3), "ALL") .EQ. 0) THEN
	            CALL SET_PYTEXT_GROUP (fnt_all, buff1, status)
		    iset = iset + 1
		 ENDIF

	      ENDDO
	      IF (iset .EQ. 0) GOTO 5480
	   ENDDO

 	ENDIF  ! SET TEXT arguments

*  SET TEXT with no qualifiers is an error
	IF (num_qualifiers .EQ. 0) THEN
	   GOTO 5470
	ENDIF

* Set logical flag
	pyfont = (STR_SAME(pyfnt_fontname, 'Hershey') .NE. 0 )
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
 5086	CALL ERRMSG( ferr_invalid_command, status, 
     .     'Movies are discontinued as of Ferret v6.6', *5000 )
 5090	CALL ERRMSG( ferr_TMAP_error, status, ' ', *5000 )
 5100	CALL ERRMSG( ferr_invalid_command, status, 
     .			'SET what?', *5000 )
 5110	CALL ERRMSG( ferr_unknown_data_set, status,
     .			cmnd_buff( item_start(item):item_end(item) ), *5000 )
 5120	CALL ERRMSG( ferr_unknown_grid, status, 
     .			'nothing saved from past commands', *5000 )
 5130	CALL ERRMSG( ferr_syntax, status,
     .		     'Only a single .agg file can be opened in a command'
     .		     //pCR//cmnd_buff(:len_cmnd), *5000 )
 5200	CALL ERRMSG( ferr_syntax, status,
     .		     'To set CACHE CHUNK size use SET CCACHE/SIZE=megabytes'
     .		     //pCR//cmnd_buff(:len_cmnd), *5000 )
 5210	CALL ERRMSG( ferr_syntax, status,
     .		     'To set CACHE CHUNK size use SET CCACHE/SIZE=megabytes'
     .		     //pCR//cmnd_buff(:len_cmnd), *5000 )
 5320	CALL ERRMSG( ferr_invalid_command, status,
     .			'delta value not permitted on SET REGION'//
     .			pCR//'check '//ww_dim_name(idim)//' axis', *5000 )
 5470   CALL ERRMSG( ferr_invalid_command, status,
     .			'SET TEXT what? /FONT/COLOR/BOLD/ITALIC/ISIZ', *5000 )
 5480   CALL ERRMSG( ferr_invalid_command, status,
     .			'SET TEXT  with unknown text-group name', *5000 )
 5700	CALL ERRMSG( ferr_invalid_command, status,
     .			'qualifier(s) valid only with SET DATA/EZ command',
     .			*5000 )
 5710	CALL ERRMSG( ferr_invalid_command, status,
     .		     'illegal qualifier or argument'//pCR//cmnd_buff(:len_cmnd),
     .			*5000 )
 5720	CALL ERRMSG( ferr_invalid_command, status,
     .		     '/ORDER must be permutation of XYZTEF (with minus signs)'//
     .			pCR//cmnd_buff(:len_cmnd), *5000 )
 5800	CALL ERRMSG( ferr_syntax, status,
     .	 'To reconfigure use SET MEMORY/MAX=xxMb or /MAX=xxGb. If no units, Mb assumed'
     .		     //pCR//cmnd_buff(:len_cmnd), *5000 )
 5803	CALL ERRMSG( ferr_invalid_command, status,
     .		     'Unspecified or invalid SET MEMORY/MAX= value: '
     .		     //pCR//cmnd_buff(:len_cmnd), *5000)
 5805	CALL ERRMSG( ferr_invalid_command, status,
     .		     'Negative SET MEMORY/MAX= value: '
     .		     //pCR//cmnd_buff(:len_cmnd), *5000)
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
 6088   CALL ERRMSG( ferr_invalid_command, status,
     .           'it is not a netcdf dataset',
     .               *5000 )
 6110	CALL ERRMSG( ferr_invalid_command, status,
     .		     'no grid has been saved', *5000 )
 6210	CALL ERRMSG( ferr_invalid_command, status,
     .		 'unknown axis: '//cmnd_buff(item_start(1):item_end(1)),
     .               *5000 )
 6220	CALL ERRMSG( ferr_invalid_command, status,
     .		     buff1(:buff_len)//pCR//cmnd_buff(:len_cmnd),
     .               *5000 )
 6310	CALL ERRMSG( ferr_invalid_command, status,
     .	         'SET AXIS/CALENDAR=what? ', 
     .               *5000 )
 6320	CALL ERRMSG( ferr_invalid_command, status,
     .	         'invalid calendar name: '
     .           //cmnd_buff(item_start(1):item_end(1)), 
     .               *5000 )
 6410	CALL ERRMSG( ferr_invalid_command, status,
     .	         'SET AXIS/t0= what? ', 
     .               *5000 )
 6420	CALL ERRMSG( ferr_syntax, status,
     .	         'invalid date string: '//buff1, 
     .               *5000 )
 6510	CALL ERRMSG( ferr_invalid_command, status,
     .	         'SET AXIS/UNITS= what? ', 
     .               *5000 )
 6520	CALL ERRMSG( ferr_syntax, status,
     .	         'specify SET AXIS/T0 separately from /UNITS ',
     .               *5000 )
 6610   CALL ERRMSG( ferr_invalid_command, status,
     .           'SET AXIS/stride= what? ',
     .               *5000 )
 6620   CALL ERRMSG( ferr_invalid_command, status,
     .           'illegal operation for a non-dynamic axis',
     .               *5000 )
 6630   CALL ERRMSG( ferr_invalid_command, status,
     .           'it is a child axis already',
     .               *5000 )
 6640   CALL ERRMSG( ferr_invalid_command, status,
     .           'offset must be less than stride value',
     .               *5000 )
 6650   CALL ERRMSG( ferr_invalid_command, status,
     .           'fail to create a strided axis',
     .               *5000 )
 6660   CALL ERRMSG( ferr_invalid_command, status,
     .           'offset must be greater than or equal to zero',
     .               *5000 )
 6670   CALL ERRMSG( ferr_invalid_command, status,
     .           'stride must be greater than zero',
     .               *5000 )
     
 
 6680	CALL ERRMSG( ferr_invalid_command, status,
     .	         'SET AXIS/NAME= what? ', 
     .               *5000 )

 6690	CALL ERRMSG( ferr_invalid_command, status,
     .               'illegal new name. Axis '
     .			//buff1(:TM_LENSTR1(buff1))//' is already defined ',
     .                                                          *5000 )
 6692	CALL ERRMSG( ferr_invalid_command, status,
     .               'illegal new name given for axis '
     .			//buff1(:TM_LENSTR1(buff1)),
     .               						*5000 )
 
 
 5900 CALL ERRMSG( ferr_invalid_command, status,
     .		'SET what name.attribute?', *5000 )
 5910 slen = TM_LENSTR(buff1)
      CALL ERRMSG( ferr_invalid_command, status,
     .   'variable, axis, or attribute does not exist '//
     .    buff1(:slen), *5000 )
 5920 slen = TM_LENSTR(buff1)
      CALL ERRMSG( ferr_invalid_command, status,
     .   'attribute name missing. SET ATT varname.attname '//
     .    buff1(:slen), *5000 )
 5930 slen = TM_LENSTR(buff1)
      CALL ERRMSG( ferr_invalid_command, status,
     .   'SET ATT/OUT=[all or none] varname should NOT '//
     .   'include attribute. '//
     .    buff1(:slen), *5000 )
 5940 slen = TM_LENSTR(varname)
      CALL ERRMSG( ferr_invalid_command, status,
     .   'error editing attribute for'//
     .    varname(:slen), *5000 )
 5950 slen = TM_LENSTR(buff1)
      CALL ERRMSG( ferr_invalid_command, status,
     .   'cannot set T0 unless Units are also set'//
     .    buff1(:slen), *5000 )
 6700   CALL ERRMSG( ferr_invalid_command, status,
     .           'cache chunk size must be greater than zero',
     .               *5000 )
 6710   CALL ERRMSG( ferr_invalid_command, status,
     .           'cache nelems must be greater than zero',
     .               *5000 )
 6720   CALL ERRMSG( ferr_invalid_command, status,
     .           'cache preemption must be 0 or 1',
     .               *5000 )
 6800   CALL ERRMSG( ferr_erreq, status, title, *5000 )
 6810	CALL ERRMSG( ferr_invalid_command, status,
     .		'Unrecognized option/'//
     .          cmnd_buff(qual_start(lp):qual_end(lp)) //pCR//
     .          ' Valid options are /OUTTYPE=DOUBLE,FLOAT,INT,SHORT,or BYTE', *5000 )
     
 6900	CALL ERRMSG( ferr_unknown_arg, status,
     .'Can SET AXIS/OUTTYPE=INPUT, DOUBLE, FLOAT, INT, SHORT, BYTE, or NONE',
     .               						*5000 )
     
 6910	CALL ERRMSG( ferr_unknown_arg, status,
     . 'Can SET DATA/FEATIRE=trajectory, profile, timeseries, or none',
     .               						*5000 )
     
 6920	CALL ERRMSG( ferr_invalid_command, status,
     .  'SET MODE DSG when datasets are open. '//
     .  'Must first CANCEL DATA/ALL.', *5000 )

	END
