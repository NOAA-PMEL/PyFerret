	BLOCK DATA XPROG_STATE_DATA

*  This software was developed by the Thermal Modeling and Analysis
*  Project(TMAP) of the National Oceanographic and Atmospheric
*  Administration''s (NOAA) Pacific Marine Environmental Lab(PMEL),
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
* variable declarations for COMMON/ XPROG_STATE /
* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 - 3/26/86
* revision 0.1 - 3/18/87 - new TMAP library:
*			   eliminated TMAP_DIMS.PARM and data_set_open
* revision 0.2 - 5/1/87  - added FRAME and LIST output files
* revision 0.3 - 5/19/87 - added mode_ignore_err, mode_stupid and 
*			   err_lun = ttout_lun to take
*			   advantage of 255 byte RECLEN in ttout_lun OPEN
* revision 0.4 - 8/26/87 - added mode_calendar for date/time format time output
*			   and variables for unformatted listing
* revision 0.4 - 10/5/87 - revised list_ options and added mode_gaps_ok
* revision 0.5 - 2/25/88 - added mode_geographical
* revision 0.6 - 3/7/88  - added list_TMAP
* revision 0.7 - 3/23/88 - calendar and geog modes --> long,lat,depth,time
*			   added font_ASCII
* revision 0.8 - 4/15/88 - added mode_state_last
* revision 0.9 - 6/22/88 - GFDL.MGM --> FERRET.MGM
* revision 1.0 -  7/1/88 - added logical interactive and mode segment
* revision 1.1 -  8/2/88 - added mode wait (wait after plots  - for terminals)
* revision 1.2 - 8/11/88 - changed default MODE VERIFY to CANCEL
* revision 1.3 -  1/9/89 - added MODE SCREEN (how to handle missing data)
* revision 1.4 -  2/2/89 - eliminated MODE GAPS_OK (for new TM library)
*			 - added MODE JOURNAL
*			 - changed MODE SCREEN to MODE REJECT
* revision 1.5 - 4/24/89 - MODE INTERPOLATE defaults to "FALSE" to minimize
*			   encountering its bugs !!!
* V200:   6/8/89 - added mode_desperate
*	11/29/89 - added mode_rpn
*	 12/8/89 - added mode_GKS, saved_dset, saved_abstract_grid
*	12/20/89 - added mode_xwindows
*	  1/3/89 - added mode_metafile
* Unix/RISC port - 2/91 - changes to logical unit numbers: standard err and out
*                - 10/91 - mode verify to default to true
* V230:  5/14/92 - replaced mode REMOTE_X with mode REFRESH
*         6/9/92 - replaced list_* logicals with list_fmt_type
*        11/9/92 - eliminated arguments from MODES GKS and META
*        1/14/93 - meta_device --> frame_compress
* V300:   2/3/93 - added argument FULL to MODE VERIFY
* V300:  5/12/93 - added mode ppllist and journal_file
*	 6/16/93 - added mode GUI
*	25/10/93 - default state for mode refresh set to TRUE
* V301: 26/ 1/94 - larger MODE DESPERATE argument to reflect larger memory
*	 2/24/94 - incorporated a stack of past SET MODE states
*	 3/14/94 - set default state of mode refresh (xwindows) to .FALSE.
* V310:  11/4/94 - eliminated MODE REJECT (never implemented)
* V400:  7/10/94 - added list_format for LIST/HEADING=enhanced
*
*        6/15/95 -*kob* added ifdef for STDERR_UNIT_SEVEN because stderr on
*                  HP machines was 7 instead of 0
* V420:	 4/28/96 - added initialization for IF-stack (ifstk)
* V450:	 11/19/96 - eliminated MODE POLISH
* V510 5/00 *sh* - added denig_x/ylim_msg_done
* V540 11/01 *sh* - initialize last_cmnd_buff
* V550:12/02 *acm*- new MODE LOGO and MODE LABELS
* V570: *acm* 5/04 - add MODE GRATICULE[:argument] 
* v580 10/04 *acm* - add flag silent_err for SHOW VAR/XML This is set for  
*                    SHOW VAR/XML when checking for user variables not  
*                    relevant to the context
* V600: *acm* 5/06 - add new MODE LINECOLORS:[arg],Flag no_plot_yet
* V600  *acm* 6/06 - add new MODE UPCASE_OUTPUT
* V610  *acm* 3/08 - add new MODE NLEVELS
* V614  *acm* 10/08 - add new MODE NODATA_LAB
* V6.2  *acm*  5/09 - new default number levels is 30 pmode_nlevels
*                If this is changed, also change it in ppl/plot/ppldata.F
* V6.7  *kms* 8/11 - moved one_cmnd_mode here since needed outside ferret_dispatch.F
*       *kms*  2/12 Add pmode_ensemble_lab, pmode_forecast_lab,
*                   mode_ensemble_lab, and mode_forecast_lab;
*                   include tmap_dims.parm for nferdims needed by xprog_state.cmn
*       *acm* 4/12 Add new MODE FOURDEE to allow E,F,M,N,etc as user-vars
*       *acm* 4/12 Remove MODE FOURDEE, using _E, _F, _M, _N instead, throughout.
* V6.8  acm  6/12 Instead of mode_ensemble_lab, mode_forecast_lab, use mode_F_LAB, 
*                 which will be on by default. If canceled, the various SHOW and 
*                 LIST commands, etc, will not list info for E,F dimensions.
* V683  *acm* 8/12 - add new MODE SHRINK_YLAB
* V683  *acm* 8/12 - frame file FERRET.MGM  > ferret.gif
* PyFr  *kms* 8/12 - frame file ferret.gif -> ferret.png
* V685  *sh* 11/13 - added allowed_err for SH VAR/TREE
* V690  *sh* 12/13 - added mode_curvilinear, mode_sigma
* V694 11/14 *sh* renaming 'sigma' as 'layerz' throughout
* V695  *acm* 5/15 logical flag its_pyferret 
* V695  *acm* 5/13 - Ticket 2281: mode_e_lab and mode_f_lab to act as mode_lon_lab, etc, 
*                    precision of E and F labeling. The values are equivalenced to 
*                    ax_dec_pt(MODE_6D_LAB still turns on and off the listing of all 
*                    6 dimensions on SHOW commands.)
* V698  4/16 *acm* For ticket 2399: Apply MODE CALENDAR settings to F axes too
* V702  3/17 *sh* new mode FRUGAL
*                 SPAWN command returns spawn_status
* v720 7/17 *acm* ticket 2552 Make SPAWN_STATUS return the return status, so 0=succcess
*                 SPAWN_STATUS returns the status value, used for SPAWN_OK and SPAWN_STATUS
* v720 7/17 *kms* remove unused its_pyferret flag
* v745 *acm* 12/18  Issue 1909, option to write dates in ISO8601 format
*                  new mode datefmt for formatted date output, setting stored in date_fmt_out
* V751 6/19 *acm* TM_FMT rewrite: formatting of lon/lat coordinate strings set to match previous
* V745 *acm*  5/20  added mode_dsg to control turning off all DSG ops
** 10/2022 *acm* Code cleanup: lengths of strings not to overflow length of character variable

        IMPLICIT NONE
        include 'tmap_dims.parm'
	include 'ferret.parm'
	include 'movies.parm'
	include 'xprog_state.cmn'

* internal variable declarations:
	INTEGER	i

* for getting commands
	DATA	prompt		/'yes? '/,
     .		prompt_len	/ 5 	/

	DATA	input_source	/ input_from_tty /,
     .		mouse_input 	/.FALSE./,
     .          one_cmnd_mode   /.FALSE./

	DATA	last_cmnd_buff/' '/,
     .		len_last_cmnd/1/

* I/O logical unit numbers
	DATA	ttin_lun	/ 5	/,	! Unix standard input
     .		ttout_lun	/ 6	/,	! Unix standard output
     .		list_lun	/ 6	/,	! same as ttout to start
     .		jrnl_lun	/ 20	/,
#ifdef STDERR_UNIT_SEVEN
     .		err_lun		/ 7	/,	! HP Unix standard error
#else
     .		err_lun		/ 0	/,	! Unix standard error
#endif
     .		cmnd_lun	/ 21	/

* for SET REDIRECT
        DATA    redirect_stdout_flags  / redirect_none /,
     .          redirect_stderr_flags  / redirect_none /

* for SET MODE
	DATA	(mode_state( pmode_diagnostic  ,i), i=1,mode_stack_size)
     .					/ mode_stack_size*.FALSE.   /,
     .		(mode_state( pmode_verify      ,i), i=1,mode_stack_size)
     .					/ mode_stack_size*.TRUE.   /,   ! 10/91
     .		(mode_state( pmode_interpolate ,i), i=1,mode_stack_size)
     .					/ mode_stack_size*.FALSE.   /,
     .		(mode_state( pmode_ignore_err  ,i), i=1,mode_stack_size)
     .					/ mode_stack_size*.FALSE.   /,
     .		(mode_state( pmode_stupid      ,i), i=1,mode_stack_size)
     .					/ mode_stack_size*.FALSE.   /,
     .		(mode_state( pmode_journal     ,i), i=1,mode_stack_size)
     .					/ mode_stack_size*.TRUE.    /,
     .		(mode_state( pmode_long_lab    ,i), i=1,mode_stack_size)
     .					/ mode_stack_size*.TRUE.    /,
     .		(mode_state( pmode_lat_lab     ,i), i=1,mode_stack_size)
     .					/ mode_stack_size*.TRUE.    /,
     .		(mode_state( pmode_depth_lab   ,i), i=1,mode_stack_size)
     .					/ mode_stack_size*.TRUE.    /,
     .		(mode_state( pmode_time_lab    ,i), i=1,mode_stack_size)
     .					/ mode_stack_size*.TRUE.    /,
     .		(mode_state( pmode_E_LAB      ,i), i=1,mode_stack_size)
     .					/ mode_stack_size*.TRUE.    /,
     .		(mode_state( pmode_F_LAB      ,i), i=1,mode_stack_size)
     .					/ mode_stack_size*.TRUE.    /
	DATA	(mode_state( pmode_font_ascii  ,i), i=1,mode_stack_size)
     .					/ mode_stack_size*.TRUE.    /,
     .		(mode_state( pmode_segment     ,i), i=1,mode_stack_size)
     .					/ mode_stack_size*.TRUE.    /,
     .		(mode_state( pmode_wait        ,i), i=1,mode_stack_size)
     .					/ mode_stack_size*.FALSE.   /,
     .		(mode_state( pmode_reject      ,i), i=1,mode_stack_size)
     .					/ mode_stack_size*.FALSE.   /,
     .		(mode_state( pmode_desperate   ,i), i=1,mode_stack_size)
     .					/ mode_stack_size*.FALSE.   /,
     .		(mode_state( pmode_rpn         ,i), i=1,mode_stack_size)
     .					/ mode_stack_size*.FALSE.   /,
     .		(mode_state( pmode_GKS         ,i), i=1,mode_stack_size)
     .					/ mode_stack_size*.FALSE.   /,	! irrelevant
     .		(mode_state( pmode_xwindows    ,i), i=1,mode_stack_size)
     .					/ mode_stack_size*.FALSE.   /
	DATA	(mode_state( pmode_metafile    ,i), i=1,mode_stack_size)
     .					/ mode_stack_size*.FALSE.   /,
     .		(mode_state( pmode_ppllist     ,i), i=1,mode_stack_size)
     .					/ mode_stack_size*.FALSE.   /,
     .		(mode_state( pmode_gui        ,i), i=1,mode_stack_size)
     .					/ mode_stack_size*.FALSE.   /,
     .		(mode_state( pmode_logo        ,i), i=1,mode_stack_size)
     .					/ mode_stack_size*.TRUE.   /,
     .		(mode_state( pmode_labels      ,i), i=1,mode_stack_size)
     .					/ mode_stack_size*.TRUE.   /,
     .		(mode_state( pmode_grat      ,i), i=1,mode_stack_size)
     .					/ mode_stack_size*.FALSE.   /,
     .		(mode_state( pmode_linecolors,i), i=1,mode_stack_size)
     .					/ mode_stack_size*.TRUE.   /,
     .		(mode_state( pmode_upcase_output,i), i=1,mode_stack_size)
     .					/ mode_stack_size*.TRUE.   /,
     .		(mode_state( pmode_nlevels,i), i=1,mode_stack_size)
     .					/ mode_stack_size*.TRUE.   /,
     .		(mode_state( pmode_nodata_lab,i), i=1,mode_stack_size)
     .					/ mode_stack_size*.TRUE.   /,
     .		(mode_state( pmode_shrink_ylab,i), i=1,mode_stack_size)
     .					/ mode_stack_size*.FALSE.   /
     .		(mode_state( pmode_curvilinear,i), i=1,mode_stack_size)
     .					/ mode_stack_size*.FALSE.   /
     .		(mode_state( pmode_layerz,i), i=1,mode_stack_size)
     .					/ mode_stack_size*.FALSE.   /
     .		(mode_state( pmode_6d_lab,i), i=1,mode_stack_size)
     .					/ mode_stack_size*.TRUE.   /
     .		(mode_state( pmode_frugal,i), i=1,mode_stack_size)
     .					/ mode_stack_size*.TRUE.   /
     .		(mode_state( pmode_datefmt,i), i=1,mode_stack_size)
     .					/ mode_stack_size*.TRUE.   /
     .		(mode_state( pmode_dsg     ,i), i=1,mode_stack_size)
     .					/ mode_stack_size*.TRUE.    /

	DATA	mode_name ( pmode_diagnostic  )	/ 'DIAGNOSTIC  ' /,
     .		mode_name ( pmode_verify      )	/ 'VERIFY      ' /,
     .		mode_name ( pmode_interpolate )	/ 'INTERPOLATE ' /,
     .		mode_name ( pmode_ignore_err  )	/ 'IGNORE_ERROR' /,
     .		mode_name ( pmode_stupid      )	/ 'STUPID'	 /,
     .		mode_name ( pmode_journal     )	/ 'JOURNAL'	 /,
     .		mode_name ( pmode_long_lab    )	/ 'LONG_LABEL'   /,
     .		mode_name ( pmode_lat_lab     )	/ 'LATIT_LABEL'  /,
     .		mode_name ( pmode_depth_lab   )	/ 'DEPTH_LABEL'  /,
     .		mode_name ( pmode_time_lab    )	/ 'CALENDAR'     /,
     .		mode_name ( pmode_E_LAB)	/ 'E_LABEL'      /,
     .		mode_name ( pmode_F_LAB)	/ 'F_LABEL'      /,
     .		mode_name ( pmode_font_ascii  )	/ 'ASCII_FONT'   /,
     .		mode_name ( pmode_segment     )	/ 'SEGMENT'      /,
     .		mode_name ( pmode_wait        )	/ 'WAIT'         /,
     .		mode_name ( pmode_reject      )	/ pmode_unused   /,
     .		mode_name ( pmode_desperate   )	/ 'DESPERATE'    /,
     .		mode_name ( pmode_rpn         )	/ pmode_unused /,
     .		mode_name ( pmode_gks         )	/ 'GKS'          /,
     .		mode_name ( pmode_xwindows    )	/ 'REFRESH'      /,
     .		mode_name ( pmode_metafile    )	/ 'METAFILE'     /,
     .		mode_name ( pmode_ppllist     )	/ 'PPLLIST'      /,
     .		mode_name ( pmode_gui         )	/ 'GUI'          /,
     .		mode_name ( pmode_logo        )	/ 'LOGO'         /,
     .		mode_name ( pmode_labels      )	/ 'LABELS'       /,
     .		mode_name ( pmode_grat        )	/ 'GRATICULE'    /,
     .		mode_name ( pmode_linecolors  )	/ 'LINECOLORS'   /,
     .		mode_name ( pmode_upcase_output)/ 'UPCASE_OUTPU'/,
     .		mode_name ( pmode_nlevels     ) / 'NLEVELS'      /,
     .		mode_name ( pmode_nodata_lab  ) / 'NODATA_LAB'   /,
     .		mode_name ( pmode_shrink_ylab ) / 'SHRINK_YLAB'  /,
     .		mode_name ( pmode_curvilinear ) / 'CURVILINEAR'  /,
     .		mode_name ( pmode_layerz      ) / 'LAYERZ'       /,
     .		mode_name ( pmode_6d_lab      ) / '6D_LAB'       /,
     .		mode_name ( pmode_frugal      ) / 'FRUGAL'	 /,
     .		mode_name ( pmode_datefmt     ) / 'DATE_FORMAT'	 /,
     .		mode_name ( pmode_dsg         ) / 'DSG'	 /,
     .		mode_name ( 38     ) / pmode_unused /,
     .		mode_name ( 39     ) / pmode_unused /,
     .		mode_name ( 40     ) / pmode_unused /

	DATA	(mode_arg  ( pmode_diagnostic  ,i), i=1,mode_stack_size)
     .				/ mode_stack_size*99 /,
     .		(mode_arg  ( pmode_verify      ,i), i=1,mode_stack_size)
     .				/ mode_stack_size*1 /,
     .		(mode_arg  ( pmode_interpolate ,i), i=1,mode_stack_size)
     .				/ mode_stack_size*unspecified_int4 /,
     .		(mode_arg  ( pmode_ignore_err  ,i), i=1,mode_stack_size)
     .				/ mode_stack_size*unspecified_int4 /,
     .		(mode_arg  ( pmode_stupid      ,i), i=1,mode_stack_size)
     .				/ mode_stack_size*unspecified_int4 /,
     .		(mode_arg  ( pmode_journal     ,i), i=1,mode_stack_size)
     .				/ mode_stack_size*unspecified_int4 /,
     .		(mode_arg  ( pmode_long_lab    ,i), i=1,mode_stack_size)
     .				/ mode_stack_size* 4 /, ! 4 sig. digs.  
     .		(mode_arg  ( pmode_lat_lab     ,i), i=1,mode_stack_size)
     .				/ mode_stack_size* 3 /, ! 3 sig. digs.  
     .		(mode_arg  ( pmode_depth_lab   ,i), i=1,mode_stack_size)
     .				/ mode_stack_size*4 /, ! 4 sig. digs. 
     .		(mode_arg  ( pmode_time_lab    ,i), i=1,mode_stack_size)
     .				/ mode_stack_size*5 /, ! dd-mmm-yyyy:hh:mm
     .		(mode_arg  ( pmode_e_lab     ,i), i=1,mode_stack_size)
     .				/ mode_stack_size*4 / 
     .		(mode_arg  ( pmode_f_lab     ,i), i=1,mode_stack_size)
     .				/ mode_stack_size*5 /  ! dd-mmm-yyyy:hh:mm
	DATA	(mode_arg  ( pmode_font_ascii  ,i), i=1,mode_stack_size)
     .				/ mode_stack_size*unspecified_int4 /,
     .		(mode_arg  ( pmode_segment     ,i), i=1,mode_stack_size)
     .				/ mode_stack_size*unspecified_int4 /,
     .		(mode_arg  ( pmode_wait        ,i), i=1,mode_stack_size)
     .				/ mode_stack_size*unspecified_int4 /,
     .		(mode_arg  ( pmode_reject      ,i), i=1,mode_stack_size)
     .				/ mode_stack_size*unspecified_int4 /,
     .		(mode_arg  ( pmode_desperate   ,i), i=1,mode_stack_size)
     .				/ mode_stack_size*0 /,
     .		(mode_arg  ( pmode_rpn         ,i), i=1,mode_stack_size)
     .				/ mode_stack_size*unspecified_int4 /,
     .		(mode_arg  ( pmode_gks         ,i), i=1,mode_stack_size)
     .				/ mode_stack_size*unspecified_int4 /,
     .		(mode_arg  ( pmode_xwindows    ,i), i=1,mode_stack_size)
     .				/ mode_stack_size*unspecified_int4 /,
     .		(mode_arg  ( pmode_metafile    ,i), i=1,mode_stack_size)
     .				/ mode_stack_size*unspecified_int4 /,
     .		(mode_arg  ( pmode_ppllist     ,i), i=1,mode_stack_size)
     .				/ mode_stack_size*unspecified_int4 /
	DATA	(mode_arg  ( pmode_gui        ,i), i=1,mode_stack_size)
     .				/ mode_stack_size*unspecified_int4 /,
     .		(mode_arg  ( pmode_logo        ,i), i=1,mode_stack_size)
     .				/ mode_stack_size*unspecified_int4 /,
     .		(mode_arg  ( pmode_labels      ,i), i=1,mode_stack_size)
     .				/ mode_stack_size*unspecified_int4 /,
     .		(mode_arg  ( pmode_grat      ,i), i=1,mode_stack_size)
     .				/ mode_stack_size*unspecified_int4 /,
     .		(mode_arg  ( pmode_linecolors,i), i=1,mode_stack_size)
     .				/ mode_stack_size*6 /,
     .		(mode_arg  ( pmode_upcase_output,i), i=1,mode_stack_size)
     .				/ mode_stack_size*unspecified_int4 /,
     .		(mode_arg  ( pmode_nlevels,i), i=1,mode_stack_size)
     .				/ mode_stack_size*30 /,
     .		(mode_arg  ( pmode_nodata_lab,i), i=1,mode_stack_size)
     .				/ mode_stack_size*unspecified_int4 /,
     .		(mode_arg  ( pmode_shrink_ylab,i), i=1,mode_stack_size)
     .				/ mode_stack_size*unspecified_int4 /
     .		(mode_arg  ( pmode_curvilinear,i), i=1,mode_stack_size)
     .				/ mode_stack_size*unspecified_int4 /
     .		(mode_arg  ( pmode_layerz,i), i=1,mode_stack_size)
     .				/ mode_stack_size*unspecified_int4 /
     .		(mode_arg  ( pmode_6d_lab,i), i=1,mode_stack_size)
     .				/ mode_stack_size*unspecified_int4 /
     .		(mode_arg  ( pmode_frugal,i), i=1,mode_stack_size)
     .				/ mode_stack_size*30 /
     .		(mode_arg  ( pmode_datefmt,i), i=1,mode_stack_size)
     .				/ mode_stack_size*pdate_vax /
     .		(mode_arg  ( pmode_dsg     ,i), i=1,mode_stack_size)
     .				/ mode_stack_size*unspecified_int4 /

* If the value of nlevels is changed, also change NLEV in plot/ppldata.F


* for MODE WAIT
	DATA	ppl_wait / 'NOWAIT' /

* for LIST
	DATA	list_format_given / .FALSE. /
	DATA	list_format	  / '(5(1PG12.5))' /
!	DATA	list_redirected	  / .FALSE. /   ! unused
!	DATA	list_dsf	  / .FALSE. /	! equiv to list_fmt_type
!	DATA	list_TMAP	  / .FALSE. /	! unused
!	DATA	list_unformatted  / .FALSE. /   ! unused
	DATA	list_fmt_type     / plist_default /
	DATA	list_append	  / .FALSE. /
	DATA	list_heading	  / .TRUE.  /
	DATA	list_file	  / 'AUTO'	 /
	DATA	list_outtype	  / 'DFLT'	 /
	DATA	list_digits	  / default_output_precision /

* for FRAME
	DATA	frame_file	  / 'ferret.png' /
        DATA    frame_compress    / 'RLE' /  ! matching pframe_hdf_runlen

* other filenames
	DATA	journal_file	  / 'ferret.jnl'  /
	DATA	ppllist_file	  / 'ppllist.out' /

* stack pointer for nested IF statements
	DATA	ifstk		 / 0 /
     .		if_conditional	 / .FALSE. /

* for issuing messages about the denigrated status of qualifiers
	DATA	denig_xylim_msg_done / .FALSE. /
	DATA	denig_pen_msg_done / .FALSE. /

* auto layerz and curvilinear states
	DATA	mode_auto_curvi       /.FALSE./,
     .		mode_auto_curvi_last  /.FALSE./,
     .		mode_auto_layerz      /.FALSE./,
     .		mode_auto_layerz_last /.FALSE./

* miscellaneous
	DATA	saved_dset		/ unspecified_int4 /
	DATA	saved_abstract_grid	/ unspecified_int4 /
	DATA	xwind_node		/ ' ' /
	DATA	mode_grat_buff		/ ' ' /
        DATA    silent_err              / .FALSE. /  ! will set true for SHOW VAR/XML
        DATA    allowed_err             / .FALSE. /  ! briefly set true in SHOW VAR/TREE
        DATA    grat_on                 / .FALSE. /
        DATA    num_linecolors          / 6 /
        DATA    spawn_status            / 0 /  ! zero means last cmnd was successful
	DATA	date_fmt_out		/ pdate_vax /

	END
