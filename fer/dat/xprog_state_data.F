	BLOCK DATA XPROG_STATE_DATA

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
* V301: 26/ 1/94 - larger MODE DESPARATE argument to reflect larger memory
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

	include 'ferret.parm'
	include 'movies.parm'
	include 'xprog_state.cmn'

* internal variable declarations:
	INTEGER	i

* for getting commands
	DATA	prompt		/'yes? '/,
     .		prompt_len	/ 5 	/

	DATA	input_source	/ input_from_tty /,
     .		mouse_input 	/.FALSE./

	DATA	last_cmnd_buff/' '/,
     .		len_last_cmnd/1/

* I/O logical unit numbers
#ifdef unix
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
#else
	DATA	ttin_lun	/ 5	/,
     .		ttout_lun	/ 18	/,
     .		list_lun	/ 18	/,	! same as ttout to start
     .		jrnl_lun	/ 20	/,
     .		err_lun		/ 18	/,	! same as ttout (to start)
     .		cmnd_lun	/ 21	/
#endif

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
     .					/ mode_stack_size*.TRUE.    /
	DATA	(mode_state( pmode_font_ascii  ,i), i=1,mode_stack_size)
     .					/ mode_stack_size*.TRUE.    /,
     .		(mode_state( pmode_segment     ,i), i=1,mode_stack_size)
     .					/ mode_stack_size*.TRUE.    /,
     .		(mode_state( pmode_wait        ,i), i=1,mode_stack_size)
     .					/ mode_stack_size*.FALSE.   /,
     .		(mode_state( pmode_reject      ,i), i=1,mode_stack_size)
     .					/ mode_stack_size*.FALSE.   /,
     .		(mode_state( pmode_desparate   ,i), i=1,mode_stack_size)
     .					/ mode_stack_size*.FALSE.   /,
     .		(mode_state( pmode_rpn         ,i), i=1,mode_stack_size)
     .					/ mode_stack_size*.FALSE.   /,
     .		(mode_state( pmode_GKS         ,i), i=1,mode_stack_size)
     .					/ mode_stack_size*.FALSE.   /,	! irrelevant
     .		(mode_state( pmode_xwindows    ,i), i=1,mode_stack_size)
     .					/ mode_stack_size*.FALSE.   /,
     .		(mode_state( pmode_metafile    ,i), i=1,mode_stack_size)
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
     .					/ mode_stack_size*.FALSE.   /
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
     .		mode_name ( pmode_font_ascii  )	/ 'ASCII_FONT'   /,
     .		mode_name ( pmode_segment     )	/ 'SEGMENT'      /,
     .		mode_name ( pmode_wait        )	/ 'WAIT'         /,
     .		mode_name ( pmode_reject      )	/ pmode_unused   /,
     .		mode_name ( pmode_desparate   )	/ 'DESPERATE'    /,
     .		mode_name ( pmode_rpn         )	/ pmode_unused /,
     .		mode_name ( pmode_gks         )	/ 'GKS'          /,
     .		mode_name ( pmode_xwindows    )	/ 'REFRESH'      /,
     .		mode_name ( pmode_metafile    )	/ 'METAFILE'     /,
     .		mode_name ( pmode_ppllist     )	/ 'PPLLIST'      /,
     .		mode_name ( pmode_gui         )	/ 'GUI'          /,
     .		mode_name ( pmode_logo        )	/ 'LOGO'         /,
     .		mode_name ( pmode_labels      )	/ 'LABELS'       /,
     .		mode_name ( pmode_grat      )	/ 'GRATICULE'    /
	DATA  ( mode_name(i),i=pmode_grat+1,max_modes)/3*pmode_unused/


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
     .				/ mode_stack_size* 1 /, ! 1 decimal place
     .		(mode_arg  ( pmode_lat_lab     ,i), i=1,mode_stack_size)
     .				/ mode_stack_size* 1 /, ! 1 decimal place
     .		(mode_arg  ( pmode_depth_lab   ,i), i=1,mode_stack_size)
     .				/ mode_stack_size*-4 /, ! 4 sig. digs. less 0's
     .		(mode_arg  ( pmode_time_lab    ,i), i=1,mode_stack_size)
     .				/ mode_stack_size*-5 /	! dd-mmm-yyyy:hh:mm
	DATA	(mode_arg  ( pmode_font_ascii  ,i), i=1,mode_stack_size)
     .				/ mode_stack_size*unspecified_int4 /,
     .		(mode_arg  ( pmode_segment     ,i), i=1,mode_stack_size)
     .				/ mode_stack_size*unspecified_int4 /,
     .		(mode_arg  ( pmode_wait        ,i), i=1,mode_stack_size)
     .				/ mode_stack_size*unspecified_int4 /,
     .		(mode_arg  ( pmode_reject      ,i), i=1,mode_stack_size)
     .				/ mode_stack_size*unspecified_int4 /,
     .		(mode_arg  ( pmode_desparate   ,i), i=1,mode_stack_size)
     .				/ mode_stack_size*320000 /,
     .		(mode_arg  ( pmode_rpn         ,i), i=1,mode_stack_size)
     .				/ mode_stack_size*unspecified_int4 /,
     .		(mode_arg  ( pmode_gks         ,i), i=1,mode_stack_size)
     .				/ mode_stack_size*unspecified_int4 /,
     .		(mode_arg  ( pmode_xwindows    ,i), i=1,mode_stack_size)
     .				/ mode_stack_size*unspecified_int4 /,
     .		(mode_arg  ( pmode_metafile    ,i), i=1,mode_stack_size)
     .				/ mode_stack_size*unspecified_int4 /,
     .		(mode_arg  ( pmode_ppllist     ,i), i=1,mode_stack_size)
     .				/ mode_stack_size*unspecified_int4 /,
     .		(mode_arg  ( pmode_gui        ,i), i=1,mode_stack_size)
     .				/ mode_stack_size*unspecified_int4 /,
     .		(mode_arg  ( pmode_logo        ,i), i=1,mode_stack_size)
     .				/ mode_stack_size*unspecified_int4 /,
     .		(mode_arg  ( pmode_labels      ,i), i=1,mode_stack_size)
     .				/ mode_stack_size*unspecified_int4 /,
     .		(mode_arg  ( pmode_grat      ,i), i=1,mode_stack_size)
     .				/ mode_stack_size*unspecified_int4 /

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
	DATA	list_digits	  / default_output_precision /

* for FRAME
	DATA	frame_file	  / 'ferret.mgm' /
        DATA    frame_compress    / 'RLE' /  ! matching pframe_hdf_runlen

* other filenames
#ifdef unix
	DATA	journal_file	  / 'ferret.jnl'  /
#else
	DATA	journal_file	  / 'FERRET_JOURNAL'  /
#endif
	DATA	ppllist_file	  / 'ppllist.out' /

* stack pointer for nested IF statements
	DATA	ifstk		 / 0 /
     .		if_conditional	 / .FALSE. /

* for issuing messages about the denigrated status of qualifiers
	DATA	denig_xylim_msg_done / .FALSE. /
	DATA	denig_pen_msg_done / .FALSE. /

* miscellaneous
	DATA	saved_dset		/ unspecified_int4 /
	DATA	saved_abstract_grid	/ unspecified_int4 /
	DATA	xwind_node		/ ' ' /
	DATA	mode_grat_buff		/ ' ' /
        DATA    silent_err              / .FALSE. /  ! will set true for SHOW VAR/XML

	END
