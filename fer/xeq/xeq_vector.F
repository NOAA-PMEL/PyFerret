	SUBROUTINE XEQ_VECTOR( memory )

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
* execute the VECTOR command

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 - 1/7/87
* revision 0.1 - 2/25/87 - changes for INTERPRETATION STACK
* revision 0.2 - 6/24/87 - provisions to permit 90 degree data-plane transpose
*			   and to separate DISP_INIT from DISP_SET_UP
* revision 0.3 - 7/8/87  - DISP_LABELS separated from DISP_SET_UP
* revision 0.4 - 7/28/87 - added CALL DISP_PREP
* revision 0.5 - 8/11/87 - VECKEY relocates according to title position
* revision 0.6 - 8/26/87 - dont transpose on XT plots
* revision 0.7 - 3/2/88  - check for valid data range before loading
* revision 0.8 - 4/14/88 - added /NOLABELS and range check --> DISP_SET_UP
* revision 0.9 - 4/28/88 - added only_x to DISP_SET_UP
* revision 1.0 - 6/22/88 - compute axis aspect correction using PPLUS AXIS.INC
*			   axis lengths
* revision 1.1 - 8/1/88  - corrected bug: x_skip said "Y" and visa versa
*			   added MODE WAIT
* revision 1.2 - 9/21/88 - do_labels --> no_labels: /NOLABEL processing chged
* V200:  5/19/89 - 4D symmetrical
*	  1/4/89 - using keys for labelling
*	 3/13/90 - check status from DISP_PREP
* V301: 12/8/93  - added VECTOR/XSKIP/YSKIP
* 	  1/7/94 - bug fix: (see DISP_DATA_SET_UP) must unprotect plot data
*	 2/10/94 - added /PEN=
* V312: 5/94 - array "memory" as a calling argument
* V320: 12/29/94 - use SPLIT_LIST to redirect tty output for GUI
*	 1/20/94 - changes to SAVE_FRAME call reflecting C-based procedures
* V450:  5/29/97 - added /NOAXES qualifier (eventually for curvilinear plots)
* V510 *sh* 12/99 - allow plots of all-missing data
*		  - replace /PEN qualifier with /COLOR, supporting named colors
*      5/00 *sh* - restored the /PEN qualifier (with "denigrated" remark
*      10/00 *AM* - added VECTOR/FLOWLINE (alias FLOWLINE)
* V530 *sh* 1/01 - change "denigrated" to "deprecated"
*      *acm*6/01 - add arguments is_logh, is_logv to DISP_SET_UP call.
*                  (only used for PLOT command)

#ifdef unix
	include 'tmap_dims.parm'
	include	'ferret.parm'
	include 'errmsg.parm'
	include	'plot_setup.parm'
	include	'slash.parm'
	include 'xcontext.cmn'
	include 'xvariables.cmn'
	include 'xprog_state.cmn'
	include 'xplot_state.cmn'
	include 'xplot_setup.cmn'
	include 'xpplus.cmn'
        include 'axis_inc.decl'       ! with axis lengths	
        include 'AXIS.INC'       ! with axis lengths	
	include 'xrisc.cmn'          ! 12/94 SPLIT_LIST buffer
#else
	INCLUDE 'TMAP_FORMAT:TMAP_DIMS.PARM'
	INCLUDE	'FERRET_CMN:FERRET.PARM'
	INCLUDE 'FERRET_CMN:ERRMSG.PARM'
	INCLUDE	'FERRET_CMN:PLOT_SETUP.PARM'
	INCLUDE	'FERRET_CMN:SLASH.PARM'
	INCLUDE 'FERRET_CMN:XCONTEXT.CMN'
	INCLUDE 'FERRET_CMN:XVARIABLES.CMN'
	INCLUDE 'FERRET_CMN:XPROG_STATE.CMN'
	INCLUDE 'FERRET_CMN:XPLOT_STATE.CMN'
	INCLUDE 'FERRET_CMN:XPLOT_SETUP.CMN'
	INCLUDE 'FERRET_CMN:XPPLUS.CMN'
        INCLUDE 'TMAP_PPLV11INC:AXIS_INC.DECL/LIST'       ! with axis lengths	
        INCLUDE 'PPLV11INC:AXIS.INC/LIST'       ! with axis lengths	
	INCLUDE 'FERRET_CMN:XINTERRUPT.CMN'
#endif

* calling argument declarations:
	REAL	memory( mem_blk_size, max_mem_blks )

* internal variable declarations:
	LOGICAL		overlay, transpz, no_labels, no_x_rng,
     .			set_up, noaxes, no_data, flows, 
     .			is_logh, is_logv
	INTEGER		status, cx, loc, ipen, thick,
     .			dim1, dim2, asp,
     .			ax1_blks, ax1_start,
     .			ax2_blks, ax2_start,
     .			x_skip, y_skip,
     .                  density
	REAL		only_x, aspect, vbuf


* internal parameter declarations:
	LOGICAL		norm_dims_only, vector
	PARAMETER     ( norm_dims_only  = .FALSE.,
     .			vector		= .TRUE.  )
	INTEGER ncolors
	PARAMETER (ncolors = 6)


* set up the graphics environment and get the data
	CALL START_PPLUS
	CALL DISP_DATA_SET_UP( memory,
     .		ax1_start, ax1_blks, ax2_start, ax2_blks, status )
	IF ( status .NE. ferr_ok ) GOTO 5100

* make sure the number of expressions given is even ( x and y components )
	IF ( MOD( num_uvars_in_cmnd, 2 ) .NE. 0 ) CALL ERRMSG
     .		(ferr_invalid_command,status,
     .		 'VECTOR requires paired components',*5000)

* initialize
	overlay   = qual_given( slash_vect_over_plot ).GT.0 .AND. twodee_on
	no_labels = qual_given( slash_vect_nolabels  ).GT.0
	transpz   = qual_given( slash_vect_transpz   ).GT.0
	set_up	  = qual_given( slash_shad_set_up    ).GT.0
	noaxes    = qual_given( slash_noaxes         ).GT.0
	flows     = qual_given( slash_vect_flowline  ).GT.0
	is_logh   = qual_given( slash_is_logh        ).GT.0  ! Not valid quals
	is_logv   = qual_given( slash_is_logv        ).GT.0  ! for vector

	IF ( .NOT.overlay ) CALL DISP_INIT(no_labels,is_cx,num_uvars_in_cmnd)
* ... default orientation for YT and ZT contours has time on horizontal
	IF ( plot_axis(2).EQ.t_dim .AND. plot_axis(1).NE.x_dim )
     .							transpz = .NOT.transpz
	CALL DISP_PREP( status )
	IF ( status .NE. ferr_ok ) GOTO 5000

* possibly transpose the axes
	IF ( transpz ) THEN
	   dim2 = plot_axis(1)
	   dim1 = plot_axis(2)
	ELSE
	   dim1	= plot_axis(1)
	   dim2	= plot_axis(2)
	ENDIF

* scale-ratio correction requested ?
	IF ( qual_given( slash_vect_aspect ) .GT. 0 ) THEN
	   asp = qual_given( slash_vect_aspect )
	   CALL EQUAL_VAL( cmnd_buff(qual_start(asp):qual_end(asp)),
     .			   aspect, status )
	   IF ( status .NE. ferr_ok ) RETURN
	   IF ( aspect .EQ. unspecified_val4 ) aspect = ylen/xlen
	   IF ( aspect .LE. 0.0 ) CALL ERRMSG( ferr_out_of_range,
     .		status, cmnd_buff(qual_start(asp):qual_end(asp)), *5000 )
	ELSE
	   aspect = 0.0
	ENDIF

* user-specified pen color ?  (2/94)
	loc = MAX( qual_given(slash_pen), qual_given(slash_color))
	IF ( loc .GT. 0 ) THEN
	   IF (.NOT.denig_pen_msg_done .AND. qual_given(slash_pen).GT.0)
     .									THEN
	      CALL WARN( '/PEN is deprecated. Use /COLOR.')
	      denig_pen_msg_done = .TRUE.
	   ENDIF
	   CALL EQUAL_COLOR( cmnd_buff(qual_start(loc):qual_end(loc)),
     .                           ipen, status )
           IF ( status .NE. ferr_ok ) GOTO 5000
	ELSE
	   ipen = -1	! auto-select in DISP_SET_UP
	ENDIF
* ... VECTOR/THICKNESS=n
	loc = qual_given( slash_vect_thick )
	IF ( loc .GT. 0  ) THEN
	   IF ( ipen .EQ. -1 ) ipen = 1  ! /THICK alone means "BLACK"
	   ipen = MOD(ipen-1,ncolors)+1  ! eliminate thickness
	   CALL EQUAL_VAL( cmnd_buff(qual_start(loc):qual_end(loc)),
     .			   vbuf, status )	! syntax check
	   IF ( status .NE. ferr_ok ) GOTO 5000
	   IF ( vbuf .NE. unspecified_val4 ) THEN
	      IF ( vbuf.LE.0.0 .OR. vbuf.GT.3 ) GOTO 5400
	      thick = NINT( vbuf )	! user-specified line style
	   ELSE
	      thick = 2  ! default if no value is specified
	   ENDIF
	   ipen = ncolors*(thick-1) + ipen
	ENDIF

* set up title and load the x component data into PPLUS
	CALL DISP_SET_UP   ( 	memory,
     .				vector,
     .				aspect,
     .				overlay,
     .				transpz,
     .				no_labels,
     .				no_x_rng,
     .				only_x,
     .				is_logh, 
     .				is_logv,
     .				is_mr,
     .				num_uvars_in_cmnd,
     .				is_cx,
     .				is_uvar,
     .				memory( 1, ax1_start ),
     .				memory( 1, ax2_start ),
     .				ipen,
     .				status )
	IF ( status .NE. ferr_ok ) GOTO 5000

* is there no valid input data?
        no_data =  no_x_rng .AND. only_x.EQ.bad_val4
        IF (no_data) CALL PUT_LABEL( ppl_movlab, 0.5, 0.5,
     .                  ppl_centered, letsize_title,
     .                  'No Valid Data',
     .                  lab_loc_axlen_times, lab_loc_axlen_times )

* increment counter of vector fields on the screen
	nvect_on = nvect_on + 1

* special set-up required only for the VECTOR command
	IF ( .NOT.overlay .AND. .NOT.no_data ) THEN
* ... vector length key below plot title (overlay written as key)
	   WRITE ( ppl_buff, '(2F6.1)' ) 3.5, -1.2
	   CALL PPLCMD (' ',' ',0,'VECKEY/NOUSER '//ppl_buff, 1, 1 )
	ENDIF	

* turn off axes? (5/29/97)
	IF ( noaxes ) CALL PPL_AXES_OFF(set_up)

* done now if /set_up, only
	IF ( set_up) GOTO 1000

* sub-sample the vectors to make a cleaner plot ?
* *ACM 10/00*  do not subsample if its a flowline plot

        IF (.NOT. flows) THEN
	   cx   = is_cx( 1 )
	   loc = qual_given( slash_vect_xskip )
	   IF ( loc .GT. 0 ) THEN
	      CALL EQUAL_VAL( cmnd_buff(qual_start(loc):qual_end(loc)),
     .			   vbuf, status )
	      IF ( status .NE. ferr_ok ) RETURN
	      IF ( vbuf .EQ. unspecified_val4 ) THEN
	         x_skip = 1.
	      ELSE
	         x_skip = ABS(vbuf)
	      ENDIF
	   ELSE
	      x_skip = ( cx_hi_ss(cx,dim1)-cx_lo_ss(cx,dim1) )
     .		 / max_x_vec + 1
	      IF ( x_skip .NE. 1 ) THEN
	         risc_buff = ' '	         
	         WRITE ( risc_buff, 3000 ) x_skip, 'X'
	         CALL SPLIT_LIST(pttmode_explct, ttout_lun, risc_buff, 0)
	      ENDIF
	   ENDIF
	   loc = qual_given( slash_vect_yskip )
	   IF ( loc .GT. 0 ) THEN
	      CALL EQUAL_VAL( cmnd_buff(qual_start(loc):qual_end(loc)),
     .			   vbuf, status )
	      IF ( status .NE. ferr_ok ) RETURN
	      IF ( vbuf .EQ. unspecified_val4 ) THEN
	         y_skip = 1.
	      ELSE
	         y_skip = ABS(vbuf)
	      ENDIF
	   ELSE
	      y_skip = ( cx_hi_ss(cx,dim2)-cx_lo_ss(cx,dim2) )
     .	   	    / max_y_vec + 1
	      IF ( y_skip .NE. 1 ) THEN 
	         risc_buff = ' '	         
	         WRITE ( risc_buff, 3000 ) y_skip, 'Y'
	         CALL SPLIT_LIST(pttmode_explct, ttout_lun, risc_buff, 0)
	      ENDIF
	   ENDIF
 3000	   FORMAT (' Using every ',I3,'th vector in the ',
     .            A1,' direction')
	   WRITE ( ppl_buff, '(2I4)' ) x_skip, y_skip

* plot it with vector arrows
	   IF (  overlay ) THEN
	      CALL PPLCMD (' ',' ',0,
     .		'VECTOR/OVER/' //ppl_wait//' '//ppl_buff, 1, 1 )
	   ELSE
	      CALL PPLCMD (' ',' ',0,
     .		'VECTOR/LATCH/'//ppl_wait//' '//ppl_buff, 1, 1 )
	   ENDIF

        ELSE  

* plot it with flowlines
	   loc = qual_given( slash_vect_density )
	   IF ( loc .GT. 0 ) THEN
	      CALL EQUAL_VAL( cmnd_buff(qual_start(loc):qual_end(loc)),
     .			   vbuf, status )
	      IF ( status .NE. ferr_ok ) RETURN
	      IF ( vbuf .EQ. unspecified_val4 ) THEN
	         density = 1.
	      ELSE
	         density = ABS(vbuf)
	      ENDIF
           ELSE
              density = 5
           ENDIF

           CALL flow_set_up (is_mr, is_cx, transpz)

* special set-up required only for the VECTOR/FLOW command
	IF ( .NOT.overlay .AND. .NOT.no_data ) THEN
* ... vector length key below plot title (overlay written as key)
	   WRITE ( ppl_buff, '(2F6.1)' ) 5.5, -1.2
	   CALL PPLCMD (' ',' ',0,'VECKEY/NOUSER '//ppl_buff, 1, 1 )
	ENDIF	

           WRITE (ppl_buff, '(I4)') density
	   IF (  overlay ) THEN
	      CALL PPLCMD (' ',' ',0,
     .		'FLOWLINE/OVER/' //ppl_wait//' '//ppl_buff, 1, 1 )

	   ELSE
	      CALL PPLCMD (' ',' ',0,
     .		'FLOWLINE/LATCH/'//ppl_wait//' '//ppl_buff, 1, 1 )
	   ENDIF
        ENDIF

* restore axes? (5/29/97)
	IF ( noaxes ) CALL PPL_AXES_RESTORE

* interrupt occurred during plot ?
	IF ( ppl_interrupted ) GOTO 1000

* capture the bit map ?
	IF ( qual_given( slash_vect_frame ) .GT. 0 ) THEN
	   CALL SAVE_FRAME( frame_file, "HDF", status )
	   IF ( status .NE. ferr_ok ) GOTO 5000
	ENDIF

* release the memory used for axis buffers
 1000 	CALL FREE_MEMORY( ax2_start, ax2_blks )
 	CALL FREE_MEMORY( ax1_start, ax2_blks )
	CALL UNPROTECT_CMND_DATA			! 1/94
 5100	RETURN

* error exits
 5000	GOTO 1000
 5400	CALL ERRMSG( ferr_out_of_range, status, 
     .		cmnd_buff(qual_start(loc):qual_end(loc))//pCR//
     .		'Legal range is 1 to 3', *5000 )
	END
