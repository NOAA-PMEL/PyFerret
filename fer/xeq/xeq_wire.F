	SUBROUTINE XEQ_WIRE

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
* execute the WIRE (3D wireframe) command

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* V230  6/8/92 - for new command, WIRE: this routine based on XEQ_CONTOUR
*       10/22/92 - removed unnecessary shade_vars.cmn
*       12/2/92  - added /ZSCALE qualifier
* V301:	  1/7/94 - bug fix: (see DISP_DATA_SET_UP) must unprotect plot data
* 	 1/10/94 - change to DISP_SET_UP call to accomodate CONTOUR/PEN
* V312: 5/94 - array "memory" as a calling argument
* V320:   1/20/94 - changes to SAVE_FRAME call reflecting C-based procedures
* 12/96 *kob* - Linux Port - had to move variable assignments off of
*               declaration line.
* V530: *acm*6/01 - add arguments is_logh, is_logv to DISP_SET_UP call.
*                  (only used for PLOT command)
* V550: *acm*11/02 add argument is_shade to DISP_SET_UP call (not used by wire)
*                  remove VMS includes.
* V552: *acm* 5/03 add argument use_cell to DISP_SET_UP call (not used by wire)
*       *acm* 3/12 Add E and F dimensions (use nferdims in tmap_dims.parm)
*                  A time axis may be a T axis or F axis
* V702 1/17 *sh* for trac enhancement #2369 -- dynamic memory management 
* V72  5/17 acm* Ticket 2530 add WIRE/COLOR= 

	include 'tmap_dims.parm'
	include	'ferret.parm'
	include 'errmsg.parm'
	include 'plot_setup.parm'
	include 'slash.parm'
	include 'xcontext.cmn'
	include 'xvariables.cmn'
	include 'xprog_state.cmn'
	include 'xplot_state.cmn'
	include 'xplot_setup.cmn'

	
* calling argument declarations:

* internal variable declarations:
	LOGICAL	  no_range, overlay, no_labels, transpz, got_vpoint,
     .		  is_logh, is_logv, is_shade, use_cell, set_up, inc_grid
	INTEGER	  status, qp,
     .		  ax1_work, ax2_work, density, ipen
	REAL	  only_val, aspect
	PARAMETER (aspect = 0.0) 		!kob 12/96
	CHARACTER vpoint*24, zlims*24, zscale*24

* internal parameter declarations:
	LOGICAL		norm_dims_only, not_vector
	PARAMETER     ( norm_dims_only  = .FALSE.,
     .			not_vector	= .FALSE.)

* set up the graphics environment
	CALL START_PPLUS

* get the data (possibly ?? on a slightly oversized region)
	CALL DISP_DATA_SET_UP(  ax1_work, ax2_work, status )
	IF ( status .NE. ferr_ok ) GOTO 5100

* initialize
	overlay   = qual_given( slash_wire_over      ).GT.0 .AND. twodee_on
	no_labels = qual_given( slash_wire_nolabels  ).GT.0
	transpz   = qual_given( slash_wire_transpz   ).GT.0
	is_logh   = qual_given( slash_is_logh        ).GT.0 ! Not valid quals
	is_logv   = qual_given( slash_is_logv        ).GT.0 ! for wire fill
        is_shade  = .FALSE.
        use_cell  = .FALSE.

	IF ( .NOT.overlay ) CALL DISP_INIT(no_labels,
     .                                     is_cx, num_uvars_in_cmnd)
* ... default orientation for YT and ZT contours has time on horizontal
	IF ( (plot_axis(2).EQ.t_dim .OR. plot_axis(2).EQ.f_dim) .AND. 
     .		plot_axis(1).NE.x_dim ) transpz = .NOT.transpz

* get wire frame parameters (just get 'em as text strings)
* /VIEW = x,y,z
        vpoint = ' '
        qp = qual_given( slash_wire_view )
        got_vpoint = qp .GT. 0
        IF ( got_vpoint ) THEN
           CALL EQUAL_STRING( cmnd_buff(qual_start(qp):qual_end(qp)),
     .                     vpoint, status )
           IF ( status .NE. ferr_ok ) RETURN
        ENDIF
* /ZLIMITS = lo,hi
        zlims = ' '
        qp = qual_given( slash_wire_zlimits )
        IF ( qp .GT. 0 ) THEN
           CALL EQUAL_STRING( cmnd_buff(qual_start(qp):qual_end(qp)),
     .                     zlims, status )
           IF ( status .NE. ferr_ok ) RETURN
        ENDIF
* /ZSCALE=scale
        zscale = ' '
        qp = qual_given( slash_wire_zscale )
        IF ( qp .GT. 0 ) THEN
           CALL EQUAL_STRING( cmnd_buff(qual_start(qp):qual_end(qp)),
     .                     zscale, status )
           IF ( status .NE. ferr_ok ) RETURN
        ENDIF

* ... /COLOR=color name or pen number
	qp = qual_given(slash_wire_color)
	IF ( qp .GT. 0 ) THEN
           CALL EQUAL_COLOR( cmnd_buff(qual_start(qp):qual_end(qp)),
     .                           ipen, status )
           IF ( status .NE. ferr_ok ) GOTO 5000
	ELSE
	   ipen = -1	! auto-select in DISP_SET_UP
	ENDIF

* prepare PLOT+ environment
	CALL DISP_PREP( status )
	IF ( status .NE. ferr_ok ) GOTO 5000
	
* /set_up only requested ?
	set_up    = qual_given( slash_cont_set_up    ).GT.0

        density = 0	! not used; vec/flow needs it in set up

* set up the axes and title and load the data into PPLUS

	inc_grid = set_up
	CALL DISP_SET_UP   ( 	
     .				not_vector,
     .                          is_shade,
     .                          use_cell,
     .				aspect,
     .				overlay,
     .				transpz,
     .				no_labels,
     .				no_range,
     .				only_val,
     .                          density,
     .				is_logh, 
     .				is_logv,
     .				is_mr,		! pass list
     .				num_uvars_in_cmnd,
     .				is_cx,		! pass list
     .				is_uvar,	
     .				workmem(ax1_work)%ptr,
     .				workmem(ax2_work)%ptr,
     .				ipen,
     .				set_up,
     .				status )
	IF ( status .NE. ferr_ok ) GOTO 5000

* change "window" state in PLOT+
	CALL PPLCMD ( from, line, 0, 'WINDOW OFF', 1, 1 )

* set the view point
        IF ( got_vpoint ) 
     .       CALL PPLCMD ( from, line, 0, 'VPOINT '//vpoint, 1, 1 )

* /set_up only requested ?
	IF ( set_up ) GOTO 1000

* draw it 
	IF ( overlay ) THEN
	   CALL PPLCMD ( ' ', ' ', 0,
     .	                 'VIEW/OVER/' //ppl_wait//' '
     .                                //zscale//',,'//zlims, 1, 1 )
	ELSE
	   CALL PPLCMD ( ' ', ' ', 0,
     .			'VIEW/LATCH/'//ppl_wait//' '
     .                               //zscale//',,'//zlims, 1, 1 )
	ENDIF

* interrupt occurred during plot ?
	IF ( ppl_interrupted ) GOTO 1000

* capture the bit map ?
	IF ( qual_given( slash_wire_frame ) .GT. 0 ) THEN
	   CALL SAVE_FRAME( frame_file, "HDF", status )
	   IF ( status .NE. ferr_ok ) GOTO 5000
	ENDIF

* release the memory used for axis buffers
 1000 	CALL FREE_WS_DYNMEM( ax1_work )
	CALL FREE_WS_DYNMEM( ax2_work )
	CALL UNPROTECT_CMND_DATA			! 1/94
 5100	RETURN

* error exits
 5000	GOTO 1000
	END
