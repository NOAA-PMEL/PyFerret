 	SUBROUTINE XEQ_STREAMLN (memory)

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
* execute the STREAMLN command
* 9/00  *ACM* based on xeq_shade, xeq_vector

#ifdef unix
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
#else
	INCLUDE 'TMAP_FORMAT:TMAP_DIMS.PARM'
	INCLUDE	'FERRET_CMN:FERRET.PARM'
	INCLUDE 'FERRET_CMN:ERRMSG.PARM'
	INCLUDE 'FERRET_CMN:PLOT_SETUP.PARM'
	INCLUDE 'FERRET_CMN:SLASH.PARM'
	INCLUDE 'FERRET_CMN:XCONTEXT.CMN'
	INCLUDE 'FERRET_CMN:XVARIABLES.CMN'
	INCLUDE 'FERRET_CMN:XPROG_STATE.CMN'
	INCLUDE 'FERRET_CMN:XPLOT_STATE.CMN'
	INCLUDE 'FERRET_CMN:XPLOT_SETUP.CMN'
#endif

* calling argument declarations:
	REAL	memory( mem_blk_size, max_mem_blks )

* internal variable declarations:
	LOGICAL	  no_range, overlay, transpz, no_labels, 
     .            set_up, noaxes, no_data
	INTEGER	  status, loc, sp, ipen, thick,
     .		  ax1_blks, ax1_start,
     .		  ax2_blks, ax2_start
	REAL	  only_val, aspect, vbuf
	PARAMETER ( aspect = 0.0)			!kob 12/96

* internal parameter declarations:
	LOGICAL		norm_dims_only, vector
	PARAMETER     ( norm_dims_only  = .FALSE.,
     .			vector	= .TRUE. )
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
     .		 'STREAMLN requires paired components',*5000)

* initialize
	overlay   = qual_given( slash_strm_over_plot ).GT.0 .AND. twodee_on
	no_labels = qual_given( slash_strm_nolabels  ).GT.0
	transpz   = qual_given( slash_strm_transpz   ).GT.0
        sp        = qual_given( slash_cont_spectrum  )
	set_up	  = qual_given( slash_strm_set_up    ).GT.0
	noaxes    = qual_given( slash_noaxes         ).GT.0
	
	IF ( .NOT.overlay ) CALL DISP_INIT(no_labels,is_cx,num_uvars_in_cmnd)
* ... default orientation for YT and ZT contours has time on horizontal
	IF ( plot_axis(2).EQ.t_dim .AND. plot_axis(1).NE.x_dim )
     .							transpz = .NOT.transpz
	CALL DISP_PREP( status )
	IF ( status .NE. ferr_ok ) GOTO 5000

C ACM from xeq_vector:
* user-specified pen color ?  (2/94)
	loc = MAX( qual_given(slash_pen), qual_given(slash_color))
	IF ( loc .GT. 0 ) THEN
	   IF (.NOT.denig_pen_msg_done .AND. qual_given(slash_pen).GT.0)
     .									THEN
	      CALL WARN( '/PEN is denigrated. Use /COLOR.')
	      denig_pen_msg_done = .TRUE.
	   ENDIF
	   CALL EQUAL_COLOR( cmnd_buff(qual_start(loc):qual_end(loc)),
     .                           ipen, status )
           IF ( status .NE. ferr_ok ) GOTO 5000
	ELSE
	   ipen = -1	! auto-select in DISP_SET_UP
	ENDIF
* ... STREAMLN/THICKNESS=n
	loc = qual_given( slash_strm_thick )
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


* set up the axes and title and load the data into PPLUS
	CALL DISP_SET_UP   (    memory,
     .				vector,
     .				aspect,
     .				overlay,
     .				transpz,
     .				no_labels,
     .				no_range,
     .				only_val,
     .				is_mr,
     .				num_uvars_in_cmnd,
     .				is_cx,
     .				is_uvar,
     .				memory( 1, ax1_start ),
     .				memory( 1, ax2_start ),
     .				-1, 			! auto pen select
     .				status )
	   IF ( status .NE. ferr_ok ) GOTO 5000

* is there no valid input data?
           no_data =  no_range .AND. only_val.EQ.bad_val4

* PPLUS blows up setting auto contour levels if all data have same value
c ACM -- what if all same data?
c	   IF ( no_range .AND. .NOT.no_data 
c     .          .AND. qual_given( slash_strm_lev_same ) .EQ. 0 )
c     .		CALL PPLCMD ( from, line, 0, 
c     .		'LEV () ('//TM_FMT(only_val, 3, 12, slen)//')' , 1, 1 )

c ACM  remove commands re KEY 

c ACM  do we need this?
* protect colors already on-screen before a color overlay
           IF ( overlay ) CALL PPL_SHASET( 'PROTECT' )

c ACM  remove commands re color specrum, pattern from xeq_shade

* turn off axes? (5/29/97)
	IF ( noaxes ) CALL PPL_AXES_OFF(set_up)

* /set_up only requested ?
	   IF ( set_up ) GOTO 1000

* draw it 
	   IF ( overlay ) THEN
	      CALL PPLCMD ( ' ', ' ', 0,
     .            'STREAMLN/OVER/' //ppl_wait//' ' , 1, 1 )
	   ELSE
	      CALL PPLCMD ( ' ', ' ', 0,
     .            'STREAMLN/LATCH/'//ppl_wait//' ', 1, 1 )
	   ENDIF

* restore axes? (5/29/97)
	IF ( noaxes ) CALL PPL_AXES_RESTORE

* interrupt occurred during plot ?
	IF ( ppl_interrupted ) GOTO 1000

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
