	SUBROUTINE XEQ_CONTOUR( memory )

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
* execute the CONTOUR command

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 - 1/5/87 copied from rev. 0.0 of XEQ_PLOT (6/3/86) with changes
*		 to allow multiple mrs in a single expression
*		 and with separate subroutines for graphics set-up
* revision 0.1 - 2/25/87 - changes for INTERPRETATION STACK
* revision 0.2 - 6/24/87 - incorporated 90 degree in-plane transpose
*			   and DISP_INIT separated from DISP_SET_UP
* revision 0.3 - 7/8/87  - DISP_LABELS separated from DISP_SET_UP
* revision 0.4 - 7/28/87 - added CALL DISP_PREP
* revision 0.5 - ????    - plots not automatically transposed in XT plane
* revision 0.6 - 3/2/88  - check for valid data range before PLOT+ auto-scaling
* revision 0.7 - 4/14/88 - added /NOLABELS option and range check-->DISP_SET_UP
* revision 0.8 - 4/28/88 - set up PPLUS LEV command if no data range
* revision 0.9 - 8/2/88  - added MODE WAIT
* revision 1.0 - 9/21/88 - do_labels --> no_labels: /NOLABEL processing chged
* revision 1.1 - 1/25/89 - PPLUS interruptibility - chg for multiple plots
* revision 1.2 - 5/1/89  - PPL LEVELS not cleared for no_range if /LEVELS_SAME
* V200:  5/19/89 - 4D symmetrical
*		 - obtain oversized data region to fill contours to edges
*	  1/4/89 - using keys for labelling
*	 3/13/90 - check status from DISP_PREP
* V230:   6/8/92 - added CONTOUR/FILL/KEY/LINE/NOKEY
*       10/22/92 - removed dependence on shade_vars.cmn using PPLCMND(SHAKEY)
*        11/9/92 - protect colors before doing a color overlay
*                - default for KEY is off on overlays
* V300:  3/11/93 - added /SPECTRUM=
* V301:	  1/7/94 - bug fix: (see DISP_DATA_SET_UP) must unprotect plot data
*	 2/10/94 - added /PEN=
* V312: 5/94 - array "memory" as a calling argument
* V320:   1/20/94 - changes to SAVE_FRAME call reflecting C-based procedures
* 12/96 *kob* - Linux Port - had to move variable assignments off of
*               declaration line.
* V450:  5/29/97 - added /NOAXES qualifier (useful in curvilinear plots)
* V491:  12.98   - added /PATTERN qualifier *jd*
* V510 12/99 *sh* - allow plots of all-missing data
*		  - added /SIZE=/SPACING=/SIGDIG= controls over contour labels
*      5/00 *sh* - restored the /PEN qualifier (with "denigrated" remark
* V530 *sh* 1/01 - change "denigrated" to "deprecated"
*      *acm*6/01 - add arguments is_logh, is_logv to DISP_SET_UP call.
*                  (only used for PLOT command)
*      *acm* 7/01 - Add /AXES=TOP,BOT,LEFT,RIGHT
*		    & call ax_on_off to implement /NOAXES.
*      *acm* 10/01  remove VMS includes
* V540 *acm*  2/02  restore persistence of PPL AXSET settings
* V541 *acm*  3/02  Make PPL AXSET settings persist, but not settings
*                   done with qualifiers /AXES= or /NOAXES
*      *acm*  4/02  do not set is_logh, is_logv.  They arent used by CONTOUR.
*      *acm*  4/02  If this is an overlay, and there is no valid data, 
*                   do not put "No Valid Data" across the middle of the plot.
*                   Instead (in disp_set_up) add this information to the
*                   variable name in the key below the plot.
* v541 *acm*  5/02 add option to control outline on the shade key boxes 
*                   /KEY=continuous
* V550: *acm*11/02 add argument is_shade to DISP_SET_UP call (not used by contour)
* V552: *acm* 5/03 add argument use_cell to DISP_SET_UP call (not used by contour)


	include 'tmap_dims.parm'
	include	'ferret.parm'
	include 'errmsg.parm'
	include 'slash.parm'
	include 'xplot_state.cmn'
	include 'xcontext.cmn'
	include 'xvariables.cmn'
	include 'xprog_state.cmn'
	include 'plot_setup.parm'
	include 'xplot_setup.cmn'
	
* calling argument declarations:
	REAL	memory( mem_blk_size, max_mem_blks )

* internal variable declarations:
	LOGICAL	  no_range, overlay, transpz, no_labels, fill, usa_line,
     .            do_key, spectrum, set_up, noaxes, axes, pattern, 
     .		  no_data, is_logh, is_logv
	INTEGER	  TM_LENSTR1, status, slen, sp, ipen,
     .		  ax1_blks, ax1_start, loc, sigdig,
     .		  ax2_blks, ax2_start, patt, i, nparm, num_it, 
     .		  s1, s2, it_start(4), it_end(4), iax(4), density
	REAL	  vbuf, only_val, aspect, lab_size, lab_spacing
	PARAMETER (aspect = 0.0)			!kob 12/96
	CHARACTER TM_FMT*12, fname*128, buff*25

* internal parameter declarations:
	LOGICAL		norm_dims_only, not_vector, cont_key, 
     .                  is_shade, use_cell
	PARAMETER     ( norm_dims_only  = .FALSE.,
     .			not_vector	= .FALSE. )

* set up the graphics environment

	CALL START_PPLUS

* get the data (possibly ?? on a slightly oversized region)
!	cx_ovsiz(cx_last) = ovsiz_plane	! ... causes too many headaches !
	CALL DISP_DATA_SET_UP( memory,
     .		ax1_start, ax1_blks, ax2_start, ax2_blks, status )
	IF ( status .NE. ferr_ok ) GOTO 5100

* initialize
	overlay   = qual_given( slash_cont_over_plot ).GT.0 .AND. twodee_on
	no_labels = qual_given( slash_cont_nolabels  ).GT.0
	transpz   = qual_given( slash_cont_transpz   ).GT.0
	set_up    = qual_given( slash_cont_set_up    ).GT.0
	fill      = qual_given( slash_cont_fill      ).GT.0
	noaxes    = qual_given( slash_noaxes         ).GT.0
	axes      = qual_given( slash_cont_axes      ).GT.0
        sp        = qual_given( slash_cont_spectrum  )
        spectrum  = sp .GT. 0 .AND. fill
        is_shade  = .FALSE.
        use_cell  = .FALSE.

        patt     = qual_given( slash_cont_pattern  )
        pattern  = patt .GT. 0 .AND. fill

	usa_line  = .NOT.fill
     .          .OR. qual_given( slash_cont_line      ).GT.0
	IF ( .NOT.overlay ) CALL DISP_INIT(no_labels,is_cx,
     .                                       num_uvars_in_cmnd)
* ... default orientation for YT and ZT contours has time on horizontal
	IF ( plot_axis(2).EQ.t_dim .AND. plot_axis(1).NE.x_dim )
     .							transpz = .NOT.transpz

* ... CONTOUR/SIZE=float
	loc = qual_given( slash_cont_size )
	IF ( loc .GT. 0  ) THEN
	   CALL EQUAL_VAL( cmnd_buff(qual_start(loc):qual_end(loc)),
     .			      vbuf, status )
	   IF ( status .NE. ferr_ok ) GOTO 5000
	   IF ( vbuf .NE. unspecified_val4 ) THEN
	      IF ( vbuf.LE. 0.0 ) GOTO 5210
	      lab_size = vbuf	
	   ENDIF
	ELSE
	   lab_size = 0.08
	ENDIF
* ... CONTOUR/SPACING=float
	loc = qual_given( slash_cont_spacing )
	IF ( loc .GT. 0  ) THEN
	   CALL EQUAL_VAL( cmnd_buff(qual_start(loc):qual_end(loc)),
     .			      vbuf, status )
	   IF ( status .NE. ferr_ok ) GOTO 5000
	   IF ( vbuf .NE. unspecified_val4 ) THEN
	      IF ( vbuf .LE. 0.0 ) GOTO 5210
	      lab_spacing = vbuf	
	   ENDIF
	ELSE
	   lab_spacing = 5.0
	ENDIF
* ... CONTOUR/SIGDIG=integer
	loc = qual_given( slash_cont_sigdig )
	IF ( loc .GT. 0  ) THEN
	   CALL EQUAL_VAL( cmnd_buff(qual_start(loc):qual_end(loc)),
     .			      vbuf, status )
	   IF ( status .NE. ferr_ok ) GOTO 5000
	   IF ( vbuf .NE. unspecified_val4 ) THEN
	      IF ( vbuf.LE.0.5 ) GOTO 5210
	      sigdig = NINT(vbuf)
	   ENDIF
	ELSE
	   sigdig = 2
	ENDIF
* ... CONTOUR/COLOR=color name or pen number
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

	CALL DISP_PREP( status )
	IF ( status .NE. ferr_ok ) GOTO 5000

        density = 0	! not used; vec/flow needs it in set up

* set up the axes and title and load the data into PPLUS
	CALL DISP_SET_UP   ( 	memory,
     . 			 	not_vector,
     .				is_shade,
     .				use_cell,
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
     .				memory( 1, ax1_start ),
     .				memory( 1, ax2_start ),
     .				ipen,
     .				status )
	IF ( status .NE. ferr_ok ) GOTO 5000

* increment counter of line-style contours on screen
	IF ( usa_line ) ncont_on = ncont_on + 1

* is the no valid input data?
        no_data =  no_range .AND. only_val.EQ.bad_val4

* set contour label details (3/00)
	WRITE ( ppl_buff, 3100 ) lab_size, sigdig, lab_spacing
	CALL PPLCMD ( from, line, 0, ppl_buff, 1, 1 )
 3100	FORMAT('CONSET ',F8.3,',',I4,', , , , , ,',F7.2)

* PPLUS blows up setting auto contour levels if all data have same value
	IF ( no_range  .AND. .NOT.no_data 
     .	     .AND. qual_given( slash_cont_lev_same ) .EQ. 0 )
     .		CALL PPLCMD ( from, line, 0, 
     .		'LEV () ('//TM_FMT(only_val, 3, 12, slen)//')' , 1, 1 )

* default key
	   do_key =  (qual_given(slash_cont_key)  .GT. 0)
     .		.OR. (.NOT.usa_line .AND. .NOT.overlay)
	   IF ( no_data ) do_key = .FALSE.
	   IF (qual_given(slash_cont_nokey) .GT. 0) do_key = .FALSE.
           IF ( do_key ) THEN
              CALL PPLCMD ( from, line, 0, 'SHAKEY 1,1', 1, 1 )

* ... /KEY=continouus ?
              loc = qual_given( slash_cont_key )
              IF ( loc .GT. 0) THEN
	         CALL EQUAL_STR_LC( cmnd_buff(qual_start(loc):qual_end(loc)),
     .                        buff, status )
                 IF ( status .NE. ferr_ok ) GOTO 5000
	         IF ( buff(1:3) .EQ. 'con' ) THEN
                    cont_key = .TRUE.
                    CALL PPLCMD (from, line, 0, 'S_KEY 1', 1, 1 )
                 ENDIF
              ENDIF

           ELSE
              CALL PPLCMD ( from, line, 0, 'SHAKEY 0,1', 1, 1 )

c when it's an overlay, 'No Valid Data' is part of the plot key,
c done in disp_set_up.

        IF (.NOT. overlay .AND. no_data) 
     .                  CALL PUT_LABEL( ppl_movlab, 0.5, 0.5,
     .                  ppl_centered, letsize_title,
     .                  'No Valid Data',
     .                  lab_loc_axlen_times, lab_loc_axlen_times )
           ENDIF

* protect colors already on-screen before a color overlay
           IF ( overlay .AND. fill ) CALL PPL_SHASET( 'PROTECT' )

* save old and reset to new color spectrum if requested
           IF ( spectrum ) THEN
              CALL EQUAL_STR_LC( cmnd_buff(qual_start(sp):qual_end(sp)),
     .             fname, status )
              IF ( status .NE. ferr_ok ) RETURN
              IF ( fname .EQ. ' ' ) fname = 'default'
              CALL PPL_SHASET( 'SAVE' )
              CALL PPL_SHASET( 'SPECTRUM='//fname(1:TM_LENSTR1(fname)) )
           ENDIF

* save old and set to new pattern if requested (a la spectrum)
           IF ( pattern ) THEN
              CALL EQUAL_STR_LC( 
     .             cmnd_buff(qual_start(patt):qual_end(patt)),
     .             fname, status )
              IF ( status .NE. ferr_ok ) RETURN
              IF ( fname .EQ. ' ' ) fname = 'default'
              CALL PPL_PATSET( 'SAVE' )
              CALL PPL_PATSET( 'PATTERN='//fname(1:TM_LENSTR1(fname)) )
           ENDIF


* turn off axes? 
	IF ( noaxes ) THEN 
           DO 300 i = 1, 4
              iax(i) = 0
 300       ENDDO
        ENDIF

* turn on (one or more) axes? (7/11/01)

* ... PLOT/AXES=(TOP,BOTTOM,LEFT,RIGHT)
	loc = qual_given( slash_cont_axes )
	IF ( loc .GT. 0) THEN
	   CALL EQUAL_STR_LC( cmnd_buff(qual_start(loc):qual_end(loc)),
     .                        buff, status )
           IF ( status .NE. ferr_ok ) GOTO 5000

	   IF ( buff .NE. ' ' ) THEN

* Get arguments, if any.  Otherwise turn on all axes
              slen = TM_LENSTR1(buff)
              s1 = 1
              s2 = slen
              IF (buff(1:1) .EQ. '(' )THEN
                  s1 = 2
                  s2 = slen-1
                  IF ( buff(slen:slen) .NE. ')' ) GOTO 5440
              ENDIF

              nparm = 4
              CALL parse_comma_list ( buff, s1, s2, nparm, 
     .             num_it, it_start, it_end, status)

              IF (status .EQ. ferr_ok .AND. it_start(1) .GT. it_end(1)) THEN	! /AXES=()
                 DO 400 i = 1, 4
                    iax(i) = 0
 400             ENDDO
              ELSE IF (status .NE. ferr_ok .OR. num_it .LT. nparm) THEN
                 GOTO 5440
              ELSE
                 DO 500 i = 1, nparm
                    READ (buff(it_start(i):it_end(i)), *) iax(i)
                    IF (iax(i) .LT. 0  .OR.  iax(i) .GT. 1) GOTO 5450
 500             CONTINUE
              ENDIF
           ELSE		! turn on all axes
	      DO 600 i = 1, 4
		 iax(i) = 1
 600	      CONTINUE
           ENDIF
              
	ENDIF	

*  Turn off or on axes, including time axes.
        IF (axes .OR. noaxes) CALL ax_on_off (iax)

* /set_up only requested ?
	IF ( set_up ) GOTO 1000

* contour it 
        IF ( fill ) THEN
* ... filled contours
	   IF ( overlay ) THEN
	      IF ( usa_line ) THEN
		 CALL PPLCMD ( ' ', ' ', 0,
     .			'FILL/OVER/' //ppl_wait//' ' , 1, 1 )
	         IF ( ppl_interrupted ) GOTO 1000
		 CALL PPLCMD ( ' ', ' ', 0,
     .			'CONTOUR/OVER/' //ppl_wait//' ' , 1, 1 )
	      ELSE
	         CALL PPLCMD ( ' ', ' ', 0,
     .			'FILL/OVER/' //ppl_wait//' ' , 1, 1 )
	      ENDIF
	   ELSE
	      IF ( usa_line ) THEN
		 CALL PPLCMD ( ' ', ' ', 0,
     .			'FILL/LATCH/'//ppl_wait//' ', 1, 1 )
	         IF ( ppl_interrupted ) GOTO 1000
		 CALL PPLCMD ( ' ', ' ', 0,
     .			'CONTOUR/OVER/' //ppl_wait//' ' , 1, 1 )
	      ELSE
		 CALL PPLCMD ( ' ', ' ', 0,
     .			'FILL/LATCH/'//ppl_wait//' ', 1, 1 )
	      ENDIF
	   ENDIF
        ELSE
* ... line-style contour
	   IF ( overlay ) THEN
	      CALL PPLCMD ( ' ', ' ', 0,
     .			'CONTOUR/OVER/' //ppl_wait//' ' , 1, 1 )
	   ELSE
	      CALL PPLCMD ( ' ', ' ', 0,
     .			'CONTOUR/LATCH/'//ppl_wait//' ', 1, 1 )
	   ENDIF
        ENDIF

* protect the new colors and restore the original color spectrum
           IF ( spectrum ) THEN
              CALL PPL_SHASET( 'PROTECT' )
              CALL PPL_SHASET( 'SPECTRUM' )
           ENDIF

* restore the original pattern
           IF ( pattern ) THEN
              CALL PPL_PATSET( 'PATTERN' )
           ENDIF

* restore axes? (5/29/97  revised 3/2002 

        IF ( axes .or. noaxes ) CALL PPL_AXES_RESTORE

* restore default (non-continuous) key style?
        IF (cont_key ) CALL PPLCMD (from, line, 0, 'S_KEY 0', 1, 1 )

* interrupt occurred during plot ?
	IF ( ppl_interrupted ) GOTO 1000

* capture the bit map ?
	IF ( qual_given( slash_cont_frame ) .GT. 0 ) THEN
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
! 5200   CALL ERRMSG( ferr_out_of_range, status,
!     .          cmnd_buff(qual_start(loc):qual_end(loc))//pCR//
!     .          'Legal range is 1 to '//LEFINT(limit,slen), *1000 )
 5210	CALL ERRMSG( ferr_out_of_range, status, 
     .		cmnd_buff(qual_start(loc):qual_end(loc))//pCR//
     .		'Value must be positive', *1000 )
 5440	CALL ERRMSG( ferr_syntax, status, 
     .		cmnd_buff(qual_start(loc):qual_end(loc))//pCR//
     .	'AXES[=(TOP,BOTTOM,LEFT,RIGHT)] takes 4 arguments or no argument', *1000 )
 5450	CALL ERRMSG( ferr_out_of_range, status, 
     .		cmnd_buff(qual_start(loc):qual_end(loc))//pCR//
     .		'AXES arguments must be 0 or 1', *1000 )

	END
