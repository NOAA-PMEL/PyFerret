	SUBROUTINE XEQ_PLOT( memory )

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
* execute the plot command

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 - 7/8/87  - based on XEQ_CONTOUR rev. 0.3
* revision 0.1 - 7/28/87 - added CALL DISP_PREP
* revision 0.2 - 3/2/88  - added safety check for data range prior to plotting
* revision 0.3 - 4/12/88 - added /NOLABELS
* revision 0.4 - 4/28/88 - moved rev. 0.2 check to PLOT_SET_UP
* revision 0.5 - 8/2/88  - added mode wait
* revision 1.0 - 9/21/88 - do_labels --> no_labels: /NOLABEL processing chged
* V200:   1/8/89 - using keys - major rewrite
*	 3/13/90 - check status from DISP_PREP
*	  5/8/90 - allow PLOT/VS to overlay 2D fields
* V230:  7/23/92 - when /LINE_STYLE is given /SYMBOLS must default to OFF
* V301:	  1/7/94 - bug fix: (see DISP_DATA_SET_UP) must unprotect plot data
*        1/27/94 - raise limit on /LINE=# from num_plain (6) to num_fancy_lines
* V312: 5/94 - array "memory" as a calling argument
* V320:   1/20/94 - changes to SAVE_FRAME call reflecting C-based procedures
*	  1/27/95 - default style for /VS is now /SYMBOLS_ONLY
* V450:  5/29/97 - added /NOAXES qualifier
* V500   4.27.99 - *jd* NLINES command added to clean up after/before POLYGON
* V510 *sh* 3/00 - added PLOT/COLOR=/SIZE= qualifiers
* V521 6/00 *sh* - bug fix: PLOT/LINE/SYM should auto-select colors
* V530 *sh* 1/01 - added PLOT/STEP
* V530 *acm* 5/01 - added PLOT/DASH
*	            and PLOT/SYMBOL=dot
* V533 *acm* 6/01 - add arguments is_logh, is_logv to PLOT_SET_UP call, 
*                  set log axes, inverse log axes and log depth axes with
*                  /VLOG or /HLOG
*      *acm* 7/01 - Add /AXES=TOP,BOT,LEFT,RIGHT
*		    & call ax_on_off to implement /NOAXES.
*      *acm* 10/01  remove VMS includes
* V540 *acm*  2/02  restore persistence of PPL AXSET settings
* V541 *acm*  3/02  Make PPL AXSET settings persist, but not settings
*                   done with qualifiers /AXES= or /NOAXES
* V541 *acm*  3/02  Allow PLOT/VS/LINE/DASH
* V541 *ACM*  8/02  Remove the call with dot_sym=99 for PLOT/SYM=DOT-- need
*                      to resolve PPLUS issues with polymark first.
* V550 *acm* 12/02  Restore dot_sym = 99 and warnings that /THICK and /SIZE 
!                    do not apply to DOT
* v554 *acm* 11/03  Bug fix: plot/sym/size=  didnt change the size; the size
*                   was always being reset to the default, not just on sym=dot.
* v554: *acm* 2/04 add /HGRAT /VGRAT to draw optional graticule lines at tic marks
* V580: *acm*11/04 add PLOT/NOKEY qualifier (bug 1089)
* V580 *acm*11/04  (bug 1085) Allow thickness to be set when cycling through colors 
*                  automatically. In call to line_style, when color = -11  then 
*                  thin line, color=-21 then thick=2, when color=-31 then thick=3
* V581 *acm* 5/05  fix bug 1250: step=connected. Need to use EQUAL_STR to get the
*                  argument to /STEP, then compare to see if it equals uppercase 'CON'
* V600 *acm* 5/06  increase length of buff, for longer arugments to /DASH=
* V600 *acm* 5/06  with MODE LINECOLORS, number of lines not limited to num_fancy_lines
* V6.11 4/08 *acm* Fixes for bug 1571; call to PPL_AXES_RESTORE moved to DISP_INIT
* V612 *acm*  8/08 Fixing bug 1038. No Valid Data labeled as on 2D plots. If 
*                  there is no valid data, get the info from plot_set_up and add 
*                  a to the plot. If this is an overlay, that is added to the label
*                  in plot_set_up.
* V612 *acm* 10/08 No Valid Data labeled If there is no valid data, only add this
*                  information to the label in plot_set_up. (see bugs 1038 and 1565)

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
        include 'switch_inc.decl'
        include 'SWITCH.INC'
        include 'data_inc.decl'
        include 'DATA.INC'
        include 'PPLDAT.INC'
        include 'parampl5_dat.decl'
        include 'PARAMPL5.DAT'
        include 'lines_inc.decl'
        include 'LINES.INC'
	
* calling argument declarations:
	REAL	memory( mem_blk_size, max_mem_blks )

* internal variable declarations:
	LOGICAL		overlay, transpz, versus,
     .			no_labels, set_up, noaxes, axes, use_line,
     .			is_logh, is_logv, nokey, no_data, no_range
	INTEGER		TM_LENSTR1, STR_UPCASE,
     .                  symbol, color, status,
     .			loc, limit, slen, thick, step_inc,
     .			ax1_blks, ax1_start,
     .			ax2_blks, ax2_start,
     .			do_dash, s1, s2, it_start(4), it_end(4),
     .			nparm, dot_sym, num_it, i, iax(4), i1,
     .                  loch, locv
	REAL		val, sym_size, dashstyle(4), only_val
	CHARACTER	LEFINT*4, buff*128

* internal parameter declarations
	INTEGER ncolors
	PARAMETER (ncolors = 6, dot_sym=99)

* set up the graphics environment and get the data
	CALL START_PPLUS
	CALL DISP_DATA_SET_UP( memory,
     .		ax1_start, ax1_blks, ax2_start, ax2_blks, status )
	IF ( status .NE. ferr_ok ) GOTO 5100

* Initialize the lines buffer in PPL before setting up
	overlay   = qual_given( slash_plot_over_plot ).GT.0
	if (overlay) SOVER = .true.

	if (sover) then
CC      *jd* NLINES initializes SOVER, and resets MIN/MAX. Don't execute. 
CC       => Set values here instead.
	   LINEN=0
	   IBASE=1
	   PBUF=0
	   LNUM=1
	else
	   CALL PPLCMD ( from, line, 0, 'NLINES', 1, 1 )
	endif

* initialize
	no_labels = qual_given( slash_plot_nolabels  ).GT.0
	transpz   = qual_given( slash_plot_transpz   ).GT.0
	versus    = qual_given( slash_plot_vs	     ).GT.0
	set_up	  = qual_given( slash_shad_set_up    ).GT.0
	noaxes    = qual_given( slash_noaxes         ).GT.0
	axes      = qual_given( slash_plot_axes      ).GT.0
	is_logh   = qual_given( slash_is_logh        ).GT.0
	is_logv   = qual_given( slash_is_logv        ).GT.0

	symbol	  = unspecified_int4
	sym_size  = 0.08	! default plot mark size
	thick	  = 1		! default thickness
	color     = qual_on	! automatic selection as default
	step_inc  = 1
        do_dash   = 0		! default no dashes

	IF ( .NOT.overlay ) CALL DISP_INIT(no_labels,is_cx,num_uvars_in_cmnd)
	CALL DISP_PREP( status )
	IF ( status .NE. ferr_ok ) GOTO 5000

* change default style to symbols, only if PLOT/VS (1/27/95)
	IF ( versus ) THEN
	   symbol = qual_on
	   use_line = .FALSE.
	ELSE
	   use_line = .TRUE.
	ENDIF

* decode qualifiers
* ... PLOT/SYMBOL=n
	loc = qual_given( slash_plot_symb )
	IF ( loc .GT. 0  ) THEN

*  Is it SYMBOL=DOT?
	   CALL EQUAL_STRING( cmnd_buff(qual_start(loc):qual_end(loc)),
     .                        buff, status )
	   IF ( status .NE. ferr_ok ) GOTO 5000
	   IF (buff .EQ. 'DOT') THEN
              symbol = dot_sym
           ELSE

	      CALL EQUAL_VAL( cmnd_buff(qual_start(loc):qual_end(loc)),
     .			      val, status )	! syntax check
	      IF ( status .NE. ferr_ok ) GOTO 5000

	      IF ( val .NE. unspecified_val4 ) THEN
	         limit = max_sym 
	         IF ( val.LE. 0.0 .OR. val.GT.limit ) GOTO 5200
	         symbol = NINT( val )	! user-specified symbol
	      ELSE
	         symbol = qual_on		! auto selection of symbols
	      ENDIF
           ENDIF
	   use_line = .FALSE.		! default to no line w/ symbol
! 6/00	   color = 1			! default symbol color	      
	ENDIF
* ... PLOT/SYM_SIZE=float
	loc = qual_given( slash_plot_symsize )
	IF ( loc .GT. 0  ) THEN
	   IF ( symbol .EQ. unspecified_int4 ) THEN
	       symbol = qual_on		! turn on symbols if needed
	       use_line = .FALSE.
	       color = 1	
	   ENDIF
	   CALL EQUAL_VAL( cmnd_buff(qual_start(loc):qual_end(loc)),
     .			      val, status )	! syntax check
	   IF ( status .NE. ferr_ok ) GOTO 5000
	   IF ( val .NE. unspecified_val4 ) THEN
	      IF ( val.LE. 0.0 ) GOTO 5210
	      sym_size = val		! user-specified symbol
              IF ( symbol .EQ. dot_sym) THEN
                 CALL WARN ('/SIZE= ignored  on /SYMBOL=DOT ')
                 sym_size = 0.08
              ENDIF
	   ENDIF
	ENDIF
* ... PLOT/COLOR=n
	loc = qual_given( slash_plot_color )
	IF ( loc .GT. 0  ) THEN
	   CALL EQUAL_COLOR( cmnd_buff(qual_start(loc):qual_end(loc)),
     .                           color, status )
           IF ( status .NE. ferr_ok ) GOTO 5000
	ENDIF
* ... PLOT/THICKNESS=n
 	loc = qual_given( slash_plot_thick )
	IF ( loc .GT. 0  ) THEN

	   CALL EQUAL_VAL( cmnd_buff(qual_start(loc):qual_end(loc)),
     .			   val, status )	! syntax check
	   IF ( status .NE. ferr_ok ) GOTO 5000

           IF (val .NE. 1  .AND. color .EQ. 19) GOTO 5460  ! Only have thin white line

	   IF ( val .NE. unspecified_val4 ) THEN
	      IF ( val.LE.0.0 .OR. val.GT.3 ) GOTO 5400
	      thick = NINT( val )	! user-specified line style
	   ELSE
	      thick = 2  ! default if no value is specified
	   ENDIF

	   IF ( color .EQ. qual_on ) THEN
              IF ( symbol .EQ. dot_sym) THEN
                 CALL WARN ('/THICKNESS ignored  on /SYMBOL=DOT ')
	         color = MOD(color-1,ncolors)+1  ! eliminate thickness
              ELSE
                 color = color-10*(thick-1)
              ENDIF
           ELSE
	      color = MOD(color-1,ncolors)+1  ! eliminate thickness
	      color = ncolors*(thick-1) + color
              IF ( symbol .EQ. dot_sym) THEN
                 CALL WARN ('/THICKNESS ignored  on /SYMBOL=DOT ')
	         color = MOD(color-1,ncolors)+1  ! eliminate thickness
              ENDIF
           ENDIF

	ENDIF
* ... PLOT/LINE_STYLE=n
	loc = qual_given( slash_plot_line )
	IF ( loc .GT. 0  ) THEN
	   use_line = .TRUE.
	   CALL EQUAL_VAL( cmnd_buff(qual_start(loc):qual_end(loc)),
     .			      val, status )	! syntax check
	   IF ( status .NE. ferr_ok ) GOTO 5000
	   IF ( val .NE. unspecified_val4 ) THEN
c	      limit = num_fancy_lines 			!was num_plain 1/94
              limit = 3*mode_arg(pmode_linecolors,1)
	      IF ( val.LE. 0.0 .OR. val.GT.limit ) GOTO 5200
	      color = NINT( val )	! user-specified line style
	   ENDIF
* ... default to no symbols when /LINE_STYLE, alone, is given
* 	or for PLOT/VS/LINE  (1/27/95)
           IF ( symbol .EQ. unspecified_int4
     .	   .OR. (versus.AND.qual_given(slash_plot_symb).EQ.0) ) 
     .							symbol = qual_off
	ENDIF
* ... PLOT/STEP[=CONNECTED]
	loc = qual_given( slash_plot_step )
	IF ( loc .GT. 0  .AND. .NOT.versus ) THEN
	   CALL EQUAL_STRING( cmnd_buff(qual_start(loc):qual_end(loc)),
     .                        buff, status )
           IF ( status .NE. ferr_ok ) GOTO 5000
	   IF ( buff(1:3) .EQ. 'CON' ) THEN
	      step_inc = 2	    
	   ELSEIF ( buff(1:3) .EQ. ' ' ) THEN
	      step_inc = 3
	   ELSE
	      GOTO 5220
	   ENDIF
	ENDIF	   

* ... PLOT/DASH=(dn1,up1,dn2,up2)
*     3/02 No reason we can't do PLOT/VS/DASH 

	loc = qual_given( slash_plot_dash )
c	IF ( loc .GT. 0  .AND. .NOT.versus ) THEN
	IF ( loc .GT. 0 ) THEN
	   CALL EQUAL_STR_LC( cmnd_buff(qual_start(loc):qual_end(loc)),
     .                        buff, status )
           IF ( status .NE. ferr_ok ) GOTO 5000
           do_dash = 1

           DO 100 i = 1, 4
              dashstyle(i) = 0.
 100       ENDDO

	   IF ( buff .NE. ' ' ) THEN
* Check for optional parentheses

              slen = TM_LENSTR1(buff)
              s1 = 1
              s2 = slen
              IF (buff(1:1) .EQ. '(' )THEN
                  s1 = 2
                  s2 = slen-1
                  IF ( buff(slen:slen) .NE. ')' ) GOTO 5420
              ENDIF

              nparm = 4
              CALL parse_comma_list ( buff, s1, s2, nparm, num_it,
     .             it_start, it_end, status)
              IF (status .EQ. ferr_ok .AND. it_start(1) .GT. it_end(1)) THEN
                 DO 200 i = 1, 4
                    dashstyle(i) = 0.
 200             ENDDO
              ELSE IF (status .NE. ferr_ok .OR. num_it .LT. nparm) THEN
                 GOTO 5420
              ELSE
                 DO 300 i = 1, nparm
                    READ (buff(it_start(i):it_end(i)), *) dashstyle(i)
                    IF (dashstyle(i) .LT. 0.) GOTO 5430
 300             CONTINUE
              ENDIF
           ENDIF
              
	ENDIF	   


* ... PLOT/hgrat[=dash or line, color=, thick[=] ] default is line
*     hgrat line type: 0=tics only, 1=dash 2=line

	loch = qual_given(slash_plot_hgrat)

* ... PLOT/vgrat[=dash or line, color=, thick[=] ] default is line
*     vgrat line type: 0=tics only, 1=dash 2=line

	locv = qual_given(slash_plot_vgrat)

        IF (loch .EQ. 0  .AND. locv .EQ. 0) THEN
           loch = qual_given(slash_plot_grat)
           locv = qual_given(slash_plot_grat)
        ENDIF

        CALL set_graticules(loch, locv, status)

* ... PLOT/NOKEY  (default is key on)

	nokey = qual_given(slash_plot_nokey) .GT. 0

* set up the axes and title and load the data into PPLUS
	CALL PLOT_SET_UP   ( 	memory,
     .				overlay,
     .				transpz,
     .				versus,
     .				no_labels,
     .				symbol,
     .				sym_size,
     .				color,
     .				use_line,
     .				step_inc,
     .				do_dash,
     .				dashstyle,
     .				is_logh, 
     .				is_logv,
     .                          nokey,
     .				no_range,
     .				only_val,
     .				is_mr,
     .				is_cx,
     .				num_uvars_in_cmnd,
     .				memory( 1, ax1_start ),
     .				memory( 1, ax2_start ),
     .				status )
	IF ( status .NE. ferr_ok ) GOTO 5000

c when it's an overlay, 'No Valid Data' is part of the plot key,
c done in plot_set_up.

* is the no valid input data?
              no_data =  no_range .AND. only_val.EQ.bad_val4

* turn off axes? 
	IF ( noaxes ) THEN 
           DO 400 i = 1, 4
              iax(i) = 0
 400       ENDDO
        ENDIF

* turn on (one or more) axes? (7/11/01)

* ... PLOT/AXES=(TOP,BOTTOM,LEFT,RIGHT)
	loc = qual_given( slash_plot_axes )
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
                 DO 500 i = 1, 4
                    iax(i) = 0
 500             ENDDO
              ELSE IF (status .NE. ferr_ok .OR. num_it .LT. nparm) THEN
                 GOTO 5440
              ELSE
                 DO 600 i = 1, nparm
                    READ (buff(it_start(i):it_end(i)), *) iax(i)
                    IF (iax(i) .LT. 0  .OR.  iax(i) .GT. 1) GOTO 5450
 600             CONTINUE
              ENDIF
           ELSE		! turn on all axes
	      DO 700 i = 1, 4
		 iax(i) = 1
 700	      CONTINUE
           ENDIF
              
	ENDIF

*  Turn off or on axes, including time axes.
        IF (axes .OR. noaxes) CALL ax_on_off (iax)

* /set_up only requested ?
	IF ( set_up ) GOTO 1000

* plot it 
	IF ( overlay ) THEN
	   CALL PPLCMD ( ' ', ' ', 0, 'PLOT/OVER/' //ppl_wait//' ', 1, 1 )
	ELSE
	   CALL PPLCMD ( ' ', ' ', 0, 'PLOT/LATCH/'//ppl_wait//' ', 1, 1 )
	ENDIF

* restore axes? (5/29/97  revised 3/2002
!        IF ( axes .or. noaxes ) CALL PPL_AXES_RESTORE

* restore dash settings? (5/29/97  revised 3/2002
        IF ( do_dash .EQ. 1 ) CALL PPL_DASH_RESTORE

* interrupt occurred during plot ?
	IF ( ppl_interrupted ) GOTO 1000

* capture the bit map ?
	IF ( qual_given( slash_plot_frame ) .GT. 0 ) THEN
	   CALL SAVE_FRAME( frame_file, "HDF", status )
	   IF ( status .NE. ferr_ok ) GOTO 5000
	ENDIF

*  Save log axis state

	IF (overlay) THEN
	   is_logh = h_logaxis
	   is_logv = v_logaxis
	ELSE
	   h_logaxis= is_logh
	   v_logaxis = is_logv 
	ENDIF


* release the memory used for axis buffers
 1000	CALL FREE_MEMORY( ax2_start, ax2_blks )
 	CALL FREE_MEMORY( ax1_start, ax2_blks )
	CALL UNPROTECT_CMND_DATA			! 1/94
 5100	RETURN

* error exits
 5000	GOTO 1000
 5200	CALL ERRMSG( ferr_out_of_range, status, 
     .		cmnd_buff(qual_start(loc):qual_end(loc))//pCR//
     .		'Legal range is 1 to '//LEFINT(limit,slen), *1000 )
 5210	CALL ERRMSG( ferr_out_of_range, status, 
     .		cmnd_buff(qual_start(loc):qual_end(loc))//pCR//
     .		'Symbol size must be positive ', *1000 )
 5220	CALL ERRMSG( ferr_syntax, status, 
     .		cmnd_buff(qual_start(loc):qual_end(loc))//pCR//
     .		'optionally /STEP=connected', *1000 )
 5400	CALL ERRMSG( ferr_out_of_range, status, 
     .		cmnd_buff(qual_start(loc):qual_end(loc))//pCR//
     .		'Legal range is 1 to 3', *1000 )
 5420	CALL ERRMSG( ferr_syntax, status, 
     .		cmnd_buff(qual_start(loc):qual_end(loc))//pCR//
     .	'DASH[=(dn1,up1,dn2,up2)] takes 4 arguments or no argument', *1000 )
 5430	CALL ERRMSG( ferr_out_of_range, status, 
     .		cmnd_buff(qual_start(loc):qual_end(loc))//pCR//
     .		'DASH arguments must be positive', *1000 )
 5440	CALL ERRMSG( ferr_syntax, status, 
     .		cmnd_buff(qual_start(loc):qual_end(loc))//pCR//
     .	'AXES[=(TOP,BOTTOM,LEFT,RIGHT)] takes 4 arguments or no argument', *1000 )
 5450	CALL ERRMSG( ferr_out_of_range, status, 
     .		cmnd_buff(qual_start(loc):qual_end(loc))//pCR//
     .		'AXES arguments must be 0 or 1', *1000 )
 5460	CALL ERRMSG( ferr_out_of_range, status, 
     .		cmnd_buff(qual_start(loc):qual_end(loc))//pCR//
     .		'/COLOR=WHITE only available with /THICK=1', *1000 )

	END
