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
        include 'switch_inc.decl'
        include 'SWITCH.INC'
        include 'data_inc.decl'
        include 'DATA.INC'
        include 'ppldat_inc.decl'
        include 'PPLDAT.INC'
        include 'parampl5_dat.decl'
        include 'PARAMPL5.DAT'
        include 'lines_inc.decl'
        include 'LINES.INC'
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
        INCLUDE 'tmap_pplv11inc:tmap_SWITCH.INC'
        INCLUDE 'TMAP_PPLV11INC:DATA_INC.DECL'
        INCLUDE 'PPLV11INC:DATA.INC'
        INCLUDE 'TMAP_PPLV11INC:PPLDAT_INC.DECL'
        INCLUDE 'PPLV11INC:PPLDAT.INC'
        INCLUDE 'TMAP_PPLV11INC:PARAMPL5_DAT.DECL'
        INCLUDE 'PPLV11INC:PARAMPL5.DAT'
        INCLUDE 'TMAP_PPLV11INC:LINES_INC.DECL'
        INCLUDE 'PPLV11INC:LINES.INC'
#endif
	
* calling argument declarations:
	REAL	memory( mem_blk_size, max_mem_blks )

* internal variable declarations:
	LOGICAL		overlay, transpz, versus,
     .			no_labels, set_up, noaxes, use_line
	INTEGER		TM_LENSTR1, symbol, color, status,
     .			loc, limit, slen, thick, step_inc,
     .			ax1_blks, ax1_start,
     .			ax2_blks, ax2_start,
     .			do_dash, it_start(4), it_end(4),
     .			ndashparm, num_it, i
	REAL		val, sym_size, dot_sym, dashstyle(4)
	CHARACTER	LEFINT*4, buff*4, dashbuf*25

* internal parameter declarations
	INTEGER ncolors
	PARAMETER (ncolors = 6, dot_sym = 99)

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
	   CALL EQUAL_STR_LC( cmnd_buff(qual_start(loc):qual_end(loc)),
     .                        buff, status )
	   IF ( status .NE. ferr_ok ) GOTO 5000
	   IF (buff .EQ. 'dot') THEN
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
              IF ( symbol .EQ. dot_sym) 
     .           CALL WARN ('/SIZE= ignored  on /SYMBOL=DOT ')
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
	   IF ( color .EQ. qual_on ) color = 1  ! /THICK alone means "BLACK"
	   color = MOD(color-1,ncolors)+1  ! eliminate thickness
	   CALL EQUAL_VAL( cmnd_buff(qual_start(loc):qual_end(loc)),
     .			   val, status )	! syntax check
	   IF ( status .NE. ferr_ok ) GOTO 5000
	   IF ( val .NE. unspecified_val4 ) THEN
	      IF ( val.LE.0.0 .OR. val.GT.3 ) GOTO 5400
	      thick = NINT( val )	! user-specified line style
	   ELSE
	      thick = 2  ! default if no value is specified
	   ENDIF
	   color = ncolors*(thick-1) + color
           IF ( symbol .EQ. dot_sym) THEN
              CALL WARN ('/THICKNESS ignored  on /SYMBOL=DOT ')
	      color = MOD(color-1,ncolors)+1  ! eliminate thickness
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
	      limit = num_fancy_lines 			!was num_plain 1/94
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
	   CALL EQUAL_STR_LC( cmnd_buff(qual_start(loc):qual_end(loc)),
     .                        buff, status )
           IF ( status .NE. ferr_ok ) GOTO 5000
	   IF ( buff(1:3) .EQ. 'con' ) THEN
	      step_inc = 2	    
	   ELSEIF ( buff(1:3) .EQ. ' ' ) THEN
	      step_inc = 3
	   ELSE
	      GOTO 5220
	   ENDIF
	ENDIF	   

* ... PLOT/DASH=(dn1,up1,dn2,up2)
	loc = qual_given( slash_plot_dash )
	IF ( loc .GT. 0  .AND. .NOT.versus ) THEN
	   CALL EQUAL_STR_LC( cmnd_buff(qual_start(loc):qual_end(loc)),
     .                        dashbuf, status )
           IF ( status .NE. ferr_ok ) GOTO 5000
           do_dash = 1

	   IF ( dashbuf .NE. ' ' ) THEN
* Check parentheses
              slen = TM_LENSTR1(dashbuf)
              IF (dashbuf(1:1) .NE. '(' .OR. dashbuf(slen:slen) .NE. ')' ) 
     .           GOTO 5420
              ndashparm = 4
              CALL parse_comma_list ( dashbuf, 2, slen-1, ndashparm, num_it,
     .             it_start, it_end, status)
              IF (status .EQ. ferr_ok .AND. it_start(1) .GT. it_end(1)) THEN	! /DASH=()
                 DO 100 i = 1, 4
                    dashstyle(i) = 0.
 100             ENDDO
              ELSE IF (status .NE. ferr_ok .OR. num_it .LT. ndashparm) THEN
                 GOTO 5420
              ELSE
                 DO 200 i = 1, ndashparm
                    READ (dashbuf(it_start(i):it_end(i)), *) dashstyle(i)
                    IF (dashstyle(i) .LT. 0.) GOTO 5430
 200             CONTINUE
              ENDIF
           ENDIF
              
	ENDIF	   

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
     .				is_mr,
     .				is_cx,
     .				num_uvars_in_cmnd,
     .				memory( 1, ax1_start ),
     .				memory( 1, ax2_start ),
     .				status )
	IF ( status .NE. ferr_ok ) GOTO 5000

* turn off axes? (5/29/97)
	IF ( noaxes ) CALL PPL_AXES_OFF(set_up)

* /set_up only requested ?
	IF ( set_up ) GOTO 1000

* plot it 
	IF ( overlay ) THEN
	   CALL PPLCMD ( ' ', ' ', 0, 'PLOT/OVER/' //ppl_wait//' ', 1, 1 )
	ELSE
	   CALL PPLCMD ( ' ', ' ', 0, 'PLOT/LATCH/'//ppl_wait//' ', 1, 1 )
	ENDIF

* restore axes? (5/29/97)
	IF ( noaxes ) CALL PPL_AXES_RESTORE

* interrupt occurred during plot ?
	IF ( ppl_interrupted ) GOTO 1000

* capture the bit map ?
	IF ( qual_given( slash_plot_frame ) .GT. 0 ) THEN
	   CALL SAVE_FRAME( frame_file, "HDF", status )
	   IF ( status .NE. ferr_ok ) GOTO 5000
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

	END
