	SUBROUTINE XEQ_PLOT

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
* V580 *acm*11/04  (bug 1085) Allow JFness to be set when cycling through colors 
*                  automatically. In call to line_style, when color = -11  then 
*                  thin line, color=-21 then JF=2, when color=-31 then thick=3
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
* V6.2  *acm* 11/08Fix bug 1609: If the plot is an overlay then do not process /AXES=
*                  or /NOAXES. Axes are never drawn for overlays anyway.
* V6.2  *acm& 2/09 Fix choice of color settings with /PEN and /THICK when we have 
*                  increased the num of colors with MODE linecolors
* V6.85 *acm* 3/13 Fix ticket 2049; check we can open palette file
* PyFr  *kms* 7/13 EQUAL_COLOR now returns a color ID, and EQUAL_THICK 
*                  returns a pen ID from a color ID and a thickness.
* PyFr  *kms* 7/13 Added /OPACITY
* V6.85 *acm* 9/13 New qualifier PLOT/%%%%%%% for first color in multi-line plot
* http://www.pmel.noaa.gov/maillists/tmap/ferret_users/fu_2013/msg00536.html

* V6.92 *acm* 6/14 New color /key options: any of /key=horizontal,centerlabel
* V6.94 *acm* 2/15 Ticket 2231: New /key option for multiple plot lines: /key=titles
* v696 *acm*  6/15 PLOT/VS/GAPLOC to define seperate line segments on
*                  PLOT/VS/RIBBON plots (LAS ticket 1597 and others). Could implement
*                  this later for other PLOT/VS plots.
* v697 *acm* 12/15 Ticket 2331, add flag changed_key to common, to restore the color key
*                  style after a color plot and also after SHADE/SET; SHADE.
* v698 *sh* 3/16   added PLOT/ALONG=axis
* V698  4/16 *acm* New PLOT/SKIP=
* V7    *acm* 6/16 Ticket 2434: command qualifiers /DEGMINSEC= /HALFSPAC
*                  DEGMINSEC=1 for Deg-Min axis labels, =2 for Deg-Min-Sec. Default = 1
*                  /HALFSPAC adds a half space between the degree sign and E/W or N/S
* V702 1/17 *sh* for trac enhancement #2369 -- dynamic memory management 
* V71  *acm* 2/17  More on ticket 2231. Initialize flag use_keys.
* PyFr *kms*  4/17 Remove SAVE_FRAME for HDF format since no longer supported

	include 'tmap_dims.parm'
	include 'ferret.parm'
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
        include 'gkscm1_inc.decl'
        include 'GKSCM1.INC'    ! wsid
        include 'cont_inc.decl' ! with iautoc and nlev
        include 'CONT.INC'

* calling argument declarations:

* internal variable declarations:
	LOGICAL		TM_HAS_STRING, 
     .			overlay, transpz, versus,
     .			no_labels, set_up, noaxes, axes, use_line,
     .			is_logh, is_logv, nokey, no_data, no_range,
     .			do_key, spectrum, addgaps, along
	INTEGER		TM_LENSTR1, TM_LENSTR, STR_UPCASE, STR_SAME,
     .                  symbol, icolor, ipen, icolor1, ipen1, status,
     .			loc, limit, slen, thick, step_inc,
     .			ax1_work, ax2_work,
     .			do_dash, s1, s2, it_start(4), it_end(4),
     .			nparm, dot_sym, num_it, i, iax(4), i1,
     .                  loch, locv, nvs, skipsym
	REAL		val, sym_size, dashstyle(4), only_val
        REAL*4          thickness, opacity
	CHARACTER	LEFINT*4, buff*128
	
	CHARACTER	TM_FMT*12, TM_INQ_PATH*2048, sfile*2048
	INTEGER ncolors

* internal parameter declarations
	PARAMETER (dot_sym=99)

* set up the graphics environment and get the data
	CALL START_PPLUS(.FALSE.)
	CALL DISP_DATA_SET_UP(  ax1_work, ax2_work, status )
	IF ( status .NE. ferr_ok ) GOTO 5100

        ncolors = mode_arg(pmode_linecolors,1)

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
	set_up	  = qual_given( slash_plot_set_up    ).GT.0
	noaxes    = qual_given( slash_noaxes         ).GT.0
	axes      = qual_given( slash_plot_axes      ).GT.0
	along     = qual_given( slash_plot_along     ).GT.0

	IF (along .AND. versus) THEN
	   CALL WARN("/ALONG ignored.  Incompatible with PLOT/VS") 
	   along = .FALSE.
	ENDIF

        IF (overlay) THEN
	   noaxes = .FALSE.
	   axes = .FALSE.
	ENDIF

	is_logh   = qual_given( slash_is_logh        ).GT.0
	is_logv   = qual_given( slash_is_logv        ).GT.0

	symbol	  = unspecified_int4
	sym_size  = 0.08	! default plot mark size
	thick	  = 1		! default thickness
	icolor    = qual_on	! automatic selection as default
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
	ENDIF
* ... PLOT/SYM_SIZE=float
	loc = qual_given( slash_plot_symsize )
	IF ( loc .GT. 0  ) THEN
	   IF ( symbol .EQ. unspecified_int4 ) THEN
	       symbol = qual_on		! turn on symbols if needed
	       use_line = .FALSE.
	       icolor = 1
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
* ... PLOT/COLOR[=n]
        icolor1 = 1
        thickness = 1.0
	loc = qual_given( slash_plot_color )
	IF ( loc .GT. 0  ) THEN
	   icolor1 = icolor
	   CALL EQUAL_COLOR( cmnd_buff(qual_start(loc):qual_end(loc)),
     .                       icolor, thickness, status )
           IF ( status .NE. ferr_ok ) GOTO 5000
           s1 = INDEX( cmnd_buff(qual_start(loc):qual_end(loc)), '=' )
           IF ( s1 .EQ. 0 ) THEN
	      icolor = icolor1
	      icolor1 = 2
	      IF ( overlay ) GOTO 5510  ! /COLOR w/o argument not allowed for overlay
	   ENDIF
	ENDIF
* ... PLOT/OPACITY=n
        loc = qual_given( slash_plot_opacity )
        IF ( loc .GT. 0 ) THEN
           CALL EQUAL_VAL(cmnd_buff(qual_start(loc):qual_end(loc)),
     .                    val, status)
           IF ( status .NE. ferr_ok ) GOTO 5000
           IF ( (val .LT. 0.0) .OR. (val .GT. 100.0) ) GOTO 5500
           opacity = val / 100.0
           IF ( icolor .GE. 0 )
     .        CALL DUP_COLOR_SET_OPACITY(wsid, icolor, opacity)
           CALL SET_OVERRIDE_OPACITY(opacity)
        ELSE
*          overriding opacity not specififed
           opacity = -1.0
           CALL SET_OVERRIDE_OPACITY(opacity)
        ENDIF
* ... PLOT/THICKNESS=n
 	loc = qual_given( slash_plot_thick )
	IF ( loc .GT. 0  ) THEN
           IF ( symbol .EQ. dot_sym) THEN
              CALL WARN ('/THICKNESS ignored  on /SYMBOL=DOT ')
*             convert the color ID into a (thin) pen ID
*             note that thickness is explicitly REAL*4 (not REAL)
              thickness = 1.0
              CALL EQUAL_THICK(' ', thickness, icolor, ipen, status)
              IF ( status .NE. ferr_ok ) GOTO 5000
           ELSE
              CALL EQUAL_THICK(cmnd_buff(qual_start(loc):qual_end(loc)),
     .                         thickness, icolor, ipen, status)
              IF ( status .NE. ferr_ok ) GOTO 5000
           ENDIF
        ELSE
*          /THICK not given; convert color ID to pen ID using default thickness
           CALL EQUAL_THICK(' ', thickness, icolor, ipen, status)
           IF ( status .NE. ferr_ok ) GOTO 5000
        ENDIF
* ... PLOT/LINE_STYLE=n
	loc = qual_given( slash_plot_line )
	IF ( loc .GT. 0  ) THEN
	   use_line = .TRUE.
	   CALL EQUAL_VAL( cmnd_buff(qual_start(loc):qual_end(loc)),
     .			      val, status )	! syntax check
	   IF ( status .NE. ferr_ok ) GOTO 5000
	   IF ( val .NE. unspecified_val4 ) THEN
          limit = 3*mode_arg(pmode_linecolors,1)
	      ipen = NINT( val )	! user-specified line style
	      IF ( ipen .LT. 1 .OR. ipen .GT. limit ) GOTO 5200
              IF ( opacity .GE. 0.0 )
     .           CALL DUP_STD_PEN_SET_OPACITY(wsid, ipen, opacity)
	   ENDIF
* ... default to no symbols when /LINE_STYLE, alone, is given
* 	or for PLOT/VS/LINE  (1/27/95)
           IF ( symbol .EQ. unspecified_int4
     .	   .OR. (versus.AND.qual_given(slash_plot_symb).EQ.0) ) 
     .							symbol = qual_off
	ENDIF

* ... PLOT/SKIP=n

	skipsym = 1
	loc = qual_given( slash_plot_skip )
	IF ( loc .GT. 0) THEN
	   CALL EQUAL_VAL( cmnd_buff(qual_start(loc):qual_end(loc)),
     .			      val, status )	! syntax check
	   IF ( status .NE. ferr_ok ) GOTO 5000
	   IF ( val .NE. unspecified_val4 ) skipsym = NINT( val )
	ENDIF 

* ... /KEY[=title]  for control over labeling the multi-line legend

	loc = qual_given( slash_plot_key )
	use_keys = .TRUE.

	IF ( loc .GT. 0) THEN
	   CALL EQUAL_STRING( cmnd_buff(qual_start(loc):qual_end(loc)),
     .                        buff, status )
	   IF ( status .NE. ferr_ok ) GOTO 5000
	   IF (TM_LENSTR(buff) .GT. 0) THEN
	      IF ( TM_HAS_STRING(buff, 'TIT') ) use_keys = .FALSE.
	   ENDIF
	ENDIF
	

* ... /DEGMINSEC[=1 or 2] Request deg/min or deg/min/sec axis labels
	loc = qual_given( slash_plot_dms )
        dms = 0
	IF ( loc .GT. 0) THEN
	   CALL EQUAL_VAL( cmnd_buff(qual_start(loc):qual_end(loc)),
     .			      val, status )
           dms = 1
           IF (val .GT. 0) THEN
	      dms = val
	      IF (dms .GT. 2) GOTO 5530
	   ENDIF
	ENDIF

* ... /SPACELAB Request deg/min or deg/min/sec axis labels
	loc = qual_given( slash_plot_halfsp )
	IF ( loc .GT. 0) THEN
	   CALL EQUAL_VAL( cmnd_buff(qual_start(loc):qual_end(loc)),
     .			      val, status )
           lonlatspace = 1
           IF (val .GE. 0) lonlatspace = val
	ENDIF

* PLOT/RIBBON for multi-colored "ribbon" line
	ribbon_plot = qual_given(slash_plot_ribbon) .GT. 0

* A "gaps" variable specifies the end of line segments to draw.
* Implemented only for PLOT/VS/RIBBON
	addgaps   = qual_given( slash_plot_gap       ).GT.0
	IF (addgaps .AND. .NOT.ribbon_plot) GOTO 5600

*     3-argument plot/vs/lev xpts, ypts, var
*     or 4-argument plot/vs/lev xpts, ypts, var, gap

	IF ( ribbon_plot  ) THEN

	   IF (versus) THEN
	      nvs = 3
	      IF (addgaps) nvs = 4
	      IF (num_uvars_in_cmnd .LT. nvs) GOTO 5480
	      IF (num_uvars_in_cmnd .GT. nvs) THEN
	         buff = 'PLOT/RIBBON/VS uses just 3 variables. Ignoring extra variables.'
		 IF (addgaps) buff = 'PLOT/RIBBON/VS/GAPLOC uses 4 variables. Ignoring extra variables.'
		 CALL WARN (buff)
	      ENDIF
           ELSE
	      IF (num_uvars_in_cmnd .LT. 2) GOTO 5480
	      IF (num_uvars_in_cmnd .GT. 2)
     .         CALL WARN (
     .         'PLOT/RIBBON/LEVELS uses just 2 variables. Ignoring extra variables.')
           ENDIF

* If a color was given, that's not valid for Ribbon plots.
	   loc = qual_given( slash_plot_color )
	   IF ( loc .GT. 0  ) GOTO 5490

* If a thickness is given, set pen number to the thickness, pen=1,2,or 3

	   ipen = 1
 	   loc = qual_given( slash_plot_thick )
	   IF ( loc .GT. 0  ) THEN

	      CALL EQUAL_VAL( cmnd_buff(qual_start(loc):qual_end(loc)),
     .			   val, status )

	      IF ( val .NE. unspecified_val4 ) THEN
	         thick = NINT( val )	! user-specified line style
	      ELSE
	         thick = 2  ! default if no value is specified
	      ENDIF
	      ipen = thick 
	   ENDIF
	   
	   ribbon_plot = .TRUE.
	ENDIF

* A "gaps" variable specifies the end of line segments to draw for PLOT/VS/RIBBON.
* Issue notes for cases not already handled.

	IF (addgaps .AND. .NOT.versus ) GOTO 5600

* Style: fast for one-color per point, default is to interpolate
* colors between points

           loc = qual_given( slash_plot_fast  )
           IF ( loc.GT.0 ) THEN
     		CALL PPLCMD ( from, line, 0, 'RIBFAST 1', 1, 1 )
           ELSE
     		CALL PPLCMD ( from, line, 0, 'RIBFAST 0', 1, 1 )
           ENDIF

* Missing: how to represent missing color-values. Default is
* thin black line. Can request /missing=dash, /missing=blank, 
* /missing= color-palette (e.g. gray_light). The color palette 
* will typically be a one-color plot, or if a multicolor palette 
* is given its first color will be used.

* (call this BEFORE setting the main color palette for ribbon plot; so that
* the setting for palette type: percent, by_level, or by_value is set for the
* specified palette.)

           loc = qual_given( slash_plot_missing  )
           IF ( loc.GT.0 ) THEN
              CALL EQUAL_STR_LC( cmnd_buff(qual_start(loc):qual_end(loc)),
     .             buff, status )
              IF ( status .NE. ferr_ok ) RETURN

	      IF (STR_SAME (buff(1:4), 'blan') .EQ. 0) THEN
                 CALL PPLCMD ( from, line, 0, 'RIBMISS BLANK', 1, 1 )
              ELSEIF ( buff .EQ. ' ' ) THEN
                 CALL PPLCMD ( from, line, 0, 'RIBMISS DEFAULT', 1, 1 )
	      ELSE
                 CALL PPLCMD ( from, line, 0, 'RIBMISS SPECTRUM='//buff, 1, 1 )
	      ENDIF
           ELSE
	      CALL PPLCMD ( from, line, 0, 'RIBMISS DEFAULT', 1, 1 )
           ENDIF

* ... PLOT/PALETTE=  for multi-colored "ribbon" line 
*     3-agument plot/lev/palette xpts, ypts, var

* save old and reset to new color spectrum if requested

        IF (ribbon_plot) THEN
           loc = qual_given( slash_plot_spectrum  )
           IF ( loc.GT.0 ) THEN
              CALL EQUAL_STR_LC( cmnd_buff(qual_start(loc):qual_end(loc)),
     .             buff, status )
              IF ( status .NE. ferr_ok ) RETURN
              IF ( buff .EQ. ' ' ) buff = 'default'

	      slen = TM_LENSTR1(buff)
              sfile = TM_INQ_PATH (buff(:slen),'FER_PALETTE','.spk',
     .        	 .false.,status)
              IF (status .NE. ferr_ok) go to 5300 

              CALL PPL_SHASET( 'SAVE' )
              CALL PPL_SHASET( 'SPECTRUM='//buff(1:TM_LENSTR1(buff)) )
	      spectrum = .TRUE.

           ENDIF


* default color key
	   do_key =  (qual_given(slash_plot_key)  .GT. 0)
     .		.OR. ( (qual_given(slash_plot_key) .EQ. 0)
     .           .AND. .NOT.overlay
     .           .AND. .NOT.no_data )
	   IF (qual_given(slash_plot_nokey) .gt. 0 ) do_key = .FALSE.
           IF ( do_key ) THEN
              CALL PPLCMD ( from, line, 0, 'SHAKEY 1,1', 1, 1 )
* ... /KEY[=continuous,horizontal,centerlabel] ?
*     /KEY=title is a legal option, but ignored for ribbon plot.
              loc = qual_given( slash_plot_key )
	      changed_key = .FALSE.

              IF ( loc .GT. 0) THEN
                 CALL EQUAL_STRING( cmnd_buff(qual_start(loc):qual_end(loc)),
     .                        buff, status )
                 IF ( status .NE. ferr_ok ) GOTO 5000
                 IF (TM_LENSTR(buff) .GT. 0) THEN
                    changed_key = .FALSE.
                    IF ( TM_HAS_STRING(buff, 'CON') ) THEN
                       changed_key = .TRUE.
                       CALL PPLCMD (from, line, 0, 'S_KEY 1', 1, 1 )
                    ENDIF
                    IF ( TM_HAS_STRING(buff, 'CEN') ) THEN
                       changed_key = .TRUE.
                       CALL PPLCMD (from, line, 0, 'S_KEY 2', 1, 1 )
                    ENDIF
                    IF ( TM_HAS_STRING(buff, 'HOR') ) THEN
                       changed_key = .TRUE.
                       CALL PPLCMD (from, line, 0, 'S_KEY 3', 1, 1 )
                    ENDIF
                    IF ( TM_HAS_STRING(buff, 'TIT') ) THEN  
                       changed_key = .TRUE.
                       use_keys = .FALSE.
                    ENDIF
                    IF (.NOT. changed_key) GOTO 5520
                 ENDIF
              ENDIF
           ELSE
              CALL PPLCMD ( from, line, 0, 'SHAKEY 0,1', 1, 1 )
           ENDIF


	ENDIF  ! ribbon plot


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


* ... PLOT/hgrat[=dash or line, color=, JF[=] ] default is line
*     hgrat line type: 0=tics only, 1=dash 2=line

	loch = qual_given(slash_plot_hgrat)

* ... PLOT/vgrat[=dash or line, color=, JF[=] ] default is line
*     vgrat line type: 0=tics only, 1=dash 2=line

	locv = qual_given(slash_plot_vgrat)

        IF (loch .EQ. 0  .AND. locv .EQ. 0) THEN
           loch = qual_given(slash_plot_grat)
           locv = qual_given(slash_plot_grat)
        ENDIF

        CALL SET_GRATICULES(loch, locv, status)

* ... PLOT/NOKEY (default is key on)
* Note: "/KEY" refers to the legend on PLOT and to the color bar on SHADE

	nokey = qual_given(slash_plot_nokey) .GT. 0
     .     .OR. (qual_given(slash_plot_title) .GT. 0 .AND. .NOT.overlay)

* set up the axes and title and load the data into PPLUS

	CALL PLOT_SET_UP   ( 	
     .				overlay,
     .				transpz,
     .				versus,
     .				no_labels,
     .				symbol,
     .				sym_size,
     .				ipen,
     .				icolor1,
     .				use_line,
     .				step_inc,
     .				do_dash,
     .				dashstyle,
     .				is_logh, 
     .				is_logv,
     .				nokey, 
     .				addgaps,
     .				no_range,
     .				only_val,
     .				skipsym,
     .				is_mr,
     .				is_cx,
     .				num_uvars_in_cmnd,
     .				workmem(ax1_work)%ptr,
     .				workmem(ax2_work)%ptr,
     .				status )
	IF ( status .NE. ferr_ok ) GOTO 5000
	
	IF (need_histo) iautoc = 1

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
	IF ( axes ) THEN
           loc = qual_given( slash_plot_axes )
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
              CALL PARSE_COMMA_LIST ( buff, s1, s2, nparm, 
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
	IF (ribbon_plot) THEN
	   IF ( overlay ) THEN
	      CALL PPLCMD ( ' ', ' ', 0, 'RIBBON/OVER/' //ppl_wait//' ', 1, 1 )
	   ELSE
	      CALL PPLCMD ( ' ', ' ', 0, 'RIBBON/LATCH/'//ppl_wait//' ', 1, 1 )
	   ENDIF
 
	ELSE
	
	   IF ( overlay ) THEN
	      CALL PPLCMD ( ' ', ' ', 0, 'PLOT/OVER/' //ppl_wait//' ', 1, 1 )
	   ELSE
	      CALL PPLCMD ( ' ', ' ', 0, 'PLOT/LATCH/'//ppl_wait//' ', 1, 1 )
	   ENDIF
	ENDIF

* clear the overriding opacity
        opacity = -1.0
        CALL SET_OVERRIDE_OPACITY(opacity)

* restore axes? (5/29/97  revised 3/2002
        IF ( axes .or. noaxes ) CALL PPL_AXES_RESTORE

* restore default color key style?
        IF (changed_key ) CALL PPLCMD (from, line, 0, 'S_KEY 0', 1, 1 )

* restore dash settings? (5/29/97  revised 3/2002
        IF ( do_dash .EQ. 1 ) CALL PPL_DASH_RESTORE

* interrupt occurred during plot ?
	IF ( ppl_interrupted ) GOTO 1000

*  Save log axis state

	IF (overlay) THEN
	   is_logh = h_logaxis
	   is_logv = v_logaxis
	ELSE
	   h_logaxis= is_logh
	   v_logaxis = is_logv 
	ENDIF
	
* Ribbon plot? protect the new colors and restore the original color spectrum
           IF ( spectrum ) THEN
*              PyFerret does not need to protect colors after using them
*              CALL PPL_SHASET( 'PROTECT' )
              CALL PPL_SHASET( 'SPECTRUM' )
           ENDIF


* release the memory used for axis buffers
! PLOT/ALONG does a hack to num_uvars_in_cmnd that can lead to a problem
! in GET_PROT_CMND_DATA if the same argument string is re-used in the next cmnd
 1000	IF (along) num_uvars_in_cmnd = 1
 	CALL FREE_WS_DYNMEM( ax1_work )
	CALL FREE_WS_DYNMEM( ax2_work )
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
 5300   CALL ERRMSG (ferr_file_not_found, status, 
     .          'unable to open "' //
     .	buff(:slen)//'.spk" to set ribbon-plot colors.', *1000)
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
 5470	CALL ERRMSG( ferr_syntax, status, 
     .		cmnd_buff(qual_start(loc):qual_end(loc))//pCR//
     .	'Ribbon plot PLOT/LEVELS valid only with /VS', *1000 )
 5480	IF (versus .AND. addgaps .AND. ribbon_plot) THEN
           CALL ERRMSG( ferr_syntax, status, 
     .	   'PLOT/VS/RIBBON/GAPLOC plot needs 4 arguments', *1000 )
        ELSEIF (versus .AND. addgaps) THEN
           CALL ERRMSG( ferr_syntax, status,
     .     'PLOT/VS/GAPLOC implemented only for RIBBON plots', *1000)
        ELSEIF (versus) THEN
           CALL ERRMSG(ferr_syntax, status,
     .     'PLOT/VS/RIBBON plot needs 3 arguments', *1000 )
	ELSE
           CALL ERRMSG( ferr_syntax, status, 
     .	   'RIBBON plot needs 2 arguments', *1000 )
	ENDIF
 5490	CALL ERRMSG( ferr_syntax, status, 
     .		cmnd_buff(qual_start(loc):qual_end(loc))//pCR//
     .	'/COLOR not valid for PLOT/LEVELS ribbon plot', *1000 )
 5500   CALL ERRMSG( ferr_out_of_range, status,
     .          cmnd_buff(qual_start(loc):qual_end(loc))//pCR//
     .          'Value from 0 to 100 must be given', *1000 )
 5600	CALL ERRMSG( ferr_syntax, status, 
     .	'PLOT/GAPLOC applies only to PLOT/VS/RIBBON', *1000 )

 5510	CALL ERRMSG( ferr_syntax, status, 
     .		cmnd_buff(qual_start(loc):qual_end(loc))//pCR//
     .	'/COLOR must have an argument for PLOT/OVERLAY', *1000 )
     
 5520	CALL ERRMSG( ferr_syntax, status, 
     .		cmnd_buff(qual_start(loc):qual_end(loc))//pCR//
     .		'Unrecognized argument to KEY command', *1000 )

 5530	CALL ERRMSG( ferr_syntax, status, 
     .		cmnd_buff(qual_start(loc):qual_end(loc))//pCR//
     .		'Argument to DegMinSec qualifier must be 1 or 2', *1000 )

	END
