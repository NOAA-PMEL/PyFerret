	SUBROUTINE LINE_STYLE( symbol, sym_size, color,
     .			       use_line, do_dash, dashstyle,
     .			       line_number, so_far )

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
* In Ferret the concept of the PLOT+ PEN command is generalized to combine
* line type (dashing), color and symbol into this single call
* Line number ipen will always be associated with line style ipen.
* Prior to this call colors and line types have already been combined via GKS
* bundles to define 6 distinct line styles as bundle indices 1-6
* Symbols are now added to extend this number to 6x17=102
* 1/94: number of line bundles is now num_fancy_lines.  All of these are
* used if /LINES is specified without /SYMBOLS

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system

* V200:	  1/2/90
* 	 2/21/90 - changed line style order so all dash patterns are used for
*		   each symbol choice, then the next symbol ...
* 	 5/18/90 - added args symbol and color to force styling of lines
* Unix/RISC port - 10/91 - /symb"/noline" -> successive line w/ new symbol
* V230:  7/23/92 - when /LINE_STYLE is given /SYMBOLS must default to OFF
* V301:  1/27/94 - when /LINE_STYLE is given w/out /SYMBOL use num_fancy_lines
* V510 *sh* 3/00 - added args use_line and sym_size to suport PLOT/COLOR=/SIZE=
* V530 *acm 5/01 - added args do_dash and dashstyle for 
*		   PLOT/DASH[=(dn1,up1,dn2,up2)]
* V580 *acm*11/04  Allow thickness to be set when cycling through colors automatically.
*                  When color = -11  then thin line, color=-21 then thick=2, when 
*                  color=-31 then thick=3
* V590 *acm* 5/06  settable number of line colors, no longer hard-wired to 6 colors.
                 
	include	'slash.parm'
	include	'plot_setup.parm'
	include	'xplot_setup.cmn'
        include 'parampl5_dat.decl'
        include 'PARAMPL5.DAT'
        include 'lines_inc.decl'
        include 'LINES.INC'
	include 'xprog_state.cmn'
        include 'ppl_dash_save.cmn'

* calling argument declarations
	LOGICAL use_line
	INTEGER symbol, color, line_number, so_far, do_dash
	REAL	sym_size, dashstyle(*)

* internal variable declarations
        LOGICAL unique_symbol, symbol_only1, select_lines
	INTEGER	style, ppl_draw_line, sym_so_far, i, ltyp
        SAVE sym_so_far     ! # symbols on the screen ... *sh* 10/91

* internal parameter declarations
	INTEGER	num_line, ncolors, num_linebundles
c	PARAMETER     ( num_line = num_plain*num_sym, ncolors=6)

* internal variable declarations
	INTEGER	ityp, imrk, style0, sym(num_sym)
        LOGICAL setdash
	DATA	sym/1,3,17,19,21,23,25,27,29,31,33,35,39,41,43,47,51/

* initialize

* num_plain colors was in common as 6, now use line_colors from mode setting
* Use num_linebundles in place of hardwired num_fancy_lines

        ncolors = mode_arg(pmode_linecolors,1)
        num_linebundles = 3 * ncolors
        num_line = ncolors*num_sym

        IF ( so_far .EQ. 1 ) sym_so_far = 0      ! first line on screen
* ... each line needs a new symbol for /SYMBOL or /LINE=n/SYMBOL
        unique_symbol = ( symbol.EQ.qual_on
     .             .AND. .NOT.use_line )
        symbol_only1  = symbol.GT.0 .AND. .NOT.use_line
	select_lines = symbol.EQ.qual_off .AND. use_line

* respond to qualifiers that force actions
	IF ( use_line ) THEN
	   ppl_draw_line = 1	! draw the line by default
	ELSE
	   ppl_draw_line = 3	! PPL LINE command: mark points only
	ENDIF
	IF ( symbol .EQ. qual_on ) THEN
* ... bypass plain lines
	   style = so_far + ncolors
	ELSE
	   style = so_far		! default: no qualifiers
	ENDIF

* select line type and symbols
        IF ( unique_symbol .OR. symbol_only1 ) THEN 
           IF (color.GT.0) THEN
	      ityp = color		! turn on symbol color 3/00
	   ELSE
	      ityp = 1
	   ENDIF
           imrk = MOD(sym_so_far,num_sym) + 1
           sym_so_far = sym_so_far + 1
	ELSEIF ( select_lines .AND. style .LE. num_linebundles ) THEN  ! 1/94
	   ityp = style
	   imrk = 0
	ELSEIF ( style .LE. ncolors ) THEN
	   ityp = style
	   imrk = 0
	ELSEIF ( style .LE. num_line ) THEN
	   style0 = (style-ncolors) - 1
	   ityp  = MOD(style0,ncolors) + 1
	   imrk  = MOD(style0/ncolors,num_sym) + 1
           sym_so_far = MAX( sym_so_far, imrk )    ! ... *sh* 10/91
	ELSE
	   ityp = 1
	   imrk = 0
	ENDIF	   
* replace selections with forced values if supplied
	IF ( symbol    .GT. 0 ) THEN
	  imrk = symbol
	ELSEIF ( symbol .EQ. qual_off ) THEN
          imrk = 0                       ! 7/92
	ELSEIF ( imrk .NE. 0 ) THEN
	  imrk = sym(imrk)
	ENDIF
	IF ( color .GT. 0 ) THEN
           ityp = color
           ltyp = color
        ELSE
           IF (color .EQ. -21) THEN
              ltyp = ityp + ncolors
           ELSE IF (color .EQ. -31) THEN
              ltyp = ityp + ncolors*2
           ELSE
              ltyp = ityp
           ENDIF
        ENDIF

	WRITE ( ppl_buff, 3000 ) line_number, ltyp
 3000	FORMAT('PEN ',2I4)
	CALL PPLCMD ( from, line, 0, ppl_buff, 1, 1 )

	IF ( imrk .GT. 0 ) THEN
* ... "LINE 8 3 1" - symbol number 3 on line 8
	   WRITE ( ppl_buff, 3100 ) line_number, imrk, ppl_draw_line
 3100	   FORMAT ('LINE ',3I4)
	ELSE
* ... "LINE 4 1 0" - no symbols on line 4
	   WRITE ( ppl_buff, 3200 ) line_number
 3200	   FORMAT ('LINE ',I4,' 1 0')
	ENDIF
	CALL PPLCMD ( from, line, 0, ppl_buff, 1, 1 )
	IF ( imrk .GT. 0 ) THEN
* ... "MARKH 8 0.08" - symbol size 0.08 on line 8
	   WRITE ( ppl_buff, 3300 ) line_number, sym_size
 3300	   FORMAT ('MARKH ',I4,F8.3)
	   CALL PPLCMD ( from, line, 0, ppl_buff, 1, 1 )
	ENDIF

*  User-requested dash overrides automatic selections
        dash_changed(line_number) = .FALSE.
        linenum_save = 0
        IF (do_dash .EQ. 1) THEN
	   ityp = 4

           dash_changed(line_number) = .TRUE.
           dashsave(line_number,1) = dn1(line_number)
           dashsave(line_number,2) = up1(line_number)
           dashsave(line_number,3) = dn2(line_number)
           dashsave(line_number,4) = up2(line_number)
           linenum_save = line_number

* communicate this to PLOT+
           setdash = .FALSE.
           DO 100 i = 1, 4
              IF (dashstyle(i) .NE. 0.) setdash = .TRUE.
  100	   CONTINUE
           IF (setdash) THEN
              WRITE (ppl_buff, 3400) line_number, ityp, (dashstyle(i),i=1,4)
 3400         FORMAT ('LINE ', I4, ' 0 ', I4, ' 0 0 ',4F7.3)
           ELSE 
              WRITE (ppl_buff, 3500) line_number, ityp
 3500         FORMAT ('LINE ', I4, ' 0 ', I4)
           ENDIF

	   CALL PPLCMD ( from, line, 0, ppl_buff, 1, 1 )
        ENDIF


	RETURN
	END

