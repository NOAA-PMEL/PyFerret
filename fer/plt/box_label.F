	SUBROUTINE BOX_LABEL( lab_type, string, xpos, ypos, xsize, ysize,
     .			      lab_ht, justify, xaxrel, yaxrel )

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
* create a PLOT+ label that fits within the designated box
* if necessary, use commas in the text as places where the string can be broken
* into 2 lines
* the position given refers to the lower left of the box

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system

* V200:	  2/2/90
*	 3/30/90 - use font info in calling SYMWID
* Unix/RISC port - 1/91 - cant use "//" with CHAR*(*)
* v552 *acm* 3/03 multi-line strings in titles and labels.
*                 If the string starts with @ do not set ASCII font
*                 for title and X and Y label; lets user set pen and font.
*                 Note can do colors and ASCII fonts, if set pen first: @P2@AC

	include	'ferret.parm'
	include	'plot_setup.parm'
	include 'xprog_state.cmn'
	include 'xrisc.cmn'

* calling argument declarations
	INTEGER lab_type, justify, xaxrel, yaxrel, k, numlines
	REAL	xpos, ypos, xsize, ysize, lab_ht
	CHARACTER string*(*), font*3

* internal variable declarations
        INTEGER lnbeg(250), lnend(250)
	INTEGER	TM_LENSTR1, slen, nfont, brk_pt, nex_brk_pt, 
     .          las_brk_pt, nlines
	REAL	SYMWID, ymid, width, ht, frac, frac_width, las_frac, xend,
     .          htsave

* initialize
	ymid = ypos + ysize*0.5
	ht = MIN( lab_ht, ysize )
	slen = TM_LENSTR1( string )
	brk_pt = INDEX( string(:slen), ',' )
	risc_buff = string			! for RISC/FORTRAN

* compute the width of the string in the font that it will be presented
        numlines = 1
        call tm_break_lines (string, lnbeg, lnend, numlines)

* select PLOT+ font

        nfont = 0

	IF ( .NOT.mode_font_ascii ) THEN
	   font = ' '
	ELSEIF ( lab_type .EQ. ppl_title ) THEN
	   font = '@AC'
           nfont = 3
	ELSE
	   font = '@AS'
           nfont = 3
	ENDIF

* Let user select the font if they want

        IF (string(1:1) .EQ. "@") THEN
           font = ' '
           nfont = 0
        ENDIF

        width = 0.
        DO 330 k = 1, numlines
           slen = lnend(k) - lnbeg(k) + 1
           risc_buff = string(lnbeg(k):lnend(k) )
	      width = MAX(width, SYMWID( ht, 
     .                           slen+nfont, font//risc_buff(:slen) ))
  330      CONTINUE

	slen = TM_LENSTR1( string )
	risc_buff = string			! for RISC/FORTRAN

* left or right justify or center
	xend = xpos
	IF ( xaxrel .EQ. lab_loc_absolute ) THEN
	   IF ( justify .EQ. ppl_right_just ) THEN
	      xend = xpos + xsize   
	   ELSEIF ( justify .EQ. ppl_centered ) THEN
	      xend = xpos + 0.5*xsize
	   ENDIF
	ENDIF

* titles and axis labels cannot be broken into multiple lines
	IF ( lab_type .EQ. ppl_movlab ) THEN
	   nlines = 2
	ELSE
	   nlines = 1
	ENDIF

* should we write the label in a single line ?
	IF ( nlines .LT. 2
     .	.OR. width .LE. xsize*1.8
     .	.OR. brk_pt.EQ.0 .OR. brk_pt.EQ.slen ) GOTO 500

* it's too big - we will try to write it as 2 lines
* ... break into pieces as close to halves as possible
	las_frac = 999.0
 100	IF ( .NOT.mode_font_ascii ) THEN
	   frac_width = SYMWID( ht,brk_pt,string(:brk_pt) )
	ELSEIF ( lab_type .EQ. ppl_title ) THEN
c	   frac_width = SYMWID( ht, brk_pt+3, '@AC'//risc_buff(:brk_pt)  )
	   frac_width = SYMWID( ht, brk_pt, risc_buff(:brk_pt)  )
	ELSE
c	   frac_width = SYMWID( ht, brk_pt+3, '@AS'//risc_buff(:brk_pt)  )
	   frac_width = SYMWID( ht, brk_pt, risc_buff(:brk_pt)  )
	ENDIF
	frac = frac_width / width
	IF ( ABS(0.5-frac) .LT. ABS(0.5-las_frac) ) THEN
	   nex_brk_pt = INDEX( string(brk_pt+1:), ',' ) + brk_pt
	   IF ( nex_brk_pt.EQ.brk_pt .OR. nex_brk_pt.EQ.slen ) GOTO 200
	   las_frac   = frac
	   las_brk_pt = brk_pt
	   brk_pt     = nex_brk_pt
	   GOTO 100

	ELSEIF ( frac .GT. las_frac ) THEN
	   brk_pt = las_brk_pt
	   frac = las_frac

	ENDIF

* don't break into 2 lines unless both lines are reasonably long
	IF ( frac.LT.0.2 .OR. frac.GT.0.8 ) GOTO 500

* WRITE STRING AS 2 LINES
 200	width = MAX(frac, 1.-frac) * width
	ht = MIN( ysize*0.4, ht*MIN(1.,xsize/width) )
* ... line 1
	CALL PUT_LABEL( lab_type, xend, ymid, justify, ht,
     .			string(:brk_pt), xaxrel, yaxrel )
* ... line 2
	CALL PUT_LABEL( lab_type, xend, ypos, justify, ht,
     .			string(brk_pt+1:), xaxrel, yaxrel )
	RETURN


* WRITE STRING AS 1 LINE
 500	htsave = ht

* Adjust character size for the length of the longest line
        ht = ht * MIN(1., 1.25* xsize/width)

* Adjust character size for the number of lines
        IF (lab_type .NE. ppl_movlab) THEN
           IF  (numlines .GE. 4 .AND. ht .EQ. htsave) ht = 0.5*ht
           IF  (numlines .EQ. 3 .AND. ht .EQ. htsave) ht = 0.65*ht
           IF  (numlines .EQ. 2 .AND. ht .EQ. htsave) ht = 0.8*ht
        ENDIF

	CALL PUT_LABEL( lab_type, xend, ymid-0.5*ht, justify, ht,
     .			string(:slen), xaxrel, yaxrel )
	RETURN
	END
