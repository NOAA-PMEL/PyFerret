	SUBROUTINE BOX_LABEL( lab_type, string, xpos, ypos, xsize, ysize,
     .			      lab_ht, justify, xaxrel, yaxrel )

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

#ifdef unix
	include	'ferret_cmn/ferret.parm'
	include	'ferret_cmn/plot_setup.parm'
	include 'ferret_cmn/xprog_state.cmn'
	include 'ferret_cmn/xrisc.cmn'
#else
	INCLUDE	'FERRET_CMN:FERRET.PARM'
	INCLUDE	'FERRET_CMN:PLOT_SETUP.PARM'
	INCLUDE 'FERRET_CMN:XPROG_STATE.CMN'
	INCLUDE 'FERRET_CMN:XRISC.CMN'
#endif

* calling argument declarations
	INTEGER lab_type, justify, xaxrel, yaxrel
	REAL	xpos, ypos, xsize, ysize, lab_ht
	CHARACTER string*(*)

* internal variable declarations
	INTEGER	TM_LENSTR1, slen, brk_pt, nex_brk_pt, las_brk_pt, nlines
	REAL	SYMWID, ymid, width, ht, frac, frac_width, las_frac, xend

* initialize
	ymid = ypos + ysize*0.5
	ht = MIN( lab_ht, ysize )
	slen = TM_LENSTR1( string )
	brk_pt = INDEX( string(:slen), ',' )
	risc_buff = string			! for RISC/FORTRAN

* compute the width of the string in the font that it will be presented
	IF ( .NOT.mode_font_ascii ) THEN
	   width = SYMWID( ht, slen, string )
	ELSEIF ( lab_type .EQ. ppl_title ) THEN
	   width = SYMWID( ht, slen+3, '@AC'//risc_buff(:slen) )
	ELSE
	   width = SYMWID( ht, slen+3, '@AS'//risc_buff(:slen) )
	ENDIF

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
	   frac_width = SYMWID( ht, brk_pt+3, '@AC'//risc_buff(:brk_pt)  )
	ELSE
	   frac_width = SYMWID( ht, brk_pt+3, '@AS'//risc_buff(:brk_pt)  )
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
 500	ht = ht * MIN(1., xsize/width)
	CALL PUT_LABEL( lab_type, xend, ymid-0.5*ht, justify, ht,
     .			string(:slen), xaxrel, yaxrel )
	RETURN
	END
