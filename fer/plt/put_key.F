	SUBROUTINE PUT_KEY( line_num, string, xpos, ypos, xsize, ysize,
     .			      lab_ht, style, xaxrel, yaxrel )

* create a key for a plot using a PLOT+ movable label and an ALINE
* (xsize,ysize) is the size of the "key box" (including the line specimen
* (xpos,ypos) is the lower left corner of the box
* as of V2.00 of FERRET the "style" is always line specimen to left of text

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system

* V200:	  2/2/90
* V301:  11/5/93 - define PLOT+ symbol "LABKEY" with the key string

#ifdef unix
	include	'ferret_cmn/ferret.parm'
	include	'ferret_cmn/plot_setup.parm'
	include	'ferret_cmn/xplot_setup.cmn'
	include 'tmap_pplv11inc/axis_inc.decl'	! axis lengths
	include 'pplv11inc/AXIS.INC'	! axis lengths
#ifdef NO_PASSED_CONCAT
	include 'ferret_cmn/xrisc.cmn'
#endif
#else
	INCLUDE	'FERRET_CMN:FERRET.PARM'
	INCLUDE	'FERRET_CMN:PLOT_SETUP.PARM'
	INCLUDE	'FERRET_CMN:XPLOT_SETUP.CMN'
	INCLUDE 'TMAP_PPLV11INC:AXIS_INC.DECL'	! axis lengths
	INCLUDE 'PPLV11INC:AXIS.INC'	! axis lengths
#endif

* calling argument declarations
	INTEGER line_num, style, xaxrel, yaxrel
	REAL	xpos, ypos, xsize, ysize, lab_ht
	CHARACTER string*(*)

* internal variable declarations
	REAL	xmid, ymid, xll, yll

* adjust positions relative to axis lengths if requested
	xll = xpos
	yll = ypos
	IF ( xaxrel .EQ. lab_loc_axlen_plus ) THEN
	   xll = xll + xlen
	ELSEIF ( xaxrel .EQ. lab_loc_axlen_times ) THEN
	   xll = xll * xlen
	ENDIF
	IF ( yaxrel .EQ. lab_loc_axlen_plus ) THEN
	   yll = yll + ylen
	ELSEIF ( yaxrel .EQ. lab_loc_axlen_times ) THEN
	   yll = yll * ylen
	ENDIF

* x,y coords. of line specimen right end
	xmid = xll + xsize*key_line_frac
	ymid = yll + ysize*0.5

* instruct PLOT+ to draw line specimen: "ALINE/NOUSER 3 2.5 -1.5 3.0 -1.5 ON"
	WRITE ( ppl_buff, 3000) line_num, xll, ymid, xmid, ymid
 3000	FORMAT( 'ALINE/NOUSER ',I3,4F9.3,' ON')
	CALL PPLCMD ( from, line, 0, ppl_buff, 1, 1 )

* instruct PLOT+ to write the key text
* ... x coord of text left end
	xmid = xmid + xsize*0.05
	CALL BOX_LABEL( ppl_movlab,
     .			string, xmid, yll, xsize*(.95-key_line_frac), ysize,
     .			lab_ht, ppl_left_just, xaxrel, yaxrel )

* define a PLOT+ symbol LABKEY
#ifdef NO_PASSED_CONCAT
	risc_buff = string
	CALL PPLCMD ( from, line, 0, 'SET LABKEY '//risc_buff, 1, 1 )
#else
	CALL PPLCMD ( from, line, 0, 'SET LABKEY '//string, 1, 1 )
#endif

	RETURN
	END
