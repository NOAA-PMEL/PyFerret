	SUBROUTINE PUT_CMND_KEY( cmnd, string, xll, yll, xsize, ysize, lab_ht )

* create a key for a plot using PLOT+ movable labels
* that will document a SHADE, CONTOUR or VECTOR overlay
* (xsize,ysize) is the size of the "key box" (including the line specimen
* (xll,yll) is the lower left corner of the box

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system

* V200:	  2/9/90

#ifdef unix
	include	'ferret_cmn/ferret.parm'
	include	'ferret_cmn/plot_setup.parm'
	include	'ferret_cmn/command.parm'
	include	'ferret_cmn/xplot_setup.cmn'
#else
	INCLUDE	'FERRET_CMN:FERRET.PARM'
	INCLUDE	'FERRET_CMN:PLOT_SETUP.PARM'
	INCLUDE	'FERRET_CMN:COMMAND.PARM'
	INCLUDE	'FERRET_CMN:XPLOT_SETUP.CMN'
#endif

* calling argument declarations
	INTEGER cmnd
	REAL	xll, yll, xsize, ysize, lab_ht
	CHARACTER string*(*)

* internal variable declarations
	REAL	ymid, xmid
	CHARACTER	cmnd_name*8

	ymid = yll + ysize*0.5
	IF ( cmnd .EQ. cmnd_vector ) THEN
	   xmid = xsize*(1.-key_line_frac)
	   WRITE ( ppl_buff, '(2F6.1)' ) xll+(1.05*xmid), ymid
	   CALL PPLCMD (' ',' ',0,'VECKEY/NOUSER '//ppl_buff, 1, 1 )
	   CALL BOX_LABEL( ppl_movlab, string,
     .			   xll, yll, xmid , ysize,
     .			   lab_ht, ppl_right_just,
     .			   lab_loc_absolute, lab_loc_absolute )
	ELSE
	   IF ( cmnd .EQ. cmnd_contour ) THEN
	      cmnd_name = 'CONTOUR:'
	   ELSE
	      cmnd_name = 'SHADE:  '
	   ENDIF
	   xmid = xsize*key_line_frac
	   CALL BOX_LABEL( ppl_movlab, cmnd_name,
     .			   xll, yll, .95*xmid, ysize,
     .			   lab_ht, ppl_right_just,
     .			   lab_loc_absolute, lab_loc_absolute )
	   CALL BOX_LABEL( ppl_movlab, string,
     .			   xll+xmid, yll, xsize-xmid, ysize,
     .			   lab_ht, ppl_left_just,
     .			   lab_loc_absolute, lab_loc_absolute )

	ENDIF

	RETURN
	END
