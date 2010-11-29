	SUBROUTINE PUT_CMND_KEY( cmnd, string, xll, yll, xsize, ysize, lab_ht )

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
* create a key for a plot using PLOT+ movable labels
* that will document a SHADE, CONTOUR or VECTOR overlay
* (xsize,ysize) is the size of the "key box" (including the line specimen
* (xll,yll) is the lower left corner of the box

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system

* V200:	  2/9/90

#ifdef unix
	include	'ferret.parm'
	include	'plot_setup.parm'
	include	'command.parm'
	include	'xplot_setup.cmn'
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
