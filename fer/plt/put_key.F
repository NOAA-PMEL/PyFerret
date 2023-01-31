	SUBROUTINE PUT_KEY( line_num, string, xpos, ypos, xsize, ysize,
     .			      lab_ht, style, xaxrel, yaxrel )

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
* create a key for a plot using a PLOT+ movable label and an ALINE
* (xsize,ysize) is the size of the "key box" (including the line specimen
* (xpos,ypos) is the lower left corner of the box
* as of V2.00 of FERRET the "style" is always line specimen to left of text

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system

* V200:	  2/2/90
* V301:  11/5/93 - define PLOT+ symbol "LABKEY" with the key string
* V68  *acm* 1/12  changes for double-precision ferret, single-precision pplus.
*       *acm* 3/12 6D Ferret (common uses nferdims in tmap_dims.parm)

        IMPLICIT NONE
	include 'tmap_dims.parm'
	include	'ferret.parm'
	include	'plot_setup.parm'
	include	'xplot_setup.cmn'
	include 'axis_inc.decl'	! axis lengths
	include 'AXIS.INC'	! axis lengths
	include 'xrisc.cmn'

* calling argument declarations
	INTEGER line_num, style, xaxrel, yaxrel
	REAL	xpos, ypos, xsize, ysize, lab_ht
	CHARACTER string*(*)

* internal variable declarations
	REAL	xmid, ymid, xll, yll, dxlen, dylen

	dxlen = xlen
	dylen = ylen
* adjust positions relative to axis lengths if requested
	xll = xpos
	yll = ypos
	IF ( xaxrel .EQ. lab_loc_axlen_plus ) THEN
	   xll = xll + dxlen
	ELSEIF ( xaxrel .EQ. lab_loc_axlen_times ) THEN
	   xll = xll * dxlen
	ENDIF
	IF ( yaxrel .EQ. lab_loc_axlen_plus ) THEN
	   yll = yll + dylen
	ELSEIF ( yaxrel .EQ. lab_loc_axlen_times ) THEN
	   yll = yll * dylen
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
	risc_buff = string
	CALL PPLCMD ( from, line, 0, 'SET LABKEY '//risc_buff, 1, 1 )

	RETURN
	END
