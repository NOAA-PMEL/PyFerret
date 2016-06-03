	SUBROUTINE LON_LAT_FMT (idim, x_or_y)
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
* V7    *acm* 6/16 Ticket 2434: command qualifiers /DEGMINSEC= /HALFSPAC
*                  DEGMINSEC=1 for Deg-Min axis labels, =2 for Deg-Min-Sec. Default = 1
*                  /HALFSPAC adds a half space between the degree sign and E/W or N/S


* Set up PPLUS with info on longitude formatting with Deg-Min-Sec, and/or 
* adjusting spacing on labels with E,W,N, or S.

	include 'tmap_dims.parm'
	include 'ferret.parm'
	include 'plot_setup.parm'
	include 'xplot_setup.cmn'
	include 'xplot_state.cmn'

	INTEGER idim
	CHARACTER*(*) x_or_y

	IF ( idim .EQ. x_dim) THEN

	   ppl_buff = x_or_y//'FOR (dd)'
	   IF (dms .NE. 0) THEN
	      IF (dms .EQ. 1) ppl_buff = x_or_y//'FOR (dm)' 
	      IF (dms .EQ. 2) ppl_buff = x_or_y//'FOR (dms)'
	   ENDIF
	   CALL PPLCMD ( from, line, 0, ppl_buff, 1, 1 )

	   ppl_buff = x_or_y//'FOR (SPC0)'
	   IF (lonlatspace .NE. 0) ppl_buff = x_or_y//'FOR (SPC1)' 
	   CALL PPLCMD ( from, line, 0, ppl_buff, 1, 1 )
	ENDIF
	
	IF ( idim .EQ. y_dim) THEN
* latitude format for axis tic labling

	   ppl_buff = x_or_y//'FOR (dd)'
	   IF (dms .NE. 0) THEN
	      IF (dms .EQ. 1) ppl_buff = x_or_y//'FOR (dm)' 
	      IF (dms .EQ. 2) ppl_buff = x_or_y//'FOR (dms)'
	   ENDIF
	   CALL PPLCMD ( from, line, 0, ppl_buff, 1, 1 )

	   ppl_buff = x_or_y//'FOR (SPC0)'
	   IF (lonlatspace .NE. 0) ppl_buff = x_or_y//'FOR (SPC1)' 
	   CALL PPLCMD ( from, line, 0, ppl_buff, 1, 1 )
	ENDIF
	
	RETURN
	END
