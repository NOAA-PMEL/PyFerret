	SUBROUTINE GET_COLOR_NAME (rd,gr,bl, cname, len)

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
* Ansley Manke 4/2015 
*
* return the name of color of the default pen # that corresponds
* to the given red, green, blue, only if they match the color setting.

        IMPLICIT NONE
	INTEGER TM_LENSTR, len
	CHARACTER*(*) cname
	REAL	rd, gr, bl

	cname = ' '
	IF (rd.EQ.    0. .AND. gr.EQ.    0. AND. bl.EQ.    0.) cname = 'black'
	IF (rd.EQ.  100. .AND. gr.EQ.    0. AND. bl.EQ.    0.) cname = 'red'
	IF (rd.EQ.    0. .AND. gr.EQ.  100. AND. bl.EQ.    0.) cname = 'green'
	IF (rd.EQ.    0. .AND. gr.EQ.    0. AND. bl.EQ.  100.) cname = 'blue'
	IF (rd.EQ.    0. .AND. gr.EQ.  100. AND. bl.EQ.  100.) cname = 'lightblue'
	IF (rd.EQ.  100. .AND. gr.EQ.  100. AND. bl.EQ.    0.) cname = 'purple'
	IF (rd.EQ.  100. .AND. gr.EQ.  100. AND. bl.EQ.  100.) cname = 'white'

	len = TM_LENSTR(cname)

	END

