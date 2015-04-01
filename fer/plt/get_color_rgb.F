	SUBROUTINE GET_COLOR_RGB (icolor, v1,v2,v3, status)

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
* return the name of color corresponding to the default pen # icolor

* V6.95 *acm* 3/15 Changes for PyFerret fonts. 


	INTEGER icolor, status
	REAL	v1, v2, v3

        include 'errmsg.parm'

! Default to black
	v1 = 0.
	v2 = 0.
	v3 = 0.

	IF (icolor .LT. 1  .OR. icolor.GT.7) GOTO 5100

	IF (icolor .EQ. 2) v1 = 100.  ! red

	IF (icolor .EQ. 3) v2 = 100.  ! green

	IF (icolor .EQ. 4) v3 = 100.  ! blue

	IF (icolor .EQ. 5) THEN       ! cyan
	   v2 = 100.
	   v3 = 100.
	ENDIF

	IF (icolor .EQ. 6) THEN       ! purple
	   v1 = 100.
	   v3 = 100.
	ENDIF

	IF (icolor .EQ. 7) THEN      ! white
	   v1 = 100.
	   v2 = 100.
	   v3 = 100.
	ENDIF

 5000	RETURN
 5100	status = ferr_invalid_command
	GOTO 5000 
	END

