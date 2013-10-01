	SUBROUTINE CHECK_2D_SIZE (mx, my, err_string, status)
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
C**    @(#)check_2d_size.F	1.1    9/16/12
C**
C**
C***********************************************************************
C**
C**                 PLOT+ Scientific Graphics System
C**
C***********************************************************************
C**
C**
c V685  *acm* 9/13 Fixing ticket 2092. Check that the pplus internal storage
c                  for coordinate data will be enough for the grid and if not
c                  return an error message. 
c
c	mx	dimensioned x size of z(mx,my)
c	my	dimensioned y size of z
c       status  returned 
C
C
C
	INTEGER		mx, my, status
	CHARACTER	err_string*128

        include 'errmsg.parm'
	INCLUDE 'ZGRIDD.INC'

	LOGICAL		ITS_CURVILINEAR

	status = ferr_ok

* This check is only needed for rectilinear grids

        IF ( ITS_CURVILINEAR() ) RETURN

	IF (mx .GT. IZSE) THEN
	   err_string = 
     .      'grid in horizontal direction too large for 2D graphics call'
	   status = ferr_out_of_range
	ENDIF

	IF (my .GT. IZSE) THEN
	   err_string = 
     .      'grid in vertical direction too large for 2D graphics call'
	   status = ferr_out_of_range
	ENDIF

	RETURN
	END
