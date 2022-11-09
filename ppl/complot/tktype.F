      SUBROUTINE TKTYPE(ICODE)
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
C**    @(#)tktype.f	1.1    3/10/88
C**
C**
C***********************************************************************
C**
C**                 PLOT+ Scientific Graphics System
C**
C***********************************************************************
C**
C**
C
C     CALL TKTYPE(ICODE)
C
C     COMPLOT ASSUMES THAT A TEKTERMINAL IS A 4010.  IF A DIF-
C     FERENT TYPE OF TEKTRONIX GRAPHICS TERMINAL IS USED INSTEAD,
C     THIS SUBROUTINE MUST BE CALLED TO MAKE THE APPROPRIATE
C     CHANGES TO CONTROL INFORMATION.
C
C
	include 'PLTCOM.DAT'
C
C     CHECK THE TEKTERMINAL MODEL.
C
	TTYPE=ICODE
C
      IF(ICODE .EQ. 4010)GO TO 40
      IF(ICODE .EQ. 4014)GO TO 50
      IF(ICODE .EQ. 4107)GO TO 50
      IF(ICODE .EQ. 4115)GO TO 50
      IF(ICODE .EQ. 4051)GO TO 70
      IF(ICODE .EQ. 4052)GO TO 70
      IF(ICODE .EQ. 4662)GO TO 170
      RETURN
C
C     CHANGE CONTROL INFORMATION.
C
   40 TWIDTH = 7.5
      THIGHT = 5.625
      TSCLA = 1023.0
      TSCLB = 779.0
      GO TO 250
C
C     CHANGE CONTROL INFORMATION.
C
   50 TWIDTH = 7.8448
      THIGHT = 6.0
      TSCLA = 4095.0
      TSCLB = 3132.0
      GO TO 250
C
C     CHANGE CONTROL INFORMATION.
C
   70 TWIDTH = 7.875
      THIGHT = 6.0
      TSCLA = 1023.0
      TSCLB = 779.0
      GO TO 250
  170 IDTK = 67
      TTYPE = -4662
C
C     CHANGE CONTROL INFORMATION.
C
      TWIDTH = 15.0
      THIGHT = 10.0
      TSCLA = 4095.0
      TSCLB = 2731.0
  250 FACTOR = 1000.0
C
C     DEFINE THE DEFAULT DIMENSIONS OF THE PLOTTING AREA.
C
c
c	keep from resizing the display
c
c      CALL SIZE(TWIDTH,THIGHT)
      RETURN
C
      END
