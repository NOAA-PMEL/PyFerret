      SUBROUTINE SCALE(XSCL,YSCL,ABIAS,BBIAS,XLO,YLO)
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
C**    @(#)scale2.F	1.1    3/10/88
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
C     CALL SCALE(XSCL,YSCL,ABIAS,BBIAS,XLO,YLO)
C
C     THIS SUBROUTINE DEFINES SCALE FACTORS AND DISPLACEMENTS
C     FOR SUBSEQUENT PLOTTING.  THE SCALE FACTORS DETERMINE THE
C     RELATIVE SIZE OF THE DISPLAY, AND THE DISPLACEMENTS DETER-
C     MINE ITS LOCATION WITHIN THE PLOTTING AREA.
C
C     THE VARIOUS PARAMETERS AND THEIR DEFINITIONS ARE LISTED
C     BELOW:
C        XSCL   - SCALE FACTOR TO CONVERT FROM USER'S UNITS TO
C                 INCHES (OR MM) ON THE X-AXIS.
C        YSCL   - SCALE FACTOR TO CONVERT FROM USER'S UNITS TO
C                 INCHES (OR MM) ON THE Y-AXIS.
C        ABIAS  - HORIZONTAL DISPLACMENT OF THE DISPLAY IN IN-
C                 CHES (OR MM).
C        BBIAS  - VERTICAL DISPLACEMENT OF THE DISPLAY IN INCHES
C                 (OR MM).
C        XLO    - MINIMUM X-VALUE TO BE PLOTTED IN USER'S UNITS.
C        YLO    - MINIMUM Y-VALUE TO BE PLOTTED IN USER'S UNITS.
C
C
	INCLUDE 'PLTCOM.DAT'
	SAVE XLOW,YLOW
C
C     CONVERT THE SCALE FACTORS TO INTERNAL INCREMENTS.
C
      XSCALE = FACTOR * XSCL
      YSCALE = FACTOR * YSCL
C
C     CALCULATE THE LOCATION OF THE ORIGIN.
C
	XLOW = XLO * XSCALE
	YLOW = YLO * YSCALE
C
      AORIGN = FACTOR * ABIAS - XLOW
      BORIGN = FACTOR * BBIAS - YLOW
C
	IF(ROTAT)THEN
	    T=AORIGN
	    AORIGN=ASIZE-BORIGN
	    BORIGN=T
	ENDIF
C
      RETURN
	ENTRY INQSCL(XSCL,YSCL,ABIAS,BBIAS,XLO,YLO)
C
C	INQUIRE SCALE
C
	XSCL = XSCALE / FACTOR
	YSCL = YSCALE / FACTOR
C
	XLO = XLOW / XSCALE
	YLO = YLOW / YSCALE
C
	IF(ROTAT)THEN
	    TA=BORIGN
	    TB=ASIZE-AORIGN
	ELSE
	    TA=AORIGN
	    TB=BORIGN
	ENDIF
	ABIAS = (TA + XLOW) / FACTOR
	BBIAS = (TB + YLOW) / FACTOR
	RETURN
C
      END
