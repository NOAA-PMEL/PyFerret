      SUBROUTINE MARK(IMARK)
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

*  8/2001  acm  for PLOT/SYMBOL=DOT, call makedot
*      *acm* 10/01  remove VMS includes
* V73  3/18 *kms*  For clarity, append MD to makedot.cmn values IPT, XDOT, 
*                  and YDOT; make MDOT (99) a makedot.cmn parameter DOTSYMIM
* V76  6/20 *acm*  Issue 76: If drawing PyFerret symbols (symbol number > 88) 
*                  still need to execute the code setting the mode and the 
*                  xyzplt call at the end so line plot still gets drawn.
*
*
C**    @(#)mark.F	1.1    3/10/88
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

      INCLUDE 'makedot.cmn'

      CHARACTER IMODE*1
      LOGICAL LARGE
C
C     THE NEXT LINE IS MACHINE DEPENDENT AND MAY BE REPLACED BY
C     INTEGER MRKTAB(2,44),TABT(200)
      INTEGER*2 MRKTAB(2,44),TABT(200)
C
	include 'PLTCOM.DAT'

      DATA MRKTAB/    5,   9,  11,  15,  14,  15,  11,  12,
     *               26,  31,  32,  37,  38,  43,  44,  49,
     *                1,   5,  64,  67,   5,  15,  50,  54,
     *                1,   9,  55,  63,  15,  19,  21,  25,
     *               50,  53,  51,  54,  72,  77,  84,  98,
     *               18,  22,  11,  19,  64,  66,  68,  71,
     *               68,  70,  78,  83, 102, 106, 113, 118,
     *              119, 124, 125, 130, 131, 136, 105, 110,
     *              107, 112, 137, 139,  99, 106, 103, 108,
     *              140, 144, 140, 147, 156, 163, 148, 155,
     *              170, 183, 184, 189, 188, 193, 164, 169/
C
	DATA TABT/9,41,45,13,9,45,0,13,41,0,
     *25,29,0,11,43,29,11,25,43,11,25,29,11,43,29,18,27,34,0,27,
     *24,20,27,36,0,27,30,20,27,18,0,3,27,36,27,34,0,27,51,41,
     *13,45,9,41,4,2,16,32,50,52,38,22,4,9,29,41,9,13,25,45,
     *13,13,27,31,0,27,45,9,27,29,0,27,41,13,20,18,9,0,18,34,
     *0,20,36,0,45,36,34,41,19,35,0,21,17,33,37,21,19,35,33,17,
     *21,37,20,29,25,0,17,33,21,37,35,19,17,33,21,37,19,35,33,17,
     *21,19,43,0,37,33,21,37,25,12,44,0,42,10,0,17,37,26,30,0,
     *12,44,0,8,40,13,45,0,43,11,0,9,41,4,41,30,9,52,4,12,
     *20,21,13,12,0,9,45,0,33,41,42,34,33,14,44,10,0,9,41,0,
     *42,12,46,0,0,0,0,0,0,0/
C
C     CHECK THE DATA MARK CODE.  
C     For PLOT/SYMBOL=<name>, call makedot to call POLYMARK
C
      IF (IMARK .GE. DOTSYMIM) THEN
C         IMARK refers to a named symbol
          CALL makedot(IMARK - DOTSYMIM + 1)
      ENDIF

      IF(IMARK.LT.1 .OR. imark.EQ.DOTSYM) RETURN

C
C     DETERMINE THE SIZE OF THE DATA MARK AND ITS CODE.
C
      JMARK = (IMARK + 1) / 2
      LARGE = .FALSE.
      IF(2 * JMARK .EQ. IMARK)LARGE = .TRUE.
C
C     GET CONTROL INFORMATION FROM THE POINTER TABLE.
C
      IPOINT = MRKTAB(1,JMARK)
      ILAST = MRKTAB(2,JMARK)
C
C     SAVE THE CURRENT PLOTTING POSITION AS THE REFERENCE POINT.
C
      XREF = XTEMP
      YREF = YTEMP
C
C     SET VECTORS MODE.
C
      IMODE = PMODE
      PMODE = VECTOR
C
C     LIFT THE PEN BEFORE DRAWING THE FIRST LINE SEGMENT.
C
      PENF = .FALSE.
C
C     DRAW THE DATA MARK.
C
      IF (IMARK .LE. 88) THEN
      DO 10 IPT=IPOINT,ILAST
      IB = TABT(IPT)
C
C     CHECK FOR NO DISPLACEMENT.  THIS INDICATES THAT THE PEN IS
C     TO BE LIFTED BEFORE DRAWING THE NEXT LINE SEGMENT.
C
      IF(IB .EQ. 0)THEN
        PENF = .FALSE.
      ELSE
C
C     CALCULATE THE DISPLACEMENT FROM THE REFERENCE POINT.
C     FOR SPEED WE HAVE CODED THE NEXT TWO LINES USING THE NON STANDARD
C     AND AND RS FUNCTIONS.  HOWEVER THESE LINES COULD BE REPLACED BY
C
	movex = ISHFT(ib,-3) -3
	movey = IAND(ib,7) -3
C
C     IF THE DATA MARK IS LARGE, DOUBLE THE DISPLACEMENT.
C
        IF(LARGE)THEN
          MOVEX = 2 * MOVEX
          MOVEY = 2 * MOVEY
        ENDIF
C
C     CALCULATE THE NEW POINT.
C
        XTEMP = HMARK * FLOAT(MOVEX) + XREF
        YTEMP = HMARK * FLOAT(MOVEY) + YREF
C
C     PLOT TO THE POINT.
C
        CALL XYZPLT
        PENF = .TRUE.
      ENDIF
   10 CONTINUE

      ENDIF  ! mark .LT. 88
C
C     RESTORE THE CURRENT PLOTTING POSITION.
C
      XTEMP = XREF
      YTEMP = YREF
      PENF = .FALSE.
      CALL XYZPLT
C
C     RESTORE THE CURRENT PLOTTING MODE.
C
      PMODE = IMODE
      RETURN
C
      END

