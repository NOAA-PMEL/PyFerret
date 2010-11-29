      SUBROUTINE SQUISH(JSTRNG,JFIRST,NPOS)
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
C**    @(#)squish.f	1.1    3/10/88
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
C	Removes blank characters starting at position JFIRST
C	for the next NPOS characters and packs the characters
C	back into the original string.  Good for getting rid
C	of extra blanks.
C
C	JSTRNG -- character, input and output string
C	JFIRST -- INT, position to begin processing
C	NPOS -- INT, number of characters to process
C
C
C *kob* 04/07/96 passed in character strings had to be declared 
C                with "*(*)" for the linux port 

	CHARACTER*1 JSTRNG*(*),KLPAR,KRPAR,JBLNK,JCH
      LOGICAL REMOVE

      DATA KLPAR/'('/,KRPAR/')'/,JBLNK/' '/

      IF(NPOS.EQ.0)RETURN
      LAST=IABS(NPOS)+JFIRST-1
      REMOVE=.FALSE.
      IF(NPOS.LT.0)REMOVE=.TRUE.
      NEXT=JFIRST
      NPAREN=0

      DO 500 JPOS=JFIRST,LAST
	JCH=JSTRNG(JPOS:JPOS)
	JSTRNG(JPOS:JPOS)=JBLNK
      IF(JCH.NE.KLPAR)GO TO 100
      NPAREN=NPAREN+1
      GO TO 150

  100 IF(NPAREN.EQ.0)GO TO 200
      IF(JCH.NE.KRPAR)GO TO 300
      NPAREN=NPAREN-1
  150 IF(REMOVE)JCH=JBLNK
      GO TO 300

  200 IF(JCH.EQ.JBLNK)GO TO 500
  300 JSTRNG(NEXT:NEXT)=JCH
      NEXT=NEXT+1
  500 CONTINUE

      RETURN
      END
