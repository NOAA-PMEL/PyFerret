	SUBROUTINE UPPER(STRING,NC)
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
C**    @(#)upper.F	1.1    3/10/88
C**
C**
C***********************************************************************
C**
C**                 PLOT+ Scientific Graphics System
C**
C***********************************************************************
C**
C**
c
c	converts the string to upper case
C
c	STRING -- character*80, input string
C	nc -- length of string
C
C *kob* 04/07/96 passed in character strings had to be declared 
C                with "*(*)" for the linux port 

#ifdef unix
	CHARACTER STRING*(*),CHR(26)
	DATA CHR/'A','B','C','D','E','F','G','H','I',
     1	'J','K','L','M','N','O','P','Q','R','S','T',
     2	'U','V','W','X','Y','Z'/
	AA=ICHAR('a')
	DO 100 I=1,NC
	IF(STRING(I:I).GE.'a'.AND.STRING(I:I).LE.'z')THEN
	K=ICHAR(STRING(I:I))-AA+1
	STRING(I:I)=CHR(K)
	ENDIF
100	CONTINUE
#else
	CHARACTER STRING*80
	INTEGER*4 STR$UPCASE,ISTAT
	ISTAT = STR$UPCASE(STRING(1:NC),STRING(1:NC))
	IF(.NOT.ISTAT)CALL SYS$EXIT(%VAL(ISTAT))
#endif
	RETURN
	END
