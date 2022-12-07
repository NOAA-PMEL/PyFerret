	SUBROUTINE ZBUFFT(CHR,ICNT)
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
C**    @(#)zbufft.f	1.1    3/10/88
* 11/2022 *acm* Code cleanup: Modernize DO-loop syntax. 
C**
C**
C***********************************************************************
C**
C**                 PLOT+ Scientific Graphics System
C**
C***********************************************************************
C**
C**

	character buff*64,blnk*1,zfile*81
	character chr*(*)
C	
C	WRITES CHAR TO CURRENT ZETA BUFFER EVENTUALLY TO BE
C	WRITTEN TO LUN ATTACHED TO THE FILE
C	D. DENBO  JUNE 1983   UPDATED MARCH 1984
C

	include 'PLTCOM.DAT'
	data blnk/' '/
	J=1
	IF (NEWFIL) THEN

	len=lnblk(filez,81)
	jcnt=1
100	write(zfile,998)filez(:len),fzcnt
998	format(a,i3.3)
	fzcnt=fzcnt+1
	jcnt=jcnt+1
	if(jcnt.gt.101)stop 'can not open hp file'
	OPEN(UNIT=LUHP,FILE=zfile,STATUS='NEW',err=100)

	NEWFIL=.FALSE.
	IPT=1
	ENDIF
40	BUFF(IPT:IPT)=CHR(J:J)
	IPT=IPT+1
	J=J+1
	IF(IPT.GT.64)THEN
	WRITE(LUHP,999)BUFF
	IPT=1
999	FORMAT(A64)
	ENDIF
	IF(J.LE.ICNT)GOTO 40
	RETURN

30	WRITE(LUHP,999)BUFF
	CLOSE(LUHP)
	NEWFIL=.TRUE.
	IPT=1
	RETURN
	ENTRY HFLUSH
	DO 20 I=IPT,64
	  BUFF(I:I)=BLNK
20	CONTINUE
	GOTO 30
	END
