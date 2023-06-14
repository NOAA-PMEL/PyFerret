	SUBROUTINE GETLIN(LINE,ILEN)
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
C**    @(#)getlin.F	1.2    5/26/88
C**
C**
C***********************************************************************
C**
C**		    PLOT+ Scientific Graphics System
C**
C***********************************************************************
C**
C**
C**	Subprogram called:
C**		CALL GETLIN(LINE,ILEN)
C**
C**	Parameters:
C**		LINE -- Line from command input device   (char*2048)
C**		ILEN -- Length of LINE
C**
C**	Description:
C**		Gets the next line from the command input device.
C**
C**	History:
C**	    Written: 
C**		21-DEC-85	by Donald W. Denbo
C**	    Modified:
C**
C*** v552 *acm* 3/03 Longer strings: changes for multi-line labels
C**
	INCLUDE 'CMDCOM.INC'
	INCLUDE 'SYSTEM.INC'

C *kob* 04/07/96 passed in character strings had to be declared 
C                with "*(*)" for the linux port 

	CHARACTER PMPT*30,LINE*(*),IN*2048
	IP=INP
	PMPT=PRMPT
	K=1
100	IN=' '

	IF(CMDFIL(1:8).EQ.'/dev/tty')THEN
	    CALL CHARIN(IN,2048,PMPT,IP)
	ELSE
	    READ(CMDLUN,999,END=901)IN
999	    FORMAT(A)
	    LINCNT=LINCNT+1
	ENDIF
	ILEN=LNBLK(IN,lenlab)
	LINE(K:K+ILEN-1)=IN(1:ILEN)
	IF(IN(ILEN:ILEN).EQ.'-')THEN
	    K=K+ILEN-1
C
C	SINCE LF IS FIRST CHARACTER
C
	    PMPT=PRMPT
	    PMPT(2:2)='_'
	    PMPT(3:)=PRMPT(2:)
	    IP=INP+1
	    GOTO 100
	ENDIF
	ILEN=K+ILEN-1
	RETURN
C
C	EOF READ GO UP ONE LEVEL IN COMMAND FILE STRUCTURE
C
901	CALL ATEND
	IF(DONEF)RETURN
	GOTO 100
	END
