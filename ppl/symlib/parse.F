	SUBROUTINE PARSE(LINE,ILEN,ICOM,LAB,N,IST)
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
C**    @(#)parse.f	1.2    5/26/88
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
C**		CALL PARSE(LINE,ILEN,ICOM,LAB,N)
C**
C**	Parameters:
C**		LINE -- Line to be parsed   (char*2048)
C**		ILEN -- Length of ILEN
C**		ICOM -- Command from LINE   (char*2048)
C**		LAB -- rest of LINE  (char*2048)
C**		N -- Length of LAB
C**		IST -- Start position of LAB
C**
C**	Description:
C**		Remove the command from LINE and places the rest of LINE
C**		in LAB.
C**
C**	History:
C**	    Written: 
C**		22-DEC-85	by Donald W. Denbo
C**	    Modified:
C**
C*
C *kob* 04/07/96 passed in character strings had to be declared 
C                with "*(*)" for the linux port 
C * v552 *acm* 3/03 Longer strings: changes for multi-line labels

	CHARACTER ICOM*(*),LAB*(*),LINE*(*)
	ICOM=' '
	LAB=' '
C
C	Search for the fist space or comma
C
	N=0
	IST=0
	ISPC=INDEX(LINE,' ')
	IF(ISPC.EQ.0)ISPC=256
	ICMA=INDEX(LINE,',')
	IF(ICMA.EQ.0)ICMA=256
	I=MIN0(ISPC,ICMA,256,ILEN+1)
	ICOM=LINE(:I-1)
	CALL UPPER(ICOM,2048)
100	I=I+1
	IF(I.GT.ILEN)RETURN
	IF(LINE(I:I).EQ.' ')GOTO 100
	IF(LINE(I:I).EQ.'"')I=I+1
	LAB=LINE(I:)
	IST=I
	N=ILEN-I+1
	IF(LAB(N:N).EQ.'"')THEN
	    LAB(N:N)=' '
	    N=N-1
	ENDIF
	RETURN
	END
