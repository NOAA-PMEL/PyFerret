	SUBROUTINE LSTSYM(STR,LABEL,NC,ILST,IER)
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
C**    @(#)lstsym.F	1.2    5/26/88
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
C**		CALL LSTSYM(STR,LABEL,NC,ILST,IER)
C**
C**	Parameters:
C**		STR -- Symbol name (char*30)
C**		LABEL -- Symbol value (char*255)
C**		NC -- length of LABEL
C**		ILST -- Initially run with ILST = 0
C**			Output is a counter
C**		IER -- error code
C**			0 = success
C**			1 = error or end of file
C**
C**	Description:
C**		Gets the symbol values one by one until all have been
C**		accessed.
C**
C**	History:
C**	    Written:
C**		15-DEC-1985	by Donald W. Denbo
C**	    Modified:
C**
C**
C *kob* 04/07/96 passed in character strings had to be declared 
C                with "*(*)" for the linux port 

#ifdef unix
	INCLUDE 'cmdinc/SYMKEY.INC'
#else
	INCLUDE 'cmdinc:SYMKEY.INC'
#endif
	CHARACTER STR*(*),LABEL*(*)
	IF(ILST.EQ.0)THEN
	    call dbmfirstkey(keyln1,str,ilen,ier)
	else
	    call dbmnextkey(keyln1,str,ilen,ier)
	endif
	if(ier.ne.0)goto 2000
	call dbmfetch(keyln1,str,ilen,label,nc,ier)
	str(ilen+1:)=' '
	ILST=ILST+1
	IER=0
	RETURN
2000	IER=1
	RETURN
	END
