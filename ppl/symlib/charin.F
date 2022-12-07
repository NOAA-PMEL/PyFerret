	SUBROUTINE CHARIN(STR,IN,PRMT,INPL)
C** 
C**    @(#)charin.F	1.2    5/26/88
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
C**		CALL CHARIN(STR,IN,PRMP,INP)
C**
C**	Parameters:
C**		STR -- Output string (char*IN or byte(IN))
C**		IN -- Maximum input string length
C**		PRMP -- Prompt for terminal input
C**		INP -- Prompt lenght in characters
C**
C**	Description:
C**		CHARIN prompts the user for input from the terminal.
C**
C**	History:
C**	    Written: 
C**		15-DEC-1985	by Donald W. Denbo
C**	    Modified:
C**		30-MAY-1986	by Donald W. Denbo
C**		  Changes made for VAX/VMS version
C**
C**
C     Mod *JD* 3.5.91 problems with buffered terminal output in unix
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
*     Mod *jd* 8.30.93 to handle FORTRAN read error you get w XGKS if the 
*	window is clicked with the mouse.
C *kob* 04/07/96 passed in character strings had to be declared 
C                with "*(*)" for the linux port 
* v552 *acm* 3/03 Longer strings: changes for multi-line labels
c
C  10/2022 *acm* Code cleanup: Unclear if anything is called here
C                but fix the possible infinite loop at the READ for str()

	include 'CMDCOM.INC'
	include 'LUNITS.INC'

	character str*(*),prmt*(*)
	write(LTTOUT,999)prmt(:inpl)

999	format(a)
	call flush(LTTOUT)

	do in = 1, 2048
	   if (in .gt. 2048) exit
	   read(CMDLUN,998,err=100,end=100)str(:in)	   
	enddo

100	continue
998	format(a)

	RETURN
	END
