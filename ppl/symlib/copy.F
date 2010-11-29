	SUBROUTINE COPY(FROM,TO,IER)
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
C**    @(#)copy.f	1.2    5/26/88
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
C**		CALL COPY(FROM,TO,IER)
C**
C**	Parameters:
C**		FROM -- File to be copied (char*80)
C**		TO   -- File to be created (char*80)
C**		IER  -- Error code
C**			0 = success
C**			10 = spawn error
C**			20 = wait error
C**			2 = error - unlikly results to be expected
C**			4 = severe error - possible PIP abort
C**
C**	Description:
C**		COPY executes the MCR PIP command to copy a file.
C**
C**	History:
C**	    Written: 
C**		22-JAN-86	by Donald W. Denbo
C**	    Modified:
C**		30-MAY-86	by Donald W. Denbo
C**		  Changes made for VAX/VMS version
C**
C**
C *kob* 04/07/96 passed in character strings had to be declared 
C                with "*(*)" for the linux port 

	CHARACTER FROM*(*),TO*(*)
c
c	rsx-11m
c
c	INTEGER*2 WAIT,ERROR,FLAG,STATUS,IER
c	CHARACTER CMD*80
c	WRITE(CMD,999)TO,FROM
c999	FORMAT('PIP ',A30,'=',A30)
c	CALL SQUISH(CMD,5,61)
c	WAIT=0
c	FLAG=0
c	CALL TISPWN(CMD,WAIT,ERROR,FLAG,STATUS)
c	IF(ERROR.EQ.0.AND.STATUS.EQ.1)THEN
c	    IER=0
c	ELSE
c	    IER=10*ERROR+STATUS
c	ENDIF
c
c	end rsx-11m
c
c	vax/vms
c
	CHARACTER CMD*200	
	INTEGER IER
	WRITE(CMD,999)FROM,TO
999	FORMAT('$COPY ',A80,' ',A80)
#ifndef unix
	CALL LIB$SPAWN(CMD)
#endif
	IER=0
	RETURN
	END


