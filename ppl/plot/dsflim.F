	SUBROUTINE DSFLIM
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
C**    @(#)dsflim.F	1.1    3/10/88
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
C	Subroutine called by REDSF to keep track of the maximum and
C	minimum times for the data
C
#ifdef unix
	INCLUDE 'BIBO.COM'
	INCLUDE 'AXISL.INC'
	INCLUDE 'TAXIS.INC'
	INCLUDE 'CMRD.INC'
	INCLUDE 'DATA.INC'
#else
        INCLUDE 'pplinc:BIBO.COM'
        INCLUDE 'pplinc:AXISL.INC'
        INCLUDE 'pplinc:TAXIS.INC'
        INCLUDE 'pplinc:CMRD.INC'
        INCLUDE 'pplinc:DATA.INC'
#endif
C
	INTEGER*4 MTMIN,TSTART,TSTOP,TIME
	REAL STOP(3),TEMP(3)
C
	IF(LINEN.EQ.1)THEN
	    CALL MTMDY(P(STIME),IM,ID,IY)
	    WRITE(ITMIN,999)IY,IM,ID,IFIX(P(STIME+2))
999	    FORMAT(3I2.2,I4.4)
	    IF(IAUTOT.EQ.1)ITSTRT=ITMIN
	    TSTOP=MTMIN(P(STIME))+(P(NPTS)-1)*P(SRATE)
	    CALL MINMT(TSTOP,STOP)
	    CALL MTMDY(STOP,IM,ID,IY)
	    WRITE(ITMAX,999)IY,IM,ID,IFIX(STOP(3))
	ELSE
	    TSTART=MTMIN(P(STIME))
	    TSTOP=TSTART+(P(NPTS)-1)*P(SRATE)
	    CALL MINMT(TSTOP,STOP)
C
C	FIND TIME RANGE MINIMUM
C
	    READ(ITMIN,999)IY,IM,ID,IHR
	    CALL MDYMT(TEMP,IM,ID,IY)
	    TEMP(3)=IHR
	    TIME=MTMIN(TEMP)
	    IF(TSTART.LT.TIME)THEN
		CALL MTMDY(P(STIME),IM,ID,IY)
		WRITE(ITMIN,999)IY,IM,ID,IFIX(P(STIME+2))
	    ENDIF
	    READ(ITMAX,999)IY,IM,ID,IHR
	    CALL MDYMT(TEMP,IM,ID,IY)
	    TEMP(3)=IHR
	    TIME=MTMIN(TEMP)
	    IF(TSTOP.GT.TIME)THEN
		CALL MTMDY(STOP,IM,ID,IY)
		WRITE(ITMAX,999)IY,IM,ID,IFIX(STOP(3))
	    ENDIF
	ENDIF
	RETURN
	END
