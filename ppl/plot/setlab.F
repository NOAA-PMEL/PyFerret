	SUBROUTINE SETLAB(XF,YF,LTT,LTB,LYL,LYR,LXT,LXB,IFX,IFY)
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
C**    @(#)setlab.F	1.1    3/10/88
C**
C**
C***********************************************************************
C**
C**                 PLOT+ Scientific Graphics System
C**
C***********************************************************************
C**
C**
C *kob* 04/07/96 passed in character strings had to be declared 
C                with "*(*)" for the linux port 

#ifdef unix
	INCLUDE 'AXIS.INC'
	INCLUDE 'TAXIS.INC'
	INCLUDE 'AXISL.INC'
#else
        INCLUDE 'pplinc:AXIS.INC'
        INCLUDE 'pplinc:TAXIS.INC'
        INCLUDE 'pplinc:AXISL.INC'
#endif
	CHARACTER IFX*(*),IFY*(*)

	EXTERNAL scale


C     SETUP AXIS 
	LTT=0 
	LTB=0 
	LYL=0 
	LYR=0 
	LXT=0 
	LXB=0 
	IF(LABELY.EQ.-1)THEN 
	    LYL=-1 
	ELSE IF(LABELY.EQ.1)THEN 
	    LYR=1 
	END IF 
	IF(LABELX.EQ.-1)THEN 
	    LXB=-1 
	ELSE IF(LABELX.EQ.1)THEN 
	    LXT=1 
	END IF 
	IF(LABELT.LT.0)THEN 
	    LTB=LABELT 
	ELSE IF(LABELT.GT.0)THEN 
	    LTT=LABELT 
	END IF 
	CALL SCALE(XF,YF,XORG,YORG,XLO,YLO) 
	IF(IFRX.EQ.' ')THEN 
	    CALL FRMT(XLO,XHI,NSIGX,IFX) 
	ELSE 
	    IFX=IFRX
	ENDIF 
	IF(IFRY.EQ.' ')THEN 
	    CALL FRMT(YLO,YHI,NSIGY,IFY) 
	ELSE 
	    IFY=IFRY
	ENDIF 
	RETURN
	END
