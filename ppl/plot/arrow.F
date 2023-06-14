	SUBROUTINE ARROW(XL,YL,XL2,YL2,IAT,SHFT,XMN,CS,SN,XF,YF)
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
C**    @(#)arrow.F	1.1    3/10/88
C**
C**
C***********************************************************************
C**
C**                 PLOT+ Scientific Graphics System
C**
C***********************************************************************
C**
C**
	INCLUDE 'LABCOM.INC'
	IF(IAT.EQ.0)RETURN
	IF(XL+SHFT*CS/XF.GE.XL2)THEN
	    AX=-HLABS*.5
	    AY=HLABS*.5
	ELSE IF(XL+(XMN+SHFT*CS)/XF.LE.XL2)THEN
	    AX=(HLABS*.5+XMN)
	    AY=HLABS*.5
	ELSE IF(YL+SHFT*SN/YF.GT.YL2)THEN
	    AY=-HLABS*.5
	    AX=XMN*.5
	ELSE
	    AY=HLABS*1.5
	    AX=XMN*.5
	ENDIF
	XLT=XL+((SHFT+AX)*CS-AY*SN)/XF
	YLT=YL+((SHFT+AX)*SN+AY*CS)/YF
	IF(IAT.EQ.2)THEN
	    HS=HLABS*0.5
	    DNM=SQRT(((XL2-XLT)*XF)**2+((YL2-YLT)*YF)**2)
	    CCS=(XL2-XLT)*XF/DNM
	    SSN=(YL2-YLT)*YF/DNM
	    XN=XL2+(-CCS-.5*SSN)*HS/XF
	    YN=YL2+(-SSN+.5*CCS)*HS/YF
	    CALL PLOT(XN,YN,0,0)
	    CALL PLOT(XL2,YL2,1,0)
	    XN=XL2+(-CCS+.5*SSN)*HS/XF
	    YN=YL2+(-SSN-.5*CCS)*HS/YF
	    CALL PLOT(XN,YN,1,0)
	ENDIF
	CALL PLOT(XL2,YL2,0,0)
	CALL PLOT(XLT,YLT,1,0)
	RETURN
	END
