	SUBROUTINE VIEW(X)
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
C**    @(#)view.F	1.2    5/31/88
C**
C**
C***********************************************************************
C**
C**                 PLOT+ Scientific Graphics System
C**
C***********************************************************************
C**
C**
* V720 *acm*   Ticket 2530 add WIRE/COLOR= and draw the color with pen 1

	INCLUDE 'PARAMPL5.DAT'
	INCLUDE 'HD.INC'
	INCLUDE 'CMRD.INC'
	INCLUDE 'CMRDL.INC'
	INCLUDE 'LABELS.INC'
	INCLUDE 'LABCOM.INC'
	INCLUDE 'PEN.INC'
	INCLUDE 'PLT.INC'
	INCLUDE 'VIEW.INC'
	INCLUDE 'SYSTEM.INC'
	INCLUDE 'SWITCH.INC'

	IF(NX.EQ.0.OR.NY.EQ.0)RETURN
	call trans(0,xmin,ymin,xtm,ytm)
	call trans(0,xmax,ymax,xtx,ytx)
	IF(XYSCLE.EQ.1)THEN
	    XM=xtm
	    XMX=xtx
	ELSE
	    XM=ytm
	    XMX=ytx
	END IF
	IF(ZSCLE.NE.0)THEN
	    P1=1./ZSCLE
	ELSE
	    P1=(VEWZMX-VEWZMN)/(ytx-ytm)
	END IF
	XM=XM*P1
	XMX=XMX*P1
	YM=ytm*P1
	YMX=ytx*P1
C     TRANSFORM VIEWPOINT
	XV=(XMX-XM)*(VIEWX-xtm)/(xtx-xtm)+XM
	YV=(YMX-YM)*(VIEWY-ytm)/(ytx-ytm)+YM
	ICOUNT=ICOUNT+1

	CALL COLOR(IPEN(1))

	CALL SURFCE(X,XM,XMX,NX,YM,YMX,NY,VEWZMN,VEWZMX,
     *	XV,YV,VIEWZ)

	CALL COLOR(IPEN(0))

	RETURN
	END
