	SUBROUTINE DRAWVEC (xtail,ytail,xhead,yhead,rlen,icntrl,xf,yf)
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
C**    @(#)drawvec.F	1.1    3/10/88
C**
C**
C***********************************************************************
C**
C**                 PLOT+ Scientific Graphics System
C**
C***********************************************************************
C**
C**
	
	INCLUDE 'AXIS.INC'
	INCLUDE 'AXLWID.INC'
	INCLUDE 'CMRD.INC'
	INCLUDE 'LABCOM.INC'
	INCLUDE 'TAXIS.INC'
	
	REAL minvec,maxarw,minarw,maxvel
	
	CHARACTER veclab*20
	
C  headfact is ratio of arrowhead length to vector length
	PARAMETER (headfact = 0.25)
C  minvec is minimum plotted vector (point plotted otherwise)
	PARAMETER (minvec = .0625)
C maxarw is length of vector above which arrowhead is of fixed size
	PARAMETER (maxarw = 2.0)
C minarw is length of vector below which arrowhead is of fixed size
	PARAMETER (minarw = .0625)
C pi is the constant pi
	PARAMETER (pi = 3.141592654)
C arwang is angle between vector and arrowhead in radians
	PARAMETER (arwang = pi/8.)
	
	dist(dx,dy) = SQRT(dx**2 + dy**2)
	
C If ictrl = -1 initialize maxvec, if = 1 display maxvec
	
	IF (icntrl .eq. 1) THEN
	    maxvel = xtail
	    xtail = xlo
	    ytail = ylo - (XWID + 2.0*HXLAB * (1-ITFLG) + 2.5*HLAB1)/YF
	    xhead = xlo + rlen*maxvel/XF
	    yhead = ytail
	    WRITE (veclab,1000) maxvel
1000	    FORMAT(F7.3)
	    xl = xhead + .125/xf
	    yl = yhead
	    CALL SYMBEL(XL,YL,0.,hlabs,7,veclab)
	ENDIF
	
C  Calculate length of vector in user units and in inches
	
	userdx = xhead - xtail
	userdy = yhead - ytail
	userlen = dist(userdx,userdy)
	
	realdx = userdx*xf
	realdy = userdy*yf
	reallen = dist(realdx,realdy)
	
C  If vector is too small plot a point only
	
	IF (reallen .LT. minvec) THEN
	    CALL PLOT(xtail,ytail,0,0)
	    CALL PLOT(xtail,ytail,1,0)
	    RETURN
	ENDIF
	
C Determine factor of arrowhead length
	
	IF (reallen .GT. maxarw) THEN
	    theadfact = maxarw*headfact/reallen
	ELSE IF (reallen .LT. minarw) THEN
	    theadfact = minarw*headfact/reallen
	ELSE
	    theadfact = headfact
	ENDIF
	
C Calculate coordinates of arrowhead
	
	cosarw = cos(arwang)
	sinarw = sin(arwang)
	xarwrgt = xhead - theadfact*(realdx*cosarw-realdy*sinarw)/xf
	yarwrgt = yhead - theadfact*(realdy*cosarw+realdx*sinarw)/yf
	xarwlft = xhead - theadfact*(realdx*cosarw+realdy*sinarw)/xf
	yarwlft = yhead - theadfact*(realdy*cosarw-realdx*sinarw)/yf
	
C Plot the vector
	
	CALL PLOT(xtail,ytail,0,0)
	CALL PLOT(xhead,yhead,1,0)
	CALL PLOT(xarwlft,yarwlft,0,0)
	CALL PLOT(xhead,yhead,1,0)
	CALL PLOT(xarwrgt,yarwrgt,1,0)
	
	RETURN
	END
