	subroutine   arc(x,y,nask,eps,xx,yy,nret,
     ,      ks,kmax,xl,yl,splxp,splyp,spls,spline_tension)

* revised for TMAP
*
* *sh*  - 1/26/93 modified routine to call spline under tension fit
*                 if spline_tension is not zero.  otherwise, it will
*		  use the fit that this algorithm calculates (cubic).
*******************************************************************
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
C**    @(#)arc.f	1.1    3/10/88
C**
C**
C***********************************************************************
C**
C**                 PLOT+ Scientific Graphics System
C**
C***********************************************************************
C**
C**
c	given four successive points x(k),y(k),k=1,4 defining 3 segments, 
c	the points xx(k),yy(k),k=1,nret are generated describing a curve 
c	between points 2 and 3 trying to match slope with similar curves 
c	from 1 to 2 and from 3 to 4. if the segment (2,3) is less than eps 
c	in length,nret is set to 2 and points 2 and 3 only are returned in 
c	arrays xx,yy.  otherwise nret = the input parameter nask. 
c	the curve xx,yy is a cubic over segment (2,3). 
c	if segment (1,2) is less than eps the second derivative at 
c	point 2 is set to zero.   if segment (3,4) is less than eps 
c	the second derivative at point 3 is set to zero.  these two 
c	features may be used for starting and ending an open curve. 
c	oceanography emr  december 1969 
c 
	dimension x(*),y(*)
	dimension  xx(11),yy(11),slope(3),ds(3) 
! TMAP mod, spline fit:
	dimension xl(*),yl(*),splxp(*),splyp(*),spls(*)
c 
	if(nask-2) 10,10,20 
10	nret=2 
	xx(1)=x(2) 
	yy(1)=y(2) 
	xx(2)=x(3) 
	yy(2)=y(3) 
	return 
c 
20	continue
c
! TMAP mod, spline fit:
	if (spline_tension.ne.0.) then
	nret=nask
	do 141, k=1,nret
	  splt=(spls(ks)+float(k)/float(nret)*
     *         (spls(ks+1)-spls(ks)))/spls(kmax)
	  call kurv2(splt,xx(k),yy(k),kmax,xl,yl,
     *         splxp,splyp,spls,spline_tension)
141	continue
	else
c
	pi=3.1415926 
	hafpi =pi*.50 
	angmax =pi*.25 
	xcent = (x(2)+x(3))*.5 
	ycent = (y(2)+y(3))*.5 
	dx2= x(3)-x(2) 
	dy2= y(3)-y(2) 
	ds2 =  sqrt(dx2*dx2+dy2*dy2) 
	if(ds2-eps)10,30,30 
30	ang2=atan2(dy2,dx2) 
	cos2 = cos(ang2) 
	sin2 = sin(ang2) 
c 
	do 130 k=1,3,2 
	dx = x(k+1)-x(k) 
	dy = y(k+1)-y(k) 
	ds(k)=sqrt(dx*dx + dy*dy) 
	slope(k)=0. 
	if(ds(k)-eps) 130,40,40 
40	ang =(atan2(dy,dx) - ang2)*.5 
	if(ang-hafpi) 60,60,50 
50	ang = ang-pi 
60	if(ang+hafpi) 70,80,80 
70	ang = ang+pi 
80	if(ang-angmax)100,100,90 
90	ang=angmax 
100	if(ang+angmax)110,120,120 
110	ang=-angmax 
120	slope(k)=sin(ang)/cos(ang) 
130	continue 
	if(ds(3)-eps)132,134,134 
132	slope(3)=-slope(1)*.5 
134	if(ds(1)-eps)136,138,138 
136	slope(1)=-slope(3)*.5 
138	continue 
	c= .25*(slope(3)-slope(1)) 
	d= .25*(slope(3)+slope(1)) 
c 
	nret=nask 
	dx=2./(nret-1) 
	xt=-1.-dx 
	do 140 k=1,nret 
	  xt=xt+dx
	  yt=(c+d*xt)*(xt*xt-1.) 
	  xx(k)=(xt*cos2-yt*sin2)*ds2*.5+xcent 
	  yy(k)=(xt*sin2+yt*cos2)*ds2*.5+ycent 
140	continue
c
! TMAP mod, spline fit:
	endif
c
	return 
	end 
