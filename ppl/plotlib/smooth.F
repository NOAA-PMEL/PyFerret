	subroutine smooth(z,nxs,nys,nx,ny,nsm) 
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
C**    @(#)smooth.f	1.1    3/10/88
C**
C**
C***********************************************************************
C**
C**                 PLOT+ Scientific Graphics System
C**
C***********************************************************************
C**
C**
c     given array z(i,j),i=1,nx , j=1,ny   and nsm=0,1,2,3 ... 
c     laplacian smoothing is applied to z nsm times by means of the 
c     operation z=z + .25*(av(zn,zs,ze,zw)-z) . 
c     the sweep is alternately sw to ne and ne to sw. 
c     zxx and zyy are assumed to be zero at edges 
c     unused points in z should be .ge. 10**35 . 
c     oceanography emr   oct/69   version no. 2  improved edging 
c 
	dimension z(nxs,nys)
	if(nsm)400,400,20 
20	big=.9e35 
	r=.25/4. 
	do 330 it=1,nsm 
	itodd=it-(it/2)*2 
	do 300 ii=1,nx 
	i=itodd*ii+(1-itodd)*(nx+1-ii) 
	iintp=(i-1)*(nx-i) 
	do 300 jj=1,ny 
	j=itodd*jj+(1-itodd)*(ny+1-jj) 
	zij=z(i,j) 
	if(zij-big)210,210,300 
210	del2=0. 
	if(iintp)230,230,215 
215	del2x=z(i-1,j)+z(i+1,j)-zij-zij 
	if(del2x-big)220,230,230 
220	del2=del2x 
230	if((j-1)*(ny-j))250,250,235 
235	del2y=z(i,j-1)+z(i,j+1)-zij-zij 
	if(del2y-big)240,250,250 
240	del2=del2+del2y 
250	z(i,j)=zij+del2*r 
300	continue 
330	continue 
400	return 
	end 
