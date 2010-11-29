	subroutine double(z,nxs,nys,nx,ny) 
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
C**    @(#)double.f	1.1    3/10/88
C**
C**
C***********************************************************************
C**
C**                 PLOT+ Scientific Graphics System
C**
C***********************************************************************
C**
C**
c     the array z(i,j) i=1,nx j=1,ny  is expanded to twice its size , 
c     that is, z(i,j) i=1,2*nx-1,2  j=1,2*ny-1,2.  the spaces are filled 
c     using cubic polynomial interpolation.  undefined points of z 
c     should be set to 1.e35 . 
c     oceanography emr   december 1969. 
c 
c 
	dimension z(nxs,nys) 
	dimension  zz(4) 
c 
	nxd = 2*nx-1 
	nyd = 2*ny-1 
	nxdm2 =nxd-2 
	nydm2 =nyd-2 
	big =.9e35 
	big2 = big*2. 
c     spread existing points 
c 
c*******************************************************************************
c 
	do 20 ii = 1,nx 
	i= 1+nx-ii 
	id= i*2-1 
	do 20 jj = 1,ny 
	j= 1+ny-jj 
	jd = j*2-1 
20	z(id,jd)=z(i,j) 
c 
c     fill in half-way points of x-lines. 
c*******************************************************************************
c 
	do 90 i=1,nxdm2,2 
	do 90 j=1,nyd,2 
	z(i+1,j) = 1.e35 
	if(z(i,j)+z(i+2,j)-big)30,90,90 
30	zz(2)=z(i,j) 
	zz(3)=z(i+2,j) 
	zz(1)=1.e35 
	if(i-1)50,50,40 
40	zz(1)= z(i-2,j) 
50	zz(4)=1.e35 
	if(i-nxdm2)60,70,70 
60	zz(4)=z(i+4,j) 
70	call gethlf(zz,zzhalf)
80	z(i+1,j) = zzhalf 
90	continue 
c 
c     fill in half-way points of y-lines 
c*******************************************************************************
c 
	do 190 j = 1,nydm2,2 
	do 190 i = 1,nxd,2 
	z(i,j+1)=1.e35 
	if(z(i,j)+z(i,j+2)-big)130,190,190 
130	zz(2)= z(i,j) 
	zz(3)= z(i,j+2) 
	zz(1)=1.e35 
	if(j-1)150,150,140 
140	zz(1)= z(i,j-2) 
150	zz(4)= 1.e35 
	if(j-nydm2)160,170,170 
160	zz(4) = z(i,j+4) 
170	call gethlf(zz,zzhalf)
180	z(i,j+1)= zzhalf 
190	continue 
c 
c     fill in center-point of each square 
c*******************************************************************************
c 
	do 300 i = 1,nxdm2,2 
	do 300 j = 1,nydm2,2 
	z(i+1,j+1) = 1.e35 
	zsum = z(i,j)+z(i+2,j)+z(i  ,j+2)+z(i+2,j+2) 
	if(zsum-big)245,210,210 
210	if(zsum-big2)220,300,300 
220	if(z(i,j)+z(i+2,j+2)-big) 230,240,240 
230	z(i+1,j+1)  = (z(i,j)+z(i+2,j+2))*.5 
	go to 300 
240	z(i+1,j+1)  = (z(i+2,j)+z(i,j+2))*.5 
	go to 300 
245	zz(2)=  z(i,j+1) 
	zz(3)=  z(i+2,j+1) 
	zz(1)=1.e35 
	if(i-1) 260,260,250 
250	zz(1)= z(i-2,j+1) 
260	zz(4)=1.e35 
	if(i-nxdm2)270,280,280 
270	zz(4) =  z(i+4,j+1) 
280	call gethlf(zz,zzhalf)
290	z(i+1,j+1) = zzhalf 
300	continue 
	return 
	end
	subroutine gethlf(zz,zzhalf)
	dimension zz(4)
	big =.9e35 
c 
c     interpolation to get zz(2.5) . 
c*******************************************************************************
c 
400	if(zz(1)+zz(4)-big)420,430,430 
420	zzhalf = .5625*(zz(2)+zz(3)) -.0625*(zz(1)+zz(4)) 
	go to 480 
430	if(zz(1)-big)440,450,450 
440	zzhalf = .375*zz(3) +.75*zz(2) -.125*zz(1) 
	go to 480 
450	if(zz(4)-big) 460,470,470 
460	zzhalf=  .375*zz(2)+.75*zz(3)-.125*zz(4) 
	go to 480 
470	zzhalf = .5*(zz(2)+zz(3)) 
480	return
	end 
