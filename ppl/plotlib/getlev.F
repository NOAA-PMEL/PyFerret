	subroutine getlev(z,nxs,nys,nx,ny,dz,zlev,nlev) 
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
C**    @(#)getlev.f	1.1    3/10/88
C**
C**
C***********************************************************************
C**
C**                 PLOT+ Scientific Graphics System
C**
C***********************************************************************
C**
C**
c     given the array z(1...nx,1...ny) and the contour interval dz 
c     the array zlev(1...nlev) of contour levels is returned. 
c 
	dimension z(nxs,nys) 
	dimension zlev(2) 
	big=.9e35 
c 
	nfnd=0 
	do 100 i=1,nx 
	do 100 j=1,ny 
	if(z(i,j)-big)70,100,100 
70	nfnd=nfnd+1 
	if(nfnd-1)80,80,90 
80	zmin=z(i,j) 
	zmax=zmin 
90	zmin=amin1(zmin,z(i,j)) 
	zmax=amax1(zmax,z(i,j)) 
100	continue 
c 
	nlmin=zmin/dz+.5+sign(.5,zmin) 
	zlev(1)=dz*nlmin 
	do 200 k=1,51 
	zlevk=zlev(1)+(k-1)*dz 
	if(zlevk-zmax)190,210,210 
190	zlev(k)=zlevk 
200	continue 
210	nlev=k-1 
	return 
	end 
