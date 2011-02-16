	subroutine CONSEG_SECTION(xc, yc, nx, ny, nxs, nys, px, py, z, 
     .    xf, xoff, yf, yoff, zlev, ndecl, lwgtl, lclr, nlev, hgt, 
     .    narc, ang, ilo, ihi, jlo, jhi, xadd, lmin, lmax)

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
* 
C  v581 5/2005 *acm* From conseg.F for one modulo replication of a contour plot
C  v67  2/2011 *acm* Fix ticket 1799; do contouring in larger sections.

	include 'MISS.INC'

	dimension zz(285,285),z(nxs,nys),px(nx),py(ny)
	dimension zlev(2),ndecl(2),lwgtl(2),lclr(2)
        integer izz, lseg

        izz = 285
        lseg = (izz-1)-2

C       Calculate the beginning and ending index values (iam,ibm) and
C       the number of index values (nxx) along the X axis in this 
C       zz() subregion.

	ia = 1 
 40	ib = min0(ia+lseg,nx) 
	lrem = nx-ib 
	if (lrem*(lseg-lrem)) 60,60,50 
 50	ib = (1+ia+nx)/2 
 60	iam = max0(ia-1,1) 
	ibm = min0(ib+1,nx) 
	lxm = ibm-iam 
	iim = 1+(ia-iam)
	nxx = (ib-ia)+1 
	
	
C       Calculate the beginning and ending index values (jam,jbm) and
C       the number of index values (nyy) along the Y axis in this 
C       zz() subregion.

	ja=1 
 80	jb = min0(ja+lseg,ny) 
	lrem = ny-jb 
	if (lrem*(lseg-lrem)) 100,100,90 
 90	jb = (1+ja+ny)/2 
 100	jam = max0(ja-1,1) 
	jbm = min0(jb+1,ny) 
	lym = jbm-jam 
	jjm = 1+(ja-jam)
	nyy = (jb-ja)+1 
	
	
C       Check all the z values for "goodness" and replace the bad ones with 1.e35 ?
C       Fill in the zz() subregion with values from z().

	do 130 i=iam,ibm 
	   ii =  i-iam+1 
	   do 120 j=jam,jbm 
	      jj =  j-jam+1 
	      if ((zle.and.z(i,j).le.cmzle).or.
     1	   (zeq.and.z(i,j).eq.cmzeq).or.
     2	   (zge.and.z(i,j).ge.cmzge)) then
		 zz(ii,jj)=1.e35
	      else
		 zz(ii,jj)=z(i,j)
	      endif
 120	   continue
 130	continue
	   

C       Actually contour the zz() subregion.

	call contur(zz,izz,izz,px(iam),py(jam),nxx,nyy,xf,xoff,
     .    yf,yoff,zlev,ndecl,lwgtl,lclr,nlev,hgt,narc,ang,xc,yc,
     .    xadd,lmin,lmax) 

C       This is the Y axis loop

	ja = jb 
	if (ja-ny) 80,160,160

C       This is the X axis loop

 160	ia = ib 
	if (ia-nx) 40,170,170 

 170	return 
	end 
