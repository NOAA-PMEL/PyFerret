	subroutine zgrid_laplace(z,nxsize,nysize,nx,ny,x1,y1,dx,dy, 
     1	xp,yp,zp,zpij,knxt,n,cay,nrng) 
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
C**    @(#)zgrid.f	1.1    3/10/88
C**
C**
C***********************************************************************
C**
C**                 PLOT+ Scientific Graphics System
C**
C***********************************************************************
C**
C**
c     sets up square grid for contouring , given arbitrarily placed 
c     data points. laplace interpolation is used. 
c     the method used here was lifted directly from notes left by 
c     mr ian crain formerly with the comp.science div. 
c     info on relaxation soln of laplace eqn supplied by dr t murty. 
c     fortran ii   oceanography/emr   dec/68   jdt 
c 
c     z = 2-d array of hgts to be set up. points outside region to be 
c     contoured should be initialized to 10**35 . the rest should be 0.0 
c     nx,ny = max subscripts of z in x and y directions . 
c     x1,y1 = coordinates of z(1,1) 
c     dx,dy = x and y increments . 
c     xp,yp,zp = arrays giving position and hgt of each data point. 
c     n = size of arrays xp,yp and zp . 
c 
c     modification feb/69   to get smoother results a portion of the 
c     beam eqn  was added to the laplace eqn giving 
c     delta2x(z)+delta2y(z) - k(delta4x(z)+delta4y(z)) = 0 . 
c     k=0 gives pure laplace solution.  k=inf. gives pure spline solution. 
c     cayin = k = amount of spline eqn (between 0 and inf.) 
c     nrng...grid points more than nrng grid spaces from the nearest 
c            data point are set to undefined. 
c 
c     modification dec23/69   data pts no longer moved to grid pts. 
c 
c     modification may 5 79  common blocks work1 and work2 must 
c     be dimension at least n points long by the user.  common 
c     block work3 must be dimensioned at least ny points long. 
c 
c	modification june 17,1985 - handles data values of 1e35. if at
c	least one data value near a grid point is equal to 1e35, the z
c	array is initialized to 1e35 at that grid point
c	- by g.r. halliwell
c
c  8/26/2008   This is exactly zgrid from the PPLUS code. This function has
c              some hard-wired workspace limited to 50K points. For the external
c              functions scat2gridlplace*, instead allocate work arrays in the 
c              external functions and pass them to zgrid_laplace.F
c
c  10/2022    Ansley Manke
c	Rewrite zgrid to remove obsolete Arithmetic-if's and numbered loops
c	with shared DO termination statements. The original zgrid.F is preserved 
c       at the end of this file.
c
c
c
c*******************************************************************************
c O

* calling argument declarations:
	INTEGER nxsize,nysize,nx,ny,knxt(n),n,nrng
	REAL	z(nxsize,nysize),xp(n),yp(n),zp(n),zpij(n),x1,y1,dx,dy,cay

* internal variable declarations:
	INTEGER itmax,i,j,k,kk,im,jm,imask,imnew,jmnew,iter,kksav,nnew,npg,npt
	REAL	big,eps,cge,zmin,zmax,zrange,zbase,hrange,derzm,ze,zw,zn,zs,a,b,
     .		c,d,abz,z00,zxy,delz,dz,delzm,dzmax,dzmaxf,dzrmsp,dzrms,dzrms8,
     .		relax,relaxn,root,rootgs,tpy,wgt,x,y,zijn,zim,zimm,zip,zipp,
     .		zjm,zjmm,zjp,zjpp,zpxy,zsum
cc
	dimension imnew(1000),kksav(500)

	itmax=100 
	eps=.002 
	big=.9e35
	cge = 1.e35  ! same as PPLUS CMXGE 
c 
c     get zbase which will make all zp values positive by 20*(zmax-zmin) 
c*******************************************************************************
c 
	zmin=big
	zmax=-big
	do k=1,n 
	   if (zp(k).ge.cge) cycle
	   if (zp(k) .gt. zmax) zmax=zp(k) 
	   if (zp(k) .lt. zmin) zmin=zp(k)
	enddo

	zrange=zmax-zmin 
	zbase=zrange*20.-zmin 
	hrange=amin1(dx*(nx-1) , dy*(ny-1)) 
	derzm=2.*zrange/hrange 
c 
c     set pointer array knxt 
c*******************************************************************************
c 

	do kk=1,n 
	   k=1+n-kk 
	   knxt(k)=0 
	   i= (xp(k)-x1)/dx + 1.5 

	   if (i*(nx+1-i) .gt. 0) then
	      j= (yp(k)-y1)/dy + 1.5 
	      if(j*(ny+1-j) .gt. 0) then  
	         if(z(i,j) .lt. big) then 
	            knxt(k)=n+1 
	            if(z(i,j) .gt. 0) knxt(k)= z(i,j)+.5
	            z(i,j)=k 
	         endif
	      endif	   
	   endif
	enddo
c 
c     affix each data point zp to its nearby grid point.  take avg zp if 
c     more than one zp nearby the grid point. add zbase and complement. 

	do k=1,n 
	   if(knxt(k) .le. 0)  cycle  
	   npt=0 
	   imask=0

	   zsum=0. 
	   i= (xp(k)-x1)/dx + 1.5 
	   j =(yp(k)-y1)/dy + 1.5 
	   kk=k 

	   do while (kk .le. n)
	      npt=npt+1 
	      kksav(npt)=kk
	      if(zp(kk).gt.big)imask=1

	      zsum=zsum+ zp(kk) 
	      knxt(kk)=-knxt(kk) 
	      kk = -knxt(kk) 
	   enddo

	   if(imask.eq.0)then
	      z(i,j) = -zsum/npt-zbase 
	   else
	      z(i,j) = cge
	      do i=1,npt
	         knxt(kksav(i))=0
	     enddo
	   endif

	enddo

c 
c     initially set each unset grid point to value of nearest known pt. 
c*******************************************************************************
c 
	do i=1,nx 
	   do j=1,ny 
	      if (z(i,j) .eq. 0.) z(i,j)=-1.* cge 
	   enddo
	enddo

	do iter=1,nrng 

	nnew=0 

	do i=1,nx 
	do j=1,ny 

	if(z(i,j)+big .lt. 0) then 
	   if(j .gt. 1) then 
	      if(jmnew .le. 0) then
	         zijn=abs(z(i,j-1)) 
	         if(zijn .lt. big) goto 195
              endif
	   endif

	   if(i .gt. 1) then 
	      if(imnew(j) .le. 0) then
	         zijn=abs(z(i-1,j)) 
	         if(zijn .lt. big) goto 195
	      endif
	   endif 
	   if(j .lt. ny) then
	      zijn=abs(z(i,j+1)) 
	      if(zijn .lt. big) goto 195 
	   endif 
	   if(i .lt. nx) then
	      zijn=abs(z(i+1,j)) 
	      if(zijn .lt. big) goto 195 
	   endif
	endif

	imnew(j)=0 
	jmnew=0 
	cycle 
	
195	continue  ! target for goto

	imnew(j)=1 
	jmnew=1 
	z(i,j)=zijn 
	nnew=nnew+1 

	enddo ! j=1,ny
	enddo ! i=1,nx

	if(nnew .le. 0) exit 

	enddo  ! iter=1,nrng


	do i=1,nx 
	   do j=1,ny 
	      abz=abs(z(i,j)) 
	      if (abz .ge. big) z(i,j)=abz 
	   enddo
	enddo

c 
c     improve the non-data points by applying point over-relaxation 
c     using the laplace-spline equation  (carres method is used) 
c*******************************************************************************
c 
	dzrmsp=zrange 
	relax=1.0 

	do iter=1,itmax ! 4000 loop iter=1,itmax

	dzrms=0. 
	dzmax=0. 
	npg =0 

	do i=1,nx 
	do j=1,ny 

	z00=z(i,j) 
	if(z00 .ge. big) cycle ! j=1,ny
	if(z00 .lt. 0)   cycle ! j=1,ny
	wgt=0. 
	zsum=0. 
c 
	im=0 
	if(i-1 .gt. 0) then 
	   zim=abs(z(i-1,j)) 
	   if(zim .lt. big) then 
	      im=1 
	      wgt=wgt+1. 
	      zsum=zsum+zim 
	      if(i .gt. 2 ) then 
	         zimm=abs(z(i-2,j)) 
	         if(zimm .lt. big) then 
	            wgt=wgt+cay 
	            zsum=zsum-cay*(zimm-2.*zim) 
	         endif
	      endif
	   endif
	endif

	if(nx .gt. i) then  
	   zip=abs(z(i+1,j)) 
	   if(zip .lt. big) then
	      wgt=wgt+1. 
	      zsum=zsum+zip 
	      if(im .gt. 0) then 
	         wgt=wgt+4.*cay 
	         zsum=zsum+2.*cay*(zim+zip) 
	      endif 
	      if(nx-1 .gt. i) then 
	         zipp=abs(z(i+2,j)) 
	         if(zipp .lt. big) then 
	            wgt=wgt+cay 
	            zsum=zsum-cay*(zipp-2.*zip) 
	         endif
	      endif
	   endif
	endif
c 
	jm=0 
	if(j .gt. 1) then 
	   zjm=abs(z(i,j-1)) 
	   if(zjm .lt. big) then
	      jm=1 
	      wgt=wgt+1. 
	      zsum=zsum+zjm 
	      if(j .gt. 2) then 
	         zjmm=abs(z(i,j-2)) 
	         if(zjmm .lt. big) then 
	            wgt=wgt+cay 
	            zsum=zsum-cay*(zjmm-2.*zjm) 
	         endif
	      endif
	   endif
	endif

	if(ny .gt. j) then
	   zjp=abs(z(i,j+1)) 
	   if(zjp .lt. big) then
	      wgt=wgt+1. 
	      zsum=zsum+zjp 
	      if(jm .gt. 0) then 
	         wgt=wgt+4.*cay 
	         zsum=zsum+2.*cay*(zjm+zjp) 
	      endif
	      if(ny-1 .gt. j) then 
	         zjpp=abs(z(i,j+2)) 
	         if(zjpp .lt. big) then 
	            wgt=wgt+cay 
	            zsum=zsum-cay*(zjpp-2.*zjp) 
		 endif
	      endif
	   endif
	endif
c 
	dz=zsum/wgt-z00 
	npg=npg+1 
	dzrms=dzrms+dz*dz 
	dzmax=amax1(abs(dz),dzmax) 
	z(i,j)=z00+dz*relax 

	enddo  ! j=1,ny
	enddo  ! i=1,nx

c 
c 
c     shift data points zp progressively back to their proper places as 
c     the shape of surface z becomes evident. 
c*******************************************************************************
c 

	if(iter .ne. (iter/10)*10) goto 3600  
	do k=1,n ! 3400

	knxt(k) =iabs(knxt(k)) 
	if(knxt(k) .le. 0 )  cycle ! 3400
	x=(xp(k)-x1)/dx 
	i=x+1.5 
	x= x+1.-i 
	y=(yp(k)-y1)/dy 
	j=y+1.5 
	y=y+1.-j 
	zpxy = zp(k)+zbase 
	z00 = abs(z(i,j)) 
c 
	zw=cge 
	if(i .gt. 1) zw = abs(z(i-1,j))  
	ze=cge 
	if(i .lt. nx) ze = abs(z(i+1,j)) 
	if(ze.ge.big) then 
	   if(zw .gt. big) then 
	     ze=z00 
	     zw=z00 
	  else
	     ze=2.*z00-zw 
	  endif
	else
	  if(zw .ge. big) then
	     zw = 2.*z00-ze
	  endif
	endif
 
	zs=cge 
	if(j .gt. 1) zs = abs(z(i,j-1)) 
  	zn= cge 
	if(j .lt. ny) zn = abs(z(i,j+1)) 
	if(zn .ge. big) then 
	   if (zs .lt. big) then
	      zn = 2.*z00-zs 
	   else
	      zn= z00 
	      zs= z00 
	   endif
	else
	    if(zs .ge. big) zs = 2.*z00-zn 
	endif
	 
	a=(ze-zw)*.5 
	b=(zn-zs)*.5 
	c=(ze+zw)*.5-z00 
	d=(zn+zs)*.5-z00 
	zxy=z00+a*x+b*y+c*x*x+d*y*y 
	delz=z00-zxy 
	delzm=derzm*(abs(x)*dx+abs(y)*dy)*.80 
	if(delz .gt. delzm) delz=delzm 
	if(delz+delzm .lt. 0) delz=-delzm 
	zpij(k)=zpxy+delz 

3400	continue
	enddo ! 3400
c 
	do k=1,n  
	   if(knxt(k) .le. 0) cycle

	   npt=0 
	   zsum = 0. 
	   i= (xp(k)-x1)/dx + 1.5 
	   j= (yp(k)-y1)/dy + 1.5 
	   kk = k 

	   do while (kk .le. n)
	      npt = npt+1 
	      zsum = zsum + zpij(kk) 
	      knxt(kk)= -knxt(kk) 
	      kk = -knxt(kk) 
	   enddo

	   z(i,j) =  -zsum/npt 
	enddo ! 3500

3600	continue   ! target for goto


c 
c     test for convergence 
c*******************************************************************************
c 
	if(npg.eq.0) goto 4010

	dzrms=sqrt(dzrms/npg) 
	root =dzrms/dzrmsp 
	dzrmsp=dzrms 
	dzmaxf=dzmax/zrange 

	if(MOD(iter,10) .eq. 2) dzrms8 = dzrms 
	if(MOD(iter,10) .ne. 0) cycle  ! 4000 iter loop

	root = sqrt(sqrt(sqrt(dzrms/dzrms8))) 
	if(root .ge. .9999) cycle  ! 4000 iter loop

	if(dzmaxf/(1.-root) .le. eps) goto 4010

c     improve the relaxation factor. 
c*******************************************************************************
c 
	if((iter-20)*(iter-40)*(iter-60) .ne. 0) cycle  ! 4000 iter loop  
	if(relax-1. .ge. root) cycle  ! 4000 iter loop

	tpy =(root+relax-1.)/relax 
	rootgs = tpy*tpy/root 
	relaxn= 2./(1.+sqrt(1.-rootgs)) 

	if(iter .ne. 60) relaxn= relaxn-.25*(2.-relaxn) 
	relax = amax1(relax,relaxn) 

	enddo     ! 4000 loop iter=1,itmax

4010	continue  ! target for goto

c 
c     remove zbase from array z and return. 
c 
	do i=1,nx 
	   do j=1,ny 
	      if (z(i,j).lt. big) z(i,j)=abs(z(i,j))-zbase 
	   enddo
	enddo

	return 
	end 

czz Here is zgrid.F from ppl/plotlib
czz 
czz 	subroutine zgrid(z,nxsize,nysize,nx,ny,x1,y1,dx,dy, 
czz      1	xp,yp,zp,n,cay,nrng) 
czz 
czz *
czz C**    @(#)zgrid.f	1.1    3/10/88
czz C**
czz C**
czz C***********************************************************************
czz C**
czz C**                 PLOT+ Scientific Graphics System
czz C**
czz C***********************************************************************
czz C**
czz C**
czz c     sets up square grid for contouring , given arbitrarily placed 
czz c     data points. laplace interpolation is used. 
czz c     the method used here was lifted directly from notes left by 
czz c     mr ian crain formerly with the comp.science div. 
czz c     info on relaxation soln of laplace eqn supplied by dr t murty. 
czz c     fortran ii   oceanography/emr   dec/68   jdt 
czz c 
czz c     z = 2-d array of hgts to be set up. points outside region to be 
czz c     contoured should be initialized to 10**35 . the rest should be 0.0 
czz c     nx,ny = max subscripts of z in x and y directions . 
czz c     x1,y1 = coordinates of z(1,1) 
czz c     dx,dy = x and y increments . 
czz c     xp,yp,zp = arrays giving position and hgt of each data point. 
czz c     n = size of arrays xp,yp and zp . 
czz c 
czz c     modification feb/69   to get smoother results a portion of the 
czz c     beam eqn  was added to the laplace eqn giving 
czz c     delta2x(z)+delta2y(z) - k(delta4x(z)+delta4y(z)) = 0 . 
czz c     k=0 gives pure laplace solution.  k=inf. gives pure spline solution. 
czz c     cayin = k = amount of spline eqn (between 0 and inf.) 
czz c     nrng...grid points more than nrng grid spaces from the nearest 
czz c            data point are set to undefined. 
czz c 
czz c     modification dec23/69   data pts no longer moved to grid pts. 
czz c 
czz c     modification may 5 79  common blocks work1 and work2 must 
czz c     be dimension at least n points long by the user.  common 
czz c     block work3 must be dimensioned at least ny points long. 
czz c 
czz c	modification june 17,1985 - handles data values of 1e35. if at
czz c	least one data value near a grid point is equal to 1e35, the z
czz c	array is initialized to 1e35 at that grid point
czz c	- by g.r. halliwell
czz c
czz c*******************************************************************************
czz c 
czz c	vax/vms
czz c
czz 	dimension z(nxsize,nysize),xp(n),yp(n),zp(n),
czz      .	zpij(50000),knxt(50000)
czz c
czz c	end vax/vms
czz c
czz c	rsx-11m
czz c
czz c	virtual z(nxsize,nysize),xp(n),yp(n),zp(n),zpij(10000),knxt(10000)
czz c
czz c	end rsx-11m
czz c
czz 	dimension imnew(1000),kksav(500)
czz  
czz 	itmax=100 
czz 	eps=.002 
czz 	big=.9e35 
czz c 
czz c     get zbase which will make all zp values positive by 20*(zmax-zmin) 
czz c*******************************************************************************
czz c 
czz c**   grh modification
czz 	zmin=big
czz 	zmax=-big
czz 	do 20 k=1,n 
czz 	if(zp(k).ge.1e35)goto 20
czz c**
czz 	if(zp(k)-zmax)14,14,12 
czz 12	zmax=zp(k) 
czz 14	if(zp(k)-zmin)16,20,20 
czz 16	zmin=zp(k) 
czz 20	continue 
czz 	zrange=zmax-zmin 
czz 	zbase=zrange*20.-zmin 
czz 	hrange=amin1(dx*(nx-1) , dy*(ny-1)) 
czz 	derzm=2.*zrange/hrange 
czz c 
czz c     set pointer array knxt 
czz c*******************************************************************************
czz c 
czz 	do 60 kk=1,n 
czz 	k=1+n-kk 
czz 	knxt(k)=0 
czz 	i= (xp(k)-x1)/dx + 1.5 
czz 	if(i*(nx+1-i))60,60,35 
czz 35	j= (yp(k)-y1)/dy + 1.5 
czz 	if(j*(ny+1-j))60,60,40 
czz 40	if(z(i,j)-big)45,60,60 
czz 45	knxt(k)=n+1 
czz 	if(z(i,j))55,55,50 
czz 50	knxt(k)= z(i,j)+.5 
czz 55	z(i,j)=k 
czz 60	continue 
czz c 
czz c     affix each data point zp to its nearby grid point.  take avg zp if 
czz c     more than one zp nearby the grid point. add zbase and complement. 
czz 	do 80 k=1,n 
czz 	if(knxt(k))80,80,65 
czz 65	npt=0 
czz c**   grh modification
czz 	imask=0
czz c**
czz 	zsum=0. 
czz 	i= (xp(k)-x1)/dx + 1.5 
czz 	j =(yp(k)-y1)/dy + 1.5 
czz 	kk=k 
czz 70	npt=npt+1 
czz c**   grh modification
czz 	kksav(npt)=kk
czz 	if(zp(kk).gt.big)imask=1
czz c**
czz 	zsum=zsum+ zp(kk) 
czz 	knxt(kk)=-knxt(kk) 
czz 	kk = -knxt(kk) 
czz 	if(kk-n)70,70,75 
czz c**	grh modification
czz 75	if(imask.eq.0)then
czz 	z(i,j) = -zsum/npt-zbase 
czz 	else
czz 	z(i,j) = 1.e35
czz 	do 76 i=1,npt
czz 76	knxt(kksav(i))=0
czz 	endif
czz c**
czz 80	continue 
czz c 
czz c     initially set each unset grid point to value of nearest known pt. 
czz c*******************************************************************************
czz c 
czz 	do 110 i=1,nx 
czz 	do 110 j=1,ny 
czz 	if(z(i,j))110,100,110 
czz 100	z(i,j) = -1.e35 
czz 110	continue 
czz 	do 199 iter=1,nrng 
czz 	nnew=0 
czz 	do 197 i=1,nx 
czz 	do 197 j=1,ny 
czz 	if(z(i,j)+big)152,192,192 
czz 152	if(j-1)162,162,153 
czz 153	if(jmnew)154,154,162 
czz 154	zijn=abs(z(i,j-1)) 
czz 	if(zijn-big)195,162,162 
czz 162	if(i-1)172,172,163 
czz 163	if(imnew(j))164,164,172 
czz 164	zijn=abs(z(i-1,j)) 
czz 	if(zijn-big)195,172,172 
czz 172	if(j-ny)173,182,182 
czz 173	zijn=abs(z(i,j+1)) 
czz 	if(zijn-big)195,182,182 
czz 182	if(i-nx)183,192,192 
czz 183	zijn=abs(z(i+1,j)) 
czz 	if(zijn-big)195,192,192 
czz 192	imnew(j)=0 
czz 	jmnew=0 
czz 	go to 197 
czz 195	imnew(j)=1 
czz 	jmnew=1 
czz 	z(i,j)=zijn 
czz 	nnew=nnew+1 
czz 197	continue 
czz 	if(nnew)200,200,199 
czz 199	continue 
czz 200	continue 
czz 	do 202 i=1,nx 
czz 	do 202 j=1,ny 
czz 	abz=abs(z(i,j)) 
czz 	if(abz-big)202,201,201 
czz 201	z(i,j)=abz 
czz 202	continue 
czz c 
czz c     improve the non-data points by applying point over-relaxation 
czz c     using the laplace-spline equation  (carres method is used) 
czz c*******************************************************************************
czz c 
czz 	dzrmsp=zrange 
czz 	relax=1.0 
czz 	do 4000 iter=1,itmax 
czz 	dzrms=0. 
czz 	dzmax=0. 
czz 	npg =0 
czz 	do 2000 i=1,nx 
czz 	do 2000 j=1,ny 
czz 	z00=z(i,j) 
czz 	if(z00-big)205,2000,2000 
czz 205	if(z00)2000,208,208 
czz 208	wgt=0. 
czz 	zsum=0. 
czz c 
czz 	im=0 
czz 	if(i-1)570,570,510 
czz 510	zim=abs(z(i-1,j)) 
czz 	if(zim-big)530,570,570 
czz 530	im=1 
czz 	wgt=wgt+1. 
czz 	zsum=zsum+zim 
czz 	if(i-2)570,570,540 
czz 540	zimm=abs(z(i-2,j)) 
czz 	if(zimm-big)560,570,570 
czz 560	wgt=wgt+cay 
czz 	zsum=zsum-cay*(zimm-2.*zim) 
czz 570	if(nx-i)700,700,580 
czz 580	zip=abs(z(i+1,j)) 
czz 	if(zip-big)600,700,700 
czz 600	wgt=wgt+1. 
czz 	zsum=zsum+zip 
czz 	if(im)620,620,610 
czz 610	wgt=wgt+4.*cay 
czz 	zsum=zsum+2.*cay*(zim+zip) 
czz 620	if(nx-1-i)700,700,630 
czz 630	zipp=abs(z(i+2,j)) 
czz 	if(zipp-big)650,700,700 
czz 650	wgt=wgt+cay 
czz 	zsum=zsum-cay*(zipp-2.*zip) 
czz 700	continue 
czz c 
czz 	jm=0 
czz 	if(j-1)1570,1570,1510 
czz 1510	zjm=abs(z(i,j-1)) 
czz 	if(zjm-big)1530,1570,1570 
czz 1530	jm=1 
czz 	wgt=wgt+1. 
czz 	zsum=zsum+zjm 
czz 	if(j-2)1570,1570,1540 
czz 1540	zjmm=abs(z(i,j-2)) 
czz 	if(zjmm-big)1560,1570,1570 
czz 1560	wgt=wgt+cay 
czz 	zsum=zsum-cay*(zjmm-2.*zjm) 
czz 1570	if(ny-j)1700,1700,1580 
czz 1580	zjp=abs(z(i,j+1)) 
czz 	if(zjp-big)1600,1700,1700 
czz 1600	wgt=wgt+1. 
czz 	zsum=zsum+zjp 
czz 	if(jm)1620,1620,1610 
czz 1610	wgt=wgt+4.*cay 
czz 	zsum=zsum+2.*cay*(zjm+zjp) 
czz 1620	if(ny-1-j)1700,1700,1630 
czz 1630	zjpp=abs(z(i,j+2)) 
czz 	if(zjpp-big)1650,1700,1700 
czz 1650	wgt=wgt+cay 
czz 	zsum=zsum-cay*(zjpp-2.*zjp) 
czz 1700	continue 
czz c 
czz 	dz=zsum/wgt-z00 
czz 	npg=npg+1 
czz 	dzrms=dzrms+dz*dz 
czz 	dzmax=amax1(abs(dz),dzmax) 
czz 	z(i,j)=z00+dz*relax 
czz 2000	continue 
czz c 
czz c 
czz c     shift data points zp progressively back to their proper places as 
czz c     the shape of surface z becomes evident. 
czz c*******************************************************************************
czz c 
czz 	if(iter-(iter/10)*10) 3600,3020,3600 
czz 3020	do 3400 k=1,n 
czz 	knxt(k) =iabs(knxt(k)) 
czz 	if(knxt(k))3400,3400,3030 
czz 3030	x=(xp(k)-x1)/dx 
czz 	i=x+1.5 
czz 	x= x+1.-i 
czz 	y=(yp(k)-y1)/dy 
czz 	j=y+1.5 
czz 	y=y+1.-j 
czz 	zpxy = zp(k)+zbase 
czz 	z00 = abs(z(i,j)) 
czz c 
czz 	zw=1.e35 
czz 	if(i-1)3120,3120,3110 
czz 3110	zw = abs(z(i-1,j)) 
czz 3120	ze=1.e35 
czz 	if(i-nx)3130,3140,3140 
czz 3130	ze = abs(z(i+1,j)) 
czz 3140	if(ze-big)3160,3150,3150 
czz 3150	if(zw-big)3180,3170,3170 
czz 3160	if(zw-big)3200,3190,3190 
czz 3170	ze=z00 
czz 	zw=z00 
czz 	go to 3200 
czz 3180	ze=2.*z00-zw 
czz 	go to 3200 
czz 3190	zw = 2.*z00-ze 
czz c 
czz 3200	zs=1.e35 
czz 	if(j-1)3220,3220,3210 
czz 3210	zs = abs(z(i,j-1)) 
czz 3220  	zn= 1.e35 
czz 	if(j-ny)3230,3240,3240 
czz 3230	zn = abs(z(i,j+1)) 
czz 3240	if(zn-big)3260,3250,3250 
czz 3250	if(zs-big)3280,3270,3270 
czz 3260	if(zs-big)3300,3290,3290 
czz 3270	zn= z00 
czz 	zs= z00 
czz 	go to 3300 
czz 3280	zn = 2.*z00-zs 
czz 	go to 3300 
czz 3290	zs = 2.*z00-zn 
czz c 
czz 3300	a=(ze-zw)*.5 
czz 	b=(zn-zs)*.5 
czz 	c=(ze+zw)*.5-z00 
czz 	d=(zn+zs)*.5-z00 
czz 	zxy=z00+a*x+b*y+c*x*x+d*y*y 
czz 	delz=z00-zxy 
czz 	delzm=derzm*(abs(x)*dx+abs(y)*dy)*.80 
czz 	if(delz-delzm)3355,3355,3350 
czz 3350	delz=delzm 
czz 3355	if(delz+delzm)3360,3365,3365 
czz 3360	delz=-delzm 
czz 3365	zpij(k)=zpxy+delz 
czz 3400	continue 
czz c 
czz 	do 3500 k=1,n 
czz 	if(knxt(k))3500,3500,3410 
czz 3410	npt=0 
czz 	zsum = 0. 
czz 	i= (xp(k)-x1)/dx + 1.5 
czz 	j= (yp(k)-y1)/dy + 1.5 
czz 	kk = k 
czz 3420	npt = npt+1 
czz 	zsum = zsum + zpij(kk) 
czz 	knxt(kk)= -knxt(kk) 
czz 	kk = -knxt(kk) 
czz 	if(kk-n)3420,3420,3430 
czz 3430	z(i,j) =  -zsum/npt 
czz 3500	continue 
czz 3600	continue 
czz c 
czz c     test for convergence 
czz c*******************************************************************************
czz c 
czz 	if(npg.eq.0)goto 4010
czz 	dzrms=sqrt(dzrms/npg) 
czz 	root =dzrms/dzrmsp 
czz 	dzrmsp=dzrms 
czz 	dzmaxf=dzmax/zrange 
czz 	if(iter-(iter/10)*10-2)3715,3710,3715 
czz 3710	dzrms8 = dzrms 
czz 3715	if(iter-(iter/10)*10)4000,3720,4000 
czz 3720	root = sqrt(sqrt(sqrt(dzrms/dzrms8))) 
czz 	if(root-.9999)3730,4000,4000 
czz 3730	if(dzmaxf/(1.-root)-eps)4010,4010,3740 
czz c 
czz c     improve the relaxation factor. 
czz c*******************************************************************************
czz c 
czz 3740	if((iter-20)*(iter-40)*(iter-60))4000,3750,4000 
czz 3750	if(relax-1.-root)3760,4000,4000 
czz 3760	tpy =(root+relax-1.)/relax 
czz 	rootgs = tpy*tpy/root 
czz 	relaxn= 2./(1.+sqrt(1.-rootgs)) 
czz 	if(iter-60)3780,3785,3780 
czz 3780	relaxn= relaxn-.25*(2.-relaxn) 
czz 3785	relax = amax1(relax,relaxn) 
czz 4000	continue 
czz 4010	continue 
czz c 
czz c     remove zbase from array z and return. 
czz c 
czz 	do 4500 i=1,nx 
czz 	do 4500 j=1,ny 
czz 	if(z(i,j)-big)4400,4500,4500 
czz 4400	z(i,j)=abs(z(i,j))-zbase 
czz 4500	continue 
czz 	return 
czz 	end 
czz 