	subroutine conseg(z,nxs,nys,px,py,nx,ny,xf,xoff,yf,yoff,
     *	zlev,ndecl,lwgtl,lclr,nlev,hgt,narc,ang) 
C** 
C**    @(#)conseg.F	1.2    7/30/88
C**
C**
C***********************************************************************
C**
C**                 PLOT+ Scientific Graphics System
C**
C***********************************************************************
C**
C**
c	plots contours of grid z(i,j) in square segments of length lseg 
c	grid units per side.
c	oceanography emr  dec/1969 
c 
#ifdef unix
	include 'pplinc/MISS.INC'
#else
	include 'pplinc:miss.inc'
#endif
	dimension zz(57,57),z(nxs,nys),px(nx),py(ny)
	dimension zlev(2),ndecl(2),lwgtl(2),lclr(2)
	izz=57 
	lseg=(izz-1)-2 
c**
c**	rotation by angle ang added  (in degrees)
c**
	xc = (px(1)+px(nx))*0.5
	yc = (py(1)+py(ny))*0.5
c 
	ia=1 
40	ib = min0(ia+lseg,nx) 
	lrem = nx-ib 
	if( lrem*(lseg-lrem))60,60,50 
50	ib = (1+ia+nx)/2 
60	iam = max0(ia-1,1) 
	ibm = min0(ib+1,nx) 
	lxm = ibm-iam 
	iim = 1+(ia-iam)
	nxx =(ib-ia)+1 
c 
c 
	ja=1 
80	jb = min0(ja+lseg,ny) 
	lrem = ny-jb 
	if(lrem*(lseg-lrem))100,100,90 
90 	jb =(1+ja+ny)/2 
100	jam = max0(ja-1,1) 
	jbm = min0(jb+1,ny) 
	lym = jbm-jam 
	jjm = 1+(ja-jam)
	nyy = (jb-ja)+1 
c 
c 
	do 120 i = iam,ibm 
	ii =  i-iam+1 
	do 120 j = jam,jbm 
	jj =  j-jam+1 
	if((zle.and.z(i,j).le.cmzle).or.
     *	   (zeq.and.z(i,j).eq.cmzeq).or.
     *	   (zge.and.z(i,j).ge.cmzge))then
	    zz(ii,jj)=1.e35
	else
	    zz(ii,jj)=z(i,j)
	endif
120	continue
c 
c 
150	continue
	call contur(zz,izz,izz,px(iam),py(jam),nxx,nyy,xf,xoff,yf,yoff,
     *	zlev,ndecl,lwgtl,lclr,nlev,hgt,narc,ang,xc,yc) 
c 
c 
	ja=jb 
	if(ja-ny)80,160,160 
160	ia=ib 
	if(ia-nx)40,170,170 
170	return 
	end 
