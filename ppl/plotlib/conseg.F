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
c
C 
C       lots of places ---> PPLMOD_PPLCMD ---> PPLMOD_PLTIT 
C       ---> PPLMOD_PLOTZ ---> PPLMOD_CONSEG
C       
C       This subroutine plots contours of grid z(i,j) in square segments
C       of length lseg grid units per side.
C       
C       z      - actual 2-D array of data values to be contoured
C       nxs    - number of X grid points
C       nys    - number of Y grid points
C       px     - array of grid points along the X axis
C       py     - array of grid points along the Y axis
C       nx     - number of X grid points
C       ny     - number of Y grid points
C       xf     - factor to go from X axis to page coordinates?
C       xoff   - X axis offset in page coordinates?
C       yf     - factor to go from Y axis to page coordinates?
C       yoff   - Y axis offset in page coordinates?
C       zlev   - array of contour level values (lo val --> hi val)
C       ndecl  - number of decimal places in the contour labels
C       lwgtl  - weight of the contour line
C       lclr   - 
C       nlev   - number of levels to use
C       hgt    - height of characters of labels in inches
C       narc   - number of arc subsegments to use in contouring
C       ang    - angle to rotate the contour plot
C       
C       
C	oceanography emr  dec/1969 
C       *jc* 5.97 Major clean up and commenting
C       *jc* 5.97 Added code for curvilinear coordinates

#ifdef unix
	include 'pplinc/MISS.INC'
#else
	include 'pplinc:miss.inc'
#endif
	dimension zz(57,57),z(nxs,nys),px(nx),py(ny)
	dimension zlev(2),ndecl(2),lwgtl(2),lclr(2)
	izz=57 
	lseg=(izz-1)-2 

C       Calculate center point about which to rotate plot
C       Plot will be rotated by 'ang' degrees.

	xc = (px(1)+px(nx))*0.5
	yc = (py(1)+py(ny))*0.5

C       By the way, "if (exp) 1,2,3" means:
C       "if (exp<0) goto 1; if (exp=0) goto 2; if (exp>0) goto 3"

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

c *kob* 8/97 had to move the continuation marker to col 5 - it was in col 8
c       this bombed aix compiler, though it seems to work on others
	call contur(zz,izz,izz,px(iam),py(jam),nxx,nyy,xf,xoff,yf,yoff,
     1    zlev,ndecl,lwgtl,lclr,nlev,hgt,narc,ang,xc,yc) 
	

C       This is the Y axis loop

	ja = jb 
	if (ja-ny) 80,160,160

C       This is the X axis loop

 160	ia = ib 
	if (ia-nx) 40,170,170 

 170	return 
	end 
