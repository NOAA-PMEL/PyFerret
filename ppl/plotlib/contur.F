	subroutine contur(z,nxs,nys,px,py,nx,ny,xf,xoff,yf,yoff,
     *	zlev,ndecl,lwgtl,lclr,nlev,hgt,narc,ang,xct,yct) 
C** 
C**    @(#)contur.F   1.4    12/9/88
C**
C**
C***********************************************************************
C**
C**                 PLOT+ Scientific Graphics System
C**
C***********************************************************************
C**
C**
c     given heights z defined over a rectangular grid,contours are drawn 
c     at specified levels. there are provisions for labeling the 
c     contours,blanking regions of the rectangle,and making the contours 
c     more smooth flowing. 
c 
c     z(i,j),i=1,nx j=1,ny  is the input array of heights.points in 
c          undefined regions of z should be set to 1.e35 
c     x1,y1,xl,yl are the lower left and upper right corners of 
c          the grid on the plotter in inches. 
c     zlev(k),k=1,nlev is the array of levels to be contoured 
c     ndecl(k),k=1,nlev gives the number of decimal places in the 
c          contour labels. -1=no decimal. -2 or less=no label. 
c     lwgtl(k),k=1,nlev  gives the weight of the contour line. 
c          1 or less=standard line. 2=heavy line. 3=dotted line. 
c     hgt = height of characters of labels in inches. 
c     narc = 1,2,3 ... 10 .  an arc of narc subsegments  replaces each 
c     st line segment of a contour. the arc will match slopes with 
c     the adjacent arcs.   care should be taken here as overlapping of 
c     contours is possible when narc is used. narc=1 has no effect. 
c**
c**	ang = angle to rotate the contour plot
c**	xct = x center about which to rotate it
c**	yct = y center about which to rotate plot
c**
c     ind10 should be dimensioned at least nx by ny/10 rounded up. 
c     kab should be dimensioned at least nx/10+2 by ny/10+2 . 
c 
c     oceanography emr   dec/69 
c     Applied fix received from Don Denbo, 18 Apr 89 (NNS)
c
	dimension z(nxs,nys),px(nx),py(ny),s(1200),ind10(57,6)
	dimension x(1200),y(1200)
	dimension zmaxb(10,10),zminb(10,10),kab(07,07) 
	dimension lclr(2)
	dimension  zlev(2),ndecl(2),lwgtl(2),xc(11),yc(11),ixpon(10) 
	dimension isin(4),ind(4),xx(4),yy(4),zz(4),lused(4),kabov(4) 
	common/rotcon/xk,yk,snthta,csthta
#ifdef unix
	include 'pplinc/DASHZZ.INC'
	include 'pplinc/CONLAB.INC'
	include 'pltinc/PLTCOM.DAT'
#else
	include 'pplinc:dashzz.inc'
	include 'pplinc:conlab.inc'
	include 'pltinc:pltcom.dat'
#endif
	integer*4 ixpon,ind10,jbit,ind1,indl
	character frmt*20,str*40
	integer colold
c 
c     initialize 
c*******************************************************************************
c 
	if(nlev)1200,1200,12 
12	pi=3.1415926 
c
c	apply the rotation and scaling
c
	thta=ang*pi/180.
	snthta=sin(thta)
	csthta=cos(thta)
	xxct=xct*xf+xoff
	yyct=yct*yf+yoff
	xk=-xxct*csthta+yyct*snthta+xxct
	yk=-yyct*csthta-xxct*snthta+yyct
c	dslab=100. 
c	dslab=5.
	colold=colour
	icolro=0
	slab1f=.4 
	isin(1)=0 
	isin(2)=1 
	isin(3)=0 
	isin(4)=-1 
	big=.9e35 
	nxm1=nx-1 
	nym1=ny-1 
	hhgt = hgt /2. 
	ny10=(ny-1)/10+1 
	do 17 jbit=1,10 
17	ixpon(jbit)=4**(jbit-1) 
c 
c     get zmax,zmin,zrng,weezee 
c*******************************************************************************
c 
	iblk=max0(nx/10+1,5) 
	jblk=max0(ny/10+1,5) 
	zmax=-1.e35 
	zmin=1.e35 
	ib=0 
	do 55 istrt=1,nx,iblk 
	ib=ib+1 
	iend=min0(istrt+iblk,nx) 
	jb=0 
	do 55 jstrt=1,ny,jblk 
	jb=jb+1 
	jend=min0(jstrt+jblk,ny) 
	zmaxbl=-1.e35 
	zminbl=1.e35 
	do 50 i=istrt,iend 
	do 50 j=jstrt,jend 
	zij=z(i,j) 
	if(zij.gt.big) go to 50 
	zmaxbl=amax1(zij,zmaxbl) 
	zminbl=amin1(zminbl,zij) 
50	continue 
	zmax=amax1(zmaxbl,zmax) 
	zmin=amin1(zminbl,zmin) 
	zmaxb(ib,jb)=zmaxbl 
	zminb(ib,jb)=zminbl 
55	continue 
	zrng=(zmax-zmin)*1.1 
	if(zrng)1200,1200,60 
60	weezee = zrng*.0002 
c 
c     main loop over all contour levels 
c*******************************************************************************
c 
	do 1100  lev=1,nlev 
	zc = zlev(lev) 
c**	if((zmax-zc)*(zc-zmin))1100,1100,70 
c** replace with a less prone to overflow calculation
c**
	if(zc.ge.zmax .or. zc.le.zmin)goto 1100
70	ndec = ndecl(lev) 
	icolr=lclr(lev)
	lwgt = lwgtl(lev) 
	do 75 i=1,nx 
	do 75 j=1,ny10 
75	ind10(i,j)=0 
c 
c     get label width 
c*******************************************************************************
c
	if(ndec.ge.0)then
	    write(frmt,999)ndec
999	    format('(a10,f20.',i2.2,',a10)')
	    write(str,frmt)conpre,zc,conpst
	else
	    write(str,998)conpre,int(zc+sign(0.5,zc)),conpst
998	    format(a10,i20,a10)
	endif
	call squish(str,1,40)
	nchar=lnblk(str,40)
	width=symwid(hgt,nchar,str)+hhgt
	if(icolr.ne.icolro)then
	    if(icolr.eq.0)then
		call color(colold)
	    else
		call color(icolr)
	    endif
	    icolro=icolr
	endif
c	nchar = ndec+2 
c	if(nchar)120,120,100 
c100	abzrnd = abs(zc)+.5/10.**max0(0,ndec) 
c	do 110 k=1,10 
c	if(abzrnd-10**k)120,110,110 
c110	nchar=nchar+1 
c120	nchar=nchar+1 
c	if(ndec.lt.0)cthen
c	    width= hgt *(nchar*9.+8.)/8.
c	else
c	    width= hgt *((nchar-1)*9.+13.)/8.
c	endif 
c 
c     loop over the grid searching for a contour entering square ii,jj 
c     thru side 1 or 4.      (the contour must be unused) 
c*******************************************************************************
c 
	ib=0 
	do 1010 istrt=1,nx,iblk 
	iend=min0(istrt+iblk,nx+1) 
	ib=ib+1 
	jb=0 
	do 1010 jstrt=1,ny,jblk 
	jb=jb+1 
	if(zmaxb(ib,jb).lt.zc .or. zminb(ib,jb).gt.zc) go to 1010 
	jend=min0(jstrt+jblk,ny+1) 
	do 200 i=istrt,iend 
	ikab=i-istrt+1 
	do 200 j=jstrt,jend 
	jkab=j-jstrt+1 
	kabij=10 
	if(i.gt.nx .or. j.gt.ny) go to 200 
	zij=z(i,j) 
	if(zij.gt.big) go to 200 
	if(abs(zij-zc).le.weezee) zij=zc+sign(weezee,zij-zc) 
	kabij=-1 
	if(zij.gt.zc) kabij=1 
200	kab(ikab,jkab)=kabij 
	iendm1=iend-1 
	jendm1=jend-1 
	do 1000 ii=istrt,iendm1 
	ikab=ii-istrt+1 
	do 1000 jj=jstrt,jendm1 
	jkab=jj-jstrt+1 
	i=ii 
	j=jj 
	kabij=kab(ikab,jkab) 
	kabip1=kab(ikab+1,jkab) 
	kabjp1=kab(ikab,jkab+1) 
	jwrd=(j-1)/10+1 
	j2bit=j-(jwrd-1)*10 
	ind1=ind10(i,jwrd)/ixpon(j2bit) 
	ind1=ind1-(ind1/4)*4 
	lused4=ind1/2 
	lused1=ind1-2*lused4 
	iretrn=300 
	ll=1 
	if(kabij+kabip1+lused1.eq.0) go to 540 
	ll=4 
	if(kabij+kabjp1+lused4.eq.0) go to 540 
	go to 1000 
300	continue 
340	lin = ll 
	k=1 
	nseg = 1 
c 
c     given entrance to square(i,j) on side lin, record the entrance 
c     point x(k),y(k).  set lin to used. 
c*******************************************************************************
c 
350	lp1 = lin+1-(lin/4)*4 
	lp2 = lp1+1-(lp1/4)*4 
	lm1 = lp2+1-(lp2/4)*4 
	if(iretrn-360)355,360,360 
355	k=k+1 
	frac = (zc-zz(lin))/(zz(lp1)-zz(lin)) 
	x(k) = xx(lin)+ (xx(lp1)-xx(lin))*frac 
	y(k) = yy(lin)+ (yy(lp1)-yy(lin))*frac 
	iind=i+lp2/4 
	jind=j+lp1/4 
	lhor = lin-(lin/2)*2 
	jwrd=(jind-1)/10+1 
	j2bit=jind-(jwrd-1)*10 
#ifdef unix
	ind10(iind,jwrd)=ind10(iind,jwrd)+int(2.-lhor)*ixpon(j2bit) 
#else
	ind10(iind,jwrd)=ind10(iind,jwrd)+jint(2.-lhor)*ixpon(j2bit) 
#endif
c 
c     see if an exit exists on side l-1,l+1,or l+2. if so choose the one 
c     closest to side l. if the exit is already used terminate x,y. 
c*******************************************************************************
c 
360	iretrn = 350 
	lexit = lm1 
	if(kabov(lin)+kabov(lm1))380,370,380 
370	if(kabov(lp1)+kabov(lp2))450,390,450 
380	if(kabov(lp1)+kabov(lp2))410,400,410 
390	flm1 = (zc-zz(lin))/(zz(lm1)-zc) 
	flp1 = (zc-zz(lp1))/(zz(lp2)-zc) 
	if(lused(lp1).eq.1) go to 450 
	if(flm1.le.flp1 .and. lused(lm1).eq.0) go to 450 
400	lexit = lp1 
	go to 450 
410	lexit = lp2 
	if(kabov(lp2)+kabov(lm1))470,450,470 
450	if(lused(lexit))530,530,460 
460	kmax=k+1 
	x(kmax)=x(2) 
	y(kmax)=y(2) 
	x(1) = x(k) 
	y(1)=y(k) 
	x(k+2)=x(3) 
	y(k+2)=y(3) 
	klosed=1 
	go to 700 
c 
c     no exit.  if nseg=1 reverse x,y and continue. 
c               if nseg=2 terminate x,y. 
c*******************************************************************************
c 
470	if(kabov(lp2)+kabov(lm1)-15)480,480,500 
480	kda=lin 
	kdb=lp2 
	if(kabov(lp2)-5)495,495,490 
490	kda = lm1 
	kdb = lp1 
495	k=k+1 
	frac = (zc-zz(kda))/(zz(kdb)-zz(kda)) 
	x(k) = xx(kda) + (xx(kdb)-xx(kda))*frac 
	y(k) = yy(kda) + (yy(kdb)-yy(kda))*frac 
500	if(nseg-1)510,510,505 
505	kmax=k 
	x(1)=x(2) 
	y(1)=y(2) 
	x(k+1)=x(k) 
	y(k+1)=y(k) 
	klosed=0 
	go to 700 
510	iretrn=360 
	nseg = 2 
	kh = 1+k/2 
	do 520 kk=2,kh 
	kkr = k+2-kk 
	xkk=x(kk) 
	ykk=y(kk) 
	x(kk)= x(kkr) 
	y(kk)= y(kkr) 
	x(kkr)=xkk 
520	y(kkr)=ykk 
	i=ii 
	j=jj 
	lexit=ll 
c 
c     find square entered by present exit. get xx,yy,zz,kabov for each 
c     corner and lused for each side. 
c*******************************************************************************
c 
530	i= i+ isin(lexit) 
	jsub=5-lexit 
	j= j +isin(jsub) 
	lin=lexit+2 - ((lexit+1)/4)*4 
c 
c 
540	do 620 l=1,4 
	jl=j+(l-1)/2 
	lp1=l+1-(l/4)*4 
	il=i+(lp1-1)/2 
	ind(l)=0 
	zz(l)=1.e35 
	kabov(l)=10 
	if(il*(nx+1-il))610,610,550 
550	if(jl*(ny+1-jl))610,610,560 
560	if(z(il,jl)-big)570,610,610 
570	jwrd=(jl-1)/10+1 
	j2bit=jl-(jwrd-1)*10 
	indl=ind10(il,jwrd)/ixpon(j2bit) 
	ind(l)=indl-(indl/4)*4 
	zz(l)=z(il,jl) 
	if(abs(zz(l)-zc)-weezee)580,580,590 
580	zz(l)=zc+sign(weezee,zz(l)-zc) 
590	kabov(l)=-1 
	if(zz(l)-zc)610,610,600 
600	kabov(l)=1 
610	xx(l)= px(il)*xf + xoff
620	yy(l)= py(jl)*yf + yoff
	lused(4) = ind(1)/2 
	lused(1) = ind(1)-2*lused(4) 
	lused(2) = ind(2)/2 
	lused(3) = ind(4)-(ind(4)/2)*2 
	if(iretrn-350) 300,350,350 
c 
c     the arrays x,y are now complete. calculate distance s along 
c     the contour. start plotting the contour 
c*******************************************************************************
c 
700	s(2)=0. 
	if(kmax-2)1000,1000,720 
720	do 750 k=3,kmax 
	dxx = x(k)-x(k-1) 
	dyy = y(k)-y(k-1) 
750	s(k) = s(k-1)+ sqrt(dxx*dxx+dyy*dyy) 
	smax = s(kmax) 
	slab1 = smax*slab1f 
	stest = max1(0.0,dslab-slab1)
	k=2 
	call plotny(x(2),y(2),3,0) 
c 
c     check conditions for labelling. 
c*******************************************************************************
c 
755	if(ndec+2)900,900,760 
760	km1= max0(k-1,2) 
	stest = stest + s(k)-s(km1) 
	if(stest-dslab)900,770,770 
770	if(smax-s(k)-2.0*width)900,900,780 
780	kp1=k+1 
	if(lev.eq.1) go to 785 
	dlev=abs(zlev(lev)-zlev(lev-1)) 
	xt = (x(k) - xoff)/xf
	yt = (y(k) - yoff)/yf
	do 781 i=1,nx-1
	if(xt.ge.px(i) .and. xt.le.px(i+1))goto 782
781	continue
782	do 783 j=1,ny-1
        if(yt.ge.py(j) .and. yt.le.py(j+1))goto 784
783	continue
784	i=min0(i,nx-1) 
	j=min0(j,ny-1) 
	dx=(px(i+1)-px(i))*xf
	dy=(py(j+1)-py(j))*yf
	if(z(i+1,j).ge.big)then
	    dzdx=0.0
	else
	    dzdx=(z(i+1,j)-z(i,j))/dx 
	endif
	if(z(i,j+1).ge.big)then
	    dzdy=0.0
	else
	    dzdy=(z(i,j+1)-z(i,j))/dy 
	endif
c**	dzdg=sqrt(dzdx*dzdx+dzdy*dzdy) 
c** replace with a less prone to overflow calculation
c**
	dzdg=abs(dzdx)+abs(dzdy)
	if(dzdg.eq.0.) go to 785 
	cspace=dlev/dzdg 
	if(cspace.lt.hgt*.90) go to 900 
785	continue 
	do 800 kk = kp1,kmax 
	dxx = x(kk)-x(k) 
	dyy = y(kk)-y(k) 
	space = sqrt(dxx*dxx+dyy*dyy) 
	ark =s(kk)-s(k) 
	if(space-width)800,790,790 
790	if(space/ark-.95)900,810,810 
800	continue 
	go to 900 
c 
c     draw the label 
c*******************************************************************************
c 
810	stest=0. 
c	do 812 kkk=1,5 
c	dxx=x(kk-1)+(x(kk)-x(kk-1))*.2*kkk-x(k) 
c	dyy=y(kk-1)+(y(kk)-y(kk-1))*.2*kkk-y(k) 
c	space=sqrt(dxx*dxx+dyy*dyy) 
c	if(space-width)812,812,815 
c812	continue 
c815	continue 
c	xendl = x(k) + width*dxx/space 
c	yendl = y(k) + width*dyy/space 
	xa=x(kk-1)-x(k)
	xb=x(kk)-x(kk-1)
	ya=y(kk-1)-y(k)
	yb=y(kk)-y(kk-1)
	aa=xb*xb+yb*yb
	bb=xa*xb+ya*yb
	cc=xa*xa+ya*ya-width*width
	zxy=(-bb+sqrt(bb*bb-aa*cc))/aa
	dxx=xa+xb*zxy
	dyy=ya+yb*zxy
	xendl=x(k)+dxx
	yendl=y(k)+dyy
c**
c**	add rotation code for label
c**
	xst=x(k)*csthta-y(k)*snthta+xk
	yst=y(k)*csthta+x(k)*snthta+yk
	xstp=xendl*csthta-yendl*snthta+xk
	ystp=yendl*csthta+xendl*snthta+yk
	dxx=xstp-xst
	dyy=ystp-yst
	angle = 90. 
        if(dyy.lt.0.0)angle=-90.
	if(dxx)820,830,820 
820	angle = atan(dyy/dxx)*180./pi 
830	if(dxx) 850,840,840 
840	xlab = xst+ hhgt*(0.5*dxx+dyy)/width
	ylab = yst+ hhgt*(0.5*dyy-dxx)/width
	go to 860 
850	xlab = xstp-hhgt*(0.5*dxx+dyy)/width
	ylab = ystp-hhgt*(0.5*dyy-dxx)/width
860	call symbel(xlab,ylab,angle,hgt,nchar,str)
	call plotny(xendl,yendl,3,0) 
	call plotny(x(kk),y(kk),2,lwgt) 
	k=kk 
	go to 920 
c 
c     plot the segment from xk,yk to  x(k+1),y(k+1). 
c*******************************************************************************
c 
900	call   arc(x(k-1),y(k-1),narc+1,.001,xc,yc,npt)      
	do 910 kk = 2,npt                          
910	call plotny(xc(kk),yc(kk),2,lwgt)                    
	k=k+1 
920	if(k-kmax)755,1000,1000 
1000	continue 
1010	continue 
c 
1100	continue 
1200	if(colour.ne.colold)call color(colold)
	return 
	end 
