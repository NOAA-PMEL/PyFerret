#ifndef unix
	OPTIONS /CHECK=(NOBOUNDS,NOUNDERFLOW) 	! OVERFLOWS BOUNDS (see below)
#endif

	subroutine contur(z,nxs,nys,px,py,nx,ny,xf,xoff,yf,yoff,
     *	zlev,ndecl,lwgtl,lclr,nlev,hgt,narc,ang,xct,yct)

* revised for TMAP
* from PPLUS :
*	Directory PMEL:[PLOT5.PPLV11.PLOTLIB]CONTUR.FOR

* search for "TMAP" to find all revisions
* *sh*  - 1/23/89 added interrupt detection
* *sh*  - 2/15/89 corrected ATAN error for vertical labels
*		  ... no explicit declarations (too much trouble !) so it must
*		      be compiled with /WARNINGS=NODECLARATIONS
* !!! ALSO :  Routine overflows bounds at line
*		610	xx(l)= px(il)*xf + xoff
*	      ...bounds checking is OFF to match Denbo's compile stmt
* *sh*  - applied Don's 12/9/88 mods (creating v1.4 of code CONTUR)
*		... replaced earlier ATAN patch with Don's version
*
* *tc*  - 1/15/93 put in option for 'draftsman' style (horz) contour labels, 
*            ...logical variable draftsman turns on or off
* *tc*  - 1/26/93 put in spline under tension fit for contour lines.  uses
*                 spline_tension variable to determine tension.
* *tc*  - 5/04/93 mod to correct bug resulting zag near labels, with tension
*		  > 0, and recapture original behavior w tension=0.
*
* *jd	- 3/29/94 fix bug where specification of zero height contour labels
*		  bombs out on ALPHA machines (divide by zero)

****************************************************************************
C**
C**    @(#)contur.F	1.2    7/30/88
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
	dimension z(nxs,nys),px(nx),py(ny),s(1200),ind10(57,6)
	dimension x(1200),y(1200)
	dimension zmaxb(10,10),zminb(10,10),kab(07,07)

	dimension  zlev(2),  ndecl(2),lwgtl(2),xc(11),yc(11),ixpon(10)
	dimension lclr(2)

	dimension isin(4),ind(4),xx(4),yy(4),zz(4),lused(4),kabov(4)
	common/rotcon/xk,yk,snthta,csthta
! TMAP mod, spline fit:
! added gkscm2.cmn which includes spline_tension and draftsman
#ifdef unix
	include 'tmap_pplv11inc/gkscm2.cmn'
	include 'tmap_pplv11inc/dashzz_inc.decl'
	include 'pplv11inc/DASHZZ.INC'
	include 'tmap_pplv11inc/conlab_inc.decl'
	include 'pplv11inc/CONLAB.INC'
	include 'tmap_pplv11inc/pltcom_dat.decl'
	include 'pplv11inc/PLTCOM.DAT'
#else
	INCLUDE 'tmap_pplv11inc:tmap_gkscm2.inc'
	INCLUDE 'tmap_pplv11inc:tmap_dashzz.inc'
	INCLUDE 'tmap_pplv11inc:tmap_conlab.inc'
	INCLUDE 'tmap_pplv11inc:tmap_pltcom.dat'
#endif

* TMAP interrupt flag
#ifdef unix
	include 'tmap_pplv11inc/xinterrupt.cmn'
#else
	INCLUDE 'tmap_pplv11inc:tmap_xinterrupt.cmn'
#endif

	integer*4 ixpon,ind10,jbit,ind1,indl
	character frmt*20,str*40
	integer colold
! TMAP mod, horz contour labels:
! set up arrays for storing labels and such.  mlab is the maximum number
! of contour labels expected on any ONE contour line in the plot.
	parameter (mlab=200)
	real xlabp(mlab),ylabp(mlab),xlabc(mlab),ylabc(mlab),widthp

C	Place init into data stmt *jd* 12.96 linux
	integer	nlabp,nlabc
	data	nlabp /0/,nlabc /0/

	integer iatime(3)
! TMAP mod, spline fit:
	parameter (msplsz=2000)
	real splxp(msplsz),splyp(msplsz),spltmp(msplsz),spls(msplsz)
	integer splier
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
! TMAP mod, horz contour labels:
! remember where the contour labels were for the previous contour line.
! these will be compared to the location of labels on the current line to
! determine any conflicts.
	if (draftsman) then
	  do 1102, k1=1,nlabc
	    xlabp(k1)=xlabc(k1)
	    ylabp(k1)=ylabc(k1)
1102	  continue
	  nlabp=nlabc
	  nlabc=0
	  widthp=width
	endif
	IF ( interrupted ) GOTO 1200		! TMAP interrupt processing
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
! TMAP mod, spline fit:
! initialize spline work arrays if using spline under tension fit.
	if (spline_tension.ne.0.) then
	  call kurv1(kmax-1,x(2),y(2),0.,0.,3,splxp,splyp,
     ,      spltmp,spls,spline_tension,splier)
	  if (splier.ne.0) then
	    print*,' error in pplmod_contur.F, call to kurv1 ',splier
	  endif
	endif
! TMAP mod, horz contour labels:
! this is where the initial offset is set for a contour label on a contour
! line.  the first label is shifted in dslab-stest distance from the edge, not 
! dslab as might be expected.  modify the stest line to get something you like.
	if (draftsman) then
	  stest = max1(0.0,dslab-slab1)                 ! original code
c
c	  stest = max1(0.0,dslab-dslab*0.5*		! shifted by 1/2 the
c     .      (smax/dslab-float(int(smax/dslab))))	! excess of n dslab labels
c
c	  call itime(iatime)				! more of a random
c	  slab1 = smax*0.4*rand(iatime(3)*k)+0.1	! starting point
c	  stest = max1(0.0,dslab-slab1)			! iatime(3) is sec
c
c	  stest = max1(0.0,dslab-width)			! inset width
c
c	  stest = max1(0.0,dslab*(1.0-0.2))		! inset 0.2 * dslab
c
	else
	  stest = max1(0.0,dslab-slab1)
	endif
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
! TMAP mod 4/19/89 - applied fixes from DON
!	if(yt.ge.py(i) .and. yt.le.py(i+1))goto 784	! original code
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
! TMAP mod, horz contour labels:
! this section checks to make sure that at this x,y, the contour label
! will not hit another contour line if it gets written.  it does this
! by comparing the amount of space between contour lines versus the 
! width and height of the label.  this should work most of the time, but
! not all the time because it only checks at this one point.  if the contour 
! lines are very wigly, it's possible that a label could be written that would
! overlie another contour line.  hopefully the check on space/arc would
! remove this case though.
	if (draftsman) then
          if (dzdy.ne.0.) then
	    cspacey=abs(dlev/dzdy)
            if (cspacey.lt.hgt*1.05) go to 900
	  endif
	  if (dzdx.ne.0.) then
	    cspacex=abs(dlev/dzdx)
            if (cspacex.lt.width*1.05) go to 900
	  endif
        else
	  if(dzdg.eq.0.) go to 785
	  cspace=dlev/dzdg
	  if(cspace.lt.hgt*.90) go to 900
        endif
785	continue
	do 800 kk = kp1,kmax
	dxx = x(kk)-x(k)
	dyy = y(kk)-y(k)
	space = sqrt(dxx*dxx+dyy*dyy)
	ark =s(kk)-s(k)
! TMAP mod, horz contour labels:
! this section decides whether a label can fit along this section of contour 
! line.  the label will fit if the gap in the contour line is EITHER wide enough 
! (dxx >~ spacex) OR high enough (dyy >~ hgt) where spacex and spacey is the 
! amount of space (in x or y) that is needed to write the label.  change spacex 
! and spacey to alter the gap size.  spacex and spacey are also used farther 
! below.  spacex=width is pretty straight forward.  spacey=hgt should be the
! appropriate y space but this doesn't work so well.  adding a little extra
! hgt was necessary and then the slope of the contour line needed to be taken
! into account.  this was found by trial and error.
	if (draftsman) then
	  spacey=hgt+0.4*hgt+hgt*abs(dyy/space)*abs(dyy/space)
	  spacex=width
	  if(abs(dyy)-spacey)791,792,792
791	  if(abs(dxx)-spacex)800,792,792
792	  if(space/ark-.95)900,810,810
	else
	  if(space-width)800,790,790
790	  if(space/ark-.95)900,810,810
        endif
800	continue
	go to 900
c
c     draw the label
c*******************************************************************************
c
! TMAP mod, horz contour labels:
! move the stest=0. to just after symbel is called.  this has the same
! effect and allows for further conflicts to be checked below without reseting
! this parameter.  setting this parameter to 0. effectively causes the next
! label to be written dslab distance over while not reseting stest indicates
! that the label was not written and that there should only be a minor shift
! before trying to write the label again.
!810	stest=0.
810	continue
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
! TMAP mod, horz contour labels:
! this section determines the x and y extent of the gap in the contour
! line.  dxx and dyy depend on whether the label fits in an x-dir
! gap or a y-dir gap.  basically, dxx =~ width of label and dyy =~ ya+yb
! if x-dir and dyy =~ hgt and dxx =~ xa+xb if y-dir.  need to watch for
! xb or yb equal to zero.  if there is a divide by zero, handle it a 
! little differently.
	if (draftsman) then
          if (abs(dyy).ge.spacey.and.abs(dxx).ge.spacex) then
            if (abs(dyy)/spacey.lt.abs(dxx)/spacex) then
	      dxx=spacex*sign(1.0,(x(kk)-x(k)))
	      if (xb.eq.0.) then
	        dyy=spacey*sign(1.0,(y(kk)-y(k)))
	        print *,' pplmod_contur.F, note xb.eq.0 in contur ',str(1:4)
	      else
	        dyy=ya+yb*(spacex-abs(xa))/abs(xb)
	      endif
            else
	      dyy=spacey*sign(1.0,(y(kk)-y(k)))
	      if (yb.eq.0.) then
	        dxx=spacex*sign(1.0,(x(kk)-x(k)))
	        print *,' pplmod_contur.F, note yb.eq.0 in contur ',str(1:4)
	      else
	        dxx=xa+xb*(spacey-abs(ya))/abs(yb)
	      endif
            endif
	  elseif (abs(dyy).ge.spacey) then
	    dyy=spacey*sign(1.0,(y(kk)-y(k)))
	    if (yb.eq.0.) then
	      dxx=spacex*sign(1.0,(x(kk)-x(k)))
	      print *,' pplmod_contur.F, note yb.eq.0 in contur ',str(1:4)
	    else
	      dxx=xa+xb*(spacey-abs(ya))/abs(yb)
	    endif
          elseif (abs(dxx).ge.spacex) then
	    dxx=spacex*sign(1.0,(x(kk)-x(k)))
	    if (xb.eq.0.) then
	      dyy=spacey*sign(1.0,(y(kk)-y(k)))
	      print *,' pplmod_contur.F, note xb.eq.0 in contur ',str(1:4)
	    else
	      dyy=ya+yb*(spacex-abs(xa))/abs(xb)
	    endif
	  else
	    print*,' error e1 in pplmod_contur.F, whoa baby, stop me'
	    stop
	  endif
	else
	  dxx=xa+xb*zxy
	  dyy=ya+yb*zxy
	endif
	xendl=x(k)+dxx
	yendl=y(k)+dyy
! TMAP mod, spline fit:
! modify xendl and yendl if doing a spline_tension fit.  xendl and yendl are
! between fixed points.  x(k),y(k) and x(kk),y(kk) are fixed points and a spline
! fit to those points will just return themselves, so don't bother.  if narc
! is less than 2, then no interpolation points will be calculated and this
! step can be skipped.  it is possible that the transformed xendl and yendl
! will not satisfy all the label criteria for gap size, arc, etc.  if this
! leads to labels looking poor, a fix will be needed.
	if (narc.gt.1.and.spline_tension.ne.0.) then
          splt=(spls(kk-1)-((x(kk)-xendl)**2+(y(kk)-yendl)**2)**0.5)/
     /         spls(kmax-1)
          call kurv2(splt,xs,ys,kmax-1,x(2),y(2),
     *               splxp,splyp,spls,spline_tension)
	  xendl=xs
	  yendl=ys
	endif
c**
c**	add rotation code for label
c**
	xst=x(k)*csthta-y(k)*snthta+xk
	yst=y(k)*csthta+x(k)*snthta+yk
	xstp=xendl*csthta-yendl*snthta+xk
	ystp=yendl*csthta+xendl*snthta+yk
	dxx=xstp-xst
	dyy=ystp-yst
c
! TMAP mod, horz contour labels:
! set angle, xlab, and ylab where xlab,ylab is the starting position where
! the label should be written.  also, loops 818 and 819 make sure that the
! contour label won't hit it's own contour line somewhere else.  this is not 
! an issue with non-draftsman labels because the label is always along a 
! segment of the contour line.  with draftsman labels, the label may only cut 
! into a small segment of the contour line.  then later, the contour line may 
! turn sharply, come back, and cross over the just written label.  if this is
! the case, don't write the label.
	if (draftsman) then
	  angle = 0.
	  xlab = (xst+xstp)*0.5-0.5*width+0.5*hhgt
	  ylab = (yst+ystp)*0.5-0.5*hgt
	  xcen = (xst+xstp)*0.5
	  ycen = (yst+ystp)*0.5
	  do 818 k1=1,k-1
	    if ((abs(x(k1)  -xcen).lt.(width*0.5)).or.
     .	        (abs(x(k1+1)-xcen).lt.(width*0.5))) then
	      if ((ycen-y(k1))*(ycen-y(k1+1)).lt.0.) goto 900
c	      if (abs(y(k1)-ycen).lt.hgt*0.5) goto 900
c	      if (abs(y(k1+1)-ycen).lt.hgt*0.5) goto 900
	    endif
818	  continue
	  do 819 k1=kk+1,kmax
	    if ((abs(x(k1)  -xcen).lt.(width*0.5)).or.
     .	        (abs(x(k1-1)-xcen).lt.(width*0.5))) then
	      if ((ycen-y(k1))*(ycen-y(k1-1)).lt.0.) goto 900
c	      if (abs(y(k1)-ycen).lt.hgt*0.5) goto 900
c	      if (abs(y(k1-1)-ycen).lt.hgt*0.5) goto 900
	    endif
819	  continue
	else
	  angle = 90. 
! TMAP patch - Don's version of ARCSIN fix - 4/19/89
          if(dyy.lt.0.0)angle=-90.
! end of TMAP patch 4/19/89
	  if(dxx)820,830,820 
820	  angle = atan(dyy/dxx)*180./pi 
830	  if(dxx) 850,840,840 
840	  if (width .eq. 0.0) then
	     xlab = xst
             ylab = yst
          else
             xlab = xst+ hhgt*(0.5*dxx+dyy)/width
	     ylab = yst+ hhgt*(0.5*dyy-dxx)/width
	  endif
	  go to 860
850	  if (width .eq. 0.0) then
	     xlab = xstp
             ylab = ystp
          else
	     xlab = xstp-hhgt*(0.5*dxx+dyy)/width
	     ylab = ystp-hhgt*(0.5*dyy-dxx)/width
	  endif
	endif
c
! TMAP mod, horz contour labels:
! this loop checks to make sure the current label is not going to conflict
! with a label written on the previous contour line.  the current contour
! label is compared with all nlabp previous labels on the previous contour
! line.  if there is an overlap, then don't write the current label.  the
! allowable 'closeness' can be adjusted by altering xdelc, ydelc, xdelp, and
! ydelp where the labels are about width/widthp wide and hgt/hgt tall.  the
! c at the end of the variable represents 'current' and the p represents the
! previous contour line.  cen is where the center of the label is and del is
! the amount of space from the center that the label wants all to itself.  line
! 860 was altered from the original contur.F.  it used to point to the call
! symbel line below.
860	continue
	if (draftsman) then
	  xcenc=(xst+xstp)*0.5
	  ycenc=(yst+ystp)*0.5
	  xdelc=width *0.5+hgt*0.15
	  ydelc=hgt   *0.5+hgt*0.12
	  xdelp=widthp*0.5+hgt*0.15
	  ydelp=hgt   *0.5+hgt*0.12
	  do 861 k1=1,nlabp
	    xcenp=xlabp(k1)+widthp*0.5
	    ycenp=ylabp(k1)+hgt*0.5
	    if (((xcenc-xdelc).lt.(xcenp+xdelp)).and.
     .	        ((xcenc+xdelc).gt.(xcenp-xdelp)).and.
     .	        ((ycenc+ydelc).gt.(ycenp-ydelp)).and.
     .	        ((ycenc-ydelc).lt.(ycenp+ydelp))) goto 900
861	  continue
	endif
! TMAP mod, horz contour labels:
! draw the label and reset stest to 0. which indicates that the label has
! been drawn.  the next label should be dslab distance over.
!860	call symbel(xlab,ylab,angle,hgt,nchar,str)	! original code (860)
	call symbel(xlab,ylab,angle,hgt,nchar,str)
	stest=0.
! TMAP mod, horz contour labels:
! this section of code stores the contour label locations for the current
! contour line so they can be compared to the labels for the next contour line
	if (draftsman) then
	  nlabc=nlabc+1
	  xlabc(nlabc)=xlab
	  ylabc(nlabc)=ylab
	endif
c
	call plotny(xendl,yendl,3,0)
! TMAP mod, spline fit:
! if spline_tension.ne.0 then you want to do a spline fit in the area
! of the label.  otherwise (original spline), just draw a straight line 
! between the 2 points and don't do the spline.  the original way of
! doing this is pretty ugly.
        if (spline_tension.eq.0.) then
	  call plotny(x(kk),y(kk),2,lwgt)		! original code
	  k=kk						! original code
	  go to 920					! original code
        else
	  k=kk-1					! start of new code
	  call arc(x(k-1),y(k-1),narc+1,.001,xc,yc,npt,
     ,      k-1,kmax-1,x(2),y(2),splxp,splyp,spls,spline_tension)
c
	  dendl=(xendl-x(k))**2+(yendl-y(k))**2
	  do 911 kk1 = 2,npt-1
	    if (((xc(kk1)-x(k))**2+(yc(kk1)-y(k))**2).gt.dendl)
     .        call plotny(xc(kk1),yc(kk1),2,lwgt)
911	  continue
	  call plotny(x(kk),y(kk),2,lwgt)
	  k=k+1
	  goto 920					! end of new code
        endif
c
c     plot the segment from xk,yk to  x(k+1),y(k+1).
c*******************************************************************************
c
! TMAP mod, spline fit:
! call arc with the ability to call the spline fit.  pass lots more variables.
900	call arc(x(k-1),y(k-1),narc+1,.001,xc,yc,npt,
     ,      k-1,kmax-1,x(2),y(2),splxp,splyp,spls,spline_tension)
c
	do 910 kk1 = 2,npt
910	call plotny(xc(kk1),yc(kk1),2,lwgt)
	k=k+1
920	if(k-kmax)755,1000,1000
1000	continue
1010	continue
c
1100	continue
1200	if(colour.ne.colold)call color(colold)
	return
	end

