        subroutine contur(z,nxs,nys,px,py,nx,ny,xf,xoff,yf,yoff,
     *	zlev,ndecl,lwgtl,lclr,nlev,hgt,narc,ang,xct,yct,xadd,lmin,lmax)

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
* *jc*  - 3/26/99 added transformation to curvilinear coordinates
*  v581 5/2005 *acm* Add CONTOUR/MOD capability for modulo replications for 
*                    curvilinear data
* V630  *acm* 9/09 change a message only
C  v67  2/2011 *acm* Fix ticket 1799; do contouring in larger sections.
* V68  *acm* 1/12  changes for double-precision ferret, single-precision pplus
* V687 *acm* 3/14 tickets 2151 and 2147: allow missing coordinate data in 3-arg 
*                 and 4-arg plot commands. 
* V6.95 *acm* 3/15 Changes for PyFerret fonts. 
* V6.96 *acm*12/15 Fix ticket 2328 - contour labels spilling outside plot axes 
* V7.2  *acm* 6/17 Fix ticket bug2544 for depth-axis plots the above needs
*                  different tests. If clipping is implented we can get rid of
*                  the changes for 2328 and 2544.
* 11/2022 *acm* Code cleanup: 
*              - Fix loops with DO termination statement which is not 
*                  END DO or CONTINUE, and loops that share termination statements
*              - Rework arithmetic IF statements 
*              - Remove blocks of code that are commented out 
*              - GOTO statements remain. Overall organization of the routine unchanged. 

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
c     If these dimensions are raised, increase the size of workspace
c     s, x, y, x_curv, y_curv proportionally.
c
c     oceanography emr   dec/69
	dimension z(nxs,nys),px(nx),py(ny),s(6000),ind10(300,40)
	dimension x(6000),y(6000)
	dimension zmaxb(10,10),zminb(10,10),kab(40,40)

	dimension  zlev(*),  ndecl(*),lwgtl(*),xc(11),yc(11),ixpon(10)
	dimension lclr(*)

	dimension isin(4),ind(4),xx(4),yy(4),zz(4),lused(4),kabov(4)
	common/rotcon/xk,yk,snthta,csthta
! TMAP mod, spline fit:
! added gkscm2.cmn which includes spline_tension and draftsman

	include 'gkscm2.cmn'
	include 'dashzz_inc.decl'
	include 'DASHZZ.INC'
	include 'conlab_inc.decl'
	include 'CONLAB.INC'
	include 'pltcom_dat.decl'
	include 'PLTCOM.DAT'
        include 'xcurvilinear.cmn'
        INCLUDE 'pyfonts.cmn'
	include 'axis_inc.decl'
	include 'AXIS.INC'
	include 'miss_inc.decl'
	include 'MISS.INC'

* TMAP interrupt flag

	include 'xinterrupt.cmn'

	integer*4 ixpon,ind10,jbit,ind1,indl, skiplab
	character frmt*20,str*40
	integer colold
! TMAP mod, horz contour labels:
! set up arrays for storing labels and such.  mlab is the maximum number
! of contour labels expected on any ONE contour line in the plot.
	parameter (mlab=200)
	real xlabp(mlab),ylabp(mlab),xlabc(mlab),ylabc(mlab),widthp
        realxadd,lmin,lmax, xloskp,xhiskp, yloskp,yhiskp

C	Place init into data stmt *jd* 12.96 linux
	integer	nlabp,nlabc
	data	nlabp /0/,nlabc /0/

	integer iatime(3)
! TMAP mod, spline fit:
	parameter (msplsz=2000)
	real splxp(msplsz),splyp(msplsz),spltmp(msplsz),spls(msplsz)
	integer splier

C     Curvilinear coordinates declarations
        integer icurv, status
	real*8 x_curv(6000), y_curv(6000)
c
c     initialize
c*******************************************************************************
c
	if (nlev.le.0) goto 1200
	pi=3.1415926

	igroup = fnt_all
        IF (pyfont .AND. pyf_set(icont)) igroup = icont

* For checking that labels are within the plot box	
        IF (pyfont) THEN
	   IF (xlo .LE. xhi) THEN 
	      xloskp = xlo
	      xhiskp = xhi
	   ELSE
	      xloskp = xhi
	      xhiskp = xlo
	   ENDIF
	   IF (ylo .LE. yhi) THEN 
	      yloskp = ylo
	      yhiskp = yhi
	   ELSE
	      yloskp = yhi
	      yhiskp = ylo
	   ENDIF
        ENDIF
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
	   ixpon(jbit)=4**(jbit-1)
17	continue	
c
c     get zmax,zmin,zrng,weezee
c*******************************************************************************
c
	iblk=max0(nx/10+1,5)
	jblk=max0(ny/10+1,5)
	zmax=-1.* cmzge
	zmin=cmzge
	ib=0

 	do 56 istrt=1,nx,iblk
 	   ib=ib+1
 	   iend=min0(istrt+iblk,nx)
 	   jb=0
 
 	   do 55 jstrt=1,ny,jblk
 	      jb=jb+1
 	      jend=min0(jstrt+jblk,ny)
 	      zmaxbl=-1*cmzge
 	      zminbl=cmzge
 
 	      do 51 i=istrt,iend
 	         do 50 j=jstrt,jend
 	            zij=z(i,j)
 	            if(zij.gt.big) cycle
 	            zmaxbl=amax1(zij,zmaxbl)
 	            zminbl=amin1(zminbl,zij)
50	         continue
51 	      continue
 	      zmax=amax1(zmaxbl,zmax)
 	      zmin=amin1(zminbl,zmin)
 	      zmaxb(ib,jb)=zmaxbl
 	      zminb(ib,jb)=zminbl
55	   continue
56 	continue
 
 	zrng=(zmax-zmin)*1.1
 	if(zrng .le. 0) goto 1200
 	weezee = zrng*.0002

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
	  do 1102 k1=1,nlabc
	    xlabp(k1)=xlabc(k1)
	    ylabp(k1)=ylabc(k1)
1102	  continue
	  nlabp=nlabc
	  nlabc=0
	  widthp=width
	endif
	IF ( interrupted ) GOTO 1200		! TMAP interrupt processing
	zc = zlev(lev)

	if(zc.ge.zmax .or. zc.le.zmin) goto 1100
	ndec = ndecl(lev)
	icolr=lclr(lev)
	lwgt = lwgtl(lev)
	do 71 i=1,nx
	   do 70 j=1,ny10
	      ind10(i,j)=0
70	   continue
71	continue
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

c     loop over the grid searching for a contour entering square ii,jj
c     thru side 1 or 4.      (the contour must be unused)
c*******************************************************************************
c
	ib=0
	do 1011 istrt=1,nx,iblk
	iend=min0(istrt+iblk,nx+1)
	ib=ib+1
	jb=0
	do 1010 jstrt=1,ny,jblk
	jb=jb+1
	if(zmaxb(ib,jb).lt.zc .or. zminb(ib,jb).gt.zc) go to 1010
	jend=min0(jstrt+jblk,ny+1)

	do i=istrt,iend
	   ikab=i-istrt+1
	   do j=jstrt,jend
	      jkab=j-jstrt+1
	      kabij=10
	      if(i.gt.nx .or. j.gt.ny) goto 200
	      zij=z(i,j)
	      if(zij.gt.big) goto 200
	      if(abs(zij-zc).le.weezee) zij=zc+sign(weezee,zij-zc)
	      kabij=-1
	      if(zij.gt.zc) kabij=1
 200	      continue
	      kab(ikab,jkab)=kabij
	   enddo
	enddo

	iendm1=iend-1
	jendm1=jend-1

	do 1001 ii=istrt,iendm1
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

	lin = ll
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
	if(iretrn .lt. 360) then 
	   k=k+1
	   frac = (zc-zz(lin))/(zz(lp1)-zz(lin))
	   x(k) = xx(lin)+ (xx(lp1)-xx(lin))*frac
	   y(k) = yy(lin)+ (yy(lp1)-yy(lin))*frac
	  iind=i+lp2/4
	   jind=j+lp1/4
	   lhor = lin-(lin/2)*2
	   jwrd=(jind-1)/10+1
	   j2bit=jind-(jwrd-1)*10
           ind10(iind,jwrd)=ind10(iind,jwrd)+int(2.-lhor)*ixpon(j2bit)
	endif 
c
c     see if an exit exists on side L-1,L+1,OR L+2. if so choose the one
c     closest to side l. if the exit is already used terminate x,y.
c*******************************************************************************
c
360	iretrn = 350
	lexit = lm1

	if(kabov(lin)+kabov(lm1) .eq. 0) then  
	   if(kabov(lp1)+kabov(lp2) .eq. 0) then
	      flm1 = (zc-zz(lin))/(zz(lm1)-zc)
	      flp1 = (zc-zz(lp1))/(zz(lp2)-zc)
	      if(lused(lp1).eq.1) go to 450
	      if(flm1.le.flp1 .and. lused(lm1).eq.0) go to 450
	      lexit = lp1
	   endif
	else
	   if(kabov(lp1)+kabov(lp2) .eq. 0) then
	      lexit = lp1
	   else
	      lexit = lp2
	      if(kabov(lp2)+kabov(lm1) .ne. 0) goto 470
	   endif
	endif

450	continue
	if(lused(lexit) .le. 0) goto 530
	kmax=k+1

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

470	continue

	if(kabov(lp2)+kabov(lm1) .le. 15) then
	   kda=lin
	   kdb=lp2
	   if(kabov(lp2) .gt. 5) then
	      kda = lm1
	      kdb = lp1
	   endif
	   k=k+1
	   frac = (zc-zz(kda))/(zz(kdb)-zz(kda))
	   x(k) = xx(kda) + (xx(kdb)-xx(kda))*frac
	   y(k) = yy(kda) + (yy(kdb)-yy(kda))*frac

	endif

	if(nseg .gt. 1) then
	   kmax=k
	   x(1)=x(2)
	   y(1)=y(2)
	   x(k+1)=x(k)
	   y(k+1)=y(k)
	   klosed=0
	   go to 700
	endif

	iretrn=360
	nseg = 2
	kh = 1+k/2
	do 520 kk=2,kh
	   kkr = k+2-kk
	   xkk=x(kk)
	   ykk=y(kk)
	   x(kk)= x(kkr)
	   y(kk)= y(kkr)
	   x(kkr)=xkk
	   y(kkr)=ykk
520	continue	

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
540	continue
	do 620 l=1,4
	   jl=j+(l-1)/2
	   lp1=l+1-(l/4)*4
	   il=i+(lp1-1)/2
	   ind(l)=0
	   zz(l)=cmzge
	   kabov(l)=10
	   if(il*(nx+1-il) .gt. 0) then 
	   if(jl*(ny+1-jl) .gt. 0) then
	      if(z(il,jl) .lt. big) then
	         jwrd=(jl-1)/10+1
	         j2bit=jl-(jwrd-1)*10
	         indl=ind10(il,jwrd)/ixpon(j2bit)
	         ind(l)=indl-(indl/4)*4
	         zz(l)=z(il,jl)
	         if(abs(zz(l)-zc) .le. weezee) zz(l)=zc+sign(weezee,zz(l)-zc)
	         kabov(l)=-1
	         if(zz(l) .gt. zc) kabov(l)=1
	      endif 
	   endif 
	   endif 
 	
	   xx(l)= px(il)*xf + xoff
	   yy(l)= py(jl)*yf + yoff
620	continue

	lused(4) = ind(1)/2
	lused(1) = ind(1)-2*lused(4)
	lused(2) = ind(2)/2
	lused(3) = ind(4)-(ind(4)/2)*2
	if(iretrn .lt. 350) then
	   goto 300
	else
	   goto 350
	endif
c
c     the arrays x,y are now complete. calculate distance s along
c     the contour. start plotting the contour
c*******************************************************************************
c
700	continue

	s(2)=0.
	if(kmax .le. 2) goto 1000
c
c     For the three argument contour command, the px and py arrays contain
c     fractional indices instead of axis values.  At this point in the code,
c     the x and y arrays contain the page coordinates.  Here we undo the
c     factor(xf/yf) and offset(xoff/yoff) calculations to retrieve the fractional
c     indices, call CURV_COORD() to converts these indices into the transformed
c     axis values, and then reapply the offset and factor.
c
	IF (curvilinear) THEN
           do 722 icurv=1, kmax
              x_curv(icurv)=(x(icurv)-xoff)/xf
              y_curv(icurv)=(y(icurv)-yoff)/yf
722	   continue

           CALL CURV_COORD(x_curv, y_curv, kmax, 1.0, 1.0, status) ! xform the pts
	   IF (curv_missing) GOTO 1000

           do 724 icurv=1, kmax
              x(icurv)=(x_curv(icurv)+xadd)*xf + xoff
              y(icurv)=y_curv(icurv)*yf + yoff
              lmin = MIN(lmin, x_curv(icurv))
              lmin = MAX(lmax, x_curv(icurv))
724	   continue
        ENDIF

        do 750 k=3,kmax
	   dxx = x(k)-x(k-1)
	   dyy = y(k)-y(k-1)
	   s(k) = s(k-1)+ sqrt(dxx*dxx+dyy*dyy)
750	continue	
	smax = s(kmax)
	slab1 = smax*slab1f

! TMAP mod, spline fit:
! initialize spline work arrays if using spline under tension fit.
	if (spline_tension.ne.0.) then
	  call kurv1(kmax-1,x(2),y(2),0.,0.,3,splxp,splyp,
     .      spltmp,spls,spline_tension,splier)
	  if (splier.ne.0) then
	  endif
	endif
! TMAP mod, horz contour labels:
! this is where the initial offset is set for a contour label on a contour
! line.  the first label is shifted in dslab-stest distance from the edge, not 
! dslab as might be expected.  modify the stest line to get something you like.
	if (draftsman) then
	  stest = max1(0.0,dslab-slab1)                 ! original code
	else
	  stest = max1(0.0,dslab-slab1)
	endif
	k=2
	call plotny(x(2),y(2),3,0)
c
c     check conditions for labelling.
c*******************************************************************************
c
	do while ( k .lt. kmax)
 	if(ndec .le. -2) goto 900

	km1= max0(k-1,2)
	stest = stest + s(k)-s(km1)
	if(stest .lt. dslab) goto 900
	if(smax-s(k)-2.0*width .le. 0) goto 900
	kp1=k+1
	if(lev.eq.1) go to 785
	dlev=abs(zlev(lev)-zlev(lev-1))
	xt = (x(k) - xoff)/xf
	yt = (y(k) - yoff)/yf
	do 781 i=1,nx-1
	   if(xt.ge.px(i) .and. xt.le.px(i+1)) exit
781	continue	

	do 783 j=1,ny-1
	   if(yt.ge.py(j) .and. yt.le.py(j+1)) exit
783	continue

	i=min0(i,nx-1)
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
	  if (abs(dyy) .lt. spacey) then
	     if(abs(dxx) .lt. spacex) cycle  ! 800 loop
	  endif
	else
	  if(space .lt. width)  cycle  ! 800 loop
        endif

        if(space/ark .lt. .95) then
           goto 900
        else 
           goto 810
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

810	continue

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
	      else
	        dyy=ya+yb*(spacex-abs(xa))/abs(xb)
	      endif
            else
	      dyy=spacey*sign(1.0,(y(kk)-y(k)))
	      if (yb.eq.0.) then
	        dxx=spacex*sign(1.0,(x(kk)-x(k)))
	      else
	        dxx=xa+xb*(spacey-abs(ya))/abs(yb)
	      endif
            endif
	  elseif (abs(dyy).ge.spacey) then
	    dyy=spacey*sign(1.0,(y(kk)-y(k)))
	    if (yb.eq.0.) then
	      dxx=spacex*sign(1.0,(x(kk)-x(k)))
	    else
	      dxx=xa+xb*(spacey-abs(ya))/abs(yb)
	    endif
          elseif (abs(dxx).ge.spacex) then
	    dxx=spacex*sign(1.0,(x(kk)-x(k)))
	    if (xb.eq.0.) then
	      dyy=spacey*sign(1.0,(y(kk)-y(k)))
	    else
	      dyy=ya+yb*(spacex-abs(xa))/abs(xb)
	    endif
	  else
	    print*,' error e1 in pplmod_contur.F'
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
     .         spls(kmax-1)
          call kurv2(splt,xs,ys,kmax-1,x(2),y(2),
     .               splxp,splyp,spls,spline_tension)
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
! the label should be written.  also, loops in this block make sure that the
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
	    endif
818	  continue	  
	  do 819 k1=kk+1,kmax
	    if ((abs(x(k1)  -xcen).lt.(width*0.5)).or.
     .	        (abs(x(k1-1)-xcen).lt.(width*0.5))) then
	      if ((ycen-y(k1))*(ycen-y(k1-1)).lt.0.) goto 900
	    endif
819	  continue	  

	else

	  angle = 90. 

! TMAP patch - Don's version of ARCSIN fix - 4/19/89
          if(dyy.lt.0.0)angle=-90.
! end of TMAP patch 4/19/89
	  if(dxx .ne. 0) angle = atan(dyy/dxx)*180./pi 

	  if(dxx .ge. 0) then 
	     if (width .eq. 0.0) then
	        xlab = xst
             ylab = yst
             else
                xlab = xst+ hhgt*(0.5*dxx+dyy)/width
	        ylab = yst+ hhgt*(0.5*dyy-dxx)/width
	     endif
	  else

	     if (width .eq. 0.0) then
	        xlab = xstp
                ylab = ystp
             else
	        xlab = xstp-hhgt*(0.5*dxx+dyy)/width
	     ylab = ystp-hhgt*(0.5*dyy-dxx)/width
	     endif
	  endif
	endif

! TMAP mod, horz contour labels:
! this loop checks to make sure the current label is not going to conflict
! with a label written on the previous contour line.  the current contour
! label is compared with all nlabp previous labels on the previous contour
! line.  if there is an overlap, then don't write the current label.  the
! allowable 'closeness' can be adjusted by altering xdelc, ydelc, xdelp, and
! ydelp where the labels are about width/widthp wide and hgt/hgt tall.  the
! c at the end of the variable represents 'current' and the p represents the
! previous contour line.  cen is where the center of the label is and del is
! the amount of space from the center that the label wants all to itself. 

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
861	  enddo	   
	endif
! TMAP mod, horz contour labels:
! draw the label and reset stest to 0. which indicates that the label has
! been drawn.  the next label should be dslab distance over.

* PyFerret Fonts locate the label by user units
* check if the start and end of the label will be inside the 
* plot box.

        IF (pyfont) THEN

          symcos = COS(angle *1.7453292E-2) 
          symsin = SIN(angle *1.7453292E-2) 
	  widx = symcos*width
	  widy = symsin*width

          xlabend = (xlab+widx - xoff)/ xf
          ylabend = (ylab+widy - yoff)/ yf

          xlab = (xlab - xoff)/ xf
	  ylab = (ylab - yoff)/ yf

	  skiplab = 0
	  if (ylab.LT.yloskp .OR. ylab.GT.yhiskp) skiplab = 1
	  if (xlab.LT.xloskp .OR. xlab.GT.xhiskp) skiplab = 1
	  if (ylabend.LT.yloskp .OR. ylabend.GT.yhiskp) skiplab = 1
	  if (xlabend.LT.xloskp .OR. xlabend.GT.xhiskp) skiplab = 1
	ENDIF

	iF (skiplab.EQ.0) call symbel(xlab,ylab,angle,hgt,nchar,str)
	stest=0.
! TMAP mod, horz contour labels:
! this section of code stores the contour label locations for the current
! contour line so they can be compared to the labels for the next contour line
	if (draftsman) then
	  nlabc=nlabc+1
	  xlabc(nlabc)=xlab
	  ylabc(nlabc)=ylab
	endif

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
     .      k-1,kmax-1,x(2),y(2),splxp,splyp,spls,spline_tension)
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

900	continue
	call arc(x(k-1),y(k-1),narc+1,.001,xc,yc,npt,
     .      k-1,kmax-1,x(2),y(2),splxp,splyp,spls,spline_tension)
c
c check for lines crossing the branch cut in curvilinear grids. Pick up the pen.
	IF (curvilinear) THEN
	   kk1 = 2
	   IF (ABS(x_curv(k+1)-x_curv(k)) .GT. 190) call plotny(xc(kk1),yc(kk1),3,lwgt)
	ENDIF

	do 910 kk1 = 2,npt
	   call plotny(xc(kk1),yc(kk1),2,lwgt)
910	continue	
	k=k+1

920	continue
	
	enddo  ! do while (k .lt. kmax)

1000	continue   ! do 1000 jj=jstrt,jendm1
1001	continue   ! do 1001 ii=istrt,iendm1

1010	continue   ! do 1010 jstrt=1,ny,jblk
1011	continue   ! do 1011 istrt=1,nx,iblk

1100	continue   ! do 1100  lev=1,nlev

1200	continue

	if(colour.ne.colold)call color(colold)

        IF (pyfont .AND. pyf_set(icont)) igroup = fnt_all

	return
	end

