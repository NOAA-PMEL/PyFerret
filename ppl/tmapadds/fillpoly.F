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
**   subroutine fillpoly(x,y,npts,index) *jc*
      subroutine fillpoly(xarg,yarg,npts,index)
      real xarg(*),yarg(*)
      integer npts,index
C** 
C**    @(#)fillpoly.F	1.2   1/27/91
C**
C**
C***********************************************************************
C**
C**                 PLOT+ Scientific Graphics System
C**
C***********************************************************************
C**
C**
C      lots of places ---> PPLMOD_PPLCMD ---> PPLMOD_PLTIT 
C              ---> PPLMOD_PLOTZ ---> PPLMOD_ARFILL ---> PPLMOD_FILLPOLY
C       
C
C     call fillpoly(x,y,npts,index)
C
C     This subroutine creates a color filled convex polygon using the
C     using the coordinates.
C
C     THE VARIOUS PARAMETERS AND THEIR DEFINITIONS ARE LISTED
C     BELOW:
C        X,Y    - COORDINATES OF THE POLYGON`S CORNERS IN USER'S UNITS.
*	 ?? jd ?? Coordinates in inches from origin == nouser ??

C        npts   - number of vertices
C        index  - color index (0 - max_color)
C
C

*	MOD *JD* to include in TMAP PPL+ 1.14.92
*	Mod *jd* 10.12.92 for new hatching scheme
*	Mod *jd* 11.09.92 to get viewport size right for clipping
*     V4.50 *jc* 04.25.97 add 'curvilinear' capability
*                'x' and 'y' are  arrays of 50 to match xt, yt, etc.
*                (What is the maximum number of vertices in a polygon?)

      real x(50),y(50),xt(50),yt(50),xc(50),yc(50),xtrns,ytrns
      integer i,nout,fill_ndx

#ifdef unix
	include 'pplv11inc/PLTCOM.DAT'
	include 'tmap_pplv11inc/pltcom_dat.decl'
	include 'tmap_pplv11inc/shade_vars.cmn'
        include 'tmap_pplv11inc/axis_inc.decl'
        include 'pplv11inc/AXIS.INC'
        include 'tmap_pplv11inc/gkscm1_inc.decl'
        include 'pplv11inc/GKSCM1.INC'
        include 'tmap_pplv11inc/gkscm2.cmn' 
        include 'ferret_cmn/xcurvilinear.cmn'

#else
	include 'pplv11inc:PLTCOM.DAT'
	include 'tmap_pplv11inc:pltcom_dat.decl'
	include 'tmap_pplv11inc:tmap_shade_vars.inc'
        INCLUDE 'tmap_pplv11inc:tmap_axis.inc'
        INCLUDE 'tmap_pplv11inc:tmap_gkscm1.inc'
        INCLUDE 'tmap_pplv11inc:tmap_gkscm2.inc'
#endif

#ifdef unix
	include 	'tmap_pplv11inc/ppl_in_ferret.cmn'	
#else
	INCLUDE 	'TMAP_PPLV11INC:PPL_IN_FERRET.CMN'	
#endif
c
      common/plyclp/acmin,acmax,bcmin,bcmax
      real acmin,acmax,bcmin,bcmax

      real vwidth,vheight,amax2,bmax2

*     Get viewport width and height in ferret; otherwise leave it alone
      if (ppl_in_ferret) then
	 call get_view_size(vwidth,vheight)
	 amax2 = vwidth * factor
	 bmax2 = vheight* factor
      else
	 amax2 = amax
	 bmax2 = bmax
      endif


*     We're not sure if the xarg and yarg indices are needed in the calling routine
*     so copy all of the values to the x() and y() arrays.
*
*     pplmod_fillpoly doesn't use the 'inverse' flag.  (Should it?)
         DO 1 i=1,npts
            x(i) = xarg(i)
            y(i) = yarg(i)
 1       CONTINUE 
         
*     Curvilinear section
      IF (curvilinear) THEN

         CALL CURV_COORD(x, y, npts, 1.0, 1.0, status) ! xform the pts

      ENDIF


c
c scale and transform the vertices

*jd* begin -- temporary -- no projections permitted

      do 5 i=1,npts
*         if(mapproj)then
*            xtemp=xtrns(x(i),y(i))*xscale
*            ytemp=ytrns(x(i),y(i))*yscale
*         else

	    if (itypex .gt. 1) then
	       xtemp = xlen*log10(x(i)/xlo)/log10(xhi/xlo)
	    else
	       xtemp = xlen*(x(i) - xlo)/(xhi-xlo)
	    endif

	    if (itypey .gt. 1) then
	       ytemp = ylen*log10(y(i)/ylo)/log10(yhi/ylo)
	    else
	       ytemp = ylen*(y(i) - ylo)/(yhi-ylo)
	    endif

*            XTEMP = X(i) * XSCALE
*            YTEMP = Y(i) * YSCALE

            XTEMP = xtemp * XSCALE
            YTEMP = ytemp * YSCALE
*         endif
*jd*	end

c
         call tform(xtemp,ytemp,ztemp)
         xt(i)=anew
         yt(i)=bnew
c
 5    continue
c
c find clipping region
c
      if(windof)then
         call tform(xmin,ymin,ztemp)
         if(anew .gt. amin) then
            acmin = anew
         else
            acmin = amin
         endif
         if(bnew .gt. bmin) then
            bcmin = bnew
         else
            bcmin = bmin
         endif
         call tform(xmax,ymax,ztemp)
         if(anew .lt. amax2) then
            acmax = anew
         else
            acmax = amax2
         endif
         if(bnew .lt. bmax2) then
            bcmax = bnew
         else
            bcmax = bmax2
         endif
      else
         acmin = amin
         acmax = amax2
         bcmin = bmin
         bcmax = bmax2
      endif
c
c clip polygon
c
      call clippoly(xt,yt,npts,xc,yc,nout)
c
      if(nout.eq.0) return
c

*jd begin
	fill_ndx = index + 2

	call set_fill_ndx (fill_ndx) 
	call gfa (nout,xc,yc)

*	TO RETAIN ABILITY FOR COLOR OR HATCHING IN METAFILE, WRITE 
*	NDX TO PRIVATE ITEM - forget it 10.22
*	if (meta_actv .and. .not. area_bundles) call meta_fill_ndx (fill_ndx)

*      goto (10,50,30,10,10,20,20),ptype+3
*c
*c tekterminal
*c
* 10   goto 100
*c
*c window device X11
*c
*#ifdef X11
* 20   if(havex)then
*         if(.not.gksopn)then
*            call xwinit
*         endif
*         call fillpolyx(xc,yc,nout,index)
*      endif
*#else
* 20   continue
*#endif
*c
*c check for other devices
*c
*100   goto (50,50,50,50,30,50,30),ptype+3
*c
*c meta file
*c
* 30   if(wrtclr)then
*c
*c write color map
*c
*         wrtclr=.false.
*         call binbuf(float(numclr),-2.)
*         do 31 i=1,numclr,2
*            call binbuf(float(rdclr(i)),float(rdclr(i+1)))
* 31      continue
*         do 32 i=1,numclr,2
*            call binbuf(float(grnclr(i)),float(grnclr(i+1)))
* 32      continue
*         do 33 i=1,numclr,2
*            call binbuf(float(blclr(i)),float(blclr(i+1)))
* 33      continue
*      endif
*c
*c write poly
*c
*      call binbuf(float(index),-3.)
*      call binbuf(float(nout),-3.)
*      do 34 i=1,nout
*         call binbuf(xc(i),yc(i))
* 34   continue
*c
*jd end

 50   return
      end
c
      subroutine clippoly(xt,yt,npts,xc,yc,nout)
      real xt(*),yt(*),xc(*),yc(*)
      integer npts,nout
c
      real x,y
      logical inregion
      integer i0,i1
c
      nout = 0
      do 10 i0 = 1,npts
         i1 = i0+1
         if(i1.gt.npts)i1=1
         if(inregion(xt(i0),yt(i0))) then
            call putunique(xt(i0),yt(i0),xc,yc,nout)
            if(inregion(xt(i1),yt(i1)))then
               call putunique(xt(i1),yt(i1),xc,yc,nout)
            else
               call getinterp(xt(i0),yt(i0),xt(i1),yt(i1),x,y)
               call putunique(x,y,xc,yc,nout)
            endif
         else
            if(inregion(xt(i1),yt(i1))) then
               call getinterp(xt(i1),yt(i1),xt(i0),yt(i0),x,y)
               call putunique(x,y,xc,yc,nout)
               call putunique(xt(i1),yt(i1),xc,yc,nout)
            endif
         endif
 10   continue
c
      if(nout .le. 2) nout = 0
      return
      end
c
      logical function inregion(x,y)
      real x,y
c
      common/plyclp/acmin,acmax,bcmin,bcmax
      real acmin,acmax,bcmin,bcmax
c
      inregion = ((x.ge.acmin) .and. (x.le.acmax)) .and. 
     *           ((y.ge.bcmin) .and. (y.le.bcmax))
      return
      end
c
      subroutine putunique(x, y, xc, yc, nout)
      real x,y,xc(*),yc(*)
      integer nout,i
c
      if(nout .eq. 0) goto 100
      do 10 i=1,nout
         if((x.eq.xc(i)) .and. (y.eq.yc(i)))return
 10   continue
 100  nout = nout + 1
      xc(nout)=x
      yc(nout)=y
      return
      end
c
      subroutine getinterp(xin,yin,xout,yout,x,y)
      real xin,yin,xout,yout,x,y
c
      common/plyclp/acmin,acmax,bcmin,bcmax
      real acmin,acmax,bcmin,bcmax
c
      real f
c
      if(xout.lt.acmin)then
         f = (acmin - xin)/(xout - xin)
      else if(xout.gt.acmax) then
         f = (acmax - xin)/(xout - xin)
      else
         goto 100
      endif
      x = xin + (f * (xout - xin))
      y = yin + (f * (yout - yin))
      if((y.ge.bcmin).and.(y.le.bcmax))return
 100  if(yout.lt.bcmin) then
         f = (bcmin - yin)/(yout - yin)
      else
         f = (bcmax - yin)/(yout - yin)
      endif
      x = xin + (f * (xout - xin))
      y = yin + (f * (yout - yin))
      return
      end


