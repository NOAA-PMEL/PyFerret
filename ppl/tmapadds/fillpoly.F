**   subroutine fillpoly(x,y,npts,index) *jc*

      subroutine fillpoly(xarg,yarg,npts,index,xadd,lmin,lmax)

      IMPLICIT NONE
      real xarg(*),yarg(*)
      integer npts,index
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
*       Mod *jd* 02.16.99 Make arrays 128 not 50 for better FILL optimization
*     v5.51 *acm* 2/3/03 Get viewport sizing as in shade routines, using
*                        gks calls; else the clipping is wrong for fill 
*                        plots in viewports that were defined with /AXES.
*     v5.81 *acm* 4/05 Changes for FILL/MOD. Draw replications for curvilinear datasets
* V68  *acm* 1/12  changes for double-precision ferret, single-precision pplus
* V687 *acm* 3/14 tickets 2151 and 2147: allow missing coordinate data in 3-arg 
*                 and 4-arg plot commands. 
* V698 *acm* 3/16 Ticket 2374: missing color when FILL extends over entire 
*                 modulo region

      real*8 x(128),y(128), xplus
      real xt(128),yt(128),xc(128),yc(128),xtrns,ytrns, temp_hi
      integer i,nout,fill_ndx
      real xadd, lmin, lmax

	include 'PLTCOM.DAT'
	include 'pltcom_dat.decl'
	include 'shade_vars.cmn'
        include 'axis_inc.decl'
        include 'AXIS.INC'
        include 'gkscm1_inc.decl'
        include 'GKSCM1.INC'
        include 'gkscm2.cmn' 
        INCLUDE 'gkspar.inc'
        include 'xcurvilinear.cmn'
	include 'ppl_in_ferret.cmn'	

      common/plyclp/acmin,acmax,bcmin,bcmax
      real acmin,acmax,bcmin,bcmax

      real vwidth,vheight,amax2,bmax2

*  From shade_sub; to use GKS calls to get scaling 

	INTEGER		error, trans_no, status
	REAL	 	w(4), v(4), wc(4)

	xplus = xadd

*     Get viewport width and height in ferret; otherwise leave it alone

      IF (ppl_in_ferret) THEN

*    Do scaling as in shade_sub, using gks calls.
*    Get proper scaling, world coordinates, etc for fill

	CALL FGD_GQCNTN(error,trans_no)
	IF (error .NE. 0) GOTO 1000
	CALL FGD_GQNT(trans_no,error,w,v)
	IF (error .NE. 0) GOTO 1000

*     Get world coordinates for for clipping region.

	CALL get_world_coords (w,v,wc)

      else
	 amax2 = amax
	 bmax2 = bmax
      endif


*     We're not sure if the xarg and yarg indices are needed in the calling routine
*     so copy all of the values to the x() and y() arrays.
*
*     pplmod_fillpoly doesn't use the 'inverse' flag.  (Should it?)
         DO i=1,npts
            x(i) = DBLE(xarg(i))
            y(i) = DBLE(yarg(i))
         ENDDO
         
*     Curvilinear section
      IF (curvilinear) THEN

         CALL CURV_COORD(x, y, npts, 1.0, 1.0, status) ! xform the pts
	 IF (curv_missing) RETURN

* add offset for modulo transformation.

           DO 48, i = 1, npts
              x(i) = x(i) + xplus
 48        CONTINUE

* What is the max and min x coordinate value covered by this call?

           DO 49, i = 1, npts
              lmin = MIN(lmin, x(i))
              lmax = MAX(lmax, x(i))
 49        CONTINUE
      ENDIF
c
c scale and transform the vertices

      do 5 i=1,npts

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

            XTEMP = xtemp * XSCALE
            YTEMP = ytemp * YSCALE
*         endif
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
         acmin = wc(1)
         acmax = wc(2)
         bcmin = wc(3)
         bcmax = wc(4)
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
c ticket 2374
c If the rectangle extends beyond the acmin,acmax,bcmin,bcmax rectangle on 
c both ends, it may be because of moduloing, where the rectangle we are
c coloring extends the entire length or height of the plot. Split the
c rectangle into two, to fill it in.

      if(nout.eq.0) THEN

c This will happen only in the horizontal direction; rectangles are gathered up 
c only in the horiz.
         if (xt(1).LT.acmin .AND. xt(2).GT.acmax) THEN
	    temp_hi = xt(2)
	    xt(2) = (acmin+acmax)/2.
	    xt(3) =  xt(2)
	    call clippoly(xt,yt,npts,xc,yc,nout)
	    if (nout .ne. 0) then
	       fill_ndx = index + 2
	       call set_fill_ndx (fill_ndx) 
	       call FGD_GFA(nout,xc,yc)

	       xt(1) = xt(2)
	       xt(4) = xt(2)
	       xt(2) = temp_hi
	       xt(3) = temp_hi
	    
	       call clippoly(xt,yt,npts,xc,yc,nout)
	       if (nout .ne. 0) then
	          fill_ndx = index + 2
	          call set_fill_ndx (fill_ndx) 
	          call FGD_GFA(nout,xc,yc)

	       endif
	    endif
	    nout = 0  ! all done

         endif

      endif
      if (nout .eq. 0) return

      fill_ndx = index + 2

      call set_fill_ndx (fill_ndx) 
      call FGD_GFA(nout,xc,yc)

 50   return
*     FATAL ERROR
1000  RETURN
      END
c
c
c
      subroutine clippoly(xt,yt,npts,xc,yc,nout)

      IMPLICIT NONE
      real xt(*),yt(*),xc(*),yc(*)
      integer npts,nout
c
      real x,y
      logical inregion
      integer i0,i1

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

      if(nout .le. 2) nout = 0
      return
      end
c
c
c
      logical function inregion(x,y)

      IMPLICIT NONE
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
c
c
      subroutine putunique(x, y, xc, yc, nout)

      IMPLICIT NONE
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
c
c
      subroutine getinterp(xin,yin,xout,yout,x,y)

      IMPLICIT NONE
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


