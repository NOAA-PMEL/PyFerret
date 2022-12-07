      subroutine kurv1 (n,x,y,slp1,slpn,islpsw,xp,yp,temp,s,
     *                  sigma,ierr)

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
*     *tc* 1.26.93  added this 'spline under tension' package.
*                   called by pplmod_contur and pplmod_arc to interpolate 
*                   on contour lines.
* 11/2022 *acm* Code cleanup: Modernize DO-loop syntax. 

      IMPLICIT NONE
      integer n,islpsw,ierr
      real x(n),y(n),slp1,slpn,xp(n),yp(n),temp(n),s(n),
     *     sigma
c
c                                 coded by alan kaylor cline
c                           from fitpack -- january 26, 1987
c                        a curve and surface fitting package
c                      a product of pleasant valley software
c                  8603 altus cove, austin, texas 78759, usa
c
c this subroutine determines the parameters necessary to
c compute a spline under tension forming a curve in the
c plane and passing through a sequence of pairs (x(1),y(1)),
c ...,(x(n),y(n)). for actual computation of points on the
c curve it is necessary to call the subroutine kurv2.
c
c on input--
c
c   n is the number of points to be interpolated (n.ge.2).
c
c   x is an array containing the n x-coordinates of the
c   points.
c
c   y is an array containing the n y-coordinates of the
c   points. (adjacent x-y pairs must be distinct, i. e.
c   either x(i) .ne. x(i+1) or y(i) .ne. y(i+1), for
c   i = 1,...,n-1.)
c
c   slp1 and slpn contain the desired values for the angles
c   (in radians) of the slope at (x(1),y(1)) and (x(n),y(n))
c   respectively. the angles are measured counter-clock-
c   wise from the x-axis and the positive sense of the curve
c   is assumed to be that moving from point 1 to point n.
c   the user may omit values for either or both of these
c   parameters and signal this with islpsw.
c
c   islpsw contains a switch indicating which slope data
c   should be used and which should be estimated by this
c   subroutine,
c          = 0 if slp1 and slpn are to be used,
c          = 1 if slp1 is to be used but not slpn,
c          = 2 if slpn is to be used but not slp1,
c          = 3 if both slp1 and slpn are to be estimated
c              internally.
c
c   xp and yp are arrays of length at least n.
c
c   temp is an array of length at least n which is used
c   for scratch storage.
c
c   s is an array of length at least n.
c
c and
c
c   sigma contains the tension factor. this value indicates
c   the curviness desired. if abs(sigma) is nearly zero
c   (e.g. .001) the resulting curve is approximately a cubic
c   spline. if abs(sigma) is large (e. g. 50.) the resulting
c   curve is nearly a polygonal line. if sigma equals zero a
c   cubic spline results. a standard value for sigma is
c   approximately 1. in absolute value.
c
c on output--
c
c   xp and yp contain information about the curvature of the
c   curve at the given nodes.
c
c   s contains the polygonal arclengths of the curve.
c
c   ierr contains an error flag,
c        = 0 for normal return,
c        = 1 if n is less than 2,
c        = 2 if adjacent coordinate pairs coincide.
c
c and
c
c   n, x, y, slp1, slpn, islpsw, and sigma are unaltered.
c
c this subroutine references package modules ceez, terms,
c and snhcsh.
c
c-----------------------------------------------------------
c
      INTEGER i, ibak, im1, nm1, np1
      REAL c1, c2, c3, dels, dels1, dels2, delsn, delsnm, 
     .     delt, diag, diag1, diag2, diagin, 
     .     dx1, dx2, dy1, dy2, sdiag1, sdiag2, sigmap, 
     .     slpp1x, slpp1y, slppnx, slppny, sx, sy

      nm1 = n-1
      np1 = n+1
      ierr = 0
      if (n .le. 1) go to 11
c
c determine polygonal arclengths
c
      s(1) = 0.
      do 1 i = 2,n
        im1 = i-1
        s(i) = s(im1)+sqrt((x(i)-x(im1))**2+
     *         (y(i)-y(im1))**2)
   1  continue
c
c denormalize tension factor
c
      sigmap = abs(sigma)*float(n-1)/s(n)
c
c approximate end slopes
c
      if (islpsw .ge. 2) go to 2
      slpp1x = cos(slp1)
      slpp1y = sin(slp1)
      go to 4
    2 dels1 = s(2)-s(1)
      dels2 = dels1+dels1
      if (n .gt. 2) dels2 = s(3)-s(1)
      if (dels1 .eq. 0. .or. dels2 .eq. 0.) go to 12
      call ceez (dels1,dels2,sigmap,c1,c2,c3,n)
      sx = c1*x(1)+c2*x(2)
      sy = c1*y(1)+c2*y(2)
      if (n .eq. 2) go to 3
      sx = sx+c3*x(3)
      sy = sy+c3*y(3)
    3 delt = sqrt(sx*sx+sy*sy)
      slpp1x = sx/delt
      slpp1y = sy/delt
    4 if (islpsw .eq. 1 .or. islpsw .eq. 3) go to 5
      slppnx = cos(slpn)
      slppny = sin(slpn)
      go to 7
    5 delsn = s(n)-s(nm1)
      delsnm = delsn+delsn
      if (n .gt. 2) delsnm = s(n)-s(n-2)
      if (delsn .eq. 0. .or. delsnm .eq. 0.) go to 12
      call ceez (-delsn,-delsnm,sigmap,c1,c2,c3,n)
      sx = c1*x(n)+c2*x(nm1)
      sy = c1*y(n)+c2*y(nm1)
      if (n .eq. 2) go to 6
      sx = sx+c3*x(n-2)
      sy = sy+c3*y(n-2)
    6 delt = sqrt(sx*sx+sy*sy)
      slppnx = sx/delt
      slppny = sy/delt
c
c set up right hand sides and tridiagonal system for xp and
c yp and perform forward elimination
c
    7 dx1 = (x(2)-x(1))/s(2)
      dy1 = (y(2)-y(1))/s(2)
      call terms (diag1,sdiag1,sigmap,s(2))
      xp(1) = (dx1-slpp1x)/diag1
      yp(1) = (dy1-slpp1y)/diag1
      temp(1) = sdiag1/diag1
      if (n .eq. 2) go to 9
      do 8 i = 2,nm1
        dels2 = s(i+1)-s(i)
        if (dels2 .eq. 0.) go to 12
        dx2 = (x(i+1)-x(i))/dels2
        dy2 = (y(i+1)-y(i))/dels2
        call terms (diag2,sdiag2,sigmap,dels2)
        diag = diag1+diag2-sdiag1*temp(i-1)
        diagin = 1./diag
        xp(i) = (dx2-dx1-sdiag1*xp(i-1))*diagin
        yp(i) = (dy2-dy1-sdiag1*yp(i-1))*diagin
        temp(i) = sdiag2*diagin
        dx1 = dx2
        dy1 = dy2
        diag1 = diag2
        sdiag1 = sdiag2
    8 continue
    9 diag = diag1-sdiag1*temp(nm1)
      xp(n) = (slppnx-dx1-sdiag1*xp(nm1))/diag
      yp(n) = (slppny-dy1-sdiag1*yp(nm1))/diag
c
c perform back substitution
c
      do 10 i = 2,n
        ibak = np1-i
        xp(ibak) = xp(ibak)-temp(ibak)*xp(ibak+1)
        yp(ibak) = yp(ibak)-temp(ibak)*yp(ibak+1)
   10 continue
      return
c
c too few points
c
   11 ierr = 1
      return
c
c coincident adjacent points
c
   12 ierr = 2
      return
      end
c
c
c
      subroutine kurv2 (t,xs,ys,n,x,y,xp,yp,s,sigma)
c
      IMPLICIT NONE
      integer n
      real t,xs,ys,x(n),y(n),xp(n),yp(n),s(n),sigma
c
c                                 coded by alan kaylor cline
c                           from fitpack -- january 26, 1987
c                        a curve and surface fitting package
c                      a product of pleasant valley software
c                  8603 altus cove, austin, texas 78759, usa
c
c this subroutine performs the mapping of points in the
c interval (0.,1.) onto a curve in the plane. the subroutine
c kurv1 should be called earlier to determine certain
c necessary parameters. the resulting curve has a parametric
c representation both of whose components are splines under
c tension and functions of the polygonal arclength
c parameter.
c
c on input--
c
c   t contains a real value to be mapped to a point on the
c   curve. the interval (0.,1.) is mapped onto the entire
c   curve, with 0. mapping to (x(1),y(1)) and 1. mapping
c   to (x(n),y(n)). values outside this interval result in
c   extrapolation.
c
c   n contains the number of points which were specified
c   to determine the curve.
c
c   x and y are arrays containing the x- and y-coordinates
c   of the specified points.
c
c   xp and yp are the arrays output from kurv1 containing
c   curvature information.
c
c   s is an array containing the polygonal arclengths of
c   the curve.
c
c and
c
c   sigma contains the tension factor (its sign is ignored).
c
c the parameters n, x, y, xp, yp, s, and sigma should be
c input unaltered from the output of kurv1.
c
c on output--
c
c   xs and ys contain the x- and y-coordinates of the image
c   point on the curve.
c
c none of the input parameters are altered.
c
c this subroutine references package modules intrvl and
c snhcsh.
c

      integer i, im1, intrvl
      real    c1, c2, d, del1, del2, dels, dummy, 
     *        s1, s2, sigdel, sigmap, ss, sumx, sumy, tn
c-----------------------------------------------------------
c
c determine interval
c
      tn = s(n)*t
      im1 = intrvl(tn,s,n)
      i = im1+1
c
c denormalize tension factor
c
      sigmap = abs(sigma)*float(n-1)/s(n)
c
c set up and perform interpolation
c
      del1 = tn-s(im1)
      del2 = s(i)-tn
      dels = s(i)-s(im1)
      sumx = (x(i)*del1+x(im1)*del2)/dels
      sumy = (y(i)*del1+y(im1)*del2)/dels
      if (sigmap .ne. 0.) go to 1
      d = del1*del2/(6.*dels)
      c1 = (del1+dels)*d
      c2 = (del2+dels)*d
      xs = sumx-xp(i)*c1-xp(im1)*c2
      ys = sumy-yp(i)*c1-yp(im1)*c2
      return
    1 sigdel = sigmap*dels
      call snhcsh(ss,dummy,sigdel,-1)
      call snhcsh(s1,dummy,sigmap*del1,-1)
      call snhcsh(s2,dummy,sigmap*del2,-1)
      d = sigdel*sigmap*(1.+ss)
      c1 = del1*(s1-ss)/d
      c2 = del2*(s2-ss)/d
      xs = sumx+xp(i)*c1+xp(im1)*c2
      ys = sumy+yp(i)*c1+yp(im1)*c2
      return
      end
c
c
c
      subroutine ceez (del1,del2,sigma,c1,c2,c3,n)
c
      IMPLICIT NONE
      real del1,del2,sigma,c1,c2,c3
      integer n
c
c                                 coded by alan kaylor cline
c                           from fitpack -- january 26, 1987
c                        a curve and surface fitting package
c                      a product of pleasant valley software
c                  8603 altus cove, austin, texas 78759, usa
c
c this subroutine determines the coefficients c1, c2, and c3
c used to determine endpoint slopes. specifically, if
c function values y1, y2, and y3 are given at points x1, x2,
c and x3, respectively, the quantity c1*y1 + c2*y2 + c3*y3
c is the value of the derivative at x1 of a spline under
c tension (with tension factor sigma) passing through the
c three points and having third derivative equal to zero at
c x1. optionally, only two values, c1 and c2 are determined.
c
c on input--
c
c   del1 is x2-x1 (.gt. 0.).
c
c   del2 is x3-x1 (.gt. 0.). if n .eq. 2, this parameter is
c   ignored.
c
c   sigma is the tension factor.
c
c and
c
c   n is a switch indicating the number of coefficients to
c   be returned. if n .eq. 2 only two coefficients are
c   returned. otherwise all three are returned.
c
c on output--
c
c   c1, c2, and c3 contain the coefficients.
c
c none of the input parameters are altered.
c
c this subroutine references package module snhcsh.
c

      real coshm1, coshm2, del, delm, delp, denom, dummy,
     *     sinhmm, sinhmp

c-----------------------------------------------------------
c
      if (n .eq. 2) go to 2
      if (sigma .ne. 0.) go to 1
      del = del2-del1
c
c tension .eq. 0.
c
      c1 = -(del1+del2)/(del1*del2)
      c2 = del2/(del1*del)
      c3 = -del1/(del2*del)
      return
c
c tension .ne. 0.
c
    1 call snhcsh (dummy,coshm1,sigma*del1,1)
      call snhcsh (dummy,coshm2,sigma*del2,1)
      delp = sigma*(del2+del1)/2.
      delm = sigma*(del2-del1)/2.
      call snhcsh (sinhmp,dummy,delp,-1)
      call snhcsh (sinhmm,dummy,delm,-1)
      denom = coshm1*(del2-del1)-2.*del1*delp*delm*
     *        (1.+sinhmp)*(1.+sinhmm)
      c1 = 2.*delp*delm*(1.+sinhmp)*(1.+sinhmm)/denom
      c2 = -coshm2/denom
      c3 = coshm1/denom
      return
c
c two coefficients
c
    2 c1 = -1./del1
      c2 = -c1
      return
      end
c
c
c
      subroutine snhcsh (sinhm,coshm,x,isw)
c
      IMPLICIT NONE
      integer isw
      real sinhm,coshm,x
c
c                                 coded by alan kaylor cline
c                           from fitpack -- january 26, 1987
c                        a curve and surface fitting package
c                      a product of pleasant valley software
c                  8603 altus cove, austin, texas 78759, usa
c
c this subroutine returns approximations to
c       sinhm(x) = sinh(x)/x-1
c       coshm(x) = cosh(x)-1
c and
c       coshmm(x) = (cosh(x)-1-x*x/2)/(x*x)
c with relative error less than 1.0e-6
c
c on input--
c
c   x contains the value of the independent variable.
c
c   isw indicates the function desired
c           = -1 if only sinhm is desired,
c           =  0 if both sinhm and coshm are desired,
c           =  1 if only coshm is desired,
c           =  2 if only coshmm is desired,
c           =  3 if both sinhm and coshmm are desired.
c
c on output--
c
c   sinhm contains the value of sinhm(x) if isw .le. 0 or
c   isw .eq. 3 (sinhm is unaltered if isw .eq.1 or isw .eq.
c   2).
c
c   coshm contains the value of coshm(x) if isw .eq. 0 or
c   isw .eq. 1 and contains the value of coshmm(x) if isw
c   .ge. 2 (coshm is unaltered if isw .eq. -1).
c
c and
c
c   x and isw are unaltered.
c
c-----------------------------------------------------------
c
      real sp13, sp12, sp11, sp10, 
     *     sp24, sp23, sp22, sp21, sp20,
     *     sp33, sp32, sp31, sq32, sq31, sq30, 
     *     sp43, sp42, sp41, sq42, sq41, sq40,
     *     cp4, cp3, cp2, cp1, cp0,
     *     ax, expx, xs

      data sp13/.3029390e-5/,
     *     sp12/.1975135e-3/,
     *     sp11/.8334261e-2/,
     *     sp10/.1666665e0/
      data sp24/.3693467e-7/,
     *     sp23/.2459974e-5/,
     *     sp22/.2018107e-3/,
     *     sp21/.8315072e-2/,
     *     sp20/.1667035e0/
      data sp33/.6666558e-5/,
     *     sp32/.6646307e-3/,
     *     sp31/.4001477e-1/,
     *     sq32/.2037930e-3/,
     *     sq31/-.6372739e-1/,
     *     sq30/.6017497e1/
      data sp43/.2311816e-4/,
     *     sp42/.2729702e-3/,
     *     sp41/.9868757e-1/,
     *     sq42/.1776637e-3/,
     *     sq41/-.7549779e-1/,
     *     sq40/.9110034e1/
      data cp4/.2982628e-6/,
     *     cp3/.2472673e-4/,
     *     cp2/.1388967e-2/,
     *     cp1/.4166665e-1/,
     *     cp0/.5000000e0/
c
      ax = abs(x)
      if (isw .ge. 0) go to 5
c
c sinhm approximation
c
      if (ax .gt. 4.45) go to 2
      xs = ax*ax
      if (ax .gt. 2.3) go to 1
c
c sinhm approximation on (0.,2.3)
c
      sinhm = xs*(((sp13*xs+sp12)*xs+sp11)*xs+sp10)
      return
c
c sinhm approximation on (2.3,4.45)
c
    1 sinhm = xs*((((sp24*xs+sp23)*xs+sp22)*xs+sp21)
     .               *xs+sp20)
      return
    2 if (ax .gt. 7.65) go to 3
c
c sinhm approximation on (4.45,7.65)
c
      xs = ax*ax
      sinhm = xs*(((sp33*xs+sp32)*xs+sp31)*xs+1.)/
     .             ((sq32*xs+sq31)*xs+sq30)
      return
    3 if (ax .gt. 10.1) go to 4
c
c sinhm approximation on (7.65,10.1)
c
      xs = ax*ax
      sinhm = xs*(((sp43*xs+sp42)*xs+sp41)*xs+1.)/
     .             ((sq42*xs+sq41)*xs+sq40)
      return
c
c sinhm approximation above 10.1
c
    4 sinhm = exp(ax)/(ax+ax)-1.
      return
c
c coshm and (possibly) sinhm approximation
c
    5 if (isw .ge. 2) go to 7
      if (ax .gt. 2.3) go to 6
      xs = ax*ax
      coshm = xs*((((cp4*xs+cp3)*xs+cp2)*xs+cp1)*xs+cp0)
      if (isw .eq. 0) sinhm = xs*(((sp13*xs+sp12)*xs+sp11)
     .                              *xs+sp10)
      return
    6 expx = exp(ax)
      coshm = (expx+1./expx)/2.-1.
      if (isw .eq. 0) sinhm = (expx-1./expx)/(ax+ax)-1.
      return
c
c coshmm and (possibly) sinhm approximation
c
    7 xs = ax*ax
      if (ax .gt. 2.3) go to 8
      coshm = xs*(((cp4*xs+cp3)*xs+cp2)*xs+cp1)
      if (isw .eq. 3) sinhm = xs*(((sp13*xs+sp12)*xs+sp11)
     .                              *xs+sp10)
      return
    8 expx = exp(ax)
      coshm = ((expx+1./expx-xs)/2.-1.)/xs
      if (isw .eq. 3) sinhm = (expx-1./expx)/(ax+ax)-1.
      return
      end
c
c
c
      subroutine terms (diag,sdiag,sigma,del)
c
      IMPLICIT NONE
      real diag,sdiag,sigma,del
c
c                                 coded by alan kaylor cline
c                           from fitpack -- january 26, 1987
c                        a curve and surface fitting package
c                      a product of pleasant valley software
c                  8603 altus cove, austin, texas 78759, usa
c
c this subroutine computes the diagonal and superdiagonal
c terms of the tridiagonal linear system associated with
c spline under tension interpolation.
c
c on input--
c
c   sigma contains the tension factor.
c
c and
c
c   del contains the step size.
c
c on output--
c
c                sigma*del*cosh(sigma*del) - sinh(sigma*del)
c   diag = del*--------------------------------------------.
c                     (sigma*del)**2 * sinh(sigma*del)
c
c                   sinh(sigma*del) - sigma*del
c   sdiag = del*----------------------------------.
c                (sigma*del)**2 * sinh(sigma*del)
c
c and
c
c   sigma and del are unaltered.
c
c this subroutine references package module snhcsh.
c

      real coshm, denom, sigdel, sinhm

c-----------------------------------------------------------
c
      if (sigma .ne. 0.) go to 1
      diag = del/3.
      sdiag = del/6.
      return
    1 sigdel = sigma*del
      call snhcsh (sinhm,coshm,sigdel,0)
      denom = sigma*sigdel*(1.+sinhm)
      diag = (coshm-sinhm)/denom
      sdiag = sinhm/denom
      return
      end
c
c
c
      integer function intrvl (t,x,n)
c
      IMPLICIT NONE
      integer n
      real t,x(n)
c
c                                 coded by alan kaylor cline
c                           from fitpack -- january 26, 1987
c                        a curve and surface fitting package
c                      a product of pleasant valley software
c                  8603 altus cove, austin, texas 78759, usa
c
c this function determines the index of the interval
c (determined by a given increasing sequence) in which
c a given value lies.
c
c on input--
c
c   t is the given value.
c
c   x is a vector of strictly increasing values.
c
c and
c
c   n is the length of x (n .ge. 2).
c
c on output--
c
c   intrvl returns an integer i such that
c
c          i =  1       if         e   t .lt. x(2)  ,
c          i =  n-1     if x(n-1) .le. t            ,
c          otherwise       x(i)  .le. t .le. x(i+1),
c
c none of the input parameters are altered.
c

      integer i, il, ih
      real    tt

c-----------------------------------------------------------
c
      save i
      data i /1/
c
      tt = t
c
c check for illegal i
c
      if (i .ge. n) i = n/2
c
c check old interval and extremes
c
      if (tt .lt. x(i)) then
        if (tt .le. x(2)) then
          i = 1
          intrvl = 1
          return
        else
          il = 2
          ih = i
        end if
      else if (tt .le. x(i+1)) then
        intrvl = i
        return
      else if (tt .ge. x(n-1)) then
        i = n-1
        intrvl = n-1
        return
      else
        il = i+1
        ih = n-1
      end if
c
c binary search loop
c
    1 i = (il+ih)/2
      if (tt .lt. x(i)) then
         ih = i
      else if (tt .gt. x(i+1)) then
         il = i+1
      else
         intrvl = i
         return
      end if
      go to 1
      end
