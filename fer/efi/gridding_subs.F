c gridding_subs.F
c Utility functions called by gridding functions


c nobsxytv 
c
c***********************************************************************
c...NOBS.....Based on gausswt, return only the # of obs in (x,y) gridbox
c............method is to call this sub for each data value
c............all calcs done in gridbox units
c............xx/x1/xf all in same units
c............yy/y1/yf all in same units

c  i	xx,yy,tt=x/y/t location of data pt (data units)
c  i	nx,ny,nt=size of grids
c  i	xaxlo,xaxyi, etc bounds of destination grid cells
c  o 	obs(nx,ny,nt) number of obs in each (x,y,t) gridbox
c--------------------------------------------------------------------------

	SUBROUTINE nobsxytv (obs_only, xx, yy, tt, vv, nx, ny, nt, 
     .             xaxlo,xaxhi, yaxlo,yaxhi, taxlo,taxhi, indx, obs)

      IMPLICIT NONE

* Argument declarations

        INTEGER nx, ny, nt, obs_only, indx(4,2)
	REAL xx, yy, tt, vv, xaxlo(*), xaxhi(*), 
     .       yaxlo(*), yaxhi(*), taxlo(*), taxhi(*), obs(nx,ny,2,nt)

* Local variable declarations

        INTEGER i, j, l, ig, jg, Lg

	lg = 1
	DO l = 1, nt
	   IF (tt.GE.taxlo(lg) .AND.
     .         tt.LT.taxhi(lg)) THEN
              jg = 1
	      DO j = 1, ny
		 IF (yy.GE.yaxlo(jg) .AND.
     .	             yy.LT.yaxhi(jg)) THEN
                    ig = 1
		    DO i = 1, nx
		       IF ( xx.GE.xaxlo(ig) .AND.
     .                      xx.LT.xaxhi(ig)) THEN

		          IF (obs_only .EQ. 0) THEN
		             obs(ig,jg,1,Lg) = obs(ig,jg,1,Lg) + vv
		             obs(ig,jg,2,Lg) = obs(ig,jg,2,Lg) + 1.
		          ELSE
		             obs(ig,jg,1,Lg) = obs(ig,jg,1,Lg) + 1.
		             ENDIF

* Store the min and max index used in this cruise
		             indx(1,1) = MIN(indx(1,1), ig)
		             indx(1,2) = MAX(indx(1,2), ig)
		             indx(2,1) = MIN(indx(2,1), jg)
		             indx(2,2) = MAX(indx(2,2), jg)
		             indx(4,1) = MIN(indx(4,1), Lg)
		             indx(4,2) = MAX(indx(4,2), Lg)
		       ENDIF !xx in grid box
		       ig = ig + 1
		    ENDDO  ! nx

		 ENDIF !yy in grid box
		 jg = jg + 1
	      ENDDO  ! ny

	   ENDIF !tt in grid box
	   lg = lg + 1
	ENDDO  ! nt

	RETURN
	END


c nobsxytv_var
c
c***********************************************************************
c...NOBS.....Based on gausswt, return only the # of obs in (x,y) gridbox
c............method is to call this sub for each data value
c............all calcs done in gridbox units
c............xx/x1/xf all in same units
c............yy/y1/yf all in same units

c this version returns the cell mean in obs(ig,jg,1,Lg)
c this version returns the cell variance in obs(ig,jg,3,Lg)

c  i	xx,yy,tt=x/y/t location of data pt (data units)
c  i	nx,ny,nt=size of grids
c  i	xaxlo,xaxyi, etc bounds of destination grid cells
c  o 	obs(nx,ny,nt) number of obs in each (x,y,t) gridbox
c--------------------------------------------------------------------------

	SUBROUTINE nobsxytv_var (obs_only, xx, yy, tt, vv, nx, ny, nt, 
     .             xaxlo,xaxhi, yaxlo,yaxhi, taxlo,taxhi, indx, obs)

      IMPLICIT NONE

* Argument declarations

        INTEGER nx, ny, nt, obs_only, indx(4,2)
	REAL xx, yy, tt, vv,xaxlo(*), xaxhi(*), yaxlo(*), yaxhi(*), 
     .       taxlo(*), taxhi(*), obs(nx,ny,3,nt)

* Local variable declarations

        REAL xxg, yyg, ttg, delta
        INTEGER i, j, l, ig, jg, Lg 

	lg = 1
	DO l = 1, nt
	   IF (tt.GE.taxlo(lg) .AND.
     .         tt.LT.taxhi(lg)) THEN
              jg = 1
	      DO j = 1, ny
		 IF (yy.GE.yaxlo(jg) .AND.
     .	             yy.LT.yaxhi(jg)) THEN
                    ig = 1
		    DO i = 1, nx
		       IF ( xx.GE.xaxlo(ig) .AND.
     .                      xx.LT.xaxhi(ig)) THEN

		          IF (obs_only .EQ. 0) THEN
		             obs(ig,jg,2,Lg) = obs(ig,jg,2,Lg) + 1.	! count
		             delta = vv - obs(ig,jg,1,Lg)
		             obs(ig,jg,1,Lg) = obs(ig,jg,1,Lg) + 
     .                          delta/obs(ig,jg,2,Lg) ! current est of mean
		             obs(ig,jg,3,Lg) = obs(ig,jg,3,Lg) - 
     .                          delta*(vv - obs(ig,jg,1,Lg)) !M2
		          ELSE
		             obs(ig,jg,1,Lg) = obs(ig,jg,1,Lg) + 1.
		          ENDIF

* Store the min and max index used in this cruise
		             indx(1,1) = MIN(indx(1,1), ig)
		             indx(1,2) = MAX(indx(1,2), ig)
		             indx(2,1) = MIN(indx(2,1), jg)
		             indx(2,2) = MAX(indx(2,2), jg)
		             indx(4,1) = MIN(indx(4,1), Lg)
		             indx(4,2) = MAX(indx(4,2), Lg)
		       ENDIF !xx in grid box
		       ig = ig + 1
		    ENDDO  ! nx

		 ENDIF !yy in grid box
		 jg = jg + 1
	      ENDDO  ! ny

	   ENDIF !tt in grid box
	   lg = lg + 1
	ENDDO  ! nt


	RETURN
	END

c ----------------
c http://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#On-line_algorithm
c def online_variance(data):
c     n = 0
c     mean = 0
c     M2 = 0
c 
c     for x in data:
c         n = n + 1
c         delta = x - mean
c         mean = mean + delta/n
c         M2 = M2 + delta*(x - mean)  # This expression uses the new value of mean
c 
c     variance_n = M2/n
c     variance = M2/(n - 1)
c     return variance
c ------------------

c pickoutxytcv
c
c***********************************************************************
      SUBROUTINE pickoutxytcv (obs_only, vars, alon, alat, atimes,
     .   cruises, n, vv, xx, yy, tt, ncc)

*  Called from the scatter* family of gridding functions.
*  Pick out nth item from a set of arrays, which are really 
*  4-dimensioned arrays with normal axes on three of the four axes.

      IMPLICIT NONE
      REAL alon(*), alat(*), atimes(*), cruises(*), vars(*)
      REAL xx, yy, tt, vv
      INTEGER obs_only, n, ncc

      vv = 1.
      IF (obs_only .EQ. 0) vv = vars(n)
      xx = alon(n)
      yy = alat(n)
      tt = atimes(n)
      ncc = cruises(n)

      RETURN
      END

c
c modscatminmax
c
c***********************************************************************

      SUBROUTINE modscatminmax (axlo, axhi, nscat, scatm)
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
* ACM 11/00
*  Called by scat2gridlaplace* gridding functions.
*  If there is a modulo axis, apply modulo-ness to the coordinates in that 
*  direction of the scattered points.

*  ax           axlo, axhi outer box range of the modulo axis
*  nscat        input: number of scattered points, output: updated number.
*  scatm        scattered points coordinate array in the direction of the 
*               modulo axis

      IMPLICIT NONE
      REAL scatm(*)
      REAL*8 axlo, axhi
      INTEGER nscat, i
      REAL ax1, axf, scatpt, diff, range

      ax1 = axlo
      axf = axhi
      range = axf - ax1
      IF (range .EQ. 0.) RETURN  ! else the WHILE loops are infinite !

      DO 100 i = 1, nscat
         scatpt = scatm(i)
         diff = scatpt - axf
         DO WHILE (diff .GE. 0)
            scatpt = scatpt - range
            diff = scatpt - axf
         ENDDO

         diff = scatpt - ax1 
         DO WHILE (diff .LT. 0.)
            scatpt = scatpt + range
            diff = scatpt - ax1
         ENDDO

         scatm(i) = scatpt
  100 CONTINUE

      RETURN
      END

