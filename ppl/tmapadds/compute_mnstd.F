	SUBROUTINE compute_mnstd(z, badz, need_std, nsize, rbad, status)

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

C**
C**
* V630  *acm* 9/09 Introduction of syntax for variance-based and histogram levels
* V664:  8/10 - implement robust method for computing variances (bug 1745)
* V666:  1/11 - fix bug 1778: missing data not treated correctly. Dont use CMZGE,
*               so variance result is the same as other Ferret variance calculations
* V68  *acm* 1/12  changes for double-precision ferret, single-precision pplus
* V69+ *acm* 7/14 ticket 2186: more-robust mean and std deviation if big outliers
* v695 *acm* 9/15 ticket 2311: variance-based levels for constant variable. Also
*                 improve accuracy of computation using real*4 bad-value sent in from
*                 the Ferret side.
* V698 *acm* 2/16 Re-fix 2186. The fix for 2311 broke the fix for 2186, so the test 
*                 for data outside 3*std correction wasn't made.
* V720 *acm* 2/17 Make sure zmin and zmax have been computed. May not be done yet for 
*                 plot/vs/ribbon plots.
* V751 *acm* 7/19 Call to MINMAX needs final argument, nok

        IMPLICIT NONE
	include 'parampl5_dat.decl'
        include 'PARAMPL5.DAT'
	include 'hd_inc.decl'
	include 'HD.INC'
	include 'cont_inc.decl'
	include 'CONT.INC'
	include 'miss_inc.decl'
	include 'MISS.INC'

* calling argument declarations:
	LOGICAL need_std
	INTEGER nsize, status
	REAL*8 z(*), badz

* internal variable declarations:
	REAL*8 sum, dev, sumsq_dev, variance, tol_lo, tol_hi, zmean2

	LOGICAL TM_FPEQ_SNGL, TM_FPEQ, zmax_test, zmin_test, ok
	REAL  zero, rbad
	REAL*8 x, xmean, sum2, sumc, variance_c, xdelta, 
     .         z_max_tol, z_min_tol, zlo, zhi
	INTEGER i, n, n2, nok

        IF (.not. need_std .AND. centered) THEN
	   zstd = lev_std
	   zmean = 0.
	   GOTO 5000
	ENDIF

c  Already have zmin, zmax? (not true for ribbon plots...)

	IF (zmin.EQ.0.0 .AND. zmax.EQ.0.0) THEN
	   CALL MINMAX( z, nsize, badz, zlo, zhi, nok )
	   zmin = zlo
	   zmax = zhi
	ENDIF

c if min and max are equal, will use linear levels. No need to issue the warning.

	if (zmin .EQ. zmax) THEN
	   zmean = zmin
	   zstd = 0
	   IF (need_std) status = 0
	   GOTO 5000
	ENDIF

        IF (.not. need_std) GOTO 5000

c Has the user set min or max levels to be used?
c If so take those into account

	zmax_test = .FALSE.
	zmin_test = .FALSE.

	IF (.NOT. TM_FPEQ_SNGL(lev_max, rbad)) THEN
	   zmax_test = .TRUE.
	   z_max_tol = DBLE(lev_max)
	ENDIF
	IF (.NOT. TM_FPEQ_SNGL(lev_min, rbad)) THEN
	   zmin_test = .TRUE.
	   z_min_tol = DBLE(lev_min)
	ENDIF

c Compute data mean.

        sum = 0.0
        n = 0
        DO 100 i = 1, nsize
	   x = z(i)
	   IF ( x .NE. badz) THEN
	      ok = .TRUE.
	      IF (zmax_test .AND. x.GE.z_max_tol) ok = .FALSE.
	      IF (zmin_test .AND. x.LE.z_min_tol) ok = .FALSE.
	      IF (ok) THEN
  	         sum = sum + x
                 n = n + 1
	      ENDIF
	   ENDIF
 100    CONTINUE
        IF (n .EQ. 0) GOTO 5000
#ifdef double_p
        zmean = sum/DBLE(n)
#else
        zmean = sum/FLOAT(n)
#endif

        IF (need_std) THEN

* Compute variance. See http://en.wikipedia.org/wiki/Algorithms_for_calculating_variance

* On-line algorithm with mean subtracted first. Noted as the most robust.
* Since we always compute the mean first anyway, use it.

	   n = 0
	   xmean = 0.D0
	   sum2 = 0.D0
 
	   DO i = 1, nsize
	   x = z(i)
	      IF ( x .NE. badz) THEN
	         ok = .TRUE.
	         IF (zmax_test .AND. x.GE.z_max_tol) ok = .FALSE.
	         IF (zmin_test .AND. x.LE.z_min_tol) ok = .FALSE.
	         IF (ok) THEN
                    n = n + 1
		    x = x - zmean
                    xdelta = x - xmean
                    xmean = xmean + xdelta/FLOAT(n)
                    sum2 = sum2 + xdelta*(x - xmean)  ! This expression uses the new value of mean
 	         ENDIF
 	      ENDIF
	   ENDDO
           variance_c = sum2/FLOAT(n - 1)
	   zstd = SQRT(variance_c)

* Ignore any data outside 3 STD
           
	   tol_lo = zmean - 3.*zstd
	   tol_hi = zmean + 3.*zstd
	   IF (zmin_test) tol_hi = MIN(tol_hi, z_max_tol)
	   IF (zmax_test) tol_lo = MAX(tol_lo, z_min_tol)

c Recompute data mean.
 
	   sum = 0.0
	   n2 = 0
	   DO 200 i = 1, nsize
              x = z(i)
	      IF ( x .NE. badz) THEN
	         ok = .TRUE.
	         IF ( x .GE. tol_hi) ok = .FALSE.
	         IF ( x. LE. tol_lo) ok = .FALSE.
	         IF (ok) THEN
                    sum = sum + x
                    n2 = n2 + 1
                 ENDIF
              ENDIF
  200      CONTINUE

	   IF (n2 .EQ. 0) GOTO 5000
	   IF (n2 .EQ. n) GOTO 4900

#ifdef double_p
	   zmean2 = sum/DBLE(n2)
#else
	   zmean2 = sum/FLOAT(n2)
#endif
	   n2 = 0
	   xmean = 0.D0
	   sum2 = 0.D0
 
	   DO i = 1, nsize
	      x = z(i)
	      IF ( x .NE. badz) THEN
	         ok = .TRUE.
	         IF ( x .GE. tol_hi) ok = .FALSE.
	         IF ( x. LE. tol_lo) ok = .FALSE.
	         IF (ok) THEN
                    n2 = n2 + 1
		    x = x - zmean2
                    xdelta = x - xmean
                    xmean = xmean + xdelta/FLOAT(n2)
                    sum2 = sum2 + xdelta*(x - xmean)  ! This expression uses the new value of mean
 	         ENDIF
	      ENDIF
	   ENDDO

	   IF (FLOAT(n2)/FLOAT(n) .GT. 0.9) THEN
              zmean = zmean2
              variance_c = sum2/FLOAT(n2 - 1)
              zstd = SQRT(variance_c)
           ENDIF

c Once more.
           tol_lo = zmean - 3.*zstd
           tol_hi = zmean + 3.*zstd
	   IF (zmin_test) tol_hi = MIN(tol_hi, z_max_tol)
	   IF (zmax_test) tol_lo = MAX(tol_lo, z_min_tol)
 
           sum = 0.0
           n2 = 0
           DO 300 i = 1, nsize
              x = z(i)
	      IF ( x .NE. badz) THEN
	         ok = .TRUE.
	         IF ( x .GE. tol_hi) ok = .FALSE.
	         IF ( x. LE. tol_lo) ok = .FALSE.
	         IF (ok) THEN
                    sum = sum + x
                    n2 = n2 + 1
                 ENDIF
	      ENDIF
  300      CONTINUE
           IF (n2 .EQ. 0) GOTO 5000
#ifdef double_p
           zmean2 = sum/DBLE(n2)
#else
           zmean2 = sum/FLOAT(n2)
#endif
	   n2 = 0
	   xmean = 0.D0
	   sum2 = 0.D0
 
	   DO i = 1, nsize
	      x = z(i)
	      IF ( x .NE. badz) THEN
	         ok = .TRUE.
	         IF ( x .GE. tol_hi) ok = .FALSE.
	         IF ( x. LE. tol_lo) ok = .FALSE.
	         IF (ok) THEN
		    n2 = n2 + 1
		    x = x - zmean2
                    xdelta = x - xmean
                    xmean = xmean + xdelta/FLOAT(n2)
                    sum2 = sum2 + xdelta*(x - xmean)  ! This expression uses the new value of mean
 	         ENDIF
	      ENDIF
	   ENDDO

	   IF (FLOAT(n2)/FLOAT(n) .GT. 0.9) THEN
              zmean = zmean2
              variance_c = sum2/FLOAT(n2 - 1)
              zstd = SQRT(variance_c)
           ENDIF

        ENDIF  ! need_std

 4900   CONTINUE

* If user is resetting the mean, do that here
	IF (centered) zmean = 0.

* If the std came out as zero, use linear color levels

	zero = 0.

	IF (need_std) THEN
	   IF ( .NOT. TM_FPEQ(DBLE(zmean), zero) )  THEN
	      IF (TM_FPEQ_SNGL(zstd/zmean, zero))  GOTO 5010
	   ELSE 
	      x = MAX(ABS(zmin), ABS(zmax))
	      IF (TM_FPEQ(zstd/x , zero) )  GOTO 5010
	   ENDIF 
	ENDIF

 5000	RETURN
	
 5010   status = 0
 
	  CALL WARN(
     . 'Could not compute Std Dev. Data too large or or not within 3 std of computed mean.')
	  CALL WARN('Using linear levels instead.')

	RETURN
	END
