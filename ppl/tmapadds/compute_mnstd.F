	SUBROUTINE compute_mnstd(z, badz, need_std, nsize, status)

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
#ifdef double_p
	REAL*8 z(*), badz
#else
	REAL z(*), badz
#endif

* internal variable declarations:
#ifdef double_p
	REAL*8 sum, dev, sumsq_dev, variance, tol_lo, tol_hi, zmean2
#else
	REAL sum, dev, sumsq_dev, variance, tol_lo, tol_hi, zmean2
#endif

	REAL*8 x, xmean, sum2, sumc, variance_c, xdelta
	INTEGER i, n, n2

c  Already have zmin, zmax

        IF (.not. need_std .AND. centered) THEN
	   zstd = lev_std
	   zmean = 0.
	   GOTO 5000
	ENDIF

c Compute data mean.

        sum = 0.0
        n = 0
        DO 100 i = 1, nsize
	   x = z(i)
	   IF ( x .NE. badz) THEN
  	      sum = sum + Z(i)
              n = n + 1
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
	      x = Z(i)
	      IF ( x .NE. badz) THEN
                 n = n + 1
		 x = x - zmean
                 xdelta = x - xmean
                 xmean = xmean + xdelta/FLOAT(n)
                 sum2 = sum2 + xdelta*(x - xmean)  ! This expression uses the new value of mean
 	      ENDIF
	   ENDDO
           variance_c = sum2/FLOAT(n - 1)
	   zstd = SQRT(variance_c)

* Ignore any data outside 3 STD
           
	   tol_lo = zmean - 3.*zstd
	   tol_hi = zmean + 3.*zstd
	   
c Recompute data mean.
 
	   sum = 0.0
	   n2 = 0
	   DO 200 i = 1, nsize
              x = z(i)
              IF ( x.NE.badz .AND. x.GT.tol_lo .AND. x.LT.tol_hi) THEN
                 sum = sum + Z(i)
                 n2 = n2 + 1
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
	      x = Z(i)
	      IF ( x.NE.badz .AND. x.GT.tol_lo .AND. x.LT.tol_hi) THEN
                 n2 = n2 + 1
		 x = x - zmean2
                 xdelta = x - xmean
                 xmean = xmean + xdelta/FLOAT(n2)
                 sum2 = sum2 + xdelta*(x - xmean)  ! This expression uses the new value of mean
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
 
           sum = 0.0
           n2 = 0
           DO 300 i = 1, nsize
              x = z(i)
              IF ( x.NE.badz .AND. x.GT.tol_lo .AND. x.LT.tol_hi) THEN
                 sum = sum + Z(i)
                 n2 = n2 + 1
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
	      x = Z(i)
	      IF ( x.NE.badz .AND. x.GT.tol_lo .AND. x.LT.tol_hi) THEN
                 n2 = n2 + 1
		 x = x - zmean2
                 xdelta = x - xmean
                 xmean = xmean + xdelta/FLOAT(n2)
                 sum2 = sum2 + xdelta*(x - xmean)  ! This expression uses the new value of mean
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

 5000	RETURN
	
 5010   status = 0
	RETURN
	END
