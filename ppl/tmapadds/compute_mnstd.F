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
	REAL z(*), badz

* internal variable declarations:
	REAL datum, sum, dev, sumsq_dev, variance
	REAL*8 x, xmean, sum2, sumc, variance_c, xdelta
	INTEGER n

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
        zmean = sum/FLOAT(n)

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

        ENDIF

* If user is resetting the mean, do that here
	IF (centered) zmean = 0.

 5000	RETURN
	
 5010   status = 0
	RETURN
	END
