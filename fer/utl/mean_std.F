	   SUBROUTINE MEAN_STD (dat, nsize, bad, need_std, dmean, dstd)
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

C** Return the mean and optionally standard deviation for the 1-D series

* calling argument declarations:

	LOGICAL need_std
	INTEGER nsize
	REAL dat(*), bad, dmean, dstd

* internal variable declarations:
	REAL sum, dev

	REAL x, xmean, sum2, xdelta
	INTEGER i, n, n2

* Initialize

	dmean = bad
	dstd = bad

c Compute data mean.

        sum = 0.0
        n = 0
        DO i = 1, nsize
	   x = dat(i)
	   IF ( x .NE. bad) THEN
	      sum = sum + x
              n = n + 1
	   ENDIF
        ENDDO

        IF (n .EQ. 0) GOTO 5000
	dmean = sum/DBLE(n)

        IF (need_std) THEN

* Compute variance. See http://en.wikipedia.org/wiki/Algorithms_for_calculating_variance

* On-line algorithm with mean subtracted first. Noted as the most robust.
* Since we always compute the mean first anyway, use it.

	   n = 0
	   xmean = 0.D0
	   sum2 = 0.D0
 
	   DO i = 1, nsize
	   x = dat(i)
	      IF ( x .NE. bad) THEN
	         n = n + 1
	         x = x - dmean
	         xdelta = x - xmean
	         xmean = xmean + xdelta/DBLE(n)
	         sum2 = sum2 + xdelta*(x - xmean)  ! This expression uses the new value of mean
 	      ENDIF
	   ENDDO

	   dstd = SQRT(sum2/DBLE(n - 1))

        ENDIF

 5000	RETURN

	END