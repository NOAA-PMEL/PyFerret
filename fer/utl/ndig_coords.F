      SUBROUTINE NDIG_COORDS (grid, ax, lo, hi, del, bad_data, ndig, ndig_default)

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
* Compute the number of digits needed to list the coordinates along
* an axis with enough resolution to distinguish them.
* Called by list_cmnd_data.F
*
* Ferret v5.80 9/02/2004  ACM
* V590:  1/06 *acm*  prevent values of LOG10(0.0)

      INCLUDE 'ferret.parm'

* calling argument declarations
      INTEGER grid, ax, lo, hi, del, ndig, ndig_default
      REAL bad_data

* Internal variable declarations
      REAL*8 TM_WORLD, val8, val_last, diff_min, diff_ave, count
      LOGICAL TM_DFPEQ
      INTEGER i

* Special formatting codes for time axes etc.

      IF (ndig_default .LE. 0) THEN
         ndig = ndig_default
         RETURN
      ENDIF

* Get number of decimal places for formatting the coordinates

      val_last = bad_data
      diff_min = ABS(bad_data)
      diff_ave = 0.D0
      count = 0.

* Find the smallest difference from one coordinate to the next.
* also compute the average difference, in case two coordinates
* are very close together.

      DO 310 i = lo, hi, del
         val8 = TM_WORLD( i, grid, ax, box_middle )

         IF (val8 .NE. bad_data .AND. val_last .NE. bad_data) THEN
            diff_min = MIN(diff_min, ABS(val8-val_last) )
            diff_ave = diff_ave + ABS(val8-val_last)
            count = count + 1
         ENDIF

         val_last = val8
 310  CONTINUE
      
      IF (count .GT. 0) THEN
         diff_ave = diff_ave/count
         IF ( TM_DFPEQ(diff_min,0.D0) ) diff_min = diff_ave
         IF (diff_min .NE. 0) THEN 
            ndig = 1 - INT(LOG10(diff_min))
         ELSE
            ndig = 1
         ENDIF
         ndig = MAX(ndig, ndig_default )
      ELSE
         ndig = ndig_default
      ENDIF

      RETURN
      END
