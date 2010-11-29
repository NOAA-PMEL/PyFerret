      subroutine day_of_year(amo, ada, ayr, status, err_msg)

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
C  ACM 
C  Find sequential day of the year corresponding to month, day, year. 
C  Return sequential day in variable ada.
C  4/2001  *ACM* add error message

      REAL daymo(12)           ! Number of days in each month
      DATA daymo/31,28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31/
      REAL amo, ada, ayr
      INTEGER iyr, imo, i, status
      CHARACTER*(*) err_msg

      status = 0
      iyr = ayr
      IF (MOD(iyr, 4) .EQ. 0) THEN
        daymo(2) = 29
      ENDIF

      imo = amo
      IF (imo .LT. 1  .OR.  imo .GT. 12) THEN
         WRITE (err_msg,*) 'Month less than 1 or greater than 12', imo
         GO TO 1000
      ENDIF

      IF (ada .LT. 0  .OR. ada .GT. daymo(imo)) THEN
	 WRITE (err_msg,10) ada, imo
   10	 FORMAT ('Day ', F4.0, ' out of range for month', I3)
         GO TO 1000
      ENDIF

      DO 100 i = 1, imo-1
       ada = ada + daymo(i)
  100 CONTINUE

      RETURN

1000  status = 1
      RETURN
      END
