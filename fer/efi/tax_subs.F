* 
*  Subroutines for TAX_* functions
* 
*  This software was developed by the Thermal Modeling and Analysis
*  Project(TMAP) of the National Oceanographic and Atmospheric
*  Administration's (NOAA) Pacific Marine Environmental Lab(PMEL),
*  hereafter referred to as NOAA/PMEL/TMAP.
*
*  Access and use of this software shall impose the following
*  obligations and understandings on the user. The user is granted the
*  right, without anx fee or cost, to use, copy, modify, alter, enhance
*  and distribute this software, and anx derivative works thereof, and
*  its supporting documentation for anx purpose whatsoever, provided
*  that this entire notice appears in all copies of the software,
*  derivative works and supporting documentation.  Further, the user
*  agrees to credit NOAA/PMEL/TMAP in anx publications that result from
*  the use of this software or in anx product that includes this
*  software. The names TMAP, NOAA and/or PMEL, however, may not be used
*  in anx advertising or publicity to endorse or promote anx products
*  or commercial entity unless specific written permission is obtained
*  from NOAA/PMEL/TMAP. The user also understands that NOAA/PMEL/TMAP
*  is not obligated to provide the user with anx support, consulting,
*  training or assistance of anx kind with regard to the use, operation
*  and performance of this software nor to provide the user with anx
*  updates, revisions, new versions or "bug fixes".
*
*  THIS SOFTWARE IS PROVIDED BY NOAA/PMEL/TMAP "AS IS" AND Anx EXPRESS
*  OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
*  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
*  ARE DISCLAIMED. IN NO EVENT SHALL NOAA/PMEL/TMAP BE LIABLE FOR Anx SPECIAL,
*  INDIRECT OR CONSEQUENTIAL DAMAGES OR Anx DAMAGES WHATSOEVER
*  RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF
*  CONTRACT, NEGLIGENCE OR OTHER TORTUOUS ACTION, ARISING OUT OF OR IN
*  CONNECTION WITH THE ACCESS, USE OR PERFORMANCE OF THIS SOFTWARE. 
*
****************************************************************
*
* V6.2 ACM 2/12/2009  Fix bug 1629 in JULIAN_DAY_OF_YEAR. Always
*          set ndaymo(2) to either 28 or 29. It does not get reset
*          by the DATA statement on re-entry to the routine.

      SUBROUTINE LOWER_CASE(TEXT)
      CHARACTER*(*) TEXT
*
*     Converts a string to lowercase.
*
      EXTERNAL LENGTH_OF_TEXT
      INTEGER  LENGTH_OF_TEXT
      CHARACTER*1 LETTER
      INTEGER ISHIFT,I
      ISHIFT=ICHAR('a')-ICHAR('A')
      DO 10 I=1,LENGTH_OF_TEXT(TEXT)
        LETTER=TEXT(I:I)
        IF('A'.LE.LETTER.AND.LETTER.LE.'Z') THEN
          TEXT(I:I)=CHAR(ICHAR(LETTER)+ISHIFT)
        END IF
   10 CONTINUE
      RETURN
      END

***************************************************************
*
      INTEGER FUNCTION LENGTH_OF_TEXT(TEXT)
      CHARACTER*(*) TEXT
*
*     Computes the length of a string without trailing blanks.
*
      INTEGER I

      DO 1 I=LEN(TEXT),1,-1
        IF(TEXT(I:I).NE.' ') THEN
          GO TO 2
        END IF
    1 CONTINUE
      I=1
    2 CONTINUE
      LENGTH_OF_TEXT=I

      RETURN
      END


************************************************************************      
*     This subroutine returns day number of current year and the total 
*     number of days of current year.
      SUBROUTINE JULIAN_DAY_OF_YEAR(cmon, day_of_mon, iyear, day_of_year, 
     .           days_this_year)

*     Convert day of month to day of year

      INTEGER i, iyear
      CHARACTER*3 cmon
      CHARACTER*3 months(12)
      INTEGER day_of_mon, day_of_year, imon, ndaymo(12)
      INTEGER days_this_year

      DATA months/'JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL',
     .            'AUG', 'SEP', 'OCT', 'NOV', 'DEC'/

      DATA ndaymo/31,28,31,30,31,30,31,31,30,31,30,31/

      imon = 0
      DO 100 i=1,12
          IF(cmon .EQ. months(i)) imon = i 
 100  CONTINUE

*     Check if it is a leap year
* (Fixing bug 1639, need ELSE ndaymo(2) = 28; the data statement 
* above does NOT reset the value to 28 next time we enter this routine!!  

      IF ( ((mod(iyear,4) .EQ. 0) .AND. (mod(iyear,100) .NE. 0)) 
     .     .OR.
     .     (mod(iyear, 400) .EQ. 0)) THEN
          ndaymo(2) = 29
      ELSE 
          ndaymo(2) = 28
      END IF

*     Compute day number within current year
      day_of_year = day_of_mon
      DO 200 i =1, imon -1
          day_of_year = day_of_year + ndaymo(i) 
 200  CONTINUE

*     Compute total days of current year
      days_this_year = 0
      DO 300 i =1, 12 
          days_this_year = days_this_year + ndaymo(i)
 300  CONTINUE

      RETURN
      END

* **********************************************************************

*     Convert day of month to day since 1900
*     Returns -1 if the year is before 1900

      SUBROUTINE JULIAN_DAY_OF_YEAR1900(cmon, day_of_mon, iyear,
     .   day_since_1900)

      INTEGER i, j, iyear
      INTEGER yrstart,yrend
      LOGICAL isafter1900
      CHARACTER*3 cmon
      CHARACTER*3 months(12)
      INTEGER day_of_mon, day_of_year, imon, day_since_1900, ndaymo(12)

      DATA months/'JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL',
     .            'AUG', 'SEP', 'OCT', 'NOV', 'DEC'/

      DATA ndaymo/31,28,31,30,31,30,31,31,30,31,30,31/

      imon = 0
      DO 100 i=1,12
          IF(cmon .EQ. months(i)) imon = i 
 100  CONTINUE

      yrstart = 1900
      yrend = iyear-1
      isafter1900 = .TRUE. 
      day_since_1900 = -1

      IF(iyear .LT. 1900) THEN
          yrstart = iyear+1
          yrend = 1899
          isafter1900 = .FALSE.          
          day_since_1900 = 0
      ENDIF

*     Total days between yrstart and yrend
      DO 200 i = yrstart, yrend

*     Check if it is a leap year
          IF ( ((mod(i,4) .EQ. 0) .AND. (mod(i,100) .NE. 0))
     .     .OR.
     .     (mod(i, 400) .EQ. 0)) THEN
              ndaymo(2) = 29
          ELSE
              ndaymo(2) = 28
          ENDIF
 
          day_since_1900 = day_since_1900 + 
     .                    (ndaymo(2) + 7*31 + 4*30)

 200  CONTINUE

*     Add days of current year
      IF ( ((mod(iyear,4) .EQ. 0) .AND. (mod(iyear,100) .NE. 0))
     .     .OR.
     .     (mod(iyear, 400) .EQ. 0)) THEN
          ndaymo(2) = 29
      ELSE
          ndaymo(2) = 28
      ENDIF

      IF(isafter1900) THEN
*     Year after 1900
          day_of_year = day_of_mon
          DO 300 j =1, imon -1
              day_of_year = day_of_year + ndaymo(j)
 300      CONTINUE
       
          day_since_1900 = day_since_1900 + day_of_year
      ELSE
*     Year before 1900
          day_of_year = day_of_mon
          DO 400 j =1, imon -1
              day_of_year = day_of_year + ndaymo(j)
 400      CONTINUE

          day_since_1900 = -1*(day_since_1900 + 
     .                     ndaymo(2) + 7*31 + 4*30- day_of_year+1)

      ENDIF

      RETURN
      END
*                                                                      ^
*                                                                      |
*                                                                      |
* **********************************************************************
