
****************************************************************
*
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
      IF ( ((mod(iyear,4) .EQ. 0) .AND. (mod(iyear,100) .NE. 0)) 
     .     .OR.
     .     (mod(iyear, 400) .EQ. 0)) THEN
          ndaymo(2) = 29
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
