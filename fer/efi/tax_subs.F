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
* V6.2 ACM 2/12/2008  Fix bug 1629 in JULIAN_DAY_OF_YEAR. Always
*          set ndaymo(2) to either 28 or 29. It does not get reset
*          by the DATA statement on re-entry to the routine.
* v745 *acm* 12/18  Issue 1909, option to write dates in ISO8601 format
* v745 *acm* 12/18  New TAX_DATE_STRINGS called by TAX_DATESTRING fcn

      SUBROUTINE LOWER_CASE(TEXT)
      IMPLICIT NONE
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
      IMPLICIT NONE
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
      SUBROUTINE JULIAN_DAY_OF_YEAR(cmon, imon, day_of_mon, iyear, 
     .           day_of_year, days_this_year)

*     Convert day of month to day of year

      IMPLICIT NONE

* calling arguments
      INTEGER imon, day_of_mon, iyear, 
     .           day_of_year, days_this_year
      CHARACTER*3 cmon

* Local variables
      INTEGER i 
      CHARACTER*3 months(12)
      INTEGER ndaymo(12)

      DATA months/'JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL',
     .            'AUG', 'SEP', 'OCT', 'NOV', 'DEC'/

      DATA ndaymo/31,28,31,30,31,30,31,31,30,31,30,31/

* If they sent in a character month, convert it to month number
      IF (cmon .NE. 'XXX') THEN
         imon = 0
         DO 100 i=1,12
             IF(cmon .EQ. months(i)) imon = i 
 100     CONTINUE
      ENDIF

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

      SUBROUTINE JULIAN_DAY_OF_YEAR1900(cmon, imon, day_of_mon, iyear,
     .   day_since_1900)

      IMPLICIT NONE
      
* calling arguments

      INTEGER imon, day_of_mon, iyear, day_since_1900
      CHARACTER*3 cmon

      INTEGER i, j
      INTEGER yrstart,yrend
      LOGICAL isafter1900
      CHARACTER*3 months(12)
      INTEGER day_of_year, ndaymo(12)

      DATA months/'JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL',
     .            'AUG', 'SEP', 'OCT', 'NOV', 'DEC'/

      DATA ndaymo/31,28,31,30,31,30,31,31,30,31,30,31/

* If they sent in a character month, convert it to month number
      IF (cmon .NE. 'XXX') THEN
         imon = 0
         DO 100 i=1,12
             IF(cmon .EQ. months(i)) imon = i 
 100     CONTINUE
      ENDIF

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


      SUBROUTINE tax_format_message (L1, err_msg)
C   Create a message saying arg1 values at index are duplicate

      IMPLICIT NONE
      INTEGER L1
      CHARACTER*(*) err_msg
      
      REAL v1, v2
      INTEGER n1, n2
      CHARACTER TM_FMT*48, buff1*15, buff2*15

      v1 = L1-1
      v2 = L1
      buff1 = TM_FMT(v1, 10, 12, n1) 
      buff2 = TM_FMT(v2, 10, 10, n2) 
      
      IF (n1 .GE. 14 .OR. n2 .GE. 14) THEN
         WRITE (err_msg, 1014) L1-1, L1
      ELSEIF (n1 .GE. 12 .OR. n2 .GE. 12) THEN
         WRITE (err_msg, 1012) L1-1, L1
      ELSEIF (n1 .GE. 10 .OR. n2 .GE. 10) THEN
         WRITE (err_msg, 1010) L1-1, L1
      ELSEIF (n1 .GE. 8 .OR. n2 .GE. 8) THEN
         WRITE (err_msg, 1008) L1-1, L1
      ELSE
         WRITE (err_msg, 1000) buff1(1:n1), buff2(1:n1)
      ENDIF

1000  FORMAT ('Duplicate times in ARG1, may arise from double- to single- ',
     .          'precision conversion. At indices ', A, ',', A)
1014  FORMAT ('Duplicate times in ARG1, may arise from double- to single- ',
     .          'precision conversion. At indices ', 2i16)
1012  FORMAT ('Duplicate times in ARG1, may arise from double- to single- ',
     .          'precision conversion. At indices ', 2i14)
1010  FORMAT ('Duplicate times in ARG1, may arise from double- to single- ',
     .          'precision conversion. At indices ', 2i12)
1008  FORMAT ('Duplicate times in ARG1, may arise from double- to single- ',
     .          'precision conversion. At indices ', 2i10)

      RETURN
      END

      

      SUBROUTINE TAX_REFORMAT_DATE (indate, outdate, dlen)

* From tm_date_reformat, just the portion that calls TM_BREAK_FMT_DATE_C

*     If there is a T between the date and time (as in GRIB files) replace with a space.

        IMPLICIT NONE
#include "gt_lib.parm"

	CHARACTER*(*)   indate, outdate
	INTEGER		dlen

	CHARACTER*32 buff

	INTEGER TM_BREAK_FMT_DATE_C, slen, year, month, day,
     .          hour, minute, second, status, outform
	PARAMETER ( slen = 32)

	REAL fsecond
	INTEGER*1  dhol(slen)     !c-type Hollerith string buffers

*     If there is a T between the date and time (as in GRIB files) replace with a space.
        buff = indate
	IF (buff(11:11) .EQ. 'T') buff(11:11) = ' '
	IF (buff(11:11) .EQ. 'Z') buff(11:11) = ' '
 
* Break up into year, month, day, ...
	CALL TM_FTOC_STRNG( buff(:slen), dhol, slen)
	status = TM_BREAK_FMT_DATE_C(dhol, year, month,
     .                              day, hour, minute, fsecond)
	IF ( status .NE. 0 ) GOTO 5100

	second = NINT(fsecond)       ! information loss ...

* and reassemble it into the desired output format
	outform = pdate_vax 

	CALL TM_MAKE_FMT_DATE ( outform, outdate, year, month, day,
     .                         hour, minute, second, dlen )

 5000	RETURN
 5100	outdate = 'xxx'
        dlen = 3
	GOTO 5000
	END



* Convert time coordinates to dates.  If fractional seconds are needed
* for the formatting, this may be run a second time for consistent formatting
* when some times fall on an even second.

      SUBROUTINE TAX_DATE_STRINGS (arg_1, result, arg_lo_ss, arg_incr, 
     .                res_lo_ss, res_hi_ss, get_prec, dir, precistring, 
     .                tmodulo, bad_flag, status)


      IMPLICIT NONE
      INCLUDE 'EF_Util.cmn'
      INCLUDE 'EF_mem_subsc.cmn'

      INTEGER id

      REAL arg_1(mem1lox:mem1hix, mem1loy:mem1hiy, mem1loz:mem1hiz, 
     .           mem1lot:mem1hit, mem1loe:mem1hie, mem1lof:mem1hif)
      REAL result(memreslox:memreshix, memresloy:memreshiy, 
     .           memresloz:memreshiz, memreslot:memreshit,
     .           memresloe:memreshie, memreslof:memreshif)

      INTEGER res_lo_ss(6),
     .        res_hi_ss(6)
      INTEGER arg_lo_ss(6,EF_MAX_ARGS),
     .        arg_incr (6,EF_MAX_ARGS)

      REAL bad_flag(EF_MAX_ARGS)
      INTEGER get_prec, dir, status

      LOGICAL tmodulo, default_datefmt, first

      INTEGER i, j, k, l, m, n
      INTEGER i1, j1, k1, l1, m1, n1
      INTEGER i3, j3, k3, l3, m3, n3
      INTEGER slen, dlen, prec_full
      CHARACTER*35 datestring
      CHARACTER*(*) precistring
      REAL*8 ddate

      status = 0
      prec_full = 7
      first = .TRUE.

      n1 = arg_lo_ss(F_AXIS,ARG1)
      DO 600 n = res_lo_ss(F_AXIS), res_hi_ss(F_AXIS)

      m1 = arg_lo_ss(E_AXIS,ARG1)
      DO 500 m = res_lo_ss(E_AXIS), res_hi_ss(E_AXIS)

      l1 = arg_lo_ss(T_AXIS,ARG1)
      DO 400 l = res_lo_ss(T_AXIS), res_hi_ss(T_AXIS)

      k1 = arg_lo_ss(Z_AXIS,ARG1)
      DO 300 k = res_lo_ss(Z_AXIS), res_hi_ss(Z_AXIS)

      j1 = arg_lo_ss(Y_AXIS,ARG1)
      DO 200 j = res_lo_ss(Y_AXIS), res_hi_ss(Y_AXIS)

      i1 = arg_lo_ss(X_AXIS,ARG1)
      DO 100 i = res_lo_ss(X_AXIS), res_hi_ss(X_AXIS)

         ddate = arg_1(i1,j1,k1,l1,m1,n1) 
         ddate = ddate  
         IF ( ddate .eq. bad_flag(ARG1) ) THEN
	     datestring = '...'
             slen = 3   
	     CALL EF_PUT_STRING(datestring, slen,
     .                                  result(i,j,k,l,m,n))
	     CYCLE
	  ENDIF

*     Get the date string corresponding to time ddate for the variable ARG2
         CALL EF_GET_AXIS_DATES (id, ARG2, ddate, dir, 1, 
     .                   prec_full, dlen, datestring)

* Are we using the default date format or ISO-8601 format? 

         IF (first) THEN
	    default_datefmt = (datestring(3:3) .EQ. '-')
	    first = .FALSE.
	 ENDIF

*     Assign the appropriate substring to the result.
            
         IF (default_datefmt) THEN  ! formatting as Ferret-style date strings

	    IF ( .NOT. tmodulo ) THEN

	       IF ( precistring .EQ. 'sec' ) THEN
                  slen = MAX(dlen, 20)
                  CALL EF_PUT_STRING(datestring, slen,
     .                               result(i,j,k,l,m,n))
               ELSE IF ( precistring .EQ. 'min' ) THEN
                  slen = 17
                  CALL EF_PUT_STRING(datestring(1:17), slen,
     .                               result(i,j,k,l,m,n))
               ELSE IF ( precistring .EQ. 'hou' ) THEN
                  slen = 14
                  CALL EF_PUT_STRING(datestring(1:14), slen,
     .                               result(i,j,k,l,m,n))
               ELSE IF ( precistring .EQ. 'day' ) THEN
                  slen = 11
                  CALL EF_PUT_STRING(datestring(1:11), slen,
     .                               result(i,j,k,l,m,n))
               ELSE IF ( precistring .EQ. 'mon' ) THEN
                  slen = 8
                  CALL EF_PUT_STRING(datestring(4:11), slen,
     .                               result(i,j,k,l,m,n))
               ELSE IF ( precistring .EQ. 'yea' ) THEN
                  slen = 4
                  CALL EF_PUT_STRING(datestring(8:11), slen,
     .                               result(i,j,k,l,m,n))
               ELSE
                  slen = 20
                  CALL EF_PUT_STRING (datestring, slen,
     .                               result(i,j,k,l,m,n))
               ENDIF

            ELSE  ! modulo time axis.  datestring looks like 15-FEB 02:54:36

               IF ( precistring .EQ. 'sec' ) THEN
                  slen = 15
                  CALL EF_PUT_STRING(datestring(1:15), slen,
     .                               result(i,j,k,l,m,n))
               ELSE IF ( precistring .EQ. 'min' ) THEN
                  slen = 12
                  CALL EF_PUT_STRING(datestring(1:12), slen,
     .                               result(i,j,k,l,m,n))
               ELSE IF ( precistring .EQ. 'hou' ) THEN
                  slen = 9
                  CALL EF_PUT_STRING(datestring(1:9), slen,
     .                               result(i,j,k,l,m,n))
               ELSE IF ( precistring .EQ. 'day' ) THEN
                  slen = 6
                  CALL EF_PUT_STRING(datestring(1:6), slen, 
     .                               result(i,j,k,l,m,n))
               ELSE IF ( precistring .EQ. 'mon' ) THEN
                  slen = 3
                  CALL EF_PUT_STRING(datestring(4:6), slen,
     .                               result(i,j,k,l,m,n))
               ELSE IF ( precistring .EQ. 'yea' ) THEN
                  datestring = '...'
                  slen = 3
                  CALL EF_PUT_STRING(datestring, slen,
     .                               result(i,j,k,l,m,n))
               ELSE
                           slen = 15
                           CALL EF_PUT_STRING(datestring, slen,
     .                                        result(i,j,k,l,m,n))
               ENDIF

            ENDIF  ! Modulo time

         ELSE  ! Iso8601, datestring = 'yyyy-mm-dd hh:mm:ss"
            IF ( precistring .EQ. 'sec' ) slen = MAX(dlen, 20)
            IF ( precistring .EQ. 'min' ) slen = 16
            IF ( precistring .EQ. 'hou' ) slen = 13
            IF ( precistring .EQ. 'day' ) slen = 10
            IF ( precistring .EQ. 'mon' ) slen = 7
            IF ( precistring .EQ. 'yea' ) slen = 4
            CALL EF_PUT_STRING(datestring, slen, result(i,j,k,l,m,n))

         ENDIF

         i1 = i1 + arg_incr(X_AXIS,ARG1)
 100  CONTINUE

         j1 = j1 + arg_incr(Y_AXIS,ARG1)
 200  CONTINUE

         k1 = k1 + arg_incr(Z_AXIS,ARG1)
 300  CONTINUE

         l1 = l1 + arg_incr(T_AXIS,ARG1)
 400  CONTINUE

         m1 = m1 + arg_incr(E_AXIS,ARG1)
 500  CONTINUE

         n1 = n1 + arg_incr(F_AXIS,ARG1)
 600  CONTINUE

	status = 0
	
	RETURN

	END
