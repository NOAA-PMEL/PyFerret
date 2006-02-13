	CHARACTER*(*) FUNCTION SECS_TO_DATE_OUT ( num_secs, cal_id, 
     .            prec )
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
* Convert secs since 01-JAN-0000 00:00:00 to a date of the form
* yyyy-mm-dd hh:mm:ss
* difference from TM_SECS_TO_DATE: if year=0 or 1 then blank year field 

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 - code pulled from TM_SECS_TO_DATE
* revision 1.0 - 2/22/89 - calls TM_SECS_TO_DATE
* V530: *acm* 12/00 add calendar types; cal_id

* Calling argument declarations:
	REAL*8		num_secs
	INTEGER		cal_id, prec

* internal variable declarations
	INTEGER		year
	CHARACTER	TM_SECS_TO_DATE*20, temp_str*20
        CHARACTER*3     month_names(12)
        DATA month_names/
     .	'JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN',
     .    'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC'/

* convert to date string always including year
	temp_str = TM_SECS_TO_DATE( num_secs, cal_id )
        CALL TM_BREAK_DATE (temp_str, cal_id, year, month, day,
     .		             hour, minute, second, status)

* Select the requested precision and form the string.  If year = 0000 
* or 0001 do not write the year

	IF ( prec .GE. 6 ) THEN		! full precision

	   IF (year .GT. 1) THEN
              WRITE (temp_str,1000,ERR=5000) day, month_names(month), 
     .               year, hour, minute, second
           ELSE
              WRITE (temp_str,1010,ERR=5000) day, month_names(month), 
     .               hour, minute, second
           ENDIF

1000       FORMAT(I2.2,'-',A3,'-',I4.4,' ',2(I2.2,':'),I2.2)
1010	   FORMAT(I2.2,'-',A3,' ',2(I2.2,':'),I2.2)


	ELSEIF ( prec .EQ. 5 ) THEN	! dd-mmm-yyyy hh:mm
	   IF (year .GT. 1) THEN
              WRITE (temp_str,1020,ERR=5000) day, month_names(month), 
     .               year, hour, minute
           ELSE
              WRITE (temp_str,1030,ERR=5000) day, month_names(month), 
     .               hour, minute

           ENDIF
1020       FORMAT(I2.2,'-',A3,'-',I4.4,' ',I2.2,':',I2.2)
1030	   FORMAT(I2.2,'-',A3,' ',I2.2,':',I2.2)

	ELSEIF ( prec .EQ. 4 ) THEN	! dd-mmm-yyyy hh
	   IF (year .GT. 1) THEN
              WRITE (temp_str,1040,ERR=5000) day, month_names(month), 
     .               year, hour
           ELSE
              WRITE (temp_str,1050,ERR=5000) day, month_names(month), 
     .               hour

           ENDIF
1040       FORMAT(I2.2,'-',A3,'-',I4.4,' ',I2.2)
1050	   FORMAT(I2.2,'-',A3,' ',I2.2)

	ELSEIF ( prec .EQ. 3 ) THEN	! dd-mmm-yyyy
	   IF (year .GT. 1) THEN
              WRITE (temp_str,1060,ERR=5000) day, month_names(month), 
     .               year
           ELSE
              WRITE (temp_str,1070,ERR=5000) day, month_names(month)

           ENDIF
1060       FORMAT(I2.2,'-',A3,'-',I4.4)
1070	   FORMAT(I2.2,'-',A3)

	ELSEIF ( prec .EQ. 2 ) THEN	! mmm-yyyy
	   IF (year .GT. 1) THEN
              WRITE (temp_str,1080,ERR=5000) month_names(month), year
           ELSE
              WRITE (temp_str,1090,ERR=5000) month_names(month)

           ENDIF
1080       FORMAT(A3,'-',I4.4)
1090	   FORMAT(A3)

	ELSE				! yyyy
	   IF (year .GT. 1) THEN
              WRITE (temp_str,1100,ERR=5000) year
           ELSE
              temp_str = ' '

           ENDIF
1100       FORMAT(I4.4)
	ENDIF

* done
        SECS_TO_DATE_OUT = temp_str
	RETURN

* internal error
 5000	STOP 'SECS_TO_DATE'
	END

