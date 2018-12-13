	SUBROUTINE TM_MAKE_FMT_DATE (form, date, year, month, day,
     .				     hour, minute, second, dlen)
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
*
* Will take the components of a date and convert it to a string of the
* requested date format

* *sh* 2/13/95
* V6.74 3/12 *acm* cleanup ifdefs and unnecessary include files
* V7.44  11/18 *acm* Note numslashnum  formatting is ISO 8601 yyyy-mm-dd
* V7.44 12/18 *acm* Allow for fractional seconds 20-DEC-1982 12:52:00.24
* v745 *acm* 12/18 Issue 1909, option to write dates in ISO8601 format
*                  Also more on writing out fractional seconds

* ARGUMENT DEFINITIONS
*  form  - code for the date string format
*  date  - date to process
*  day, month, year, hour, minute, second - obvious
*  status - returned status

	CHARACTER*(*)	date
	INTEGER	form, day, month, year, hour, minute, dlen
	REAL	second
*
* INCLUDE FILES
	include 'tmap_errors.parm'
#include "tmap_dset.parm"
#include "gt_lib.parm"

* internal variable declarations:
	INTEGER		isecond, ifrac
	REAL		frac
	CHARACTER*3    cmon(12)

	DATA cmon /'JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN',
     .             'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC'/


	IF ( form .EQ. pdate_vax ) THEN
* model date: 20-DEC-1982 12:52:00
1000       FORMAT(I2.2,'-',A3,'-',I4.4,' ',2(I2.2,':'),I2.2)
1010       FORMAT(I2.2,'-',A3,'-',I4.4,' ',2(I2.2,':'),F4.2)
	   IF (AINT(second) .EQ. second) THEN
	      isecond = AINT(second)
	      WRITE (date,1000) day, cmon(month), year, hour,
     .  			minute, isecond
	      dlen = 20
	   ELSE
	      WRITE (date,1010) day, cmon(month), year, hour,
     .  			minute, second
	      dlen = 23
	   ENDIF
	ELSEIF ( form .EQ. pdate_numslashnum ) THEN
* model date: 1982-12-20 12:52:00
1100       FORMAT(I4.4,'-',I2.2,'-',I2.2,' ',2(I2.2,':'),I2.2)
1110       FORMAT(I4.4,'-',I2.2,'-',I2.2,' ',2(I2.2,':'),I2.2,'.',I2.2)
	   IF (AINT(second) .EQ. second) THEN
	      isecond = AINT(second)
	      WRITE (date,1100) year, month, day, hour,
     .  			minute, isecond
	      dlen = 19
	   ELSE
	      isecond = AINT(second)
	      frac = second - AINT(second)
	      ifrac = 100*frac
	      WRITE (date,1110) year, month, day, hour,
     .  			minute, isecond, ifrac
	      dlen = 22
	   ENDIF
	ELSE
	   STOP 'Unknown date format code'
	ENDIF

	RETURN
	END
