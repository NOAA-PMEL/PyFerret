	CHARACTER*(*) FUNCTION SECS_TO_DATE ( num_secs )
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
* $_-mm-dd hh:mm:ss
* difference from TM_SECS_TO_DATE: if year=0 or 1 then blank year field 

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 - code pulled from TM_SECS_TO_DATE
* revision 1.0 - 2/22/89 - calls TM_SECS_TO_DATE

* Calling argument declarations:
	REAL*8		num_secs

* internal variable declarations
	INTEGER		year
	CHARACTER	TM_SECS_TO_DATE*20, temp_str*20

* convert to date string always including year
	temp_str = TM_SECS_TO_DATE( num_secs )

* determine year from dd-mmm-$_:hh:mm:ss
	READ ( temp_str, '(7X,I4)', ERR=5000 ) year

* blank out year if 0000 or 0001
	IF ( year .LE. 1 ) temp_str(7:11) = ' '

* done
	SECS_TO_DATE = temp_str
	RETURN

* internal error
 5000	STOP 'SECS_TO_DATE'
	END
