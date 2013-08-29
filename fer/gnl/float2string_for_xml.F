	SUBROUTINE FLOAT2STRING_FOR_XML (val, attype, strng, slen)
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
* Write a floating-point value accurately to a string for XML output
*  Input is double precision floating-point number and the attribute
*  type that it had in the original dataset.
*  Return the string and string length

* programmer - Ansley Manke
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* 27-Jun-2013

        include 'netcdf.inc'

* Argument declarations	
	REAL val
	CHARACTER*(*) strng
	INTEGER attype, slen

* Internal declarations
	LOGICAL TM_FPEQ
	CHARACTER TM_FMT*16
	INTEGER TM_LENSTR, ival
	REAL test_val
  
  888	FORMAT(i16.0)

* Convert using TM_FMT

	strng = TM_FMT(val, 7,16,slen)

	IF (attype .EQ. nf_float .OR. attype .EQ. nf_double) THEN
	   slen = TM_LENSTR(strng)
	ELSE

* Check. Does the string correctly capture the value?

	   READ (strng, *) test_val
	   IF ( .NOT. TM_FPEQ (test_val, val) ) THEN

* Try with more digits
	      strng = TM_FMT(val, 9,16,slen)
	      READ (strng, *) test_val

* Still does not match. Write unformatted value. If it is
* known to be an integer, use a long integer format.

	      IF ( .NOT. TM_FPEQ (test_val, val) ) THEN
	         WRITE (strng, *) val
	         ival = val
	         IF (attype .EQ. nf_int) WRITE (strng, 888) ival
	      ENDIF

	      slen = TM_LENSTR(strng)
	   ENDIF
	ENDIF

	RETURN
	END

