	CHARACTER*(*) FUNCTION ARG_STRING( val )

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
* return a text string representing the value "val"
* if val is unspecified (bad_val4) then return blank, else start with a colon
* if val is integer then format it as such
* declare length of ARG_STRING in calling program to be at least 11

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 - 9/21/87
* V230: 8/18/92 - eliminated string(1:) notation for MAC compatibility

#ifdef unix
	include 'ferret.parm'
#else
	INCLUDE 'FERRET_CMN:FERRET.PARM'
#endif

* calling argument declarations:
	REAL val

* internal variable declarations:
	INTEGER	  i, llen, arg_string_len
	CHARACTER LEFINT*4

* initialiaze
        arg_string_len = LEN( ARG_STRING )

	IF ( val .EQ. bad_val4 ) THEN
	   ARG_STRING = ' '

	ELSE
	   ARG_STRING(1:1) = ':'
* ... truncate to integer if possible
	   IF ( ABS(val) .LT. 10000 ) THEN
	      i = INT( val )
	   ELSE
	      i = 0
	   ENDIF
* ... use left justified I format if appropriate
	   IF ( FLOAT( i ) .EQ. val ) THEN
	      ARG_STRING(2:arg_string_len) = LEFINT( i, llen )
	   ELSE
	      WRITE ( ARG_STRING(2:arg_string_len), '(1PG10.4)' ) val
	   ENDIF
	ENDIF

	RETURN
	END
