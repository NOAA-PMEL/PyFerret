	LOGICAL FUNCTION MATCH_QUOTED_NAME ( test_string, len_test, model_string, len_model_chars)

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
* check to see if test string is same as model string to full length of model
* string
* model string must be all upper case - test string ignores case
* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 - 5/1/86
* revision 0.1 - 12/9/86 - allowed model_string of arbitrary length
* V495: 7/7/98 - *sh* if test name is in single quotes use case-sensitive comp.
*     : 9/30/98 - NAG F90 (linux) received error on above change.  
*                 need to put single  tick between double quotes  *kob*
* ywei 05/04   changed function prototype to allow speed up
*              before calling this function, make sure
*              len_test = TM_LENSTR(test_string)
*              len_model_chars = TM_LENSTR(model_string)

* calling argument declarations:
	CHARACTER*(*)	model_string
	CHARACTER*(*)	test_string
        INTEGER         len_test
        INTEGER         len_model_chars

* internal variable declarations:
	LOGICAL         is_quoted
	INTEGER		len_model, i
        LOGICAL         result

	len_model	= LEN( model_string )

* (7/98) is test string in quotations?
	is_quoted = test_string(1:1) .EQ. "'" 
	IF (is_quoted) is_quoted = test_string(len_test:len_test) .EQ. "'"
	IF (is_quoted) len_test = len_test - 2

* test string cannot be shorter than model or longer than fully terminated model
	IF (   len_test        .LT. len_model_chars
     .	 .OR.( len_model_chars .LT. len_model		! fully terminated ?
     .	 .AND. len_test	       .GT. len_model_chars )	) GOTO 200


	IF (is_quoted) THEN
	   IF (model_string(1:len_model_chars) .NE. test_string(2:len_model_chars+1)) GOTO 200
	ELSE
           CALL tm_match_captial_name(test_string, model_string, 
     .          len_model_chars, result)
           IF(.NOT.result) GOTO 200	   
	ENDIF

* it matches
	MATCH_QUOTED_NAME = .TRUE.
	RETURN

* it doesn't match
 200	MATCH_QUOTED_NAME = .FALSE.
	RETURN

	END
