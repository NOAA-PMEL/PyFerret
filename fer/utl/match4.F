	LOGICAL FUNCTION MATCH4 (test_string,len_test,model_string,len_model)

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
* check to see if test string is same as model string for as many characters
* as it is given (to a max of 4)
* model string must be all upper case - test string ignores case
* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 - 3/24/86
*
* V5.80 *acm* 7/6/04 If either string is a single blank character, then
*                    len_to_test = 0 and MATCH4 was set true for all values 
*                    of the other string.  Check for lengths of zero.

	INTEGER		TM_LENSTR, len_model, len_test, len_to_test, i

	CHARACTER*1	UPPER_CASE
	CHARACTER*4	model_string
	CHARACTER*(*)	test_string
        LOGICAL         result

*	len_model = TM_LENSTR( model_string )
*	len_test  = TM_LENSTR( test_string(1:LEN(test_string)) )

* if model string is fewer than 4 characters the test string must not be longer
	IF (      len_model .LT. 4
     .	    .AND. len_test  .GT. len_model ) GOTO 200

        IF (len_test .EQ. 0 .AND. len_model .NE. 0) GOTO 200 
        IF (len_test .NE. 0 .AND. len_model .EQ. 0) GOTO 200 

	len_to_test = MIN( len_test, len_model )

*	DO 100 i = 1,len_to_test
*	IF ( model_string(i:i) .NE. UPPER_CASE( test_string(i:i) ) ) GOTO 200
* 100	CONTINUE
        CALL tm_match_captial_name(test_string, model_string, 
     .          len_to_test, result)

        IF(.NOT.result) GOTO 200
* it matches
	MATCH4 = .TRUE.
	RETURN

* it doesn't match
 200	MATCH4 = .FALSE.
	RETURN

	END
