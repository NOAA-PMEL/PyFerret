	LOGICAL FUNCTION MATCH_TEMPLATE (name, template)

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
* check to see if the name matches the template
* the template may contain
*	"*" which matches 0 or more of any characters 
*	"?" which match any single character

* Note: this routine IS case sensitive so make sure to UPCASE strings as
*	needed prior to calling this

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program

* V420  4/18/96

* calling argument declarations:
	CHARACTER*(*)	template
	CHARACTER*(*)	name

* internal variable declarations:
	LOGICAL		star_skip
	INTEGER		TM_LENSTR1, in, it, nlen, tlen
	CHARACTER*1	ch

* initialize
	nlen	= TM_LENSTR1( name )
	tlen	= TM_LENSTR1( template )
	star_skip = .FALSE.
	in = 1

* LOOP through the characters of the template
* (advance the char ptr for name, "in", as needed)
	DO 500 it = 1, tlen

	  ch = template(it:it)		! next character to match

	  IF ( ch .EQ. '*' ) THEN
	    star_skip = .TRUE.
	    GOTO 500

	  ELSEIF ( ch .EQ. '?' ) THEN
	    IF ( star_skip ) THEN
	      GOTO 500			! "*?" matches same as "*"
	    ELSE
	      GOTO 400
	    ENDIF

	  ELSEIF ( star_skip ) THEN
	    DO 100 in = in, nlen
	      IF ( name(in:in) .EQ. ch ) THEN
	        star_skip = .FALSE.
	        GOTO 400
	      ENDIF
 100	    CONTINUE
	    GOTO 1000			! no match for char after "*"

	  ELSEIF ( in .GT. nlen ) THEN
	    GOTO 1000			! no more name chars available

	  ELSEIF ( ch .EQ. name(in:in) ) THEN
	    GOTO 400

	  ELSE
	    GOTO 1000

	  ENDIF

* advance to next character of name
 400	  IF ( (it.EQ.tlen) ) THEN
	    IF ( (in.NE.nlen) ) GOTO 1000  ! too many chars in name
	  ENDIF
	  in = in + 1

 500	CONTINUE

* success - we've passed through the entire template with no error
	MATCH_TEMPLATE = .TRUE.
	RETURN

* something doesn't match
 1000	MATCH_TEMPLATE = .FALSE.
	RETURN

	END
