
	SUBROUTINE COACHED_STR_SUB( sym_name, 
     .		in_str, coaching, out_str, len_out, status ) 

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
* replace the input string with the text implied in the coaching
* possibly displaying an error string

* if the input string is blank and no default then report error
* if error text was supplied with coaching use it for error message

* coaching may include:
*	"default"	- default string if arg n doesn't exist
*	">default"	- default string if arg n doesn't match any options
*	"<errmsg"	- error   string 
*	"|option"     - an acceptable value for argument n
*	"|option>opt_replace"     - an acceptable value and replacement text
*  note that "*" will match any option so "|*>whatever" will substitute
*       "whatever" for anything

* USAGE EXAMPLES:
*		      "my_default_file|file1|file2"
* ... allow only instrings "file1", "file2", or blank which gets default

*		 or   "|file1|file2<This is not a valid file"
* ... allow only instrings "file1", "file2" - else an error message

*		 or   "|1>file1|2>file2<You must enter 1 or 2"
* ... allow only instrings "1" or "2" with replacement text for each

* STAR:		 or   "|file1>snoopy|file2>linus|*"
* ... allow any instring with replacement text for "file1" and "file2"
* STAR:		 or   "|file1>snoopy|file2>linus|*>no_match_file"
* ... gives replacement text for "file1" and "file2". Any other
* 	instring gets replaced by "no_match_file"
* Note: "*" effectively matches any instring given 


* table of behavior of ">default" vs "default":
*		no argument given	argument doesn't match options
*		-----------------	------------------------------
*    default  |   return default		error
*   >default  |	  return default 	   return default 

* not yet implemented:
*	$<		- input from stdin
*	$<!		- input from stdin - reprompted if not acceptable

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program

* V420	2/29/96	- extracted from dollar_command for shared use by symbols
*
* 1/03 *kob* - need to accomodate g77 and it's inability to arbitrarily concatenate
* V580 *acm* 11/04 when command contains some error text, and an error condition
*                  is found, issue an error rather than just a note. (bug 1077)
* V500 *acm*  1/05 Fix the above within the ifdef NO_PASSED_CONCAT too
* V698 *acm*  4/16 Fix ticket 2398: report the argument as given, along with the error

        IMPLICIT NONE
* calling argument declarations:
	LOGICAL   err_subst
	INTEGER	  len_out, status
	CHARACTER sym_name*(*), in_str*(*), coaching*(*), out_str*(*)

* internal variable declarations:
* d default string
* e error string
* o options string
* r replacement strings

	LOGICAL errtxt, default, options, opt_replace,
     .		def_replace, opt_failed
	INTEGER	STR_CASE_BLIND_COMPARE, TM_LENSTR1,
     .		iptr, len_in, vax_code, errstat,
     .		dstart, ostart, estart, dend, oend, eend, ostart1, oend1,
     .		rstart, rend, right_star

* Additional temp character buffer needed for concatenating error messages
*    required by g77 *kob* 1/03

	CHARACTER tmp_buff*2048

	include 'ferret.parm'
	include 'errmsg.parm'
	include 'gfdl_vms.parm'
	include 'xrisc.cmn'

* initialize
	status = ferr_ok
	err_subst = .FALSE.
	dstart = 1
	iptr = TM_LENSTR1(coaching)
	errtxt = .FALSE.
	options = .FALSE.
	default = .FALSE.
	opt_replace = .FALSE.
	def_replace = .FALSE.
	opt_failed  = .FALSE.
	IF ( in_str .EQ. ' ' ) THEN
	   len_in = 0
	ELSE
	   len_in = TM_LENSTR1(in_str)
	ENDIF
	len_out = len_in

* error string provided?
	estart = INDEX( coaching, '<' )
	errtxt = estart .GT. 0
	estart = estart + 1     ! right after "<"
	eend = iptr
	IF ( estart .GT. eend ) GOTO 5400
	IF ( errtxt ) THEN
	   dend = estart - 2
	   errstat = ferr_silent
	ELSE
	   dend = iptr
	   errstat = ferr_invalid_command
	ENDIF
	
* options provided?
	ostart = INDEX(coaching, '|')
	options = ostart .GT. 0
	ostart = ostart + dstart    ! right after first "|"
	IF ( errtxt ) THEN
	   oend = estart - 2
	ELSE
	   oend = iptr
	ENDIF

* default text provided?
	IF ( options ) THEN
	   default = ostart .GT. dstart+1
	   dend = ostart - 2
	ELSEIF ( errtxt ) THEN
	   IF ( estart .GT. dstart+1 ) GOTO 5500  ! error text AND default?
	   default = .FALSE.
	ELSE
	   default = .TRUE.
	ENDIF
* ... use the default if the given argument doesn't match any of the options?
	IF ( default .AND. options ) THEN
	   def_replace = coaching(1:1) .EQ. '>'
	   IF ( def_replace ) dstart = 2
	   IF ( dstart .GT. dend ) GOTO 5400
	   default = .NOT.def_replace
	ENDIF

* loop: compare the instring text provided with the allowable options  
	IF ( options .AND. len_out.GT.0 ) THEN
	   ostart1 = ostart
 200	   oend1 = INDEX( coaching(ostart1:oend), '|' )
	   IF ( oend1 .EQ. 0 ) THEN
	      oend1 = oend
	   ELSE
	      oend1 = ostart1 + oend1 - 2
	   ENDIF
* ... was replacement text provided ?
	   rend = oend1
 	   rstart = INDEX( coaching(ostart1:oend1), '>' )
	   opt_replace = rstart .GT. 0
	   IF ( opt_replace ) THEN
	      rstart = ostart1 + rstart
	      oend1 = rstart - 2
	      IF ( rstart .GT. rend ) GOTO 5400	
	   ENDIF
	   IF ( ostart1 .GT. oend1 ) GOTO 5400
* ... if this option is "*" it matches anything   (2/94)
	   IF ( coaching(ostart1:oend1) .EQ. '*' ) GOTO 400
* ... does the GO argument provided match this option ?
	   vax_code = STR_CASE_BLIND_COMPARE( coaching(ostart1:oend1),
     .	                                      in_str(:len_out) )
	   IF ( vax_code .EQ. vms_str_success ) THEN
	      GOTO 400          ! we got a match
	   ELSEIF ( rend+1 .LT. oend ) THEN
	      ostart1 = rend + 2
	      GOTO 200		! text didnt match - check next option
	   ELSE
	      len_out = 0        ! the argument text given was unacceptable
	      opt_replace = .FALSE.
	      opt_failed  = .TRUE.
	   ENDIF
	ENDIF

* if no argument text is provided then use alternative
 400	IF ( opt_replace ) THEN
* ... replace the given argument with the option replacement text provided

* ... 3/95 allow "*" on the right hand side replacement text  e.g. $5%|*>"*"%
	   right_star = INDEX(coaching(rstart:rend),'*')
	   IF ( right_star .GT. 0 ) THEN
* --- start of star substitution
	      IF ( len_in + rend-rstart .GT. 255 ) GOTO 5100
* ... text preceding the "*"
	      IF ( right_star .GT. 1 ) THEN
	         out_str = coaching(rstart:rstart+right_star-1)	         
	         len_out = right_star
	      ELSE
	         len_out = 1
	      ENDIF
* ... replacement text of "*", itself
	      out_str(len_out:) = in_str
	      len_out = len_out + len_in - 1
* ... text following the "*"
	      IF ( rstart+right_star-1 .LT. rend ) out_str(len_out+1:) =
     .			coaching(rstart+right_star:rend)	         
	      len_out = len_in + rend-rstart
* --- end of star substitution
	   ELSE
* ... no star involved -- a simple substitution ...
	      out_str = coaching(rstart:rend)
	      len_out = rend - rstart + 1	
	   ENDIF
	ELSEIF ( len_out.EQ.0
     .    .AND. (.NOT.opt_failed .AND. default
     .           .OR. opt_failed .AND. def_replace) ) THEN 
* ... replace the given argument with the default replacement text provided
	   out_str = coaching(dstart:dend)
	   len_out = dend - dstart + 1
	ELSEIF ( len_out.EQ.0 ) THEN
* ... nothing worked - we have a substitution error
	   GOTO 5200
	ELSE
* ... the input text will do just fine as-is
	   out_str = in_str
	   RETURN
	ENDIF

* success
	RETURN

* error exits
 5000   RETURN
 5100   CALL ERRMSG( ferr_invalid_command, status,
     .       'command argument strings are too long', *5000)
 5200	risc_buff = sym_name
	CALL ERRMSG( errstat, status,
     .       'Valid value needed for '//risc_buff, *5900)
 5400	risc_buff = coaching
	CALL ERRMSG( ferr_syntax, status,
     .       'empty text string provided with ">": '//risc_buff, *5000 )
 5500	risc_buff = coaching
	CALL ERRMSG( ferr_syntax, status,
     .       'both default and error strings given: '//risc_buff, *5000 )

* display the supplied error string or list the allowed options

 5900   risc_buff = coaching(ostart:oend)
	tmp_buff = sym_name
	IF ( errtxt ) THEN
	   CALL ERRMSG( ferr_nomessge, status,
     .       coaching(estart:eend), *5000)
!	   CALL WARN( coaching(estart:eend) )
	ELSEIF ( options ) THEN
	   tmp_buff = sym_name
	   IF (tmp_buff(1:1) .EQ. 'a') tmp_buff(1:1) = 'A'
	   iptr = TM_LENSTR1(tmp_buff)
	   IF (len_in .EQ. 0) THEN
	      CALL WARN
     . 	       (tmp_buff(:iptr)//' was missing or empty.')
           ELSE
	      CALL WARN
     . 	       (tmp_buff(:iptr)//' received was: '
     .					//in_str(:len_in))
	   ENDIF
	   tmp_buff = sym_name
	   CALL WARN
     .	    ('Valid '//tmp_buff(:TM_LENSTR1(tmp_buff))
     .	         //' choices are: '//coaching(ostart:oend))
	ENDIF
	GOTO 5000

	END
