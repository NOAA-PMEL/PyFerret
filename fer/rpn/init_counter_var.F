 	SUBROUTINE INIT_COUNTER_VAR ( name, text, txstart, txend, 
     .                 xlo, xhi, del, cvar, status )

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
* Compile a variable structure into a counter variable.
* Expression is of the form LO:HI or LO:HI:DEL
* 1/21/04 *acm* from init_user_var.F
* V554 1/04 *acm* - additions for /RANGE repeat loop control

* V554 3/04 *acm* - fix error messages

	include 'ferret.parm'	
	include 'errmsg.parm'	
	include 'rpn.parm'
	include 'xvariables.cmn'
	include 'xrisc_buff.cmn'
        include 'xalgebra.cmn'
	include	'xcontext.cmn'
	include 'xfr_grid.cmn'

* calling argument declarations:
	CHARACTER*(*)	name, text
	INTEGER		txstart, txend, xlo, xhi, del,  cvar, status

* internal variable declarations:

        INTEGER		TM_LENSTR, ATOM_POS, STR_UPCASE, iflag, 
     .                  slen, colon_pos, colon2_pos, lo_start, lo_end, 
     .                  hi_start, hi_end, del_start, del_end, dum
        LOGICAL         colon, colon2, fmat, logical_dummy, by_ss
        REAL*8          answer		
        CHARACTER*50    err_msg

* find an unused counter var slot for the variable
	DO 100 cvar = 1, num_counter_vars
	   IF ( alg_cvar( cvar ) .EQ. unspecified_name4 ) GOTO 1000
 100	CONTINUE
	GOTO 5100

* initialize the counter variable, "cvar"
* algebraic expression too long ?

1000	IF ( txend-txstart .GE. cvar_text_len ) CALL ERRMSG
     .  ( ferr_line_too_long, status, text(txend-14:txend), *5000 )

* copy name code and text into slot and change as needed to upper case
* Should not have any text: these variables are LO:HI:DEL


* copy name code and text into slot and change as needed to upper case
	iflag = STR_UPCASE( countervar_name( cvar ), name )
        iflag = STR_UPCASE( alg_cvar( cvar ), text(txstart:txend) )

* get the X range

        slen = TM_LENSTR(text)
        IF (slen .LT. 3) CALL ERRMSG
     .	 ( ferr_invalid_command, status, 
     .     'REPEAT/RANGE without range ?', *5000 )

* ... ":"
        lo_start = 1
        colon = .FALSE.
	colon_pos = ATOM_POS( text, ':' )
	IF ( colon_pos .EQ. atom_illegal_pos ) GOTO 5200
	colon  = colon_pos .NE. atom_not_found
	IF ( colon ) THEN
	   lo_end	 = colon_pos - 1
	ELSE
	   colon2	 = .FALSE.
	   lo_end	 = slen
	ENDIF
        IF (.NOT. colon) GOTO 5200

*  ... second ":"

        colon2 = .FALSE.
	IF ( colon ) THEN
	   colon2_pos = ATOM_POS(text( colon_pos+1:slen ), ':' )
	   IF ( colon2_pos .EQ. atom_illegal_pos
     .	   .OR. colon2_pos .EQ. 1 ) GOTO 5200
	   colon2 = colon2_pos .NE. atom_not_found
	   IF ( colon2 ) THEN
	      colon2_pos = colon2_pos + colon_pos
	      hi_end     = colon2_pos - 1
	      IF (INDEX(text(colon2_pos+1:slen),':').GT.0) GOTO 5200
	   ELSE
	      hi_end     = slen
	   ENDIF
	   hi_start	 = colon_pos + 1
	   del_start	 = colon2_pos + 1
           del_end       = slen
	ENDIF

* Call translate_limit to check syntax and set ss limits

        by_ss = .TRUE.
        dum = 1

	CALL TRANSLATE_LIMIT
     .		      (  text(lo_start:lo_end), x_dim,
     .			by_ss, answer, fmat, dum, status )
        xlo = INT(answer)

	CALL TRANSLATE_LIMIT
     .		      (  text(hi_start:hi_end), x_dim,
     .			by_ss, answer, fmat, dum, status )
        xhi = INT(answer)

	del = 1
        IF (colon2) THEN
           CALL TRANSLATE_LIMIT
     .		      ( text(del_start:del_end), x_dim,
     .			by_ss, answer, logical_dummy, dum, status )
           del = INT(answer)
        ENDIF

* successful completion
	status = ferr_ok
	RETURN	

* error exit
 5000	RETURN

 5100	WRITE (err_msg, 4000) num_counter_vars
        CALL ERRMSG( ferr_prog_limit, status,
     .		'too many counter variables'//pCR//
     .		err_msg, *5000 ) 
 4000   FORMAT ('maximum number is ', i3)
 5200   CALL ERRMSG( ferr_syntax, status, 
     .	          risc_buff(:slen)//pCR//
     .'":" used incorrectly. A correct example is "RANGE=1:100:10".',
     .	          *5000)
        END
