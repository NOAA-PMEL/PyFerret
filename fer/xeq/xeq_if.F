	SUBROUTINE XEQ_IF

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
* sets up the control stack (COMMON/~CONTROL/) to handle single line
* conditional "IF" or the IF-stack (ifstk) to handle multiple line IF
* Single line IF (e.g. IF cond THEN command1 ELSE command2 ENDIF)
* or a multi-line IF -- e.g.
*	IF cond THEN
*	   commands ...
*	ELIF cond2 THEN
*	   commands ...
*	ELSE
*	   commands ...
*	ENDIF

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
*
* V420:  4/96
*        5/96 - allow for multi-line IF clause inside of skipped clause
* V500:  6/99 - bug fix for final quote in IF 1  THEN SAY "I is too big" ENDIF
*       *acm* 3/12 cleanup ifdefs and unnecessary include files
*       *acm* 3/12 6D Ferret (common uses nferdims in tmap_dims.parm)
* V693+ *sh* 12/14 1- init ifstk_skipping to count nested IFs while skipping
*                  2- screen for ELIF used in a single line syntax
*	           3- on error csp drop must occur before call to ERRMSG
*	           4- improved error message for "IF some expression THEN ..."
* V695+ *sh* 2/15 - added support for _DQ_

        IMPLICIT NONE
	include 'tmap_dims.parm'
	include	'ferret.parm'
	include	'command.parm'
	include 'errmsg.parm'
	include 'gfdl_vms.parm'
	include 'xcontext.cmn'
	include 'xcontrol.cmn'
	include 'xprog_state.cmn'

* local variable declarations
	LOGICAL TRUE_OR_FALSE, condition, has_false_clause,
     .		non_executed_if
	INTEGER	STR_CASE_BLIND_COMPARE, STR_UPCASE,
     .		vax_code, if_type, iarg,
     .		true_clause_start, true_clause_end, false_clause_start,
     .		false_clause_end, status, csp1, tmp_stat, iend

* 6/96: Are we really going to process this IF or merely register that
* a (skipped) IF clause has been opened?
* e.g. in this example the first ENDIF is ignorable
*       IF yes THEN
*	  do stuff
*       ELSE
*         IF condition THEN
*           do stuff
*         ENDIF			! <=== must be skipped
*       ENDIF
	IF (if_conditional) THEN				! nested IF?
	   IF (if_doing(ifstk).NE.pif_doing_clause) THEN	! skipped IF?
	      IF ( num_args .LE. 2 ) THEN		! multi-line nested IF?
* ...       bump the IF stack - set up to skip this entire IF clause
	         IF ( ifstk .GE. max_ifstk ) GOTO 5400
	         ifstk = ifstk + 1
	         if_doing(ifstk) = pif_skip_to_endif
	         if_cs(ifstk) = if_cs(ifstk-1)
	         ifstk_skipping = 0     ! counts nested IFs while skipping
	      ENDIF
	      RETURN
	   ENDIF
	ENDIF

* check that something was given to "if" on
	IF ( num_args .LT. 2 ) GOTO 5100

* determine if the condition is true or false
	condition = TRUE_OR_FALSE(cmnd_buff(arg_start(1):arg_end(1)),
     .				  status)
	IF ( status .NE. ferr_ok ) GOTO 5000

* whether single line or multi the second argument should be "THEN"
* a common user error is to use arithmetic expression for the conditional
* without grave accents.  In this case the THEN exists, but is not argument 2
	DO iarg = 2, num_args
	   vax_code = STR_CASE_BLIND_COMPARE(
     .				cmnd_buff(arg_start(iarg):arg_end(iarg)),
     .				'THEN' )
	   IF (vax_code .EQ. vms_str_success) GOTO 100
	ENDDO
* ... no THEN was provided
	GOTO 5200
 100	IF (iarg .GT. 2) GOTO 5210

* determine the nature of the IF clause -- single line or multi
	IF ( num_args .GT. 2 ) THEN

* ***** process a single line IF *****
* Single line logic is handled on the control stack since input for dependent
* clauses comes from internal control
* allocate next spot for the IF control on the control stack
	   CALL STACK_PTR_UP( csp, cs_max, status )
	   IF ( status .NE. ferr_ok ) GOTO 5000
	   csp1 = csp

* save the full command string converted to upper case
	   vax_code = STR_UPCASE( cs_text( csp ), cmnd_buff )


* pattern: *** IF condition THEN true_clause ELSE false_clause (ENDIF) ***
	   true_clause_start = 3
* try to find "ELSE"
	   DO iarg = 3, num_args
	      IF (cs_text(csp)(arg_start(iarg):arg_end(iarg))
     .						.EQ. 'ELSE' ) THEN
		 has_false_clause   = .TRUE.
	         true_clause_end    = iarg - 1
	         false_clause_start = iarg + 1		         
	         GOTO 200
	      ENDIF
	   ENDDO
	   has_false_clause = .FALSE.

* check for ELIF used in a single line syntax -- trap as a syntax error
* (just out of laziness.  In 15 years no one has asked for this.)
 200	   DO iarg = 3, num_args
	      IF (cs_text(csp)(arg_start(iarg):arg_end(iarg))
     .						.EQ. 'ELIF' ) THEN
	         CALL STACK_PTR_DN( csp, 0, tmp_stat )
	         GOTO 5250
	      ENDIF
	   ENDDO

* is there an implicit or an explicit "ENDIF"?
	   DO 300 iarg = num_args, 3, -1
	      IF (cs_text(csp)(arg_start(iarg):arg_end(iarg))
     .						.EQ. 'ENDIF' 
     .	     .OR. cs_text(csp)(arg_start(iarg):arg_end(iarg))
     .						.EQ. 'ENDI'   ) THEN
	         IF (iarg .NE. num_args ) THEN
	            CALL STACK_PTR_DN( csp, 0, tmp_stat )
	            GOTO 5300
	         ENDIF
	         IF (has_false_clause) THEN
	            false_clause_end = num_args - 1
	         ELSE
	            true_clause_end  = num_args - 1
	         ENDIF
	         GOTO 400
	      ENDIF
 300	   CONTINUE
* ... no - there is no terminating "ENDIF"	   
	   IF (has_false_clause) THEN
	      false_clause_end = num_args
	   ELSE
	      true_clause_end  = num_args
	   ENDIF

* prepare the clause to be executed
 400	   IF ( condition ) THEN
* ... 6/99 bug fix (hack) -- final quotation mark missed in
*	IF 1 THEN SAY "I is too big" ENDIF
*     because arg_start/arg_end are pre-trimmed wrt enclosing quotes
	      iend =  arg_end(true_clause_end)
	      IF     (iend+1 .LE. len_cmnd) THEN
	         IF (cmnd_buff(iend+1:iend+1).EQ.'"') iend = iend + 1
	         IF (iend+4 .LE. len_cmnd) THEN
	            IF (cmnd_buff(iend+1:iend+4).EQ.'_DQ_') iend = iend + 4
	         ENDIF
	      ENDIF
	      cs_text(csp) = cmnd_buff(arg_start(true_clause_start):
     .				       iend)
	   ELSEIF ( has_false_clause ) THEN
* ... 6/99 bug fix (hack) -- final quotation mark missed in
*	IF 1 THEN SAY "I is too big" ENDIF
*     because arg_start/arg_end are pre-trimmed wrt enclosing quotes
	      iend =  arg_end(false_clause_end)
	      IF    (iend+1 .LE. len_cmnd) THEN
	         IF (cmnd_buff(iend+1:iend+1).EQ.'"'   ) iend = iend + 1
	         IF (iend+4 .LE. len_cmnd) THEN
	            IF (cmnd_buff(iend+1:iend+4).EQ.'_DQ_') iend = iend + 4
	         ENDIF
	      ENDIF
	      cs_text(csp) = cmnd_buff(arg_start(false_clause_start):
     .				       iend)
	   ELSE
* ... false condition and no ELSE clause
	      CALL STACK_PTR_DN( csp, 0, tmp_stat )
	      RETURN
	   ENDIF

* tell 'em who owns this level
	   cs_cmnd_num( csp ) = cmnd_if

* only for a single command, though
	   cs_phase(csp) = 0

* tell 'em we've taken over and start workin'
	   cs_in_control = .TRUE.
	   input_source  = input_from_cs
	ELSE

* ***** process a multi-line IF *****
* bump the IF stack
	   IF ( ifstk .GE. max_ifstk ) GOTO 5400
	   ifstk = ifstk + 1

* say which IF clause to execute
	   IF ( condition ) THEN
	      if_doing(ifstk) = pif_doing_clause	! IF TRUE THEN xxx
	   ELSE
	      if_doing(ifstk) = pif_skip_to_clause	! IF FALSE ... ELSE xxx
	   ENDIF

* save where we are in the control stack for error checking
	   if_cs(ifstk) = csp
	   if_conditional = .TRUE.

	ENDIF

	RETURN
	
* error exit
* ... control stack pointer, CSP, corrected in GFD_ERROR_HANDLER
 5000	RETURN
 5100	CALL ERRMSG( ferr_syntax, status, 'IF what ?', *5000)
 5200	CALL ERRMSG( ferr_syntax, status, 'IF without THEN'//
     .			pCR//cmnd_buff(:len_cmnd), *5000 )
 5210	CALL ERRMSG( ferr_syntax, status,
     .	'"IF condition THEN" requires a simple 0 or 1 (or True or False)'//
     .			pCR//cmnd_buff(:len_cmnd), *5000 )
 5250	CALL ERRMSG( ferr_syntax, status,
     .			'ELIF cannot be used in single line IF'//
     .			pCR//cmnd_buff(:len_cmnd), *5000 )
 5300	CALL ERRMSG( ferr_syntax, status, 'Trash after ENDIF'//
     .			pCR//cmnd_buff(:len_cmnd), *5000 )
 5400	CALL ERRMSG( ferr_invalid_command, status, 'IFs nested too deep',
     .							*5000 )
	END
