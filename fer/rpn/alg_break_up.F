	SUBROUTINE ALG_BREAK_UP( text,at_start,at_end,natom,
     .                               err_string,status )
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
* lexical analyzer to break up algebraic expressions into atoms of known
* type.  Possibly more than one expression separated by commas

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program

* V200: 11/24/89
* V402: 7/95 - Allow nested [] of the form vname1[G=vname2[D=dset]]
* V495: 7/7/98 - allow variable names in single quotes - add new state S10
*                and new character class 11 (single quote)
*       7/23 - fixed problem missed the first time. brackets: 'X'[stuff]
* V500: 12/98 - allow {1,3,5} notation for lists of constants
*	 1/99 - allow strings in double quotes (as fcn args)
* V530: //8/00 *sh* - allow {"string", "string"}
* V600:  3/06 *acm* - fix bug 1395: string 1.0e-4+0*8 tried to inlcude plus sign 
*                     and the 0 following it as part of the exponent. Define a new 
*                     state, in which we are interpreting an exponent and getting 
*                     the exponent value itself. Now this becomes like interpreting 
*                     a constant.
* V600:  5/05  *acm* For attribute control, allow . inside names: 
*                    change next_stat(10,4) to 4 -- 
* V685:  7/13  *acm* Bug 2084. If a string starts with . and is not numeric, make an atom.
*                    It'll generally be an error.
* V693+ 12/14 *sh* allow _QD_ and _SQ_ for quotation alternatives 
*                 Filled in state=13 (interpreting exponent) DATA inits 
*	          Hoping I did so correctly ...
* v694  6/15  *acm* fixes for 2296 -- quotes and curly braces {"{a}"}

        IMPLICIT NONE
	include	'ferret.parm'
	include 'errmsg.parm'
	include 'rpn.parm'

* calling argument declaration:
	INTEGER		at_start(maxatoms), at_end(maxatoms), natom, status
	CHARACTER*(*)	text, err_string

* internal variable declarations:
	LOGICAL		TM_DIGIT_TEST, was_operand, atom_open,
     .			requires_closure(num_state),
     .                      have_exp_digits
	INTEGER		ALG_CLASS, TM_LENSTR1,
     .			tlen, p, state, last_state, last_last_state, class,
     .			i, next_state(num_class,num_state),
     .			brkt_nest_level, pn, 
     .			curly_brace_nest, q_in_curly_brace


* possible states of lexical analysis
*	S1 - looking for start of atom
*	S2 - + or - found following operator (*sh* following name ??)
*	S3 - interpreting constant
*	S4 - interpreting name (variable or function, ...)
*	S5 - found last character of atom
*	S6 - ignoring characters inside []
*	S7 - just missed last character of atom
*	S8 - interpreting exponent
*	S9 - inside of nested []	(mod 7/95)
*       S10- inside of single quote pair
*	S11- ignoring characters inside {}
*	S12- inside of double quote pair
*	S13- interpreting exponent, found exponential value
*	S14- interpreting "_" as a possible quote signal
*	S15- inside of _DQ_ (alt double quote) pair
*	S16- inside of _SQ_ (alt single quote) pair


*	-1 ==> Syntax Error
*	-2 ==> Illegal Character
*	-9 ==> **lex state error

* classes of characters
*	C1 - + -
*	C2 - digits
*	C3 - letters
*	C4 - [
*	C5 - ]
*	C6 - * / ^ , (
*	C7 - )
*	C8 - blank tab
*	C9 - E
*	C10- .
*       C11- '
*	C12- {
*	C13- }
*	C14- "
*	C15- _
*	C16- all others

*	... CHARACTER CLASS: next_state(class, last_state) ...
*              C1 C2 C3 C4 C5 C6 C7 C8 C9 C10C11C12C13C14C15C16
	DATA (next_state(i,1),i=1,num_class)
     .        / 2, 3, 4,-1,-1, 5, 5, 1, 4,  3,10,11,-1,12,14,-2/
	DATA (next_state(i,2),i=1,num_class)
     .        / 7, 3, 7, 7, 7, 7, 7, 7, 7,  3, 7, 7, 7, 7,14,-2/
	DATA (next_state(i,3),i=1,num_class)
     .        / 7, 3, 3,-1,-1, 7, 7, 7, 8,  3, 7,-1,-1, 7,14,-2/
	DATA (next_state(i,4),i=1,num_class)
     .        / 7, 4, 4, 6,-1, 7, 7, 7, 4,  4, 7,-1,-1, 7,14,-2/
	DATA (next_state(i,5),i=1,num_class)
     .        /-9,-9,-9,-9,-9,-9,-9,-9,-9, -9,-9,-9,-9,-9,14,-2/
	DATA (next_state(i,6),i=1,num_class)
     .        / 6, 6, 6, 9, 5, 6, 6, 6, 6,  6, 6, 6, 6, 6,14, 6/
	DATA (next_state(i,7),i=1,num_class)
     .        /-9,-9,-9,-9,-9,-9,-9,-9,-9, -9,-9,-9,-9,-9,14,-2/
	DATA (next_state(i,8),i=1,num_class)
     .        / 8,13, 7, 7, 7, 7, 7, 7, 7,  7, 7, 7, 7, 7,14,-2/
	DATA (next_state(i,9),i=1,num_class)
     .        / 9, 9, 9, 9, 9, 9, 9, 9, 9,  9, 9, 9, 9, 9, 9, 9/
	DATA (next_state(i,10),i=1,num_class)
     .        /10,10,10,10,10,10,10,10,10, 10,10,10,10,10,10,10/
	DATA (next_state(i,11),i=1,num_class)
     .        /11,11,11,11,11,11,11,11,11, 11,11,11, 5,11,11,11/
	DATA (next_state(i,12),i=1,num_class)
     .        /12,12,12,12,12,12,12,12,12, 12,12,12,12,12,12,12/
	DATA (next_state(i,13),i=1,num_class)
     .        /-1,-1,-1,-1,-1,-1,-1,-1,-1, -1,-1,-1,-1,-1,-1,-1/
	DATA (next_state(i,14),i=1,num_class)
     .        /14,14,14,14,14,14,14,14,14, 14,14,14,14,14,14,14/
	DATA (next_state(i,15),i=1,num_class)
     .        /15,15,15,15,15,15,15,15,15, 15,15,15,15,15,14,15/
	DATA (next_state(i,16),i=1,num_class)
     .        /16,16,16,16,16,16,16,16,16, 16,16,16,16,16,14,16/


* which states are not valid on exit (e.g. unclosed braces)
	DATA requires_closure(1 )/.FALSE./ !S1 - looking for start of atom
	DATA requires_closure(2 )/.FALSE./ !S2 - + or - following operator
	DATA requires_closure(3 )/.FALSE./ !S3 - interpreting constant
	DATA requires_closure(4 )/.FALSE./ !S4 - interpreting name
	DATA requires_closure(5 )/.FALSE./ !S5 - found last character of atom
	DATA requires_closure(6 )/.TRUE. / !S6 - ignoring characters inside []
	DATA requires_closure(7 )/.FALSE./ !S7 - just missed last character
	DATA requires_closure(8 )/.FALSE./ !S8 - interpreting exponent
	DATA requires_closure(9 )/.TRUE. / !S9 - inside of nested []
	DATA requires_closure(10)/.TRUE. / !S10- inside of single quote pair
	DATA requires_closure(11)/.TRUE. / !S11- ignoring characters inside {}
	DATA requires_closure(12)/.TRUE. / !S12- inside of double quote pair
	DATA requires_closure(13)/.FALSE./ !S13- interpreting exponent
	DATA requires_closure(14)/.TRUE. / !S14- inside of _QD_ quote pair
	DATA requires_closure(15)/.TRUE. / !S15- inside of _SQ_ quote pair

* initialize
*    "was_operand" flags the character of the previous atom to distinguish the
*     operator "-" in "X-5" from the negative sign "-" in "X+-5"
	tlen = TM_LENSTR1( text )
	natom = 0
	p = 0
	state = 1
	last_state = 1
	was_operand = .FALSE.
	atom_open = .FALSE.
        have_exp_digits = .FALSE.
	brkt_nest_level = 0
	curly_brace_nest = 0
	q_in_curly_brace = 0

* advance to next character
 5	p = p + 1
	IF ( p .GT. tlen ) THEN
	   IF ( natom .EQ. 0 ) GOTO 9999
	   IF (requires_closure(state)) GOTO 5400
* ... SUCCESS - DONE
	   status = ferr_ok
	   at_end(natom) = tlen
	   RETURN
	ENDIF
	   
* check next character
 10	last_last_state = last_state  ! so "_" can morph to a letter
	last_state = state
	class = ALG_CLASS( text(p:p) )

* determine lexical analysis state to determine next action
 50	state = next_state( class, last_state )

	IF ( state .EQ. -2 ) GOTO 5100
	IF ( state .EQ. -1 ) GOTO 5200
	IF ( state .EQ. -9 ) GOTO 5900
	IF ( state .EQ. 1  ) GOTO 5	! just keep looking

*  "+" or "-" at start of atom is operator if last atom was an operand
	IF ( was_operand .AND. last_state.EQ.1 .AND. class.EQ.1 ) state = 5

* will this act like an operator or an operand for the next atom ?
	IF ( state.EQ.7 .OR. class.EQ.8 .OR. class.EQ.13) THEN
* ... last "significant" character hasn't changed
	   CONTINUE
	ELSEIF ( class.EQ.1 .OR. class .EQ.6 ) THEN
	   was_operand = .FALSE.
	ELSE
	   was_operand = .TRUE.
	ENDIF

* begin new atom, possibly.
	IF ( .NOT.atom_open ) THEN
	   IF ( natom .EQ. maxatoms-1 ) GOTO 5300
           IF (class .EQ. 10 ) THEN  ! var.attribute?
              atom_open = .TRUE.
              pn = MIN(p+1,tlen)
	      IF ( TM_DIGIT_TEST(text(p:pn)) ) THEN  ! looks like a number...
                 natom = natom + 1
                 at_start(natom) = p 
              ELSE
* this is a var.attr, so continue as though we are parsing a string
                 class = 3
                 state = 4
                 last_state = 4
* Non-numeric text starting with a dot. Start the atom.
		 IF (p.EQ.1) THEN
                    natom = natom + 1
                    at_start(natom) = p
                    atom_open = .TRUE.
                 ENDIF
              ENDIF
           ELSE
              natom = natom + 1
              at_start(natom) = p
              atom_open = .TRUE.
           ENDIF
	ENDIF

* Curly braces inside quotes inside curly braces
* Hack to work around the case in ticket 2296

	IF (class.EQ.13 .AND. state.EQ.5 .AND. curly_brace_nest.GT.0) state = 11

* take appropriate action
*		S1  S2  S3  S4  S5  S6  S7  S8   S9  S10 S11  S12  S13  S14 S15  S16
c	GOTO (9999,100,100,100,500,100,700,100, 900,1000,1100,1200,1300,1400,100,100)
c     .								state

6      IF (state .EQ. 1) GOTO 9999

* continue to next character

      IF (state .EQ. 2 .OR. 
     .    state .EQ.  3 .OR. 
     .    state .EQ.  4 .OR. 
     .    state .EQ.  6 .OR. 
     .    state .EQ.  8 .OR.
     .    state .EQ. 15 .OR.
     .    state .EQ. 16 ) THEN 
	   GOTO 5

* flag the last character of an atom - begin search for next atom

      ELSEIF (state .EQ. 5) THEN

	 at_end(natom) = p
	 state = 1
	
* reset - previous atoms may have had a quote not a pair of quotes inside curly braces
	curly_brace_nest = 0
	q_in_curly_brace = 0

	atom_open = .FALSE.
	GOTO 5

* flag the last character of an atom one back - begin search for next atom	

      ELSEIF (state .EQ. 7) THEN

	at_end(natom) = p-1
	last_state = 1
	atom_open = .FALSE.
	GOTO 50

* nested "[]" (mod 7/95)
* this probably violates the theoretical underpinnings of the finite state
* engine but ... what the heck, it's an easy hack
* stay in this state until the matching "]" is found

      ELSEIF (state .EQ. 9) THEN

	IF (class .EQ. 4) THEN		! increase nesting
	   brkt_nest_level = brkt_nest_level + 1
	ELSEIF ( class .EQ. 5 ) THEN	! decrease nesting
	   brkt_nest_level = brkt_nest_level - 1
	   IF ( brkt_nest_level .EQ. 0 ) state = 6	! <== leave state 
	ENDIF
	GOTO 5

* single-quoted strings (mod 7/98)
* this probably violates the theoretical underpinnings of the finite state
* engine but ... what the heck, it's an easy hack
* stay in this state until the matching \' is found
* Note: this crude code cannot deal with nested quotations or escaped quotes
      ELSEIF (state .EQ. 10) THEN

*    Also, quoted names inside of square brackets will not be handled here
	IF (class .EQ. 11) THEN		! quote found

	   IF (last_state .EQ. 10) THEN      
	      state = 4                 ! closing quote (?why not GOTO 500?)
*       ... note that we may still be interpreting names like 'X'[stuff]
	   ELSE
	      atom_open = .TRUE.        ! opening quote
	   ENDIF
	ELSE          
	   CONTINUE                     ! inside of quote
	ENDIF
	GOTO 5

* Curly braces inside quotes inside curly braces
* Work around for the case in ticket 2296

      ELSEIF (state .EQ. 11) THEN

	IF (class .EQ. 14 .AND. q_in_curly_brace.EQ.0) ! quote inside curly braces
     .    q_in_curly_brace = q_in_curly_brace + 1

	IF (class.EQ.12 .AND. q_in_curly_brace.GT.0)   ! further curly brace start found
     .    curly_brace_nest = curly_brace_nest + 1

	IF (class.EQ.13 .AND. curly_brace_nest.GT.0) THEN  !  curly brace end found
	   curly_brace_nest = curly_brace_nest - 1
	   state = 11
	ENDIF
	GOTO 5

* double-quoted strings (mod 1/99)
* this probably violates the theoretical underpinnings of the finite state
* engine but ... what the heck, it's an easy hack
* stay in this state until the matching \" is found
* Note: this crude code cannot deal with nested quotations or escaped quotes

      ELSEIF (state .EQ. 12) THEN

	IF (class .EQ. 14) THEN		! quote found
           IF (last_state .EQ. 11 .AND. q_in_curly_brace.GT.0) THEN
	      state = 11
	      q_in_curly_brace = q_in_curly_brace - 1
	   ENDIF
	   IF (last_state .EQ. 12) THEN      
	      state = 5                 ! closing quote found
	      GOTO 6

*       ... note that we may still be interpreting names like 'X'[stuff]
	   ELSE
	      atom_open = .TRUE.        ! opening quote
	   ENDIF
	ELSE          
	   CONTINUE                     ! inside of quote
	ENDIF
	GOTO 5

* digit part of exponent

      ELSEIF (state .EQ. 13) THEN

	state = 3
	GOTO 5

* "_" underscore -- possible indicator of _DQ_ or _SQ_ 

      ELSEIF (state .EQ. 14) THEN

	IF ( p .GT. tlen-3 ) THEN
	   GOTO 1410                ! cant be _DQ_ or _SQ_
	ELSEIF (text(p:p+3) .EQ. p_DQ_quote) THEN
	   IF (atom_open. AND. p.EQ.at_start(natom)) THEN
* ... start of _DQ_ enclosed string
	      p = p+3
	      state = 15
	      GOTO 5
	   ELSEIF (last_state .EQ. 15) THEN
* ... this is a closing quote.  We have been inside of _DQ_ enclosed string
	      p = p+3
	      state = 5			 ! closing quote found 
	      GOTO 6
	   ELSE
* ... the characters "_DQ_", but not start or end of string
	      GOTO 1410
	   ENDIF
	ELSEIF (text(p:p+3) .EQ. p_SQ_quote) THEN
	   IF (atom_open. AND. at_start(natom) .EQ. p) THEN
* ... start of _SQ_ enclosed string
	      p = p+3
	      state = 16
	      GOTO 5
	   ELSEIF (last_state .EQ. 16) THEN
* ... this is a closing quote.  We have been inside of _SQ_ enclosed string
	      p = p+3
	      state = 4 ! we may have _SQ_X_SQ_[stuff]
	      GOTO 5
	   ELSE
* ... the characters "_SQ_", but not start or end of string
	      GOTO 1410
	   ENDIF
	ENDIF

* ...  go back and try again, treating this underscore like any other letter
 1410	   last_state = last_last_state  ! needed?  playing it safe ...
	   class = 3  ! a letter
	   GOTO 50

      ENDIF

* error exits
 5000	status = ferr_syntax
 5005	RETURN
 5100	err_string = 'illegal character: '//text(p:p)
	GOTO 5000 
 5200	err_string = text(MAX(1,p-6):MIN(tlen,p+2))
	GOTO 5000
 5300	err_string = '['//text(MAX(1,p-8):p)//' ]'
	status = ferr_expr_too_complex
	GOTO 5005
 5400	err_string = 'unclosed quotation, parenthesis, or bracket'
	GOTO 5000
 5900	err_string = '*** lex state'
	status = ferr_internal
	GOTO 5005
 9999	STOP 'ALG_BREAK_UP'
	END



