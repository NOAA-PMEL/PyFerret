	SUBROUTINE ALG_BREAK_UP( text,at_start,at_end,natom,
     .                           err_string,status )

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
* type.  Possibly more than one expression separated by commas

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* V200: 11/24/89
* V402: 7/95 - Allow nested [] of the form vname1[G=vname2[D=dset]]
* V495: 7/7/98 - allow variable names in single quotes - add new state S10
*                and new character class 11 (single quote)
*       7/23 - fixed problem missed the first time. brackets: 'X'[stuff]

#ifdef unix
	include	'ferret_cmn/ferret.parm'
	include 'ferret_cmn/errmsg.parm'
	include 'ferret_cmn/rpn.parm'
#else
	INCLUDE	'FERRET_CMN:FERRET.PARM'
	INCLUDE 'FERRET_CMN:ERRMSG.PARM'
	INCLUDE 'FERRET_CMN:RPN.PARM'
#endif

* calling argument declaration:
	INTEGER		at_start(maxatoms), at_end(maxatoms), natom, status
	CHARACTER*(*)	text, err_string

* internal variable declarations:
	LOGICAL		was_operand, atom_open
	INTEGER		ALG_STATE, ALG_CLASS, TM_LENSTR1,
     .			tlen, p, state, last_state, class,
     .			i, next_state(num_class,num_state),
     .			brkt_nest_level

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

* classes of characters
*	C1 - + -
*	C2 - digits
*	C3 - letters _ _
*	C4 - [
*	C5 - ]
*	C6 - * / ^ , (
*	C7 - )
*	C8 - blank tab
*	C9 - E
*	C10- .
*       C11- '
*	C12- all others
*		    ... CHARACTER CLASS ...
*             C1 C2 C3 C4 C5 C6 C7 C8 C9 C10C11 C12
	DATA (next_state(i,1),i=1,num_class)
	1    / 2, 3, 4,-1,-1, 5, 5, 1, 4,  3,10, -2/
	DATA (next_state(i,2),i=1,num_class)
	1    / 7, 3, 7, 7, 7, 7, 7, 7, 7,  3, 7, -2/
	DATA (next_state(i,3),i=1,num_class)
	1    / 7, 3,-1,-1,-1, 7, 7, 7, 8,  3, 7, -2/
	DATA (next_state(i,4),i=1,num_class)
	1    / 7, 4, 4, 6,-1, 7, 7, 7, 4, -1, 7, -2/
	DATA (next_state(i,5),i=1,num_class)
	1    /-9,-9,-9,-9,-9,-9,-9,-9,-9, -9,-9, -2/
	DATA (next_state(i,6),i=1,num_class)
	1    / 6, 6, 6, 9, 5, 6, 6, 6, 6,  6, 6,  6/
	DATA (next_state(i,7),i=1,num_class)
	1    /-9,-9,-9,-9,-9,-9,-9,-9,-9, -9,-9, -2/
	DATA (next_state(i,8),i=1,num_class)
	1    / 8, 8, 7, 7, 7, 7, 7, 7, 7,  7, 7, -2/
	DATA (next_state(i,9),i=1,num_class)
	1    / 9, 9, 9, 9, 9, 9, 9, 9, 9,  9, 9, 9/
	DATA (next_state(i,10),i=1,num_class)
	1    / 10,10,10,10,10,10,10,10,10,10, 10,10/


* initialize
*    "was_operand" flags the character of the previous atom to distinguish the
*     operator "-" in "X-5" from the negative sign "-" in "X+-5"
	tlen = TM_LENSTR1( text )
	natom = 0
	p = 0
	state = 1
	was_operand = .FALSE.
	atom_open = .FALSE.
	brkt_nest_level = 0

* advance to next character
 5	p = p + 1
	IF ( p .GT. tlen ) THEN
	   IF ( natom .EQ. 0 ) GOTO 9999
* ... SUCCESS - DONE
	   status = ferr_ok
	   at_end(natom) = tlen
	   RETURN
	ENDIF
	   
* check next character
 10	last_state = state
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
	IF ( state.EQ.7 .OR. class.EQ.8 ) THEN
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
	   natom = natom + 1
	   at_start(natom) = p
	   atom_open = .TRUE.
	ENDIF

* take appropriate action
*		 S1  S2  S3  S4  S5  S6  S7  S8   S9  S10
	GOTO ( 9999,100,100,100,500,100,700,100, 900,1000 ) state

* continue to next character
 100	GOTO 5

* flag the last character of an atom - begin search for next atom
 500	at_end(natom) = p
	state = 1
	atom_open = .FALSE.
	GOTO 5
	
* flag the last character of an atom one back - begin search for next atom
 700	at_end(natom) = p-1
	last_state = 1
	atom_open = .FALSE.
	GOTO 50

* nested "[]" (mod 7/95)
* this probably violates the theoretical underpinnings of the finite state
* engine but ... what the heck, it's an easy hack
* stay in this state until the matching "]" is found
 900	IF (class .EQ. 4) THEN		! increase nesting
	   brkt_nest_level = brkt_nest_level + 1
	ELSEIF ( class .EQ. 5 ) THEN	! decrease nesting
	   brkt_nest_level = brkt_nest_level - 1
	   IF ( brkt_nest_level .EQ. 0 ) state = 6	! <== leave state 
	ENDIF
	GOTO 5

* single-quoted strings (mod 7/98)
* this probably violates the theoretical underpinnings of the finite state
* engine but ... what the heck, it's an easy hack
* stay in this state until the matching \" is found
* Note: this crude code cannot deal with nested quotations or escaped quotes
*    Also, quoted names inside of square brackets will not be handled here
 1000	IF (class .EQ. 11) THEN		! quote found
	   IF (last_state .EQ. 10) THEN      
	      state = 4                 ! closing quote
*       ... note that we may still be interpreting names like 'X'[stuff]
	   ELSE
	      atom_open = .TRUE.        ! opening quote
	   ENDIF
	ELSE          
	   CONTINUE                     ! inside of quote
	ENDIF
	GOTO 5

* error exits
 5000	status = ferr_syntax
 5005	RETURN
 5100	err_string = 'illegal character: '//text(p:p)
	GOTO 5000 
 5200	err_string = text(MAX(1,p-6):MIN(tlen,p+2))
	GOTO 5000
 5300	err_string = '['//text(MAX(1,p-8):p)//'\\\]'
	status = ferr_expr_too_complex
	GOTO 5005
 5900	err_string = '*** lex state'
	status = ferr_internal
	GOTO 5005
 9999	STOP 'ALG_BREAK_UP'
	END



