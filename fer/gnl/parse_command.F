	SUBROUTINE PARSE_COMMAND (  cmnd_buff,
     .				   max_words, max_quals,
     .				   cmnd_len, cmnd_num, subcmnd_num,
     .				   num_quals, qualifier_list,
     .				   qual_start, qual_end,
     .				   num_args, arg_start, arg_end,
     .				   err_lun, reverify, arg1_quoted, status )


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
* decode a command line into command, subcommand, qualifiers and arguments

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 - 3/24/86
* revision 0.1 - 3/4/87  - changed /star to /@
* revision 0.2 - 5/13/87 - errors processed via ERRMSG
* revision 0.3 - 3/23/88 - bug in quotation mark processing (/T="17-AUG-1982")
* revision 0.4 - 7/28/88 - treat "[" similar to '"' -  allowing commmas as in
*			   TEMP[D=1,G=U] 
* V200:  12/8/89 - correctly handle tabs and spaces preceding ! comment
* 	  2/8/90 - fill in command alias, if any
* Unix/RISC port - 1/91 - cant use "/star" in documentation
*                 10/91 - added TM_LEGAL_NAME for improved syntax checks
*!! removed - doesn't work !! - checked for buffer overflow
* V230:  6/26/92 - commands, subcommands, and qualifiers 4-->8 characters
*         8/4/92 - bug fix:  message "yy" ,xx --> error: unpaired quotation ...
*                          - formalized handling of quotations (see comments)
* V300: 2/3/93 - check for semicolons and parens in command groups
*      4/20/93 - "go back" and start again after alias substitution
*	          allowing command groups or other aliases in an alias
*      4/21/93 - replace "$n" arguments in GO command lines
*	       - ignore semicolons inside of quotation marks
*       5/6/93 - lots of little stuff for dollar argument subst incl "reverify"
* V301:2/25/94 - bug fix: num_args wrong if "/"'s follow args inside other args
* V314:8/18/94 *kob* IBM port - MATCH4 should be defined as logical
* V400: 3/20/95 - call REPL_EXPRNS to evaluated any grave accent-enclosed
*		  expressions as a part of command parsing.
*		  Added argument "memory".
*	 6/2/95  - substitute (PLOT+) symbols prior to any other parsing
* V420:10/26/95 - Avoid paren removal when line begins with "($name)"
*		- do not allow ":" to end a word (e.g. "/xlimits=12: 16")
*		- make sure "()" is retained as a single word
*       11/3/95 - allow \ to escap !,`,/ and ;	! removed quotes *kob* 12/96
*		- improve parsing to accomodate nested [] and ()
*	4/25/96 - for IF and ELIF commands only evaluate grave accents in the
*		  up to the first command argument (others will be evaluated
*		  when the sub-clause of the IF is evaluated.
* V430: 6/11/96 - Ignore "yes?" at start of command
* Linux Port - 12/96 *kob
*	     - had to close open quotes in above comment line - linux f77
*	       was freaking out because of it...
*	     - added ifdef check for double slash because f90/linux didn't
*	       need two of them together for escapes
* v4.91 12/97 *kob* - added parameter decl. for local_max_arg_list to use as
*                     number of elements for itsa_qualifer logical array
* V500 *sh* 4/99 - Increased nsubst_passes to 100 for detecting recursion
*		- so large numbers of grave accents can be in a single line
* V530 *sh* 10/00 - modify so that symbols inside of REPEAT loops get
*	translated inside of the loop (keep copy of input in risc_buff)
*	*sh* 3/01 - added arg1_quoted to indicate if the first arg in the list
*		has enclosing quotes - and avoid calling PARSE_COMMA_LIST
* V531 *sh* 4/01 - added continuation lines
* V533 *sh* 7/01 - allow both single and double quotes
* v541 *acm* 3/02 -bug fix; nested repeat loops, parsing parentheses in the command.
*                  Make another copy of cmnd_buff AFTER removal of enclosing parens
* 2/03 *kob* - g77 port - g77 won't allow intrinsic functions in PARAMETER
*                         statements.  use an octal constant instead
* V552 *acm* 6/03  The v541 fix noted above introduced bug in symbol substitution
*                  within REPEATS: yes? def sym a 0; rep/i=1:3 (def sym a `i`; say ($a))
*                  Fix in saving cmnd_copy.  Bugs 376 and 558 fixed.
* V600 *acm* 3/06  fixing bugs 439&1390, pass cmnd_num to repl_exprns so we can decide
*                  if the command is an action command, and so whether to apply command
*                  context to grave-accent expressions.
* v603 *acm* 5/07  fix bug 919; null symbol entered as the command produced an error;
*                  the fix is to return at statement 500 if word_num = 0
* v612 *acm* 7/08  Fix bug 1583: precision of grave-accent expressions upped from 5 to 7.
* v62  *acm*11/08  Fix bug 1608: semicolon inside double quotes.
* V68  *acm* 3/12  For double-precision Ferret, let the precision of grave-accent 
*                  expressions increase to 14.
* V68  *acm* 3/12 6D Ferret (common uses nferdims in tmap_dims.parm)
* V68  *acm* 7/12 return 16 digits by default for grave-accent expressions
* V685+ *acm* 12/13  bounds checking: dont refer to parts of the string outside its bounds.
* V693 *sh* 12/14 moved arg_start/end inits into PARSE_COMMAND (housekeeping)
*	          Changes in GET_FERRET_CMND simplify this routine:
*                   1) comments are never passed into this routine
*                   2) nor "yes? " or "...? "
*                   3) continuation lines are assembled before getting here
*                 Also added support for _DQ_ style of quotation marks
*      *acm*  2/15 Prevent run-time errors in string length for _DQ_ handling
* V695 *acm*  3/15 Ticket 2245: if no arguments on a REPEAT command, exit gracefully.
* V698 *sh*  12/15 avoid the need to surround filenames with slashes in quotes
*	           Applies to SET DATA, GO, SAY, DEFINE SYM and DEFINE DATA/AGG
*		   The first non-qualifier following a qualifier terminates
*                  parsing of qualifiers, supporting
*		       DEFINE DATA/AGG name = /path/filename, 
*                  Specific hacks for SET DATA (& DEFINE SYM) and GO
* V697 12/15 *acm* Ticket 2337 Increase length of cmnd buffer to 20480.
*                  change to echo back only the first part of the command on error.
* V697  1/16 *acm* ticket 2356: Allow multiple slashes in command-line between qualifiers.
* V697  2/16 *acm* DO NOT Increase length of cmnd buffer to 20480
*                  but keep the other changes at that time, echo back only the 
*                  start of command on an error
* V697  3/16 Ticket 2376: parsing backslash escapes within repeat loops.
* V697  3/16  *acm* ticket 2356 scale back change for multiple slash at end of command
* V73   7/17  *acm* ticket 2551 Report an error for unmatched parentheses in a REPEAT loop.
* V75   4/19  *acm* Issue   660 Evaluate a grave-accent expression that appears right after
*                               a slash in the position of a qualifier
* V76  11/19  *acm* Issue  1963  Check for too-short abbreviations of qualifiers, e.g.
*                  /T= for /TITLE is interpreted as a possible time qual in 
*                  evaluating grave-accent expressions.
*                  /TITLE= is the only known case of this potential conflict, could expand
*                  the too-shortness from 1 to MIN(
* V76  12/19 *acm* ticket 1963, Replace one-letter qualifiers with what is
*                  being used (so /T for /TITLE isn't misinterpreted elsewhere).

* command, subcommand and qualifiers will be identified by number in
*	 COMMON/XCOMMAND.  Arguments will be returned as positions in
*	 the command buffer
* command syntax:
*	COMMAND_NAME [/QUALx/QUALy...] [SUBCOMMAND] [ARGUMENTa ARGUMENTb ...]
*
*	qualifiers may be anywhere preceeding the first argument
*	arguments are separated by spaces or tabs unless = or comma is present
*	SUBCOMMAND is optional
*	quotes may surround any group of characters that is to be regarded as a
*	single argument
 
* note: arrays arg_start and arg_end must be large enough to accomodate the
*	sum of command, subcommand, qualifiers and args since they are used
*	for working storage

* note: quotations (8/92): quotation mark pairs will be interpreted either
*       as surrounding quotes or as embedded quotes depending on whether the
*       quote marks are both the starting and ending characters of the "word".
*       For surrounding quotes the word start/end will exclude the quotes.

        IMPLICIT NONE
* declare calling arguments of SUBROUTINE
	LOGICAL		reverify, arg1_quoted
	CHARACTER*(*)	cmnd_buff
	INTEGER		max_words,max_quals,cmnd_len,cmnd_num,subcmnd_num,
     .			num_quals,qualifier_list(max_quals),
     .			qual_start(max_quals), qual_end(max_quals),
     .			num_args,arg_start(max_words),arg_end(max_words),
     .			err_lun,status

	include	'tmap_dims.parm'
	include	'errmsg.parm'
	include	'ferret.parm'
	include	'command.parm'
	include 'xcommand.cmn'
	include 'xcontrol.cmn'
	include 'xvariables.cmn'	! for num_uvars_in_cmnd
        include 'xrisc.cmn'

* local variable declarations:
*
**kob*12/97!!! NOTE:local_max_arg_list must match max_arg_list in xprog_state
        INTEGER local_max_arg_list
        PARAMETER (local_max_arg_list = 128)

* modified def of MATCH4 from INTEGER to LOGICAL *kob* IBM port 8/94
	LOGICAL	TM_LEGAL_NAME, known_qualifier, atsin_qualifier,
     .          doub_quote, DQ_quote, a_qual_found, look_for_quals,
     .		surround_quote, surround_DQ_, end_DQ_, subst, first_paren,
     .          itsa_qualifier(local_max_arg_list), 
     .          MATCH4, escape, apply_cx, ptrslash, i3_ok,
     .		its_set_data, its_define_sym, its_go, its_say,
     .		has_lead_white_space, found_white_space, expnd_qual
	INTEGER TM_LENSTR, STR_UPCASE
	INTEGER	buff_len, bang_position, ptr, quote_end,
     .		word_num, look_ahead, look_back, isub_word, csgo,
     .		isubcmnd_ptr, iword, last_qual, iqual_word, iterm,
     .		iqual, iqual_ptr, iarg, i, ii, nparen, nsubst_passes,
     .		grave_start, grave_end, in, out, nest,
     .		paren_level,len_test,len_mchars, current_words,
     .		last_recognized_word, white_aftr_cmnd, white_aftr_subcmnd,
     .		ngone, ir, ic
	CHARACTER c1*1
	CHARACTER url_buff*2048

* local parameter declarations:
	INTEGER		grave_digits
	CHARACTER*1	tab
#ifdef NO_INTRINSIC_IN_PARAMETER
	PARAMETER     ( tab = o'011' )
#else
	PARAMETER     ( tab = CHAR(9) )
#endif
        PARAMETER ( grave_digits = 16 )

* intialize
	grave_start	= 1
	nsubst_passes   = 0
	reverify	= .FALSE.
	arg1_quoted	= .FALSE.
	status 		= ferr_ok
	buff_len	= LEN ( cmnd_buff )
 5	ptr		= 1		! current position in cmnd_buff
	word_num	= 0		! current "word" in cmnd_buff
	num_quals	= 0
	num_args	= 0
	cmnd_num 	= 0		! in case of comment line
	cmnd_buff (buff_len:buff_len) = ' '	! always end with blank
	a_qual_found	= .FALSE.
	look_for_quals	= .TRUE. 
	white_aftr_cmnd = buff_len
	white_aftr_subcmnd = buff_len

* inits moved into this code from get_ferret_command (housekeeping)
        DO i = 1, local_max_arg_list
           arg_start(i) = 0
           arg_end(i) = 0
	ENDDO

* full length of text
	cmnd_len = TM_LENSTR (cmnd_buff)

* ignore trailing slashes in parsing (further fix for ticket 2356)

	IF (cmnd_len .GE. 2) THEN
	DO WHILE (cmnd_buff(cmnd_len-1:cmnd_len) .EQ. '//' .AND. cmnd_len.GT.2)
	   cmnd_len = cmnd_len-2
	   IF (cmnd_len .LE. 1) GOTO 6
	ENDDO
	ENDIF
 6	CONTINUE

* save a copy of the command with untranslated symbols for use by REPEAT
* and for building the complete command when continuation lines are used.

	cmnd_copy = cmnd_buff
	len_cmnd_copy = cmnd_len

* * * * * * * * *
* remove parens enclosing entire command or command group
* modified 10/95 to protect "($name) at line start
 20     IF (cmnd_len .GT. 1) THEN
 	IF (cmnd_len .GT. cmnd_copy_len) GOTO 5005

	   IF( cmnd_buff(1:1) .EQ. '(' 
     .	 .AND. cmnd_buff(2:2) .NE. '$'
     .   .AND. cmnd_buff(cmnd_len:cmnd_len) .EQ. ')' ) THEN
              cmnd_buff(1:1) = ' '
              cmnd_buff(cmnd_len:cmnd_len) = ' '
              CALL LEFT_JUST( cmnd_buff(1:cmnd_len),
     .                        cmnd_buff(1:cmnd_len), cmnd_len )
           ENDIF
           DO WHILE (cmnd_buff(cmnd_len:cmnd_len) .EQ. '/' .AND. cmnd_len.GT.1)
	      cmnd_len = cmnd_len-1
	   ENDDO
        ENDIF

* Again save a copy of the command with untranslated symbols for use by REPEAT
* this time without enclosing parens.

        IF (len_cmnd_copy .GT. 1) THEN
	   IF( cmnd_copy(1:1) .EQ. '(' 
     .	 .AND. cmnd_copy(2:2) .NE. '$'
     .   .AND. cmnd_copy(len_cmnd_copy:len_cmnd_copy) .EQ. ')' ) THEN
              cmnd_copy(1:1) = ' '
              cmnd_copy(len_cmnd_copy:len_cmnd_copy) = ' '
              CALL LEFT_JUST( cmnd_copy(1:len_cmnd_copy),
     .                        cmnd_copy(1:len_cmnd_copy), len_cmnd_copy)
           ENDIF
           DO WHILE (cmnd_copy(len_cmnd_copy:len_cmnd_copy) .EQ. '/' 
     .               .AND. len_cmnd_copy.GT.1)
	      len_cmnd_copy = len_cmnd_copy-1
	   ENDDO
        ENDIF


* * * * * * * * *
* process semicolon-separated command group  (2/93)
* "REPEAT/L=lo:hi (command1;command2)"  is a single command (at this stage) but
* "command1;command2" and "(command1;command2)" are command groups
        IF ( INDEX(cmnd_buff(1:cmnd_len),';') .GT. 0 ) THEN
* ... it is a command group if there is a ";" NOT enclosed in parens
           nparen = 0
           surround_quote = .FALSE.
	   surround_DQ_   = .FALSE.
           DO 60 i = 1, cmnd_len
	      i3_ok = i+3 .LT. cmnd_len
	      c1 = cmnd_buff(i:i)
              IF ( c1 .EQ. '(' ) THEN
                 nparen = nparen + 1
              ELSEIF ( c1 .EQ. ')' ) THEN
                 nparen = nparen - 1
              ELSEIF ( c1 .EQ. '"' ) THEN
                 IF (.NOT.surround_DQ_)
     .			surround_quote = .NOT.surround_quote
              ELSEIF ( i3_ok
     .          .AND. cmnd_buff(i:i+3) .EQ. p_DQ_quote ) THEN
                 IF (.NOT.surround_quote)
     .			surround_DQ_   = .NOT.surround_DQ_
              ELSEIF ( c1 .EQ. ';' ) THEN
	         escape = i.GT.1
#ifdef NO_DOUBLE_ESCAPE_SLASH
	         IF (escape) escape = cmnd_buff(i-1:i) .EQ. '\;'
#else
	         IF (escape) escape = cmnd_buff(i-1:i) .EQ. '\\;'
#endif
                 IF ( nparen.EQ.0
     .		.AND. .NOT.escape
     .		.AND. .NOT.surround_DQ_
     .		.AND. .NOT.surround_quote ) THEN
                    cmnd_num = cmnd_semicolon
                    GOTO 1000   ! successful exit
                 ENDIF
              ENDIF
 60        CONTINUE
        ENDIF

* * * * * * * * *
* substitute GO command arguments, if any
* and return to re-process command if substitution occurs (4/93)
* Note: could have been done as a command stack operation ... maybe cleaner 
	nsubst_passes = nsubst_passes + 1 ! guard against recursion (4/93)
	IF ( nsubst_passes .GT. 100 ) GOTO 5060
	IF ( cs_in_control ) THEN
	   IF ( INDEX(cmnd_buff(:cmnd_len),'$') .GT. 0 ) THEN
* ... search backwards for a GO command in charge of the command stack
	      DO 92 csgo = csp, 1, -1
	         IF ( cs_cmnd_num(csgo) .EQ. cmnd_go ) THEN
	            CALL DOLLAR_COMMAND( cmnd_buff, cmnd_len,
     .			  	         cs_text(csgo), subst, status )
	            IF ( status .NE. ferr_ok ) GOTO 5000	
		    IF ( subst ) THEN
			reverify = .TRUE.      ! informational echo needed
			GOTO 20   ! back for recursive substitutions
		   ENDIF
	         ENDIF
 92	      CONTINUE
	   ENDIF
	ENDIF

* * * * * * * * *
* substitute symbols (PLOT+ symbols expressed as "($symname)" ), if any (6/95)
* and return to re-process command if substitution occurs
        CALL SYMBOL_COMMAND( cmnd_buff, cmnd_len, subst, status )
	IF (cmnd_len .GT. cmnd_copy_len) GOTO 5005
	IF ( status .NE. ferr_ok ) GOTO 5000
	IF ( subst ) THEN
	   reverify = .TRUE.      ! informational echo needed
	   GOTO 20   ! back for more substitutions
	ENDIF

* * * * * * * * *
* substitute command alias, if any
* and return to re-process command if substitution occurs (4/93)
* Note: could have been done as a command stack operation ... maybe cleaner 
         CALL ALIAS_COMMAND( cmnd_buff, cmnd_len, *20 )

* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
* parser: break up command into "words" storing start and end locations
* "words" are separated by blank, tab, "/" or " but not by comma or =
* 10/95 - words are also not separated by ":"
 100	CONTINUE

* skip blanks preceeding word
	has_lead_white_space = .FALSE.
	DO 110 ptr = ptr, cmnd_len
	   IF (      cmnd_buff(ptr:ptr) .NE. ' '
     .	       .AND. cmnd_buff(ptr:ptr) .NE. tab ) GOTO 200
	   has_lead_white_space = .TRUE.
 110	CONTINUE
	GOTO 500		! end of command buffer - no more words

* we have another word in the buffer.  Is this word a qualifier ?
* 12/15 - cease hunt for qualifiers when a non-qual follows qualifiers
 200	IF ( word_num .EQ. max_words ) GOTO 500
	word_num 	= word_num + 1
	IF  ( has_lead_white_space ) THEN
	   IF (ptr .LT. white_aftr_cmnd 
     .   .AND. word_num .GT. 1  ) white_aftr_cmnd = ptr-1 ! GO hack
	   IF (ptr .LT. white_aftr_subcmnd 
     .   .AND. word_num .GT. 2  ) white_aftr_subcmnd = ptr-1 ! SET DATA hack
	ENDIF
	IF ( look_for_quals .AND. cmnd_buff(ptr:ptr).EQ.'/' ) THEN
	   itsa_qualifier(word_num) = .TRUE.
	   a_qual_found = .TRUE.
	   ptr = ptr + 1		! jump over "/"

* Skip over multiple / in a row. (ticket 2356)
* (Not allowing for all possible combinations of spaces and slashes)

	   DO WHILE (cmnd_buff(ptr:ptr) .EQ. '/' .AND. ptr.LT.cmnd_len)
	      ptr = ptr+1
	   ENDDO

*	skip blanks between "/" and text of word
	   DO 210 ptr = ptr, cmnd_len
		IF (      cmnd_buff(ptr:ptr) .EQ. '/' ) GOTO 5050
		IF (      cmnd_buff(ptr:ptr) .NE. ' '
     .		    .AND. cmnd_buff(ptr:ptr) .NE. tab ) GOTO 300
 210	   CONTINUE
	   word_num = word_num - 1
	   GOTO 500			! ignore "/" at end of buffer
	ELSE
	   itsa_qualifier(word_num) = .FALSE.
	   IF (a_qual_found) look_for_quals = .FALSE.
	ENDIF

* we have found the start of a word - initialize variables
 300    arg_start (word_num) = ptr
        surround_quote = .FALSE.
	surround_DQ_   = .FALSE.

* start searching for the end of the word: blank or tab
* include as a block (i.e. skip over) quoted text 
 400	CONTINUE  ! ============= TOP OF LOOP

* Define a logical to stay away from cmnd_buff(ptr-1:ptr-1) when ptr = 1
           ptrslash = .FALSE.
	   IF (ptr .GT. 1) THEN
#ifdef NO_DOUBLE_ESCAPE_SLASH
	      IF ( cmnd_buff(ptr-1:ptr-1) .EQ. '\' ) ptrslash = .TRUE.
*                            '  closing single quote to make emacs happy
#else
	      IF ( cmnd_buff(ptr-1:ptr-1) .EQ. '\\' ) ptrslash = .TRUE.
#endif
           ENDIF
	   IF (ptrslash) THEN
	      GOTO 410
           ELSEIF ( cmnd_buff(ptr:ptr) .EQ. '"'  ) THEN
* ... double quote - skip past entire quotated string
              surround_quote = ptr .EQ. arg_start(word_num)
              DO i = ptr+1, cmnd_len
#ifdef NO_DOUBLE_ESCAPE_SLASH
                IF ( cmnd_buff(i-1:i-1) .EQ. '\'  ) CYCLE  ! ' for emacs
#else
                IF ( cmnd_buff(i-1:i-1) .EQ. '\\' ) CYCLE
#endif
                IF (cmnd_buff(i:i).EQ.'"' ) THEN
	          ptr = i
                  quote_end = ptr
                  GOTO 410
                ENDIF
	      ENDDO
              GOTO 5010
           ELSEIF ( cmnd_buff(ptr:ptr) .EQ. "'"  ) THEN
* ... single quote - skip past entire quotated string
              DO ptr = ptr+1, cmnd_len
#ifdef NO_DOUBLE_ESCAPE_SLASH
                IF ( cmnd_buff(ptr-1:ptr-1) .EQ. '\'  ) CYCLE  ! ' for emacs
#else
                IF ( cmnd_buff(ptr-1:ptr-1) .EQ. '\\' ) CYCLE
#endif
                IF (cmnd_buff(ptr:ptr).EQ."'" ) GOTO 410
	      ENDDO
              GOTO 5010
           ELSEIF ( cmnd_buff(ptr:ptr) .EQ. '_'  
     .	      .AND. ptr+3 .LE. cmnd_len           ) THEN
	      IF (cmnd_buff(ptr:ptr+3) .EQ. p_DQ_quote) THEN
* ... _DQ_ quote - skip past entire quotated string
                 surround_DQ_ = ptr .EQ. arg_start(word_num)
                 DO i = ptr+4, cmnd_len
                    IF (cmnd_buff(ptr:ptr).EQ.'_' 
     .            .AND. i+3 .LE. cmnd_len           ) THEN
	               IF (cmnd_buff(i:i+3) .EQ. p_DQ_quote) THEN
	                  ptr = i + 3
                          quote_end = ptr
                          GOTO 410
	               ENDIF
                    ENDIF
	         ENDDO
                 GOTO 5010
	      ELSEIF (cmnd_buff(ptr:ptr+3) .EQ. p_SQ_quote) THEN
* ... _SQ_ quote - skip past entire quotated string
                 DO i = ptr+4, cmnd_len
                    IF (cmnd_buff(ptr:ptr).EQ.'_' 
     .            .AND. i+3 .LE. cmnd_len           ) THEN
	               IF (cmnd_buff(i:i+3) .EQ. p_SQ_quote) THEN
	                  ptr = i + 3
                          GOTO 410
	               ENDIF
                    ENDIF
	         ENDDO
                 GOTO 5010
	      ENDIF
	      GOTO 410   ! this underscore was not part of _DQ_ or _SQ_
	   ELSEIF ( cmnd_buff(ptr:ptr) .EQ. '`' ) THEN
* ... grave accent - skip past entire immediate mode expression
	      DO 402 ptr = ptr+1, cmnd_len
#ifdef NO_DOUBLE_ESCAPE_SLASH
	         IF ( cmnd_buff(ptr-1:ptr-1) .EQ. '\'  ) GOTO 402
*                            '  closing single quote to make emacs happy
#else
	         IF ( cmnd_buff(ptr-1:ptr-1) .EQ. '\\' ) GOTO 402
#endif
	         IF ( cmnd_buff(ptr:ptr) .EQ. '`' ) GOTO 410
 402	      CONTINUE
	      GOTO 5010
	   ELSEIF ( cmnd_buff(ptr:ptr) .EQ. '[' ) THEN
* ... embedded bracket - find matching right bracket
	      nest = 1
	      DO 403 ptr = ptr+1, cmnd_len
#ifdef NO_DOUBLE_ESCAPE_SLASH
	         IF ( cmnd_buff(ptr-1:ptr-1) .EQ. '\'  ) GOTO 403
*                            '  closing single quote to make emacs happy
#else
	         IF ( cmnd_buff(ptr-1:ptr-1) .EQ. '\\' ) GOTO 403
#endif
	         IF ( cmnd_buff(ptr:ptr) .EQ. '[' ) THEN
	            nest = nest + 1
	         ELSEIF ( cmnd_buff(ptr:ptr) .EQ. ']' ) THEN
	            nest = nest - 1
	            IF ( nest .EQ. 0 ) GOTO 410
	         ENDIF
 403	      CONTINUE
	      GOTO 5010
	   ELSEIF ( cmnd_buff(ptr:ptr) .EQ. '(' ) THEN
* ... embedded paren - find matching right paren (10/95)
	      nest = 1
	      DO 404 ptr = ptr+1, cmnd_len
#ifdef NO_DOUBLE_ESCAPE_SLASH
	         IF ( cmnd_buff(ptr-1:ptr-1) .EQ. '\'  ) GOTO 404
*                            '  closing single quote to make emacs happy
#else
	         IF ( cmnd_buff(ptr-1:ptr-1) .EQ. '\\' ) GOTO 404
#endif
	         IF ( cmnd_buff(ptr:ptr) .EQ. '(' ) THEN
	            nest = nest + 1
	         ELSEIF ( cmnd_buff(ptr:ptr) .EQ. ')' ) THEN
	            nest = nest - 1
	            IF ( nest .EQ. 0 ) GOTO 410
	         ENDIF
 404	      CONTINUE
	      GOTO 5010
	   ELSE
		IF (	  cmnd_buff(ptr:ptr) .NE. ' '
     .		    .AND. cmnd_buff(ptr:ptr) .NE. '/'
     .		    .AND. cmnd_buff(ptr:ptr) .NE. tab ) THEN
                   surround_quote = .FALSE.  ! text outside of quotation
                   GOTO 410
                ENDIF
	   ENDIF

* "/" can be escaped using "\/"
	   IF (ptr .GT. 1) THEN
#ifdef NO_DOUBLE_ESCAPE_SLASH
	      IF (cmnd_buff(ptr-1:ptr) .EQ. '\/') GOTO 410
#else
	      IF (cmnd_buff(ptr-1:ptr) .EQ. '\\/') GOTO 410
#endif
	   ENDIF

* "/" is not a terminator if we are not looking for more qualifiers
	   IF (.NOT.look_for_quals
     .    .AND. cmnd_buff(ptr:ptr) .EQ. '/' ) THEN
	      surround_quote = .FALSE.  ! text outside of quotation
	      GOTO 410
	   ENDIF
	
* probable terminator found - make sure it's not hiding a comma or =
* first check ahead that the next non-blank/tab is not comma or =
* ... also check for ":"
	   DO 405 look_ahead = ptr+1, cmnd_len
	      IF (   cmnd_buff( look_ahead:look_ahead ) .EQ. ' '
     .		.OR. cmnd_buff( look_ahead:look_ahead ) .EQ. tab ) GOTO 405
	      IF (   cmnd_buff( look_ahead:look_ahead ) .EQ. ','
     .		.OR. cmnd_buff( look_ahead:look_ahead ) .EQ. ':'
     .		.OR. cmnd_buff( look_ahead:look_ahead ) .EQ. '=' ) THEN
		   GOTO 410	! no. "," ":" or "=" connects a single "word"
	      ELSE
		   GOTO 406	! all clear ahead - no = or comma
	      ENDIF
 405	   CONTINUE

* now look back to make sure this isn'a a gap following an = or comma
 406	   CONTINUE
           IF (ptr .GT. 1) THEN
           DO 408 look_back = ptr-1, 1, -1
	      IF (   cmnd_buff( look_back:look_back ) .EQ. ' '
     .	      .OR. cmnd_buff( look_back:look_back ) .EQ. tab ) GOTO 408
	      IF (   cmnd_buff( look_back:look_back ) .EQ. ','
     .	      .OR. cmnd_buff( look_back:look_back ) .EQ. ':'
     .	      .OR. cmnd_buff( look_back:look_back ) .EQ. '=' ) THEN
		 GOTO 410	! no  - , or = connects a single "word"
	      ELSE
		 GOTO 420	! yes - we found a terminator
	      ENDIF
 408	   CONTINUE
           ENDIF
		 GOTO 420	! yes - we found a terminator

* manual DO-loop to permit embedded quotation marks
 410	IF ( ptr .LT. cmnd_len ) THEN
	   ptr = ptr + 1
	   GOTO 400
	ENDIF

* end of buffer has been reached with no word terminator - assume the word ends
 415	ptr = cmnd_len + 1

* we have found the end of the word.  
 420    IF ( surround_quote ) THEN
	   arg_start(word_num) = arg_start(word_num) + 1
           arg_end  (word_num) = quote_end - 1
cc  Allow zero-length null string in quotes
           IF ( arg_start(word_num) .GT. arg_end(word_num)+1 ) GOTO 5015
	ELSEIF ( surround_DQ_ ) THEN
	   arg_start(word_num) = arg_start(word_num) + 4
           arg_end  (word_num) = quote_end - 4
cc  Allow zero-length null string in quotes
           IF ( arg_start(word_num) .GT. arg_end(word_num)+1 ) GOTO 5015
	ELSE
	   arg_end (word_num) = ptr - 1
	ENDIF
        IF ( ptr .LT. cmnd_len ) THEN
           GOTO 100                              ! go back for next word
        ELSE
	   GOTO 500				 ! done parsing
        ENDIF

* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
* identify the words in the command line	

* +++++++ COMMAND +++++++
* first word must be command name
* legal syntax of command ?  (avoid, e.g., "LISTX=160W")

 500    CONTINUE
        IF ( word_num .EQ. 0 ) RETURN
        IF ( .NOT.TM_LEGAL_NAME(cmnd_buff(arg_start(1):arg_end(1))) )
     .                                                        THEN
           word_num = 1
           GOTO 5025
        ENDIF
* search the list of options
        len_test = TM_LENSTR(cmnd_buff(arg_start(1):arg_end(1)))
	DO 510 cmnd_num = 1, total_num_commands
           len_mchars = TM_LENSTR(commands(cmnd_num)(:4))
	   IF ( MATCH4( cmnd_buff(arg_start(1):arg_end(1)),len_test,
     .			commands(cmnd_num), len_mchars ) )  GOTO 600
 510	CONTINUE
* no matches against known 4 character command names --> error
	GOTO 5020					! error exit

* +++++++ SUBCOMMAND +++++++
* now identify the subcommand (if any) as next non-qualifier word
 600    DO 610 isub_word = 2,word_num
	   IF ( .NOT.itsa_qualifier(isub_word) ) GOTO 620
 610	CONTINUE
* no other words - therefore no subcommand was given
	GOTO 650
* look for a match in the list of subcommands for this command
 620	len_test = 
     .     TM_LENSTR(cmnd_buff(arg_start(isub_word):arg_end(isub_word)))
        DO 630 subcmnd_num = 2, num_subcommands(cmnd_num)
	   isubcmnd_ptr = subcmnd_num + subcommand_pointer(cmnd_num) - 1
           len_mchars = TM_LENSTR(subcommands( isubcmnd_ptr)(:4))
	   IF ( MATCH4( cmnd_buff(arg_start(isub_word):arg_end(isub_word)),
     .                  len_test,
     .			subcommands( isubcmnd_ptr),  
     .                  len_mchars)  ) THEN
* ... legal syntax of subcommand ?  (avoid, e.g., "SET REGION@T")

*     If we passed over part of the command inside the grave accents above,
*     then pass over that part again here.  This allows
*        LET rname = "/x=1:5"; SET REGION`rname` 

              grave_start = INDEX(
     .           cmnd_buff(arg_start(isub_word):arg_end(isub_word)),'`')
              IF (grave_start .EQ. 0) grave_start = arg_end(isub_word)+1
              IF ( .NOT.TM_LEGAL_NAME
     .        (cmnd_buff(arg_start(isub_word):(grave_start-1))) )
     .                                                        THEN
                 word_num = isub_word
                 GOTO 5025
              ELSE
                 GOTO 700
              ENDIF
           ENDIF
 630	CONTINUE
* no matches against known 4 character subcommands --> no subcommand given
 650	isub_word	 = 0		! no subcommand word in cmnd_buff
	subcmnd_num	 = 1		! use "    " as  subcommand
	isubcmnd_ptr = subcommand_pointer(cmnd_num)

* special hacks to allow "SET DATA path/dset" and "GO path/script"
* without quotes surrounding the argument
 700	its_set_data = cmnd_num    .EQ. cmnd_set
     .           .AND. subcmnd_num .EQ. 7
	its_define_sym = cmnd_num    .EQ. cmnd_define
     .           .AND. subcmnd_num .EQ. 8
	its_go  = cmnd_num    .EQ. cmnd_go
	its_say = cmnd_num    .EQ. cmnd_message
	IF (its_go .OR. its_say) THEN
	   last_recognized_word = 1    ! GO and MESSAGE have no subcommand
	ELSE
	   last_recognized_word = 2    ! used by SET DATA and DEF SYM
	ENDIF

* +++++++ QUALIFIERS +++++++
* since "/" may be used for syntax purposes other that qualifiers determine the
* last "/" that is intended as a qualifier.  For example "GO path/script.jnl"
* 12/2015 note: this hack is required because the code above does not identify
* the command and subcommand at the very start and use the word count at that
* point to guide the identification of qualifier slashes.
* locate the final qualifier
	last_qual = word_num
	DO iword = MAX( isub_word+1, 2 ), word_num
	   IF ( .NOT.itsa_qualifier(iword) ) THEN
	      last_qual = iword - 1
	      EXIT
	   ENDIF
	ENDDO
* any so-called qualifiers beyond last_qual are red herrings
* they should own their slash chars, and may not be separate words at all
* note that if there is white space between a slash and the start of text
* it may not get fixed correctly ... lazy cuz that shouldn't ever happen
	DO iword = last_qual+1, word_num
	   IF (itsa_qualifier(iword)) THEN
	      arg_start(iword) = arg_start(iword) - 1  ! own the  "/"
	   ENDIF
	ENDDO
* group contiguous blocks of slash-connected chars into single args
*     e.g. set data /home/users/tmap/ferret/linux/dsets/coads.nc
* ... work from right to left, consolidate qualifier groups
	ngone = 0
	DO iword = word_num, last_qual+1, -1
	   IF (itsa_qualifier(iword) )THEN
	      itsa_qualifier(iword) = .FALSE.
	      IF (arg_start(iword).EQ.(arg_end(iword-1)+1)) THEN
	         arg_end(iword-1) = arg_end(iword)
	         arg_start(iword) = 0
	         arg_end(iword)   = 0
	         ngone = ngone + 1
	      ENDIF
	   ENDIF
	ENDDO
* ... remove the voids left behind
	iword = last_qual+1
	current_words = word_num
	DO WHILE (iword .LT. current_words)
	   IF (arg_start(iword) .EQ. 0) THEN
	      DO ii = iword+1, current_words
	         arg_start(ii-1) = arg_start(ii)
	         arg_end  (ii-1) = arg_end  (ii)
	      ENDDO
	      arg_start(current_words) = 0
	      arg_end  (current_words) = 0
	      current_words = current_words - 1
	   ENDIF
	   IF (arg_start(iword).NE.0) iword = iword + 1
	ENDDO
	word_num = word_num - ngone
*	last_qual = word_num

* finally ... loop through qualifiers to identify them
* Check for too-short abbreviations, /T= for /TITLE see ticket 1963.
 704	CONTINUE

        expnd_qual = .FALSE.
        ic = 1
        ir = 1
        DO 740 iqual_word = 2, last_qual
	   IF ( .NOT.itsa_qualifier(iqual_word) )	  GOTO 740
	   IF ( num_qualifs(isubcmnd_ptr) .EQ. 0 )        GOTO 5030	! error
	   IF ( num_quals .EQ. max_quals )		  GOTO 5040	! error

* find qualifier terminator ( eg. /K=1:5  is terminated by "=" )
	   DO 710 iterm = arg_start(iqual_word), arg_end(iqual_word)
	      IF (   cmnd_buff(iterm:iterm) .EQ. ' '
     .		.OR. cmnd_buff(iterm:iterm) .EQ. '	'
     .		.OR. cmnd_buff(iterm:iterm) .EQ. '=' ) GOTO 720
 710	   CONTINUE
	   iterm = arg_end(iqual_word) + 1	! no funny terminator

* identify against list of qualifiers for this subcommand 
* unless there's a grave-accent to evaluate (issue 660)
 720	   len_test = TM_LENSTR(cmnd_buff(arg_start(iqual_word):iterm-1))
	   IF (INDEX(cmnd_buff(arg_start(iqual_word):iterm-1),'`') .GT. 0) GOTO 740

           DO 730 iqual = 1,num_qualifs(isubcmnd_ptr)
	   iqual_ptr = iqual + qualifier_pointer(isubcmnd_ptr) - 1
           len_mchars = TM_LENSTR(qualifiers(iqual_ptr)(:4))
	   known_qualifier = 
     .		MATCH4( cmnd_buff(arg_start(iqual_word):iterm - 1 ),
     .                  len_test,
     .			qualifiers(iqual_ptr),len_mchars )
     
* Did they abbreviate to one letter when we may need more?  Fill out the qualifier.
* Could expand the too-shortness from (len_test.EQ.1) to (len_test.LT.len_mchars)

           IF (known_qualifier .AND. len_test.EQ.1 .AND. len_mchars.GT.1) THEN
c           IF (known_qualifier .AND. len_test.LT.len_mchars .AND. len_mchars.GT.1) THEN
              expnd_qual = .TRUE.
              risc_buff(ir:) = cmnd_buff(ic:arg_start(iqual_word))
	      ir = TM_LENSTR(risc_buff)   ! ir + arg_start(iqual_word)-1
              risc_buff(ir:) = qualifiers(iqual_ptr)(:len_mchars)
              ir = ir + len_mchars 
	      ir = TM_LENSTR(risc_buff)
              ic = arg_start(iqual_word) + len_test 

           ENDIF 

* "@/anything" will be permitted to pass
* (8/92 - this should be lifted outside of this loop for speed!!! *sh*)
	   atsin_qualifier  = 
     .		cmnd_buff(arg_start(iqual_word):arg_start(iqual_word)) .EQ. '@'
	   IF ( known_qualifier .OR. atsin_qualifier ) THEN
	        last_recognized_word = iqual_word
		num_quals = num_quals + 1
		qual_start(num_quals) = arg_start(iqual_word)
		qual_end  (num_quals) = arg_end  (iqual_word)
		IF ( known_qualifier ) THEN
		   qualifier_list(num_quals) = iqual
		ELSE
		   qualifier_list(num_quals) = 0
		ENDIF
		GOTO 740
	   ENDIF
 730	   CONTINUE

* checked all possible qualifiers and did't find a match

	   IF (its_set_data .OR. its_define_sym) THEN
* *sh* 2015 *HACK* for SET DATA path/filename and GO path/filename
*      since filenames may contain slashes that are mistaken as qualifiers
*      we treat the first unrecognized qualifier as potentially part of the
*      filename. 
*      It is *not* part of the filename if
*      - no white space gap has been seen to the left
*      For SET DATA it is also not part of the filename if
*      - there is a white space gap without a comma found to the right
	      IF (white_aftr_subcmnd .GT. arg_start(iqual_word)) GOTO 5030 
* ... look to right for white space with no comma
	      found_white_space = .FALSE.
	      DO iterm = arg_end(iqual_word), cmnd_len
	         IF (   cmnd_buff(iterm:iterm) .EQ. ' '
     .		   .OR. cmnd_buff(iterm:iterm) .EQ. '	' ) THEN
	            found_white_space = .TRUE.
		 ELSEIF (found_white_space) THEN
	            IF (cmnd_buff(iterm:iterm) .NE. ',') THEN
	               GOTO 5030
	            ELSE
	               EXIT
	            ENDIF
	         ENDIF
	      ENDDO
	      word_num = last_recognized_word + 1
	      arg_end(word_num) = cmnd_len
	      IF (itsa_qualifier(word_num)) arg_start(word_num)
     .			= arg_start(word_num)-1	    ! keep "/" in /home/file
	      itsa_qualifier(word_num) = .FALSE.
	      last_qual = iqual_word - 1     ! added 12/2015 -- approx true
	      EXIT
	   ELSEIF (its_go .OR. its_say) THEN
* *HACK* for GO to allow unquoted "GO path/script" and "GO /hom/path/script"
	      IF (white_aftr_cmnd .GT. arg_start(iqual_word)) GOTO 5030 
* ... its an absolute path -- count the slashes
	      ngone = 0
	      i = iqual_word+1
	      DO WHILE (i.LE. word_num .AND. itsa_qualifier(i))
	         ngone = ngone + 1
	         i  =i+1
	      ENDDO
* turn this entire absolute path into a single argument
	      word_num = word_num - ngone
	      arg_start(iqual_word) = arg_start(iqual_word)-1 ! keep start "/"
	      arg_end(iqual_word)   = arg_end(iqual_word+ngone)
	      itsa_qualifier(iqual_word) = .FALSE.
	      last_qual = iqual_word - 1     ! added 12/2015 -- approx true
	      DO i = iqual_word+1, word_num
                 arg_start(i) = arg_start(i+ngone)
                 arg_end  (i) = arg_end  (i+ngone)
	         IF (itsa_qualifier(i) ) arg_start(i) = arg_start(i) - 1
	         itsa_qualifier(i) = .FALSE.
	      ENDDO
	      EXIT
	   ELSE
	      GOTO 5030		! all other commands: unknown qual error
	   ENDIF

 740	CONTINUE

* If we replaced abbreviated qualifiers with longer versions, go back and
* and re-parse the whole thing as all the pointers now need to be revised.

        IF (expnd_qual) THEN
	   risc_buff(ir:) = cmnd_buff(ic:TM_LENSTR(cmnd_buff))
	   cmnd_buff = risc_buff
           expnd_qual = .FALSE.   
           GOTO 5  
        ENDIF

* +++++++ ARGUMENTS +++++++
* return buffer locations for all words not yet identified
* (last_qual check as bug fix 2/94 *sh*)
	DO 810 iarg = 2,word_num
	   IF ( iarg .LE. last_qual
     .      .AND. (itsa_qualifier(iarg) .OR. isub_word.EQ.iarg) ) THEN
		GOTO 810     ! command, subcommand, or qualifier
	   ELSE
		num_args		= num_args + 1
		arg_start( num_args )	= arg_start( iarg )
		arg_end  ( num_args ) 	= arg_end  ( iarg )
	   ENDIF
 810	CONTINUE

* is the first argument enclosed in quotation marks?
* the nature of arg_start,end allows safe (??) decrement/increment
	IF (num_args .GE. 1) THEN
	   arg1_quoted =
     .	        (cmnd_buff(arg_start(1)-1:arg_start(1)-1) .EQ. '"'
     .     .AND. cmnd_buff(arg_end  (1)+1:arg_end  (1)+1) .EQ. '"' )
     .   .OR.
     .	        (cmnd_buff(arg_start(1)-1:arg_start(1)-1) .EQ. "'"
     .     .AND. cmnd_buff(arg_end  (1)+1:arg_end  (1)+1) .EQ. "'" )

	   IF (.NOT.arg1_quoted
     .    .AND. (cmnd_buff(arg_start(1)-1:arg_start(1)-1) .EQ. "_")
     .    .AND. (cmnd_buff(arg_end  (1)+1:arg_end  (1)+1) .EQ. "_")
     .    .AND. (arg_start(1)-4.GE.1)
     .    .AND. (arg_end  (1)+4.LE.cmnd_len)                     )THEN
	      arg1_quoted = 
     .          (cmnd_buff(arg_start(1)-4:arg_start(1)-1) .EQ. "_DQ_"
     .      .AND.cmnd_buff(arg_end  (1)+1:arg_end  (1)+4) .EQ. "_DQ_" )
     .    .OR.
     .          (cmnd_buff(arg_start(1)-4:arg_start(1)-1) .EQ. "_SQ_"
     .      .AND.cmnd_buff(arg_end  (1)+1:arg_end  (1)+4) .EQ. "_SQ_" )
	   ENDIF
	ENDIF

* REPEAT loops postpone symbol substitutions in arguments -- all 1 arg for now
	IF ( cmnd_num .EQ. cmnd_repeat ) THEN
	   IF (num_args .EQ. 0) GOTO 5070
	   IF ( cmnd_buff(arg_end(num_args):arg_end(num_args)) .EQ. ')' 
     .    .AND. cmnd_buff(arg_start(1):arg_start(1)) .EQ. '(' ) THEN
*   ... search backwards for left paren
	    first_paren = .TRUE.
	    paren_level = 1
	    DO 850 i = len_cmnd_copy-1, 1, -1
	      IF (cmnd_copy(i:i) .EQ. ')') THEN
	        paren_level = paren_level + 1
	        first_paren = .FALSE.
	      ELSEIF (cmnd_copy(i:i) .EQ. '(') THEN
	        paren_level = paren_level - 1
	        IF (first_paren) THEN
*   ... dont confuse symbol at end "($sym)" with paren-enclosed command group 
	          IF (cmnd_copy(i+1:i+1) .EQ. '$') GOTO 890
	          first_paren = .FALSE.
	        ENDIF
	        IF ( paren_level .EQ. 0 ) THEN
*   ... found the block of REPEAT argument - replace the text with original
	          cmnd_buff(arg_start(1):) = cmnd_copy(i:len_cmnd_copy)
	          cmnd_len = arg_start(1) + (len_cmnd_copy-i)
	          num_args = 1
	          arg_end(1) = cmnd_len
	          GOTO 890
	        ENDIF
	      ENDIF
 850	    CONTINUE
*   ... no matching left paren ever found (probably a syntax error ...)
*       It would be better to let the parsing of each command catch and report
*       errors, but ticket 2551 shows a failure in that. Make it an error here.
          GOTO 5080
	  ENDIF
	ENDIF
 890	CONTINUE


* For SET DATA commands if there is an F-TDS expression, then encode that now.

	IF  (cmnd_num .EQ. cmnd_set .AND. 
     .       subcmnd_num .EQ. subcmnd_set_data+1 ) THEN
	
	   IF (num_args .GT. 0) url_buff = cmnd_buff( arg_start(1):arg_end(num_args) )
	   len_test = TM_LENSTR(url_buff)
           i = INDEX( url_buff(1:len_test), '_expr_{}' ) 
	   IF (i .GT. 0) THEN
	      num_args = 1
              CALL CD_ENCODE_URL (url_buff)
	      len_test = TM_LENSTR(url_buff)
	      i = INDEX(cmnd_buff, "http://")
	      cmnd_buff(i:i+len_test) = url_buff
	      arg_end(1) = i+len_test - 1
	   ENDIF
	ENDIF

* +++++++++ grave-accented immediate-mode expressions +++++++++++  3/95
* evaluate and substitute the first grave accent-surrounded expression, if any
* beginning at character grave_start
* and return to re-process command if substitution occurs (4/93)
* This part of the command parsing is postponed until the end so that
* the command qualifiers (which may contain region information) will have
* been identified and can be used while evaluating the grave expressions
	nsubst_passes = nsubst_passes + 1 ! guard against recursion (4/93)
	IF ( nsubst_passes .GT. 100 ) GOTO 5060
* For REPEAT commands the grave accented expressions INSIDE the REPEAT loop
* must be deferred until later
	IF ( INDEX(cmnd_buff(:cmnd_len),'`') .GT. 0 ) THEN
	   IF  ( cmnd_num .EQ. cmnd_repeat ) THEN
	      grave_end = qual_end(num_quals)

	   ELSEIF  ( cmnd_num .EQ. cmnd_if
     .	     .OR.    cmnd_num .EQ. cmnd_elif ) THEN
	      IF ( num_args .GE. 1 ) THEN
	         grave_end = arg_end(1)
	      ELSE
	         grave_end = grave_start      ! invalid IF comand syntax
	      ENDIF
	   ELSE
	      grave_end = cmnd_len
	   ENDIF
           apply_cx = its_action_command(cmnd_num)
	   CALL REPL_EXPRNS(  cmnd_buff, cmnd_len,
     .			     apply_cx, grave_start, grave_end,
     .                       grave_digits, subst, status  )
	   IF ( status .NE. ferr_ok ) GOTO 5000	
	   IF ( subst ) THEN
* ... num_uvars_in_cmnd flags expression results from previous cmnds as invalid
	      num_uvars_in_cmnd = cmnd_uvars_not_given
	      reverify = .TRUE.      ! informational echo needed
	      GOTO 5   ! back to parse again with the expresns replaced
	   ENDIF
	ENDIF

* replace backslashes that were escapes
* Ticket 2376: If the command is a REPEAT, this needs to wait until
* the command is parsed within the execution of the loop.

#ifdef NO_DOUBLE_ESCAPE_SLASH
	IF (  ( cmnd_num .NE. cmnd_repeat ) .AND. 
     .		INDEX(cmnd_buff(:cmnd_len),'\') .GT. 0 ) THEN
	   in  = 1
	   out = 1
 900	   IF (cmnd_buff(in:in+1) .EQ. '\!') THEN
	      cmnd_buff(out:out) = '!'
	      in = in + 2
	   ELSEIF (cmnd_buff(in:in+1) .EQ. '\;') THEN
	      cmnd_buff(out:out) = ';'
	      in = in + 2
	   ELSEIF (cmnd_buff(in:in+1) .EQ. '\`') THEN
	      cmnd_buff(out:out) = '`'
	      in = in + 2
	   ELSEIF (cmnd_buff(in:in+1) .EQ. '\/') THEN
	      cmnd_buff(out:out) = '/'
	      in = in + 2
#else
	IF ( ( cmnd_num .NE. cmnd_repeat ) .AND. 
     .		INDEX(cmnd_buff(:cmnd_len),'\\') .GT. 0 ) THEN
	   in  = 1
	   out = 1
 900	   IF (cmnd_buff(in:in+1) .EQ. '\\!') THEN
	      cmnd_buff(out:out) = '!'
	      in = in + 2
	   ELSEIF (cmnd_buff(in:in+1) .EQ. '\\;') THEN
	      cmnd_buff(out:out) = ';'
	      in = in + 2
	   ELSEIF (cmnd_buff(in:in+1) .EQ. '\\`') THEN
	      cmnd_buff(out:out) = '`'
	      in = in + 2
	   ELSEIF (cmnd_buff(in:in+1) .EQ. '\\/') THEN
	      cmnd_buff(out:out) = '/'
	      in = in + 2
#endif	
	   ELSE
	      cmnd_buff(out:out) = cmnd_buff(in:in)
	      in = in + 1
	   ENDIF
	   out = out + 1
	   IF (in.LT.cmnd_len) GOTO 900	
	   cmnd_buff(out:out) = cmnd_buff(in:in)
	   cmnd_len = out
* ... cover over characters exposed at line end
	   DO 910 out = out+1,in+1
 910	   cmnd_buff(out:out) = ' '
	ENDIF

* successful completion
 1000   status = ferr_ok
	RETURN

* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
* error exits
 5000	RETURN
* error exits
 5005	CALL ERRMSG( ferr_syntax, status,
     .			'command line too long -- exceeds 2048', *5000 )

 5010	CALL ERRMSG( ferr_syntax, status,
     .			'unpaired quotation marks, grave accent or brackets',
     .			*5000 )

 5015	CALL ERRMSG( ferr_syntax, status,
     .				'zero length quotation forbidden', *5000 )

 5020	CALL ERRMSG( ferr_unknown_command, status,
     .		cmnd_buff( arg_start(1):arg_end(1) ), *5000 )

 5025	IF ( arg_end(word_num) - arg_start(word_num) .GT. 200) THEN 
	   CALL ERRMSG( ferr_syntax, status,
     .		cmnd_buff( arg_start(word_num):arg_start(word_num)+200 )//'...', *5000 )
        ELSE
	   CALL ERRMSG( ferr_syntax, status,
     .		cmnd_buff( arg_start(word_num):arg_end(word_num) ), *5000 )
        ENDIF

 5030	CALL ERRMSG( ferr_unknown_qualifier, status,
     .		cmnd_buff( arg_start(iqual_word):arg_end(iqual_word) ), *5000 )

 5040	CALL ERRMSG
     .		( ferr_cmnd_too_complex, status, 'too many qualifiers', *5000 )

 5050	CALL ERRMSG( ferr_syntax, status, '"/"', *5000 )

 5060	CALL ERRMSG( ferr_syntax, status,
     .		'Recursive aliases or GO argument definitions', *5000 )

 5070	CALL ERRMSG
     .			( ferr_invalid_command, status, 'REPEAT what ?', *5000 )
 5080	CALL ERRMSG
     .			( ferr_invalid_command, status, 
     .		'Mismatched parentheses within REPEAT loop ', *5000 )
	END

