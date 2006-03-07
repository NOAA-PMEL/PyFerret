	SUBROUTINE PARSE_COMMAND ( memory, cmnd_buff,
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
* V590 *acm* 3/06  fixing bugs 439&1390, pass cmnd_num to repl_exprns so we can decide
*                  if the command is an action command, and so whether to apply command
*                  context to grave-accent expressions.

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

* declare calling arguments of SUBROUTINE
	LOGICAL		reverify, arg1_quoted
	REAL		memory(*)
	CHARACTER*(*)	cmnd_buff
	INTEGER		max_words,max_quals,cmnd_len,cmnd_num,subcmnd_num,
     .			num_quals,qualifier_list(max_quals),
     .			qual_start(max_quals), qual_end(max_quals),
     .			num_args,arg_start(max_words),arg_end(max_words),
     .			err_lun,status

	include	'errmsg.parm'
	include	'ferret.parm'
	include	'command.parm'
	include 'xcommand.cmn'
	include 'xcontrol.cmn'
	include 'xvariables.cmn'	! for num_uvars_in_cmnd

* local variable declarations:
*
**kob*12/97!!! NOTE:local_max_arg_list must match max_arg_list in xprog_state
        INTEGER local_max_arg_list
        PARAMETER (local_max_arg_list = 256)

* modified def of MATCH4 from INTEGER to LOGICAL *kob* IBM port 8/94
	LOGICAL	TM_LEGAL_NAME, known_qualifier, atsin_qualifier,
     .          surround_quote, doub_quote, subst, first_paren,
     .          itsa_qualifier(local_max_arg_list), 
     .          MATCH4, escape, apply_cx
	INTEGER TM_LENSTR
	INTEGER	buff_len, bang_position, ptr, quote_end,
     .		word_num, look_ahead, look_back, isub_word, csgo,
     .		isubcmnd_ptr, iword, last_qual, iqual_word, iterm,
     .		iqual, iqual_ptr, iarg, i, nparen, nsubst_passes,
     .		grave_start, grave_end, in, out, nest,
     .		paren_level,len_test,len_mchars
	CHARACTER c1*1

* local parameter declarations:
	INTEGER		grave_digits
	CHARACTER*1	tab
#ifdef NO_INTRINSIC_IN_PARAMETER
	PARAMETER     ( tab = o'011', 
#else
	PARAMETER     ( tab = CHAR(9),
#endif
     .			grave_digits = 5 )

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

* full length of text
	cmnd_len = TM_LENSTR (cmnd_buff)

* ignore "yes?" at start of command
 2	IF (cmnd_buff(1:5) .EQ. 'yes? '
     . .OR. cmnd_buff(1:5) .EQ. '...? ' ) THEN
	   IF ( cmnd_len .LT. 6 ) THEN
	     cmnd_buff = ' '
	     cmnd_len = 0
	   ELSE
	     cmnd_buff = cmnd_buff(6:cmnd_len)
	     cmnd_len = cmnd_len - 5
	     GOTO 2		! in case there are multiple "yes?"'s
	   ENDIF
	ENDIF

* comment line ?
	IF ( (cmnd_len .LE. 0
     .   .OR. cmnd_buff(1:1) .EQ. '*'
     .   .OR. cmnd_buff(1:1) .EQ. '!' ) 
     . .AND. .NOT.has_continuation_cmnd ) THEN
	   RETURN
	ENDIF
	IF ( cmnd_buff(1:1) .EQ. '/'
     .	.AND. .NOT.has_continuation_cmnd ) GOTO 5050

* find length of command line ("!" is a valid terminator to permit comments)
	bang_position = INDEX ( cmnd_buff, '!' )
* ... 11/95 allow "!" to be be "escaped"
	IF ( bang_position .GT. 0 ) THEN
#ifdef NO_DOUBLE_ESCAPE_SLASH
	   IF ( cmnd_buff(bang_position-1:bang_position) .EQ. '\!' ) THEN
#else
	   IF ( cmnd_buff(bang_position-1:bang_position) .EQ. '\\!' ) THEN
#endif
* ... ... this block of code could replace INDEX() but doesn't for performance
	      DO 6 bang_position = bang_position+1, cmnd_len
#ifdef NO_DOUBLE_ESCAPE_SLASH
	         IF (cmnd_buff(bang_position:bang_position).EQ.'!'
     .         .AND. cmnd_buff(bang_position-1:bang_position-1).NE.'\')
     .				GOTO 8
#else
	         IF (cmnd_buff(bang_position:bang_position).EQ.'!'
     .         .AND. cmnd_buff(bang_position-1:bang_position-1).NE.'\\')
     .				GOTO 8
#endif
 6	      CONTINUE
	      bang_position = 0
	   ENDIF
	ENDIF
 8	IF ( bang_position .GT. 0 ) THEN
	   DO 10 cmnd_len = bang_position-1, 1, -1
	      IF (  cmnd_buff(cmnd_len:cmnd_len) .NE. ' '
     .	      .AND. cmnd_buff(cmnd_len:cmnd_len) .NE. tab ) GOTO 15
 10	   CONTINUE
	   IF (.NOT.has_continuation_cmnd) RETURN	! no text before !
	ENDIF

* prepend continuation from a previous line if any
 15	IF ( has_continuation_cmnd ) THEN
	   len_cmnd_copy = len_cmnd_copy - 1  ! final char is continuation
	   IF (len_cmnd_copy .GT. 0) THEN
	     IF ( cmnd_len .GT. 0) THEN
	        cmnd_buff = cmnd_copy(:len_cmnd_copy) // cmnd_buff(:cmnd_len)
	     ELSE
	        cmnd_buff = cmnd_copy(:len_cmnd_copy)
	     ENDIF
	   ENDIF
	   cmnd_len = len_cmnd_copy + cmnd_len
	   IF ( cmnd_len .GE. 2048 ) GOTO 5005 ! must match cmnd_buff_len
	ENDIF

* save a copy of the command with untranslated symbols for use by REPEAT
* and for building the complete command when continuation lines are used.

	cmnd_copy = cmnd_buff
	len_cmnd_copy = cmnd_len

* is this an incomplete line requiring continuation (terminated w/ backslash)?
	has_continuation_cmnd =
#ifdef NO_DOUBLE_ESCAPE_SLASH
     .				cmnd_buff(cmnd_len:cmnd_len) .EQ. '\' 
#else
     .				cmnd_buff(cmnd_len:cmnd_len) .EQ. '\\'
#endif
	IF ( has_continuation_cmnd
     . .OR. cmnd_len .EQ. 0        ) THEN
	   cmnd_num = 0
	   RETURN
	ENDIF

* * * * * * * * *
* remove parens enclosing entire command or command group
* modified 10/95 to protect "($name) at line start
 20     IF (cmnd_len .GT. 1) THEN
	   IF( cmnd_buff(1:1) .EQ. '(' 
     .	 .AND. cmnd_buff(2:2) .NE. '$'
     .   .AND. cmnd_buff(cmnd_len:cmnd_len) .EQ. ')' ) THEN
              cmnd_buff(1:1) = ' '
              cmnd_buff(cmnd_len:cmnd_len) = ' '
              CALL LEFT_JUST( cmnd_buff(1:cmnd_len),
     .                        cmnd_buff(1:cmnd_len), cmnd_len )
           ENDIF
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
        ENDIF


* * * * * * * * *
* process semicolon-separated command group  (2/93)
* "REPEAT/L=lo:hi (command1;command2)"  is a single command (at this stage) but
* "command1;command2" and "(command1;command2)" are command groups
        IF ( INDEX(cmnd_buff(1:cmnd_len),';') .GT. 0 ) THEN
* ... it is a command group if there is a ";" NOT enclosed in parens
           nparen = 0
           surround_quote = .FALSE.
           DO 60 i = 1, cmnd_len
	      c1 = cmnd_buff(i:i)
              IF ( c1 .EQ. '(' ) THEN
                 nparen = nparen + 1
              ELSEIF ( c1 .EQ. ')' ) THEN
                 nparen = nparen - 1
              ELSEIF ( c1 .EQ. '"' .AND. doub_quote ) THEN
                 surround_quote = .NOT.surround_quote
              ELSEIF ( c1 .EQ. "'" .AND. .NOT.doub_quote ) THEN
                 surround_quote = .NOT.surround_quote
              ELSEIF ( c1 .EQ. ';' ) THEN
	         escape = i.GT.1
#ifdef NO_DOUBLE_ESCAPE_SLASH
	         IF (escape) escape = cmnd_buff(i-1:i) .EQ. '\;'
#else
	         IF (escape) escape = cmnd_buff(i-1:i) .EQ. '\\;'
#endif
                 IF ( nparen.EQ.0
     .		.AND. .NOT.escape
     .		.AND. .NOT.surround_quote ) THEN
                    cmnd_num = cmnd_semicolon
                    GOTO 1000   ! successful exit
                 ENDIF
              ENDIF
 60        CONTINUE
        ENDIF

!* check for buffer overflow (too-long command line)
! 90     IF ( cmnd_len .EQ. buff_len ) GOTO  5005
!	cmnd_buff (buff_len:buff_len) = ' '	! always end with blank

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
	DO 110 ptr = ptr, cmnd_len
		IF (      cmnd_buff(ptr:ptr) .NE. ' '
     .		    .AND. cmnd_buff(ptr:ptr) .NE. tab ) GOTO 200
 110	CONTINUE
	GOTO 500		! end of command buffer - no more words

* we have another word in the buffer.  Is this word a qualifier ?

 200	IF ( word_num .EQ. max_words ) GOTO 500
	word_num 	= word_num + 1
	IF ( cmnd_buff(ptr:ptr) .EQ. '/' ) THEN
	   itsa_qualifier(word_num) = .TRUE.
	   ptr = ptr + 1		! jump over "/"
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
	ENDIF

* we have found the start of a word - initialize variables
 300    arg_start (word_num) = ptr
        surround_quote = .FALSE.

* start searching for the end of the word: blank,tab,"/" or ", if its a quote
 400	CONTINUE
#ifdef NO_DOUBLE_ESCAPE_SLASH
	   IF ( cmnd_buff(ptr-1:ptr-1) .EQ. '\' ) THEN
#else
	   IF ( cmnd_buff(ptr-1:ptr-1) .EQ. '\\' ) THEN
#endif
	      GOTO 410
  
	   ELSEIF ( cmnd_buff(ptr:ptr) .EQ. '"' 
     .	   .OR.     cmnd_buff(ptr:ptr) .EQ. "'"  ) THEN
* ... quotation marks - skip past entire quotated string
	      doub_quote = cmnd_buff(ptr:ptr) .EQ. '"' 
              surround_quote = doub_quote
     .	                 .AND. ptr .EQ. arg_start(word_num)
	      DO 401 ptr = ptr+1, cmnd_len
#ifdef NO_DOUBLE_ESCAPE_SLASH
	        IF ( cmnd_buff(ptr-1:ptr-1) .EQ. '\'  ) GOTO 401
#else
	        IF ( cmnd_buff(ptr-1:ptr-1) .EQ. '\\' ) GOTO 401
#endif
	        IF (cmnd_buff(ptr:ptr).EQ.'"' .AND. doub_quote
     .	       .OR. cmnd_buff(ptr:ptr).EQ."'" .AND. .NOT.doub_quote) THEN
                  quote_end = ptr
                  GOTO 410
                ENDIF
 401	      CONTINUE
	      GOTO 5010
	   ELSEIF ( cmnd_buff(ptr:ptr) .EQ. '`' ) THEN
* ... grave accent - skip past entire immediate mode expression
	      DO 402 ptr = ptr+1, cmnd_len
#ifdef NO_DOUBLE_ESCAPE_SLASH
	         IF ( cmnd_buff(ptr-1:ptr-1) .EQ. '\'  ) GOTO 402
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
#ifdef NO_DOUBLE_ESCAPE_SLASH
	   IF (cmnd_buff(ptr-1:ptr) .EQ. '\/') GOTO 410
#else
	   IF (cmnd_buff(ptr-1:ptr) .EQ. '\\/') GOTO 410
#endif
	
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
 406	   DO 408 look_back = ptr-1, 1, -1
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
cc           IF ( arg_start(word_num) .GT. arg_end(word_num) ) GOTO 5015
cc  Allow it to be zero-length
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

 500    IF ( .NOT.TM_LEGAL_NAME(cmnd_buff(arg_start(1):arg_end(1))) )
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

* +++++++ QUALIFIERS +++++++
* since "/" may be used for syntax purposes other that qualifiers determine the
* last "/" that is intended as a qualifier
 700    DO 702 iword = MAX( isub_word+1, 2 ), word_num
	   IF ( .NOT.itsa_qualifier(iword) )	  THEN
	      last_qual = iword - 1
	      GOTO 704
	   ENDIF
 702	CONTINUE
	last_qual = word_num

* loop through qualifiers
 704	DO 740 iqual_word = 2, last_qual
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
 720	   len_test = TM_LENSTR(cmnd_buff(arg_start(iqual_word):iterm-1))
           DO 730 iqual = 1,num_qualifs(isubcmnd_ptr)
	   iqual_ptr = iqual + qualifier_pointer(isubcmnd_ptr) - 1
           len_mchars = TM_LENSTR(qualifiers(iqual_ptr)(:4))
	   known_qualifier = 
     .		MATCH4( cmnd_buff(arg_start(iqual_word):iterm - 1 ),
     .                  len_test,
     .			qualifiers(iqual_ptr),len_mchars )
* "@/anything" will be permitted to pass
* (8/92 - this should be lifted outside of this loop for speed!!! *sh*)
	   atsin_qualifier  = 
     .		cmnd_buff(arg_start(iqual_word):arg_start(iqual_word)) .EQ. '@'
	   IF ( known_qualifier .OR. atsin_qualifier ) THEN
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
	   GOTO 5030							! error

 740	CONTINUE

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
     .	   (     cmnd_buff(arg_start(1)-1:arg_start(1)-1) .EQ. '"'
     .     .AND. cmnd_buff(arg_end(1)  +1:arg_end(1)  +1) .EQ. '"' )
     . .OR.(     cmnd_buff(arg_start(1)-1:arg_start(1)-1) .EQ. "'"
     .     .AND. cmnd_buff(arg_end(1)  +1:arg_end(1)  +1) .EQ. "'" )
	ENDIF

* REPEAT loops postpone symbol substitutions in arguments -- all 1 arg for now
	IF ( cmnd_num .EQ. cmnd_repeat ) THEN
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
	  ENDIF
	ENDIF
 890	CONTINUE

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
	   CALL REPL_EXPRNS( memory, cmnd_buff, cmnd_len,
     .			     apply_cx, grave_start, grave_end,
     .                       grave_digits, subst, status  )
	   IF ( status .NE. ferr_ok ) GOTO 5000	
	   IF ( subst ) THEN
! ... num_uvars_in_cmnd flags expression results from previous cmnds as invalid
	      num_uvars_in_cmnd = cmnd_uvars_not_given
	      reverify = .TRUE.      ! informational echo needed
	      GOTO 5   ! back to parse again with the expresns replaced
	   ENDIF
	ENDIF

* replace backslashes that were escapes
#ifdef NO_DOUBLE_ESCAPE_SLASH
	IF ( INDEX(cmnd_buff(:cmnd_len),'\') .GT. 0 ) THEN
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
	IF ( INDEX(cmnd_buff(:cmnd_len),'\\') .GT. 0 ) THEN
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

! 5005   risc_buff = cmnd_buff(cmnd_len-19:cmnd_len)
! 	CALL ERRMSG( ferr_line_too_long, status,
!     .		     risc_buff(:20), *5000 )

 5005	CALL ERRMSG( ferr_syntax, status,
     .			'command line too long -- exceeds 2048', *5000 )
! 2048 must match cmnd_buff_len

 5010	CALL ERRMSG( ferr_syntax, status,
     .			'unpaired quotation marks, grave accent or brackets',
     .			*5000 )

 5015	CALL ERRMSG( ferr_syntax, status,
     .				'zero length quotation forbidden', *5000 )

 5020	CALL ERRMSG( ferr_unknown_command, status,
     .		cmnd_buff( arg_start(1):arg_end(1) ), *5000 )

 5025	CALL ERRMSG( ferr_syntax, status,
     .		cmnd_buff( arg_start(word_num):arg_end(word_num) ), *5000 )

 5030	CALL ERRMSG( ferr_unknown_qualifier, status,
     .		cmnd_buff( arg_start(iqual_word):arg_end(iqual_word) ), *5000 )

 5040	CALL ERRMSG
     .		( ferr_cmnd_too_complex, status, 'too many qualifiers', *5000 )

 5050	CALL ERRMSG( ferr_syntax, status, '"/"', *5000 )

 5060	CALL ERRMSG( ferr_syntax, status,
     .		'Recursive aliases or GO argument definitions', *5000 )
	END
