	SUBROUTINE REPL_EXPRNS( memory, cmnd,    lencmnd,
     .				istart, max_check,
     .				digits, did_sub, status  )

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
* between grave accents with text strings representing their values
* this routine will be called once for each such expression so that the
* entire command line can be re-parsed between calls.  This makes it possible
* to build the command qualifier information from fixed limits and grave
* accented expressions working from left to right

* USAGE EXAMPLE:
*	CONTOUR/Z=`TEMP[Z=0:1000@LOC:15]` SALINITY*`some_factor`

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program

* V400 3/95
* V411: 9/8/95 - serious bug fixed:  use CREATE_TEMP_MEM_VAR and COPY_GRID
*		so that the result scalar is picked correctly from within
*		the body of a larger mvar (instead of assuming that it is
*		always element (1,1,1,1) of the mvar)
* V420: 11/95 - allow grave accents to be "escaped" by '\'
* 	 1/96 - bug fix: special check if grave accents enclose white space
*	 2/96 - allow digits control with `expr,PREC=n`
*		and control of "bad string" with "BAD=xx.xx"
*	 3/96 - added RETURN=shape, isize,istart,iend, etc. for j,k, and l
* Linux Port - 5/97 *kob*
*	     - added ifdef check for double slash because f90/linux didn't
*	       need two of them together for escapes
#ifdef unix
	include 'ferret_cmn/ferret.parm'
	include 'ferret_cmn/errmsg.parm'
	include 'ferret_cmn/xvariables.cmn'
	include 'ferret_cmn/xrisc.cmn'
	include 'ferret_cmn/xcontext.cmn'
	include 'ferret_cmn/xtext_info.cmn'
#else
	INCLUDE 'FERRET_CMN:FERRET.PARM'
	INCLUDE 'FERRET_CMN:ERRMSG.PARM'
	INCLUDE 'FERRET_CMN:XVARIABLES.CMN'
	INCLUDE 'FERRET_CMN:XRISC.CMN'
	INCLUDE 'FERRET_CMN:XCONTEXT.CMN'
	INCLUDE 'FERRET_CMN:XTEXT_INFO.CMN'
#endif

* calling argument declarations:
	LOGICAL	  did_sub, info_query
	INTEGER	  lencmnd, istart, max_check, digits, status
	REAL	  memory( mem_blk_size, max_mem_blks )
	CHARACTER cmnd*(*)

* local parameter declarations
	INTEGER	   max_len
	PARAMETER (max_len = 16)

* internal variable declarations:
	INTEGER   CGRID_SIZE, TM_LENSTR1, MR_DIM_LEN,
     .		  str_len, lenbuff, iend, itmp,
     .		  mr, cx, mr_temp, i, i2, use_digits, true_end,
     .		  qual_end, brkt, idim
	REAL	  result
	CHARACTER TM_FMT*16, LEFINT*16, repl*16, bad_str*16	! max_len=16

* local parameter declarations
	CHARACTER	tab*1
	PARAMETER     ( tab = CHAR(9) )

* initialize
	did_sub = .FALSE.    	! any substitutions done?
	lenbuff = LEN( cmnd )
	use_digits = digits
	bad_str = 'bad'
	info_query = .FALSE.

* begin the search/replace at the point specified in the call.  (This is done
* to facilitate double grave accents condensed into single accents so they
* can be passed to the command line in SPAWN commands)
* locate the start of the next grave accent pair
 10	IF ( istart .GE. lencmnd ) RETURN
	itmp = INDEX(cmnd(istart:),'`')
        IF ( itmp .EQ. 0 ) RETURN
	istart = istart + itmp
#ifdef NO_DOUBLE_ESCAPE_SLASH
	IF ( istart .GT. 2 ) THEN
	   IF (cmnd(istart-2:istart-2) .EQ. '\') GOTO 10  
	ENDIF
#else
	IF ( istart .GT. 2 ) THEN
	   IF (cmnd(istart-2:istart-2) .EQ. '\\') GOTO 10  
	ENDIF
#endif


* locate the end of the next grave accent pair
* Note: no check for backslash-escaped accent here because properly paired
*    accents only need to be checked at start of accent pair
	itmp = INDEX(cmnd(istart:),'`')
	IF ( itmp .EQ. 0 ) GOTO 5100		! unpaired grave accent
	IF ( itmp .EQ. 1 ) THEN
* ... replace double grave accents with single
	   cmnd = cmnd(:istart-1)//cmnd(istart+1:)
	   istart = istart + 1			! first char following accent
	   lencmnd = lencmnd - 1
	   GOTO 1000
	ENDIF
	iend = istart + itmp - 2
	true_end = iend

* do not perform substitutions beyond max_check characters
	IF ( istart .GE. max_check ) RETURN

* ====== 2/96
* Parse ",PRECISION=n" and/or ",BAD=xx.xx" at the end of the expression
*  (may be abbreviated to "P=n" and "B=xx.xx")
* 3/96: added RETURN= (abbreviated R=)
* ... hunt for the first equal sign that is not in square brackets
	brkt = 0
	DO 90 i = istart, iend
	   IF (cmnd(i:i).EQ.'[') THEN
	      brkt = brkt + 1
	   ELSEIF (cmnd(i:i).EQ.']') THEN
	      brkt = brkt - 1
	   ELSEIF (cmnd(i:i).EQ.'=') THEN
	      IF ( brkt .EQ. 0 ) GOTO 99
	   ENDIF
 90	CONTINUE
	GOTO 149	! nope - no equal sign
* ... locate the comma following the mathematical expression
 99	DO 100 i = i, istart, -1
 100	IF ( cmnd(i:i) .EQ. ',' ) GOTO 101
	GOTO 5500			! no comma ???
 101	iend = i - 1
* ... now decode the name/value pairs following the comma
 110	i = i + 1	   ! advance to character following the comma
* ... locate the end of the name/value pair string
	DO 120 i2 = i, true_end   
	   IF ( cmnd(i2:i2) .EQ. ',' ) THEN
	      qual_end = i2 - 1	
	      GOTO 121
	   ENDIF
 120	CONTINUE
	qual_end = true_end
 121	CONTINUE
* ... locate the keyword start and decode the keyword/value pair
	DO 130 i = i, qual_end   
	   IF (cmnd(i:i).EQ.' ' .OR. cmnd(i:i).EQ.tab) THEN
	      CONTINUE
	   ELSEIF (cmnd(i:i).EQ.'p' .OR. cmnd(i:i).EQ.'P') THEN
* ... decode PRECISION = #_digits
	      CALL EQUAL_VAL(cmnd(i:qual_end), result, status)
	      IF ( status .NE. ferr_ok ) GOTO 5000
	      IF (ABS(result) .GT. 10.) GOTO 5510 	         
	      use_digits = NINT(result)
	      GOTO 131
	   ELSEIF (cmnd(i:i).EQ.'b' .OR. cmnd(i:i).EQ.'B') THEN
* ... decode BAD = string
	      CALL EQUAL_STR_LC(cmnd(i:qual_end), bad_str, status)
	      IF ( status .NE. ferr_ok ) GOTO 5000
	      GOTO 131
	   ELSEIF (cmnd(i:i).EQ.'r' .OR. cmnd(i:i).EQ.'R') THEN
* ... decode RETURN = string
	      CALL EQUAL_STRING(cmnd(i:qual_end), repl, status) !repl is buffer
	      IF ( status .NE. ferr_ok ) GOTO 5000
	      info_query = .TRUE.
	      GOTO 131
	   ELSE
	      GOTO 5500
	   ENDIF
 130	CONTINUE
 131	i = qual_end + 1	! comma or beyond end
* ... go back for another keyword?
	IF ( i .LT. true_end ) GOTO 110
* ======

* is the expression just white space?	(1/96 ... quick fix with GOTO's)
 149	DO 150 i = istart, iend
 150	IF ( cmnd(i:i) .NE. ' ' .AND. cmnd(i:i) .NE. tab ) GOTO 200
* ... weird - they gave us pure white space - no numerical result possible
	repl = bad_str
	str_len = 3
	GOTO 500

* evaluate the expression
 200	CALL EVAL_EXPR ( memory, cx_last, cmnd(istart:iend), status )
	IF ( status .NE. ferr_ok ) GOTO 5000
	cx = is_cx(1)		! always comes back as the first context
	mr = is_mr(1)

* is this a single number request or a RETURN= query?
	IF ( info_query ) THEN
* ... RETURN= information request: "repl" contains the keyword
	   IF ( repl .EQ. "SHAPE" ) THEN
	      repl = ' '
	      str_len = 0
	      DO 210 idim = 1, 4
	         IF ( MR_DIM_LEN( idim, mr ) .GT. 1 ) THEN
	            str_len = str_len + 1
	            repl(str_len:str_len) = ww_dim_name(idim)
	         ENDIF
 210	      CONTINUE
	      IF (str_len .EQ. 0) THEN
	         repl = "POINT"
	         str_len = 5
	      ENDIF
	   ELSE	
* ... axis specific requests - must start with i,j,k, or l
	      DO 220 idim = 1, 4
 220	      IF ( repl(1:1) .EQ. ss_dim_name(idim) ) GOTO 221
	      GOTO 5520
 221	      IF ( repl(2:5) .EQ. "SIZE" ) THEN
	         i2 = cx_hi_ss(cx,idim) - cx_lo_ss(cx,idim) + 1
	      ELSEIF ( repl(2:6) .EQ. "START" ) THEN
	         i2 = cx_lo_ss(cx,idim)
	      ELSEIF ( repl(2:4) .EQ. "END" ) THEN
	         i2 = cx_hi_ss(cx,idim)
	      ELSE
	         GOTO 5520
	      ENDIF
	      IF (i2 .EQ. unspecified_int4) i2 = 0
	      repl = LEFINT(i2, str_len)
	   ENDIF
	ELSE
* single number answer desired
* ... make sure the result is a single, valid value
	   IF ( CGRID_SIZE(cx) .NE. 1 ) GOTO 5200	! result not a scalar

* ... extract the result scalar from the (possibly larger) memory variable
	   CALL CREATE_TEMP_MEM_VAR( cx, mr_temp, status )
	   IF ( status .NE. ferr_ok ) RETURN
	   CALL COPY_GRID( memory(1, mr_blk1(mr)     ), mr,
     .                     memory(1, mr_blk1(mr_temp)), mr_temp )
	   result =  memory(1, mr_blk1(mr_temp))
! 9/8/95 BUG.  was:  result = memory(1, mr_blk1(mr))
	   CALL DELETE_VARIABLE( mr_temp )

* .... convert to a formatted result with requested digits
	   IF ( result .EQ. mr_bad_data(mr) ) THEN
	      repl = bad_str
	      str_len = TM_LENSTR1(bad_str)
	   ELSE
	      repl = TM_FMT( result, use_digits, max_len, str_len )
	   ENDIF
	ENDIF

* replace the expression text (incl grave accents) with the formatted string
 500	IF ( (istart-1)+str_len+(lencmnd-true_end-1) .GT. lenbuff ) THEN
           GOTO 5400					! too big to fit
        ELSE
	   IF ( iend+2 .LE. lencmnd ) THEN
	      risc_buff = cmnd(true_end+2:)
	   ELSE
	      risc_buff = ' '
	   ENDIF
           cmnd(istart-1:) = repl(:str_len) // risc_buff
           lencmnd = lencmnd - ((true_end-istart+3) - str_len)
        ENDIF

* successful translation of a grave-accented expression
 1000	did_sub = .TRUE.
	RETURN

* error exits
 5000   RETURN
 5100   CALL ERRMSG( ferr_syntax, status,
     .       'unpaired grave accents: '//cmnd(:lencmnd), *5000)
 5200   CALL ERRMSG( ferr_invalid_command, status,
     .       'grave accent doesnt evaluate to scalar'//pCR
     .                                  //cmnd(istart:iend), *5000)
 5400   CALL ERRMSG( ferr_invalid_command, status,
     .       'grave accent evaluates to string too long'//pCR
     .					//cmnd(istart:iend), *5000)
 5500   CALL ERRMSG( ferr_syntax, status,
     .       'invalid "P=prec", "B=bad", or "R=key" qualifier: '
     .		//cmnd(:lencmnd), *5000)
 5510   CALL ERRMSG( ferr_invalid_command, status,
     .       'maximum precision is 10 digits: '
     .					//cmnd(i:qual_end), *5000)
 5520   CALL ERRMSG( ferr_invalid_command, status,
     .       'RETURN="shape","isize","istart","iend","jsize",... '
     .				//pCR//cmnd(i:qual_end), *5000)
	END
