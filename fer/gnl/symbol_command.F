	SUBROUTINE SYMBOL_COMMAND( buffer, clen, did_sub, status )

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
* check to see if any PLOT+ symbols were used and replace the first one with
* the "value" string if so

* replacing only the first allows the calling routine to trap recursions
* and makes the logic simpler ... it is not optimal, however, for performance

* note that additional logic in this routine allows it to detect nested 
* PLOT+ symbols (e.g. "($lab_($lab_title))" where lab_title="4" would first
* translate to "($lab_4)" and then resolve this symbol

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system

* V400: *sh* 6/1/95 - based on ALIAS_COMMAND
* V420: *sh* 10/9/95 - added PPL$XPIXEL and PPL$YPIXEL
*	      11/95 - bug fix: single character symbols not translated
*	       3/96 - allow the same coaching (string substitution) options
*			for editing symbols that exists for numbered
*			"dollar arguments" to GO files
* V430	*kob* 11/96 - Need to avoid passing a concatenated string as
*		      on argument to COACHED_STR_SUB routine for sgi. 
*		      added crptd_cat_argument ifdef and set it to true
*		      for sgi
* Linux Port *kob* 12/96 - had to remove a single quote from a comment line
* 
* v491b2 - *kob* 3/98 - fixed sgi only bug introduced by 11/96 change.  erroneously
*                       reused "clen" argument..changed it to clen2
* v5 *js* 4/99 - yet another Ferret hack to add a byteorder symbol -- value
*                can be BIG or LITTLE

* calling argument declarations:
	LOGICAL	  did_sub
	INTEGER	  clen, status
	CHARACTER buffer*(*)

*11/96 *kob*
#ifdef sgi
#  define crptd_cat_argument
#endif


* internal variable declarations:
	LOGICAL coaching
	INTEGER	STR_UPCASE, lenbuff, spos, epos, nchar, ierr,
     .		wsid, nxpix, nypix, i, i2, coach_start, coach_end, sym_end
	CHARACTER LEFINT*16, symname*30, c1*1,
     .					symval*255	! to match PPLUS

* internal parameter declarations:
	CHARACTER*1     tab
	PARAMETER     ( tab = CHAR(9) )

* 11/96 *kob*
#ifdef crptd_cat_argument
	INTEGER clen2
	CHARACTER cat_argument*48
#endif 

#ifdef unix
	include 'ferret.parm'
	include 'errmsg.parm'
	include 'xcommand.cmn'
	include 'xplot_state.cmn'	! for PPL$XPIXEL
	include 'xrisc.cmn'
* #ifdef for endianness info. Don't you love this code?
#ifdef sun
	include 'xmachine_byte.cmn'
#else
	include 'xmachine_int1.cmn'
#endif
#else
	INCLUDE 'FERRET_CMN:FERRET.PARM'
	INCLUDE 'FERRET_CMN:ERRMSG.PARM'
	INCLUDE 'FERRET_CMN:XCOMMAND.CMN'
	INCLUDE 'FERRET_CMN:XPLOT_STATE.CMN'	! FOR PPL$XPIXEL
	INCLUDE 'FERRET_CMN:XRISC.CMN'
#endif



* initialize
	did_sub = .FALSE.
	status = ferr_ok

* immediate exit if there are no symbols in the line
	spos = INDEX( buffer(:clen), '($' )
	IF ( spos .EQ. 0 ) RETURN

* initialize to continue
	risc_buff = buffer	! ready for error messages (f77 "//" Unix bug)
	lenbuff = LEN( buffer )

* locate the closing parenthesis
* but watching out for a nested "($" as in the commens above
	epos = spos + 2
100	CONTINUE
* ... check for nested symbols
	   IF ( epos .LE. clen-2 ) THEN
	      IF ( buffer(epos:epos+1) .EQ. '($' ) THEN
	         spos = epos	! new starting point
	         epos = epos + 2
	         GOTO 100
	      ENDIF
	   ENDIF
* ... check for closing parenthesis	         
	   IF ( buffer(epos:epos) .EQ. ')' ) GOTO 110
	   epos = epos + 1
	IF ( epos .LE. clen ) GOTO 100
	GOTO 5100	! no closing paren was supplied
 110	sym_end = epos - 1

* 3/96 identify and locate any "coahing" text that accompanies the symbol
* ... look for a character between '!' and '&' in the ASCII table
*     but not a dollar sign
	coaching = .FALSE.
	DO 150 i = spos+2, epos-1
	   c1 =  buffer(i:i)
	   IF (c1.GE.'!' .AND. c1.LE.'&' .AND. c1.NE.'$') THEN
	      coaching = .TRUE.
	      sym_end = i - 1
	      coach_start = i + 1
	      GOTO 151		! break out of loop
	   ENDIF
 150	CONTINUE
 151	CONTINUE

* if coaching text was supplied then locate the coaching end
	IF ( coaching ) THEN
* ... locate the terminating coach character (if supplied)
	   coach_end = epos - 1			! default if no closing char
	   DO 170 i = coach_start, epos-1
	      IF ( c1 .EQ.  buffer(i:i) ) THEN
* ... make sure there is only white space filling out the parens
	        DO 160 i2 = i+1, epos-1
 160	        IF ( buffer(i2:i2) .NE. ' '
     .	      .AND.  buffer(i2:i2) .NE. tab ) GOTO 5300
	        coach_end = i-1
	        GOTO 171
	      ENDIF
 170	   CONTINUE
 171	   CONTINUE
	ENDIF

* translate the symbol name
* ierr=0 means success;   ierr=1 means symbol not found
 	IF ( spos+2 .LE. sym_end ) THEN 	! 11/95 bug fix: "LT" --> "LE"
!	   symname = buffer(spos+2:sym_end)	! change case for PPL$XPIXEL
	   ierr = STR_UPCASE( symname, buffer(spos+2:sym_end) )	! 10/95
	   IF (symname.EQ.'PPL$XPIXEL' .OR. symname.EQ.'PPL$YPIXEL') THEN
* ... determine the current output window (GKS wkstn ID, that is)
              DO 300 wsid = 1, max_windows
                 IF ( wn_active(wsid) ) GOTO 310
 300          CONTINUE
 310          CALL GET_WINDOW_PIXELS(wsid, nxpix, nypix)
	      IF ( symname.EQ.'PPL$XPIXEL' ) THEN
	         symval = LEFINT(nxpix,nchar)
	      ELSE
	         symval = LEFINT(nypix,nchar)
	      ENDIF
*
* Hack to get byteorder of this architecture
*
	   ELSE IF (symname .eq. 'BYTEORDER') THEN
	      if (active_cpu .eq. cptype_sun) then
		 symval = 'BIG'
		 nchar = 3
	      ELSE
		 symval = 'LITTLE'
		 nchar = 6
	      ENDIF
	   ELSE
	      CALL GETSYM(symname, symval, nchar, ierr)
	      IF ( ierr .EQ. 1 ) nchar = 0	! use NULL string if undefined
	   ENDIF
	ELSE
	   nchar = 0
	ENDIF

* perform the coached text substitution, if requested
	IF ( coaching ) THEN
	   IF (coach_start .GE. coach_end) GOTO 5400
*11/96 *kob* - concatenation w/in procedure call corrupted on sgis. !kob 12/96
#ifdef crptd_cat_argument	
	   cat_argument = 'symbol '//buffer(spos+2:sym_end)
	   clen2 = MIN(48,7 + sym_end-spos-1)
	   CALL COACHED_STR_SUB( cat_argument(:clen2),
     .		symval, buffer(coach_start:coach_end), risc_buff,
     .		nchar, status ) 
#else
	   CALL COACHED_STR_SUB( 'symbol '//buffer(spos+2:sym_end),
     .		symval, buffer(coach_start:coach_end), risc_buff,
     .		nchar, status ) 
#endif
	   IF ( status .NE. ferr_ok ) GOTO 5000
	   symval = risc_buff(:nchar)
	ENDIF

* substitute the value text for the symbol name and the enclosing quotes
        IF ( clen + (nchar-(epos-spos+1)) .GT. lenbuff ) THEN
           GOTO 5200					! too big to fit
        ELSE
* ... place the "value" string into risc_buff
	   IF ( nchar .EQ. 0 ) THEN
	      risc_buff = ' '
	      nchar = 1
	   ELSE
	      risc_buff = symval(:nchar)
	   ENDIF
* ... append onto it anything to the right of the symbol in the cmnd buffer
	   IF ( epos .LT. clen ) THEN
	      risc_buff(nchar+1:) = buffer(epos+1:)
	      nchar = nchar + (clen-epos)
	   ENDIF
* ... replace the symbol name and all that is to the right	   
           buffer(spos:) = risc_buff(:nchar)
           clen = (spos-1) + nchar
        ENDIF

* successful substitution
	did_sub = .TRUE.

* error exits
 5000	RETURN
 5100	CALL ERRMSG( ferr_syntax, status, 'no closing symbol paren: '//
     .		risc_buff(spos:clen), *5000)
 5200	CALL ERRMSG( ferr_invalid_command, status,
     .		'symbol substitution makes line too long'//pCR//
     .		symname(:epos-spos-2)//' = '//symval(:nchar), *5000)
 5300	CALL ERRMSG( ferr_syntax, status,
     .		'stray characters at symbol end: '//
     .		risc_buff(spos:epos), *5000)
 5400	CALL ERRMSG( ferr_syntax, status,
     .		'empty argument replacement string: '//
     .		risc_buff(spos:epos), *5000)
	END
