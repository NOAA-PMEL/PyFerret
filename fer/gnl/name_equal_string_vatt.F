	SUBROUTINE NAME_EQUAL_STRING_VATT (nambuff, start_pos, status)


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

* decode "NAME = expression" on a command line
! when the name is a varname.attname, do not upcase the attribute name.

* programmer - Ansley Manke - for attribute handling
* Derived from NAME_EQUAL_STRING
*       *acm* 3/12 6D Ferret (common uses nferdims in tmap_dims.parm)

        IMPLICIT NONE
	include	'tmap_dims.parm'
	include	'ferret.parm'
	include 'errmsg.parm'
	include 'xprog_state.cmn'

* calling argument declarations:
	INTEGER		start_pos, status
	CHARACTER*(*) nambuff

* local variable declarations:
	INTEGER STR_UPCASE, pos, i1

	CHARACTER*1 tab
	PARAMETER     ( tab = CHAR(9))

* find name to define
* 2 syntaxes are allowed: "DEFINE VAR NAME=TEXT", "DEFINE VAR NAME TEXT"
* in the latter be careful that "TEXT" may contain "="
	pos = INDEX(cmnd_buff(arg_start(1):len_cmnd), '=')
* ... is it a legit = sign or one used within [] further on in expression ?
	IF ( num_args .GT. 1 ) THEN
	   IF ( pos+arg_start(1)-1 .GT. arg_start(2) ) pos = 0
	ENDIF
	IF ( pos .EQ. 0 ) THEN
* ... no "=" separator
	   nambuff = cmnd_buff( arg_start(1):arg_end(1) )
	   IF ( num_args .LT. 2 ) GOTO 400
	   pos = arg_start(2)
	ELSE
* ... yes "=" separator
	   pos = pos + arg_start(1)	! character after "="
*    ... hunt backwards for end of variable name
	   nambuff = '(none)'
	   DO 101 i1 = pos-2, arg_start(1), -1
	      IF ( cmnd_buff(i1:i1).EQ.' '
     .	      .OR. cmnd_buff(i1:i1).EQ.tab ) THEN
	         CONTINUE
	      ELSE
	         GOTO 102
	      ENDIF
 101	   CONTINUE
	   GOTO 5100
 102	   nambuff = cmnd_buff( arg_start(1):i1 )
*    ... find next non-blank, non-tab after "=" -- start of expression
	   DO 105 pos = pos, len_cmnd
	      IF ( cmnd_buff(pos:pos) .NE. ' ' 
     .	     .AND. cmnd_buff(pos:pos) .NE. tab  ) GOTO 106
 105	   CONTINUE
	   GOTO 400	! no text following equal sign
 106	   CONTINUE
           IF (INDEX(nambuff,'[') .GT. 0)   ! get the rest of the [d=] spec
     .        nambuff = cmnd_buff( arg_start(1):len_cmnd )
	ENDIF
	GOTO 500

* no text follows the name -- this is just a name.
 400	start_pos = 0
	GOTO 1000

* save starting position of the text
 500	start_pos = pos


* success
 1000	status = ferr_ok
	RETURN

* error exit
 5000	RETURN
 5100 	CALL ERRMSG( ferr_syntax, status,
     .			cmnd_buff(:len_cmnd)//' ???', *5000 )


	END

