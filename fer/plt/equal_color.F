	SUBROUTINE EQUAL_COLOR( string, icolor, status )


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
* from a string of the form "name=value", return the "value" string
* if the string is simply "name" then return ' ' (blank)
* change the string to upper case

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program

* V510 *sh* 3/00 - based upon EQUAL_STRING
*      *kob* 5/00 - minor change to declaration of colors array needed to
*                   compile under nag f95 for linux
* V520 6/00 *sh* - introduced "white" as color 19 (only thin available)
*
* 1/03 *kob* g77 port - requires use of risc_buff for concats
*
* calling argument declarations
	INTEGER		icolor, status
	CHARACTER*(*)	string

* internal parameter declarations
	INTEGER ncolors
	PARAMETER (ncolors = 7)

* internal variable declarations
	LOGICAL	MATCH4, TM_DIGIT
	INTEGER	TM_LENSTR1, STR_UPCASE, equal_pos, slen, strt, vax_code,
     .          TM_LENSTR, len_test, len_mchars
	REAL	val
*	CHARACTER	LEFINT*4, str*12, colors*12(6)
	CHARACTER	LEFINT*4, str*12
	CHARACTER*12    COLORS(ncolors)

* internal data declarations
	DATA colors/'BLACK','RED','GREEN',
     .		    'BLUE','LIGHTBLUE','PURPLE','WHITE'/


	include 'ferret.parm'
	include 'errmsg.parm'
	include 'plot_setup.parm'
	include 'xrisc_buff.cmn'
* initialize
	slen	  = TM_LENSTR1( string )
	equal_pos = INDEX( string, '=' )

	IF ( equal_pos .EQ. 0 ) THEN
	   str = ' '				! "name", alone
	   icolor = 1
	   RETURN

	ELSEIF ( equal_pos .EQ. slen ) THEN	
	   GOTO 5000				! "name ="

	ELSE

* find first non-blank
	   DO 100 strt = equal_pos+1, slen
 100	   IF ( string(strt:strt) .NE. ' ' ) GOTO 110
	   STOP '=_str'		! cannot occur

* remove quotation marks
 110	   IF ( string(strt:strt) .EQ. '"' ) strt = strt + 1
	   IF ( string(slen:slen) .EQ. '"' ) slen = slen - 1

	   IF ( strt .GT. slen ) THEN
	      str = ' '
	   ELSE
	     vax_code = STR_UPCASE( str, string(strt:slen) )
	   ENDIF

	ENDIF

	IF ( TM_DIGIT(str) ) THEN 
* ... color and thickness by number
           READ ( str, *, ERR=5000 ) val
	   IF ( val.LE. 0.0 .OR. val.GT.num_fancy_lines+1 ) GOTO 5200
	   icolor = NINT( val )	! user-specified line style
	 ELSE
* ... compare with allowable colors
           len_test = TM_LENSTR(str)
	   DO 200 icolor = 1, ncolors
              len_mchars = TM_LENSTR(colors(icolor)(:4))
	      IF ( MATCH4(str, len_test,
     .             colors(icolor),len_mchars ) ) GOTO 210
 200	   CONTINUE
	   GOTO 5100   ! unknown color
 210	   IF (icolor .EQ. 7) icolor = 19   ! WHITE
	ENDIF

* success
 1000	status = ferr_ok
 4999	RETURN

* syntax error
 5000	CALL ERRMSG( ferr_syntax, status, string, *4999 )
 5100	risc_buff = string
	CALL ERRMSG( ferr_invalid_command, status, 
     .		risc_buff(:TM_LENSTR1(risc_buff))//pCR//
     .	'Colors are Black,Red,Green,Blue,LightBlue,Purple, and White',
     .		*4999 )
 5200	risc_buff = string
	CALL ERRMSG( ferr_out_of_range, status, 
     .		risc_buff(:TM_LENSTR1(risc_buff))
     .          //pCR//'Legal range is 1 to '
     .		//LEFINT(num_fancy_lines+1,slen), *4999 )
	END
