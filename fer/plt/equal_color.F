	SUBROUTINE EQUAL_COLOR( string, icolor, dfltthick, status )


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
* From a string of the form "name=value", assign the color ID of the 
* color specified in the "value" string.  If the string is just "name",
* color ID one is assigned.
*
* The default thickness of the pen for the color specified is returned
* in dfltthick.  This is for when the color is specified by the pen number.

* V510 *sh* 3/00 - based upon EQUAL_STRING
*      *kob* 5/00 - minor change to declaration of colors array needed to
*                   compile under nag f95 for linux
* V520 6/00 *sh* - introduced "white" as color 19 (only thin available)
*
* 1/03 *kob* g77 port - requires use of risc_buff for concats
* V693+ *sh* 2/15 - added support for _DQ_ quotations
* PyFr *kms*  6/13 - add support for (R,G,B) and (R,G,B,A)
* 
* PyFr *kms*  7/13 - add support for R,G,B and R,G,B,A (no parentheses);
*                    return a color ID and the default pen thickness
*                    instead of a pen ID; white is color ID 7
* 11/2022 *acm* Code cleanup: Modernize DO-loop syntax. 

        IMPLICIT NONE

* calling argument declarations
	INTEGER		icolor, status
        REAL*4          dfltthick
	CHARACTER*(*)	string

* internal parameter declarations
	INTEGER ncolor_string
	PARAMETER (ncolor_string = 7)

* internal variable declarations
	LOGICAL	MATCH4, TM_DIGIT
	INTEGER	TM_LENSTR, TM_LENSTR1, STR_UPCASE, ncolors,
     .          equal_pos, slen, strt, vax_code, len_test, len_mchars
	REAL	val
        REAL*4 redfrac, grnfrac, bluefrac, opaqfrac
	CHARACTER	LEFINT*4, str*32, parenstr*34
	CHARACTER*12    COLORS(ncolor_string)

* internal data declarations
	DATA colors/'BLACK','RED','GREEN',
     .		    'BLUE','LIGHTBLUE','PURPLE','WHITE'/

	include 'ferret.parm'
	include 'errmsg.parm'
	include 'plot_setup.parm'
        include 'tmap_dims.parm'
	include 'xrisc_buff.cmn'
        include 'xprog_state.cmn'
        include 'fgrdel.cmn'

* initialize
	slen	  = TM_LENSTR1( string )
	equal_pos = INDEX( string, '=' )
* in general, the default thickness is thin
        dfltthick = 1.0
        ncolors = mode_arg(pmode_linecolors,1)

	IF ( equal_pos .EQ. 0 ) THEN
	   str = ' '				! "name", alone
	   icolor = 1
	   RETURN

	ELSEIF ( equal_pos .EQ. slen ) THEN	
	   GOTO 5000				! "name ="

	ELSE

* find first non-blank
	   DO strt = equal_pos+1, slen
	      IF ( string(strt:strt) .NE. ' ' ) GOTO 110
 	   ENDDO
	   STOP '=_str'		! cannot occur

* remove quotation marks
 110	   IF ( string(strt:strt) .EQ. '"'
     .	  .AND. string(slen:slen) .EQ. '"' ) THEN
	      strt = strt + 1
	      slen = slen - 1
	   ELSEIF (string(strt:strt) .EQ. '_'
     .	     .AND. string(slen:slen) .EQ. '_' 
     .	     .AND. slen-strt.GE.8            ) THEN
	      IF (string(strt:strt+3) .EQ. '_DQ_'
     .	    .AND. string(slen-3:slen) .EQ. '_DQ_' ) THEN
	         strt = strt + 4
	         slen = slen - 4
	      ENDIF
	   ENDIF

	   IF ( strt .GT. slen ) THEN
	      str = ' '
	   ELSE
	      vax_code = STR_UPCASE( str, string(strt:slen) )
	   ENDIF

	ENDIF

	IF ( TM_DIGIT(str) ) THEN 
           vax_code = INDEX(str,',')
           IF ( vax_code .GT. 1 ) THEN
* ... color by R,G,B,A or R,G,B
*             add the surrounding parentheses expected by PARSE_TUPLE_COLOR
              parenstr = '(' // str // ')'
              CALL PARSE_COLOR_TUPLE(parenstr, redfrac, grnfrac, 
     .                               bluefrac, opaqfrac, status)
              IF ( status .NE. ferr_ok ) RETURN
*             Now create a temporary color using these RGBA values
              CALL FGD_CREATE_TEMP_COLOR(activewindow, icolor, 
     .                        redfrac, grnfrac, bluefrac, opaqfrac)
           ELSE
* ... color and thickness by number
              READ ( str, *, ERR=5000 ) val
              icolor = NINT( val )
              IF ((icolor .LT. 0).OR.(icolor .GE. maxcolorobjs-1)) THEN
                 GOTO 5200
              ELSE IF ( icolor .LE. ncolors ) THEN
*                Pen is thin with the color ID same as pen ID
                 CONTINUE
              ELSE IF ( icolor .LE. 2 * ncolors ) THEN
*                Pen is medium thickness with color ID - ncolors
                 icolor = icolor - ncolors
                 dfltthick = 2.0
              ELSE IF ( icolor .LE. 3 * ncolors ) THEN
*                Pen is thick with color ID - 2 * ncolors
                 icolor = icolor - 2 * ncolors
                 dfltthick = 3.0
              ENDIF
           ENDIF
        ELSEIF ( str(1:1) .EQ. '(' ) THEN
* ... color by (R,G,B,A) or (R,G,B)
           CALL PARSE_COLOR_TUPLE(str, redfrac, grnfrac, 
     .                            bluefrac, opaqfrac, status)
           IF ( status .NE. ferr_ok ) RETURN
*          Now create a temporary color using these RGBA values
           CALL FGD_CREATE_TEMP_COLOR(activewindow, icolor, 
     .                     redfrac, grnfrac, bluefrac, opaqfrac)
	ELSE
* ... compare with allowable colors
           len_test = TM_LENSTR(str)
	   DO icolor = 1, ncolor_string
              len_mchars = TM_LENSTR(colors(icolor)(:4))
	      IF ( MATCH4(str, len_test,
     .                    colors(icolor), len_mchars) ) GOTO 1000
	   ENDDO
	   GOTO 5100   ! unknown color
	ENDIF

* success
 1000	status = ferr_ok
 4999	RETURN

* syntax error
 5000	CALL ERRMSG( ferr_syntax, status, string, *4999 )
 5100	risc_buff = string
	CALL ERRMSG( ferr_invalid_command, status, 
     .		risc_buff(:TM_LENSTR1(risc_buff))//pCR//
     .          'Colors are Black, Red, Green, Blue, '//
     .          'LightBlue, Purple, and White', *4999 )
 5200	risc_buff = string
	CALL ERRMSG( ferr_out_of_range, status, 
     .		risc_buff(:TM_LENSTR1(risc_buff))
     .          //pCR//'Legal range is 0 to '
     .		//LEFINT(maxcolorobjs-2,slen), *4999 )
	END
