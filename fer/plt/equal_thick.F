        SUBROUTINE EQUAL_THICK(strng, dfltthick, icolor, ipen, istat)


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
* From a string in strng of the form "thick=value", giving the pen 
* thickness, and using the color ID in icolor, assign the pen ID to 
* ipen.  If strng is blank, the default thickness in dfltthick is used 
* for the thickness.  (The name "thick" is not checked, only the value.)
* If no "=value" is given, a value of 2.0 is used.
*
* If icolor is negative (presumably QUAL_ON, but maybe -1), then
* only a thickness of 1.0, 2.0, or 3.0 (or no "=value", which becomes
* 2.0) is accepted.  In these cases, ipen = icolor, icolor-10, or icolor-20
*
* 9/15/2021 *ACM* clarify error message for /thick setting
* 11/2022 *acm* Code cleanup: Modernize DO-loop syntax. 

        IMPLICIT NONE

        INCLUDE 'ferret.parm'
        INCLUDE 'errmsg.parm'
        INCLUDE 'plot_setup.parm'
        INCLUDE 'tmap_dims.parm'
        INCLUDE 'xrisc_buff.cmn'
        INCLUDE 'xprog_state.cmn'
        INCLUDE 'fgrdel.cmn'

* calling argument declarations
        INTEGER icolor, ipen, istat
        REAL*4  dfltthick
        CHARACTER*(*) strng

* external function declarations
        LOGICAL TM_DIGIT
        INTEGER TM_LENSTR, TM_LENSTR1

* internal variable declarations
        INTEGER slen, equal_pos, strt, numlinecolors
        REAL*4  thickness


* check if a thickness value is specified
        slen = TM_LENSTR(strng)
        IF ( slen .LE. 0 ) THEN
*          blank string - use default thickness
           thickness = dfltthick
           GOTO 200
        ENDIF

        equal_pos = INDEX(strng, '=')
        IF ( equal_pos .EQ. 0 ) THEN
*          "/thick" - make it a thicker line
           thickness = 2.0
           GOTO 200
        ENDIF

* find first non-blank after the equals sign
        IF ( equal_pos .EQ. slen ) THEN        
*          "/thick=" - syntax error
           GOTO 5010
        ENDIF
        DO strt = equal_pos+1, slen
           IF ( strng(strt:strt) .NE. ' ' ) GOTO 110
 	ENDDO
*       should never get here because of earlier checks
        GOTO 5010

*       remove quotation marks
 110    IF ( strng(strt:strt) .EQ. '"' ) strt = strt + 1
        IF ( strng(slen:slen) .EQ. '"' ) slen = slen - 1
        IF ( strt .GT. slen ) THEN
*          /thick="" - treat the same as /thick
           thickness = 2.0
           GOTO 200
        ENDIF

* get the numeric thickness value specified
        IF ( .NOT. TM_DIGIT(strng(strt:slen)) ) GOTO 5020
        READ ( strng(strt:slen), *, ERR=5010 ) thickness
*       cannot be negative; zero is okay
        IF ( thickness .LT. 0.0 ) GOTO 5020

* check the special case of negative icolor values
 200    IF ( icolor .LT. 0 ) THEN
           IF ( ABS(thickness - 1.0) .LT. 1.0E-4 ) THEN
              ipen = icolor
              GOTO 4900
           ENDIF
           IF ( ABS(thickness - 2.0) .LT. 1.0E-4 ) THEN
              ipen = icolor - 10
              GOTO 4900
           ENDIF
           IF ( ABS(thickness - 3.0) .LT. 1.0E-4 ) THEN
              ipen = icolor - 20
              GOTO 4900
           ENDIF
*          thicknesses must be 1.0, 2.0, or 3.0 in these cases
           GOTO 5030
        ENDIF

* check if this is one of the standard pens
        numlinecolors = mode_arg(pmode_linecolors,1)
        IF ( (icolor .GE. 1) .AND. (icolor .LE. numlinecolors) ) THEN
           IF ( ABS(thickness - 1.0) .LT. 1.0E-4 ) THEN
              ipen = icolor
              GOTO 4900
           ENDIF
           IF ( ABS(thickness - 2.0) .LT. 1.0E-4 ) THEN
              ipen = icolor + numlinecolors
              GOTO 4900
           ENDIF
           IF ( ABS(thickness - 3.0) .LT. 1.0E-4 ) THEN
              ipen = icolor + 2 * numlinecolors
              GOTO 4900
           ENDIF
        ENDIF

* create a temporary pen using this custom combination of color and thickness
        CALL FGD_CREATE_TEMP_PEN(activewindow, ipen, icolor, thickness)

* success
 4900   istat = ferr_ok
 5000   RETURN

* syntax error
 5010   CALL ERRMSG(ferr_syntax, istat, strng, *5000)

* invalid value
 5020   risc_buff = strng
        CALL ERRMSG(ferr_out_of_range, istat, 
     .              risc_buff(:TM_LENSTR1(risc_buff))//
     .              pCR//'Must be a non-negative number',
     .              *5000)

* restricted values for automatic colors
 5030   risc_buff = strng
        CALL ERRMSG(ferr_out_of_range, istat, 
     .              risc_buff(:TM_LENSTR1(risc_buff))//
     .              pCR//'When color is not specified or for ribbon plot '//
     .              'only 1, 2, or 3 is acceptable',
     .              *5000)

        END
