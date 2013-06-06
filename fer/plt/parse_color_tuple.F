       SUBROUTINE PARSE_COLOR_TUPLE(strng, red_frac, grn_frac, 
     .                              blue_frac, opaq_frac, istat)

* NOTE: red_frac, grn_frac, blue_frac, and opaq_frac are explicit REAL*4

*  This software was developed by the Thermal Modeling and Analysis
*  Project(TMAP) of the National Oceanographic and Atmospheric
*  Administrations (NOAA) Pacific Marine Environmental Lab(PMEL),
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
* Parse a color string given as (R,G,B) or (R,G,B,A), where R, G, B,
* and A are integer values in [0,100].  Return the color fractions 
* (values in [0.0,1.0]).
*

        IMPLICIT NONE

* calling argument declarations
        CHARACTER*(*) strng
        REAL*4 red_frac, grn_frac, blue_frac, opaq_frac
        INTEGER istat

        INCLUDE 'ferret.parm'
        INCLUDE 'errmsg.parm'
        INCLUDE 'xrisc_buff.cmn'

* function declarations
        INTEGER TM_LENSTR1

* internal variable declarations
        INTEGER maxlen, nstart, nend

* initialize
        IF ( strng(1:1) .NE. '(' ) GOTO 5100
        maxlen = INDEX(strng, ')')
        IF ( maxlen .LT. 7 ) GOTO 5100

*       Percent red
        nstart = 2
        nend = INDEX(strng(nstart:maxlen), ',')
        IF ( nend .LT. 2 ) GOTO 5100
        nend = nstart + nend - 2
        READ( strng(nstart:nend), *, ERR=5100 ) red_frac
        IF ((red_frac .LT. 0.0).OR.(red_frac .GT. 100.0)) GOTO 5100
        red_frac = red_frac / 100.0

*       Percent green
        nstart = nend + 2
        nend = INDEX(strng(nstart:maxlen), ',')
        IF ( nend .LT. 2 ) GOTO 5100
        nend = nstart + nend - 2
        READ( strng(nstart:nend), *, ERR=5100 ) grn_frac
        IF ((grn_frac .LT. 0.0).OR.(grn_frac .GT. 100.0)) GOTO 5100
        grn_frac = grn_frac / 100.0

*       Percent blue
        nstart = nend + 2
        nend = INDEX(strng(nstart:maxlen), ',')
        IF ( nend .LT. 2 ) THEN
*          Opacity not given; default to 100 percent
           opaq_frac = 1.0
           nend = maxlen - 1
        ELSE
           nend = nstart + nend - 2
        ENDIF
        READ( strng(nstart:nend), *, ERR=5100 ) blue_frac
        IF ((blue_frac .LT. 0.0).OR.(blue_frac .GT. 100.0)) GOTO 5100
        blue_frac = blue_frac / 100.0

*       Percent opacity, if given
        nstart = nend + 2
        IF ( nstart .LT. maxlen ) THEN
           nend = maxlen - 1
           READ( strng(nstart:nend), *, ERR=5100 ) opaq_frac
           IF ((opaq_frac .LT. 0.0).OR.(opaq_frac .GT. 100.0)) GOTO 5100
           opaq_frac = opaq_frac / 100.0
        ENDIF

* success
        istat = ferr_ok
 5000   RETURN

* syntax error
 5100   risc_buff = strng
        CALL ERRMSG(ferr_out_of_range, istat, 
     .              risc_buff(:TM_LENSTR1(risc_buff))
     .              //pCR//'Format is (R,G,B) or (R,G,B,A),'//
     .              'where R,G,B,A are integer values 0-100',
     .              *5000)

        END

