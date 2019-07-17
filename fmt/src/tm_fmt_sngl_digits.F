       CHARACTER*64 FUNCTION TM_FMT_SNGL_DIGITS
     .            ( val, digits, max_len, str_len )

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
* convert the value val to a formatted string, left justified, not exceeding
* max_len in length and if digits < 0, have the number of digits after the
* decimal point exactly specified by digits, e.g. for digits=-2, 0 is 0.00. 
* For consistent labels on color keys, called by key.F

* programmer - Ansley Manke, from tm_fmt_digits.F
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
*
* 1/13/2012
* Single-precision version of TM_FMT_DIGITS, to be called from PPLUS.
* *KMS* 5/19 - change to calling TM_FMT and then appending 
*              dot and zeros appropriately if needed
*              Return CHARACTER*64 to simplify life.

* calling arguments:
* val      - REAL*4 value to be converted
* digits   - IF >0 maximum number of significant digits desired
*            ELSE *EXACT* number of decimal places desired
* max_len  - longest allowable length for string to be returned
* str_len  - (returned) actual length of the string generated

        IMPLICIT NONE

* calling argument declarations:
        INTEGER digits, max_len, str_len
        REAL*4  val

* functions:
        CHARACTER*64 TM_FMT
        INTEGER   TM_CHECK_NAN, TM_CHECK_INF

* local parameters:
        INTEGER     max_str_len
        PARAMETER ( max_str_len = 64 )

* internal variable declarations:
        REAL*8    dval
        INTEGER   slen, my_max_len, num_to_add, dotidx, eidx, idx
        CHARACTER*(max_str_len) outstring, strend


        dval = DBLE(val)

* maximum length of output string
* (at least 1 character long but not longer than outstring or user-specified max)
        my_max_len = MAX( 1, MIN( max_str_len, max_len ) )

* Use TM_FMT to create the string
  100   outstring = TM_FMT(dval, digits, my_max_len, slen)

* Check for a non-numeric return
        IF ( (outstring(1:1) .EQ. '*') .OR. (outstring .EQ. 'NaN') 
     .       .OR. (outstring .EQ. 'Inf') ) GOTO 500

* If not requesting an exact number of fraction digits, done
        IF ( digits .GE. 0 ) GOTO 500

* Add dot and zeros appropriately if needed
        num_to_add = ABS(digits)
        dotidx = INDEX(outstring, '.')
        eidx = INDEX(outstring, 'E')
        IF ( eidx .GT. 0 ) THEN
           IF ( dotidx .GT. 0 ) THEN
*             eg, 25.5E-5
              num_to_add = num_to_add - eidx + dotidx + 1
           ELSE
*             eg, 25E-5
*             Add a dot, making sure still have room
              IF ( slen .EQ. max_len ) THEN
                 IF ( my_max_len .EQ. 1 ) GOTO 500
                 my_max_len = my_max_len - 1
                 GOTO 100
              ENDIF
              strend = outstring(eidx:slen)
              outstring(eidx:) = '.'//strend
              eidx = eidx + 1
              slen = slen + 1
           ENDIF
           IF ( num_to_add .GT. 0 ) THEN
              DO idx = 1, num_to_add
*                Add a zero, one at a time, making sure still have room
                 IF ( slen .EQ. max_len ) THEN
                    IF ( my_max_len .EQ. 1 ) GOTO 500
                    my_max_len = my_max_len - 1
                    GOTO 100
                 ENDIF
                 strend = outstring(eidx:slen)
                 outstring(eidx:) = '0'//strend
                 eidx = eidx + 1
                 slen = slen + 1
              ENDDO
           ENDIF
        ELSE
           IF ( dotidx .GT. 0 ) THEN
*             eg, 25.5
              num_to_add = num_to_add - slen + dotidx
           ELSE
*             eg, 25
*             Add a dot, making sure still have room
              IF ( slen .GE. max_len ) THEN
                 IF ( my_max_len .EQ. 1 ) GOTO 500
                 my_max_len = my_max_len - 1
                 GOTO 100
              ENDIF
              slen = slen + 1
              outstring(slen:slen) = '.'
           ENDIF
           IF ( num_to_add .GT. 0 ) THEN
              DO idx = 1, num_to_add
*                Add a zero, one at a time, making sure still have room
                 IF ( slen .GE. max_len ) THEN
                    IF ( my_max_len .EQ. 1 ) GOTO 500
                    my_max_len = my_max_len - 1
                    GOTO 100
                 ENDIF
                 slen = slen + 1
                 outstring(slen:slen) = '0'
              ENDDO
           ENDIF
        ENDIF

  500   str_len = slen
        TM_FMT_SNGL_DIGITS = outstring

        RETURN

        END
 
