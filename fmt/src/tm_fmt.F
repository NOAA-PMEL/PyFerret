      CHARACTER*48 FUNCTION TM_FMT(val, ndigits, max_len, str_len)

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
*  ARE DISCLAIMED. IN NO EVENT SHALL NOAA/PMEL/TMAP BE LIABLE FOR ANY
*  SPECIAL, INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER
*  RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF
*  CONTRACT, NEGLIGENCE OR OTHER TORTUOUS ACTION, ARISING OUT OF OR IN
*  CONNECTION WITH THE ACCESS, USE OR PERFORMANCE OF THIS SOFTWARE.
*
*
* convert the value val to a formatted string, left justified, not exceeding
* max_len in length and with the shortest possible format( unnecerssary zeros
* eliminated )

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Thermal Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.00 - 08/31/88
*          0.10 -  7/21/89 - check if NINT increases digits
*    1/27/95 - corrected tendency to put out tons of zeros to the right
*    3/20/95 - the 1/27/95 correction causes this routine to blow up on
*              OSF f77 **only** when compiled with -g
* *sh* 9/95  - precision output bug for large values
*     10/95  - fix missing leading zero on fixed point for some compilers (HP)
*     12/95  - fix bug with 9.9995 displayed as 4 digits: one_m_eps made
*              system dependent
* *kob* 8/96 - check for a negative w/out a leading zero to prevent output
*              looking like "0-.xxx" - for ibm and hp
* *acm* 11/04  incoming value may be NaN (e.g. in a SHOW DATA/FULL if missing
*              flag is NaN)
* *acm* 11/05  incoming value may be Inf (e.g. in a STAT, if data contains Inf)
* *acm*  1/06  When val is 0, log10(aval) is infinite.  Causes crash on IRIX;
*              take care of val=0 case individually.
* *acm*  V6.1 3/08 further corrections to last fix when writing a value with
*                  negative exponent
* *acm*  V6.5 1/10 Fix bug 1717: If the format is going to be (I0) make it (I1)
* *acm*  V7.2 3/17 Floating-point formatted output could still have trailing zeros
*                  after the decimal point. Remove them.
* *kms*  v7.5 5/19 Use DNINT instead of NINT when assigned to REAL to avoid integer
*                  overflow issues.  Change full and full10 to REAL*8.
*                  Write integer initially as Fn.0 values, then remove the '.'
*                  Return CHARACTER*64 to simplify life since 64 is max size that
*                  can be handled here. Remove the one_m_eps stuff for the VAX bug.
* *kms*  v7.5 5/19 For negative values of ndigit (previously called digit), exactly
*                  that many (as a positive number) digits after the decimal point
*                  are returned.  Revised checks for zero values.
*                  Use and return CHARACTER*48 - more than enough for even REAL*16
* *acm*  v7.5 6/19 Check for fewer digits returned with same accuracy 

* calling arguments:
* val      - REAL*8 value to be converted
* ndigits  - IF positive, maximum number of significant digits desired
*            IF zero, nearest integer value (no decimal point, no exponential)
*            IF negative, exact number of digits after the decimal point (no exponential)
* max_len  - longest allowable length for string to be returned
* str_len  - (returned) actual length of the string generated

        IMPLICIT NONE

* calling argument declarations:
        INTEGER ndigits, max_len, str_len
        REAL*8  val

* functions:
        INTEGER TM_CHECK_NAN, TM_CHECK_INF, TM_LENSTR1

* local parameters:
        INTEGER     sig_min, max_str_len
        PARAMETER ( sig_min = 1, max_str_len = 48 )

* internal variable declarations:
        LOGICAL   TM_FPEQ, negative
        INTEGER   slen, sig_max, pow10p, pow10p_save, sig_left, nleft,
     .            sig_needed, nright, xtra, idx
        REAL*8    aval, l10, full, full10
        CHARACTER frmt*10, outstring*(max_str_len), strend*(max_str_len)

* initialize
        negative     = val .LT. 0.0
        aval         = ABS( val )
        outstring    = ' '

* maximum length of output string
* (at least 1 character long but not longer than outstring or user-specified max)
        slen = MAX( 1, MIN( max_str_len, max_len ) )

* If the value is NaN, set the string here
        IF ( TM_CHECK_NAN(val) .EQ. 1   ) THEN
           IF (slen .LT. 3) GOTO 5000
           outstring  = 'NaN'
           str_len = 3
           TM_FMT = outstring
           RETURN
        ENDIF

* If the value is Inf, set the string here
        IF ( TM_CHECK_INF(val)  .EQ. 1   ) THEN
           IF (slen .LT. 3) GOTO 5000
           outstring  = 'Inf'
           str_len = 3
           TM_FMT = outstring
           RETURN
        ENDIF

* Easy case where the exact number of fractional digits is specified
        IF ( ndigits .LE. 0 ) THEN
*          Just write the value in all space provided with the indicated
*          number of fractional digits, then clean up.
           WRITE ( frmt, 20 ) slen, ABS(ndigits)
   20      FORMAT ( '(F',I2,'.',I2,')' )
           WRITE ( outstring, frmt ) val
           IF ( outstring(1:1) .EQ. '*' ) GOTO 5000
           GOTO 400
        ENDIF

        aval         = ABS( val )
        outstring    = ' '

* The number of significant digits was specified - need to decide how to represent it

*       Take care of exactly zero values
        IF ( aval .EQ. 0.0 ) THEN
           outstring = '0'
           str_len = 1
           TM_FMT = outstring
           RETURN
        ENDIF

        sig_max = ndigits

* location of leftmost digit (.01-->-1, 0.1-->0, 1.0-->1, etc.)
        l10 = LOG10(aval)
        IF ( l10 .GE. 0.0 ) THEN
           pow10p = INT(l10) + 1
        ELSE
           pow10p = INT(l10)
        ENDIF
   50   pow10p_save = pow10p

* significant digits to left of the decimal point
        sig_left = MAX(pow10p, 0)

* actual characters to the left of the decimal point
        nleft = MAX(sig_left, 1)           ! at least a zero
        IF ( negative ) nleft = nleft + 1  ! minus sign

* create the nearest integer with all the possible digits
        full = DNINT(aval * (10.0**(sig_max-pow10p)))
* special check: eg. 99.9 --> 100.0 , changing leftmost digit count
        IF ( ABS(full - 10.0**sig_max) .LT. 1.0E-7 ) THEN
           pow10p = pow10p + 1
           GOTO 50
        ENDIF
* get number of digits remaining after trimming off right hand zeros
        full10 = 10.0**(sig_max-sig_min)
        DO sig_needed = sig_min, sig_max
           IF ( ABS(DMOD(full, full10)) .LT. 1.0E-7 ) GOTO 200
           full10 = full10 / 10.0
        ENDDO
        sig_needed = sig_max
  200   nright = sig_needed - sig_left
        IF ( pow10p .LT. 0 ) nright = nright + ABS(pow10p)  ! leading zeros
        IF ( nright .LT. 0 ) nright = 0

* * * * CREATE A FORMAT STRING AND WRITE THE RESULT * * *

* compute the length if written as fixed point with unlimited string length
        str_len = nleft + 1 + nright

* EXPONENTIAL STRING
* numbers that are:
*    too big for the string length  (eg. 150000 can't fit into 4 spaces),
*    too small for fixed point (eg. 0.0000025 with 2 sig. digits), or
*    whose integer form gives too many sig. digits (eg. 555555 with 3 sig. digits)
* must be written in exponential format.
* Values in (10.0,0.1] should never be using exponential format.
* Values in (1000.0,0.001] should not use exponential format unless the
* fixed point form would give too many significant digits.
        IF ( ( (ABS(l10) .GE. 3.0) .OR. 
     .         (ABS(l10) .GE. 1.0) .AND. (ndigits .LT.3) ) .AND.
     .       ( (str_len .GT. slen) .OR.
     .         (nright .GT. sig_max) .OR.
     .         (sig_left .GT. sig_max) ) ) THEN
*          Absolute minimum needed: <n>.E+<n>, <n>.E-<n>
           IF ( negative ) THEN
              xtra = 6
           ELSE
              xtra = 5
           ENDIF
           IF ( slen .LE. xtra ) GOTO 5000
           str_len = MIN(slen, sig_needed+xtra)
           IF (pow10p .LT. 0) str_len = str_len + sig_max
           IF (str_len .GT. slen) str_len = slen
           nright = MAX(0, ndigits - 1)
           IF (str_len-(xtra+1) .LT. 0) xtra = 1
           nright = MIN(nright, str_len-(xtra+1))
*          Write out the exponential form
           WRITE ( frmt, 300 ) str_len, nright
  300      FORMAT( '(1PG',I2,'.',I2,')' )
           WRITE ( outstring, frmt ) val
           IF ( outstring(1:1) .EQ. '*' ) GOTO 5000

* INTEGER STRING
* numbers whose needed digits are all on the left (eg. -2.0001 in 4 characters)
* or whose length is too long to add a decimal point (eg. 25.5 in 3 characters)
* must be represented as integers
        ELSEIF ( (sig_left .GE. sig_needed) .OR.
     .           (nleft .GE. slen-1) ) THEN

*          get the nearest integer as a REAL*8
           aval = DNINT(aval)
           IF ( aval .EQ. 0.0 ) THEN
              outstring = '0'
              str_len = 1
              TM_FMT = outstring
              RETURN
           ENDIF
*          check if this integer value will alter formatting (e.g. 99.6-->100)
           l10  = LOG10(aval)
           IF ( l10 .GE. 0.0 ) THEN
              pow10p = INT(l10) + 1
           ELSE
              pow10p = INT(l10)
           ENDIF
           IF ( pow10p .NE. pow10p_save ) GOTO 50

           str_len = nleft
*          increment the length for the decimal point that will be removed
           IF ( str_len .LT. max_str_len ) str_len = str_len + 1
           WRITE ( frmt, 310 ) str_len
*          Write the integer form as floating point with no fraction (eg, 25.) to
*          avoid integer overflow.  The end decimal point will be removed later
  310      FORMAT ( '(F',I2,'.0)' )
           WRITE ( outstring, frmt ) DNINT(val)
           IF ( outstring(1:1) .EQ. '*' ) GOTO 5000

* FIXED POINT STRING
        ELSE
           WRITE ( frmt, 320 ) str_len, nright
  320      FORMAT ( '(F',I2,'.',I2,')' )
           WRITE ( outstring, frmt ) val
           IF ( outstring(1:1) .EQ. '*' ) GOTO 5000
        ENDIF

* Clean up the string

  400   str_len = TM_LENSTR1( outstring )

* Remove fractional zeros just before the exponent
* (but leave a trailing decimal point)
        idx = INDEX(outstring, '0E')
        DO WHILE (idx .GT. 1)
           strend = outstring(idx+1:str_len)
           outstring = outstring(1:idx-1)//strend
           str_len = str_len - 1
           idx = INDEX(outstring, '0E')
        ENDDO

* Remove fractional zeros (if not required) and
* trailing decimal point from non-exponentials
        IF ( (ndigits .GE. 0) ) CALL TM_CLEANUP_NUM_STRING (outstring, str_len)

* Add a zero before a floating point value with no integer part
        IF ( outstring(1:3) .EQ. ' -.' ) THEN
           outstring(1:3) = '-0.'
        ELSEIF ( outstring(1:2) .EQ. ' .' ) THEN
           outstring(1:2) = '0.'
        ELSEIF ( (str_len .LT. slen) .AND. (outstring(1:2) .EQ. '-.') ) THEN
           strend = outstring(2:str_len)
           outstring = '-0'//strend
           str_len = str_len + 1
        ELSEIF ( (str_len .LT. slen) .AND. (outstring(1:1) .EQ. '.') ) THEN
           strend = outstring(1:str_len)
           outstring = '0'//strend
           str_len = str_len + 1
        ENDIF

* Remove any initial spaces
        DO WHILE ( (str_len .GT. 1) .AND. (outstring(1:1) .EQ. ' ') )
           strend = outstring(2:str_len)
           outstring = strend
           str_len = str_len - 1
        ENDDO

        TM_FMT = outstring
        RETURN

* Error return of '******' (trimmed down to slen length)
 5000   str_len = 0
        DO WHILE ( (str_len .LT. 6) .AND. (str_len .LT. slen) )
           str_len = str_len + 1
           outstring(str_len:str_len) = '*'
        ENDDO
        TM_FMT = outstring
        RETURN

      END
