	CHARACTER*64 FUNCTION TM_FMT_USER( val, digits, max_len, str_len )

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
* max_len in length and with the shortest possible format( unnecerssary zeros
* eliminated )

* Ansley Manke 5/25/2016
*
* Ferret v7. This is a version of tm_fmt for use in returning values in
*            grave-accent expressions with precision set by the user.
*            It applies the # of digits more strictly than TM_FMT, and 
*            returns exponential results consistently.
*Ticket 2510, failures when values are nearly an integer.
* *kms*  v7.5 5/19 Use DNINT instead of NINT when assigned to REAL to avoid integer
*                  overflow issues.  Change full and full10 to REAL*8.
*                  Write integer initially as Fn.0 values, then remove the '.'
*                  Return CHARACTER*64 to simplify life since 64 is max size that
*                  can be handled here. Remove the one_m_eps stuff for the VAX bug.

* calling arguments:
* val		- REAL*8 value to be converted
* digits	- IF >0 maximum number of significant digits desired
*	          ELSE  maximum number of decimal places desired
* max_len	- longest allowable length for string to be returned
* str_len	- (returned) actual length of the string generated

        IMPLICIT NONE

* calling argument declarations:
	INTEGER		digits, max_len, str_len
	REAL*8		val

* functions:
        INTEGER TM_CHECK_NAN, TM_CHECK_INF, TM_LENSTR1

* local parameters:
        INTEGER     sig_min, max_str_len
        PARAMETER ( sig_min     = 1,		! minimum sig. digits
     .              max_str_len = 64 )

* internal variable declarations:
	LOGICAL		negative, fixed_pt
	INTEGER		slen, sig_left, sig_full, nleft, nright,
     .			sig_max, xtra, sig_needed, pow10, added_pow10,
     .			chk10, chk10_save, waste, i, ind,
     .			iloop, width, prec_digits, right, slenh, int_len
	REAL*8		full, full10, aval, l10
	CHARACTER	fmt*10, dummy_fmt*10,
     .                  outstring*(max_str_len), strend*(max_str_len)

* special treatment for exact zero
	IF ( val .EQ. 0.0 ) GOTO 1000

* initialize

	negative = val .LT. 0.0
	fixed_pt = digits .LE. 0
	aval     = ABS( val )
	outstring   = ' '
	added_pow10 = 0

* maximum length of output string
* (at least 1 character long but not longer than outstring or user-specified max)
	slen = MAX( 1, MIN( max_str_len, max_len ) )

* If the value is NaN, set the string here
        IF ( TM_CHECK_NAN(aval) .EQ. 1   ) THEN
           IF (slen .GE. 3) THEN
              outstring  = 'NaN'
              str_len = 3
           ELSE
              outstring = '*'
              str_len = 1
           ENDIF
           TM_FMT_USER = outstring
	   RETURN
        ENDIF

* If the value is Inf, set the string here
        IF ( TM_CHECK_INF(val)  .EQ. 1   ) THEN
           IF (slen .GE. 3) THEN
              outstring  = 'Inf'
              str_len = 3
           ELSE
              outstring = '*'
              str_len = 1
           ENDIF
           TM_FMT_USER = outstring
	   RETURN
        ENDIF

* location of leftmost digit (.01-->-1, 0.1-->0, 1.0-->1, etc.)
 10	IF ( aval .NE. 0.0 ) THEN
           l10 = LOG10(aval)
        ELSE 
           l10 = 0.0
        ENDIF
	IF ( l10 .GE. 0.0 ) THEN
	   pow10 = INT(l10) + 1
	ELSE
	   pow10 = INT(l10)
	ENDIF
	chk10_save = pow10

* compute total significant digits requested
 50	IF ( fixed_pt ) THEN
	   sig_max = pow10 + ABS(digits)
	ELSE
	   sig_max = digits
	ENDIF

* significant digits to left of the decimal point
	sig_left = MAX( pow10, 0 )

* actual characters to the left of the decimal point
	nleft = MAX( sig_left, 1 )
	IF ( negative ) nleft = nleft + 1	! minus sign

* compute "wasted" space needed for minus signs, left zeros and decimal points
	waste = nleft - sig_left			! sign and left zero
	IF ( sig_max .GT. sig_left ) waste = waste + 1	! decimal point
	IF ( pow10 .LT. 0 ) waste = waste + ABS(pow10)	! leading zeros

* modify # significant digits: cannot request more digits than 
* the "unwasted" string length but at least sig_min digits.

        sig_full = MAX( sig_min, MIN( sig_max, slen-waste ) )

* create the nearest integer with all the possible digits

        full = DNINT( aval * (10.0**(sig_full-pow10)) )

* special check: eg. 99.9 --> 100.0 , changing leftmost digit
	IF ( full .EQ. 10.0**sig_full ) THEN
	   pow10 = pow10 + 1
	   added_pow10 = added_pow10 + 1 
	   GOTO 50
	ENDIF

* get number of digits remaining after trimming off right hand zeros
	full10 = 10.0 ** (sig_full-sig_min)
	DO 100 sig_needed = sig_min, sig_full
	   IF ( DMOD(full, full10) .EQ. 0.0 ) GOTO 200
	   full10 = full10 / 10.0
 100	CONTINUE
	sig_needed = sig_full

* * * * CREATE A FORMAT STRING AND WRITE THE RESULT * * *
 200	slenh = MIN( max_str_len, slen/2 )  ! try with half the allowed lengths
	slenh = MAX( 1, slenh )
	int_len = MIN(digits+2, slen-2) 

	xtra = 5
	IF ( negative ) xtra = 6
	IF ( slenh .LE. xtra ) slenh = slen

* EXACT ZERO - fixed point numbers that would be 0.000...
 	IF ( fixed_pt .AND. sig_min-pow10 .GT. slen-waste ) THEN
	   GOTO 1000

* EXPONENTIAL STRING
* numbers too small for fixed point (eg. 0.0000025 represented as 0.0)
* or too big for the string length  (eg. 150000 can't fit in 4 spaces)
* must be written in exponential format
	ELSEIF ( sig_min-pow10 .GT. slen-waste
     .	.OR. nleft .GT. slen 
     .	.OR. nleft .GT. int_len) THEN
	   IF ( negative ) THEN
	      xtra = 6
	   ELSE
	      xtra = 5
	   ENDIF
	   IF ( slen .LE. xtra ) THEN
	      outstring  = '******'
	      str_len = slen
	   ELSE
	      str_len = MIN( slen, sig_needed+xtra)
* Correct for too-small number of slots when writing a number w neg exponent.
              IF (pow10 .LT. 0) str_len = str_len + sig_max
              IF (str_len .GT. slen) str_len = slen
              IF (str_len-(xtra+1) .LT. 0) xtra = 1
              right = MIN(str_len-(xtra+1), digits-1)
              IF (right .LT. 0) right = str_len-(xtra+1)
              WRITE ( fmt, 3000 ) str_len, right
 3000	      FORMAT( '(1PG',I2,'.',I2,')' )
	      WRITE ( outstring, fmt ) val
	   ENDIF

c ...

* EXPONENTIAL STRING
* Using half the allowed lengths. Return exponential for small 
* values and small number of digits
	ELSEIF ( sig_min-pow10 .GT. slenh-waste
     .	.OR. nleft .GT. slenh
     .	.OR. nleft .GT. int_len) THEN
	   
	      str_len = MIN( slenh, sig_needed+xtra)
* Correct for too-small number of slots when writing a number w neg exponent.
              IF (pow10 .LT. 0) str_len = str_len + sig_max
              IF (str_len .GT. slenh) str_len = slenh
              IF (str_len-(xtra+1) .LT. 0) xtra = 1
              right = MIN(str_len-(xtra+1), digits-1)
              IF (right .LT. 0) right = str_len-(xtra+1)
              WRITE ( fmt, 3000 ) str_len, right
	      WRITE ( outstring, fmt ) val
c ...

* INTEGER STRING
* numbers whose needed digits are all on the left (eg. -2.0001 in 4 characters)
* or whose length is too long to add a decimal point (eg. 25.5 in 3 characters)
* may/must be represented as integers
	ELSEIF ( sig_left .GE. sig_needed .OR. nleft .GE. slen-1 ) THEN
* ... check if nearest integer will alter formatting (e.g. 99.6-->100)

	   aval = DNINT(aval)
           IF ( aval .NE. 0.0 ) THEN
              l10  = LOG10(aval)
           ELSE
              l10 = 0.0
           ENDIF
	   IF ( l10 .GE. 0.0 ) THEN
	      chk10 = INT(l10) + 1
	   ELSE
	      chk10 = INT(l10)
	   ENDIF
	   IF ( chk10 .NE. chk10_save ) GOTO 10
	   str_len = nleft - added_pow10
	   iloop = 0
 9000	   IF (str_len .LE. 0) str_len = 1
           IF ( str_len .LT. max_str_len ) str_len = str_len + 1
	   WRITE ( fmt, 3100 ) str_len
*          Write as floating point with no fraction (eg, 25.) to avoid integer overflow
*          The end '.' will be removed later on
 3100	   FORMAT ( '(F',I2,'.0)' )
	   WRITE ( outstring, fmt ) DNINT(val)
	   IF (INDEX(outstring,'*') .EQ. 1) THEN
	      str_len = str_len + 1
	      iloop = iloop + 1
	      IF (iloop .LT. 5) GOTO 9000
	   ENDIF

* If they asked for fewer digits than written, return that,
* e.g. 55555 with 2 sig digs becomes 55000.

	   IF (digits.GT.0 .AND. digits.LT.str_len) THEN
	       strend = outstring
	       aval = DNINT(val)  / 10.0**(pow10-digits)
	       aval = DNINT(aval) * 10.0**(pow10-digits)
	       WRITE ( outstring, fmt ) DNINT(aval)
	       IF (INDEX(outstring,'*') .EQ. 1) outstring = strend
	    ENDIF

* FIXED POINT STRING
	ELSE
	   nright = MIN( sig_needed-pow10, slen-(nleft+1) )
	   str_len = nleft + nright + 1
	   WRITE ( fmt, 3200 ) str_len, nright
 3200	   FORMAT ( '(F',I2,'.',I2,')' )
	   WRITE ( outstring, fmt ) val
	   IF (slen .GE. 3) THEN 
	       IF (outstring(1:3) .EQ. ' -.') THEN
		  outstring(1:3) = '-0.'
	       ELSEIF (outstring(1:1) .EQ. ' ') THEN
                  outstring(1:1) = '0'
	       ENDIF
	   ENDIF
	ENDIF

* Ticket 2323
* the adjustment for handling large values does not always produce 
* accurate results. Compare the result to the original value and if
* needed use the formatting method used in ascii data listings.
* Ticket 2510: make the test more strict differing val and aval.
* Using the VAR_DATA_FMT call takes care of the cases where the values
* are so close that the number will be written as an integer.

* KMS - writing as Fn.0 (with the '.' to be removed below) 
*       takes care of the integer overflow issues

	str_len = TM_LENSTR1( outstring )

* Remove zeros just before the exponent.
	ind = INDEX(outstring, '0E')
	DO WHILE (ind .GT. 0)
	   strend = outstring(ind+1:str_len)
	   outstring = outstring(1:ind-1)//strend
           str_len = str_len - 1
	   ind = INDEX(outstring, '0E')
	ENDDO

* Remove zeros and . at the end of the fractional part of a fixed-point string
	IF (INDEX(outstring, '.') .GT. 0 .AND. 
     .	    INDEX(outstring, 'E') .EQ. 0) THEN
	   slen = str_len
	   DO i = slen,1,-1
	      IF (outstring(i:i) .EQ. '0') THEN
                 outstring(i:i) = ' '
                 str_len = str_len - 1
	      ELSE
                 GOTO 3500
	      ENDIF
	   ENDDO
 3500      IF (outstring(str_len:str_len) .EQ. '.') THEN
              outstring(i:i) = ' '
              str_len = str_len - 1
           ENDIF
	ENDIF

        TM_FMT_USER = outstring
	RETURN

* return exact zero
 1000   outstring = '0'
        str_len = 1
        TM_FMT_USER = outstring
	RETURN

      END
