	CHARACTER*(*) FUNCTION TM_FMT_USER( val, digits, max_len, str_len )

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

* calling arguments:
* val		- REAL value to be converted
* digits	- >0, number of significant digits desired
* max_len	- longest allowable length for string to be returned
* str_len	- (returned) actual length of the string generated

* calling argument declarations:
	INTEGER		digits, max_len, str_len
	REAL		val

* internal variable declarations:
	LOGICAL		TM_DFPEQ, exp_fmt, negative, fixed_pt
	INTEGER		slen, sig_left, sig_full, nleft, nright, full,
     .			sig_max, full10, xtra, sig_needed, pow10,
     .			chk10, chk10_save, waste, i, need_adjust, ind,
     .			iloop, width, prec_digits, slenh, right, int_len, is
	REAL		aval, l10
	CHARACTER	fmt*10, dummy_fmt*10, outstring*64, str2*64

* local parameters:
	INTEGER		TM_CHECK_NAN, TM_CHECK_INF, TM_LENSTR1, 
     .                  max_int_len, sig_min, added_pow10
	REAL		eps, one_m_eps
	PARAMETER     ( max_int_len = 8,
     .			sig_min     = 1,		! minimum sig. digits
     .			eps         = 0.6E-7,		! 1/95 - was 1.E-7
     .			one_m_eps   = 1.0 - eps )

* special treatment for exact zero
	IF ( val .EQ. 0.0 ) GOTO 1000

* initialize

	negative = val .LT. 0.0
	fixed_pt = digits .LE. 0
	aval     = ABS( val )
	outstring   = ' '
	exp_fmt  = .FALSE.
	added_pow10 = 0
	need_adjust = 0

* maximum length of output string
* (at least 1 character long but not longer than outstring or user-specified max)
	slen = MIN( LEN(TM_FMT_USER), 64, max_len )
	slen = MAX( 1, slen )

* If the value is NaN, set the string here
        IF ( TM_CHECK_NAN(aval) .EQ. 1   ) THEN
           IF (slen .GE. 3) THEN
              TM_FMT_USER  = 'NaN'
              str_len = 3
           ELSE
              TM_FMT_USER = '*'
              str_len = 1
           ENDIF
	   RETURN
        ENDIF

* If the value is NaN, or Inf, set the string here
        IF ( TM_CHECK_INF(val)  .EQ. 1   ) THEN
           IF (slen .GE. 3) THEN
              outstring  = 'Inf'
              str_len = 3
           ELSE
              outstring = '*'
              str_len = 1
           ENDIF
	   RETURN
        ENDIF

* location of leftmost digit (.01-->-1, 0.1-->0, 1.0-->1, etc.)
 10	l10      = LOG10(aval)
        IF (aval .EQ. 0.) l10 = 0.
	IF ( l10 .GE. 0.0 ) THEN
	   pow10 = INT(l10) + 1
	ELSE
	   pow10 = INT( l10 * one_m_eps )  ! eps for machine round-off problems
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

* modify # significant digits: cannot request more digits than the integer
* word size or the "unwasted" string length
* but at least sig_min digits.

* For double-precision Ferret the code below that puts the digits into a 
* integer would fail because integers cant be that large. Keep track of
* the extra length and add it back later.

        sig_full = MAX( sig_min, MIN( sig_max, slen-waste ) )

        IF (sig_full .GT. max_int_len) THEN
	  need_adjust = sig_full - max_int_len
	ENDIF
        sig_full = MAX( sig_min, MIN( sig_max, max_int_len, slen-waste ) )

* create an integer with all the possible digits
* (1-eps factor is because VAX Fortran formats round 100.5-->100 whereas VAX 
* NINT function rounds 100.5 -->101)
* Note: On DEC/OSF f77 the integer calculation blows up when compiled with -g
*       It was observed with sig_full=4, pow10=-6, aval=6.27833E-07
* 	Kludge work-around is to use the floating point calculation for cases
*	where sig_full+pow10 is too large (10 or larger crashes, expermentally)
! pre-1/95	full = NINT( aval * 10.**(sig_full-pow10) * one_m_eps )
! pre-3/95	full = NINT( aval * (10**(sig_full-pow10)) * one_m_eps )
* 9/95 ... it turns out that for large values (e.g. 2.16E8) that pow10
* can be larger than sig_full givin 10**(negative) --> zero (integer calc)
* so the IF condition has had a check added for this
! 12/95: The value 9.9995 displayed with 4 sig_full=4 generates (F5.3) and
! blows up because 9.9995 rounds to 10.000 on OSF Fortran. This occurs because
! full is computed to be 9999 when the one_m_eps factor is included. Since the
! comments just above indicate that one_m_eps was included to fix a VAX bug
! we have removed this logic to an ifdef: NINT_WRITE_UNMATCH

# ifdef NINT_WRITE_UNMATCH
	IF ( sig_full-pow10 .LT. 10
     .	.AND. sig_full      .GT. pow10 ) THEN
	   full = NINT( aval * (10**(sig_full-pow10)) * one_m_eps )
	ELSE
	   full = NINT( aval * 10.**(sig_full-pow10) * one_m_eps )
	ENDIF
# else
	IF ( sig_full-pow10 .LT. 10
     .	.AND. sig_full      .GT. pow10 ) THEN
	   full = NINT( aval * (10**(sig_full-pow10))  )
	ELSE
	   full = NINT( aval * 10.**(sig_full-pow10)  )
	ENDIF
# endif

* special check: eg. 99.9 --> 100.0 , changing leftmost digit
	IF ( full .EQ. 10**sig_full ) THEN
	   pow10 = pow10 + 1
	   added_pow10 = added_pow10 + 1  ! 

	   GOTO 50
	ENDIF

* get number of digits remaining after trimming off right hand zeros
	full10 = 10 ** (sig_full-sig_min)

c	IF (full10 .EQ. full) THEN  ! value is close to an integer
c	   sig_needed = sig_full
c	   GOTO 200
c	ENDIF

	DO 100 sig_needed = sig_min, sig_full
	   IF ( MOD(full, full10) .EQ. 0 ) GOTO 200
	   full10 = full10 / 10
 100	CONTINUE
	sig_needed = sig_full

* * * * CREATE A FORMAT STRING AND WRITE THE RESULT * * *
 200	CONTINUE
	slenh = MIN( 18, slen/2 )  ! try too with half the allowed lengths
	slenh = MAX( 1, slenh )
	int_len = max_int_len
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
* or with integer parts big enuf to ovflow the integer size (eg.123456789.123)
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

	   aval = NINT(aval)
	   l10      = LOG10(aval)
           IF (aval .EQ. 0.) l10 = 0.
	   IF ( l10 .GE. 0.0 ) THEN
	      chk10 = INT(l10) + 1
	   ELSE
	      chk10 = INT( l10 * one_m_eps )  ! eps for machine round-off problems
	   ENDIF
	   IF ( chk10 .NE. chk10_save ) GOTO 10
	   str_len = nleft - added_pow10
	   iloop = 0
 9000	   IF (str_len .LE. 0) str_len = 1
	   WRITE ( fmt, 3100 ) str_len
 3100	   FORMAT ( '(I',I2,')' )
	   WRITE ( outstring, fmt ) NINT(val)
	   IF (INDEX(outstring,'*') .EQ. 1) THEN
	      str_len = str_len + 1
	      iloop = iloop + 1
	      IF (iloop .LT. 5) GOTO 9000
	   ENDIF

* If they asked for fewer digits than written, return that,
* e.g. 55555 with 2 sig digs becomes 55000.

	   IF (digits.GT.0 .AND. digits.LT.str_len) THEN
	       str2 = outstring
	       aval = NINT(val)
	       aval = aval/ 10**(pow10 - digits)
	       aval = 10**(pow10 - digits)* NINT(aval)
	       WRITE ( outstring, fmt ) NINT(aval)
	       IF (INDEX(outstring,'*') .EQ. 1) outstring = str2
	    ENDIF

* FIXED POINT STRING
	ELSE
	   sig_needed = sig_needed + need_adjust
	   nright = MIN( sig_needed-pow10, slen-(nleft+1) )
	   str_len = nleft + nright + 1
	   WRITE ( fmt, 3200 ) str_len, nright
 3200	   FORMAT ( '(F',I2,'.',I2,')' )
	   WRITE ( outstring, fmt ) val
	   IF (slen .GE. 3) THEN 
	       IF (outstring(1:3) .EQ. ' -.') THEN
		  outstring(1:3) = '-0.'	! 8/96 for aix&hp
	       ELSEIF (outstring(1:1) .EQ. ' ') THEN
                  outstring(1:1) = '0'
	       ENDIF	
	   ENDIF
* Precision in computation of value of FULL can cause an extra
* digit at the end, and so trailing 0s may still exist. Remove them.
           slen = TM_LENSTR1( outstring )
           DO 3300 i = slen,1,-1
              IF (outstring(i:i) .EQ. '0') THEN
                 outstring(i:i) = ' '
                 str_len = str_len - 1
              ELSE
                 GOTO 3400
              ENDIF
 3300      CONTINUE
 3400      CONTINUE   

	ENDIF
	
* Ticket 2323
* the adjustment for handling large values does not always produce 
* accurate results. Compare the result to the original value and if
* needed use the formatting method used in ascii data listings.
* Ticket 2510: make the test more strict differing val and aval.
* Using the VAR_DATA_FMT call takes care of the cases where the values
* are so close that the number will be written as an integer.

	str_len = TM_LENSTR1( outstring )
	READ (outstring,*) aval
	IF (need_adjust.GT.0 .AND. .NOT.TM_DFPEQ(val, aval) ) THEN
	   width = 4
	   prec_digits = ABS(digits)
	   IF (digits .GT. 0) prec_digits = digits
	   CALL GET_SIG_DIGITS( val, prec_digits, nleft, nright )
	   nleft = MAX( 1, nleft )  
	   width = 4
	   CALL VAR_DATA_FMT( nleft, nright, prec_digits, width,
     .                         width, fmt, dummy_fmt )
	   WRITE ( outstring, fmt ) val
	   str_len = TM_LENSTR1( outstring )

* remove leading blanks.
	   ind = 1
	   DO WHILE (outstring(ind:ind) .EQ. ' ') 
              ind = ind+1
	   ENDDO
           IF (ind.GT.1) outstring = outstring(ind:str_len)
	   str_len = TM_LENSTR1( outstring )
	ENDIF
	
	IF (INDEX(outstring, '0E') .GT. 0) THEN
	   ind = INDEX(outstring, '0E')
	   DO WHILE (ind .GT. 0)
	      str2 = outstring(ind+1:str_len)
	      outstring = outstring(1:ind-1)//str2
	      ind = INDEX(outstring, '0E')
	   ENDDO
	ENDIF

	str_len = TM_LENSTR1( outstring )
	IF (INDEX(outstring, '0E') .GT. 0) THEN
	   ind = INDEX(outstring, '0E')
	   DO WHILE (ind .GT. 0)
	      str2 = outstring(ind+1:str_len)
	      outstring = outstring(1:ind-1)//str2
	      ind = INDEX(outstring, '0E')
	   ENDDO
	ENDIF

        
* Remove 00000 from the end of the fractional part of a fixed-point string

	IF (INDEX(outstring, '.') .GT. 0 .AND. INDEX(outstring, 'E').EQ.0) THEN
	   ind = INDEX(outstring, '.')
	   is = str_len
	   DO is = str_len, ind+1, -1
	      IF (outstring(is:is) .NE. '0') EXIT
	   ENDDO
	   str_len = is
	   IF (outstring(is:is) .EQ. '.') str_len = str_len - 1
	ENDIF

	TM_FMT_USER = outstring(:str_len)
	RETURN

* return exact zero
 1000	TM_FMT_USER  = '0'
	str_len = 1
	RETURN

      END
