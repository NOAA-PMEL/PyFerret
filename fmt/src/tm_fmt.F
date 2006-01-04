	CHARACTER*(*) FUNCTION TM_FMT( val, digits, max_len, str_len )

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

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Thermal Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.00 - 08/31/88
* 	   0.10 - 7/21/89 - check if NINT increases digits
*    1/27/95 - corrected tendency to put out tons of zeros to the right
*    3/20/95 - the 1/27/95 correction causes this routine to blow up on
*		OSF f77 **only** when compiled with -g
* *sh* 9/95  - precision output bug for large values
*     10/95  - fix missing leading zero on fixed point for some compilers (HP)
*     12/95  - fix bug with 9.9995 displayed as 4 digits: one_m_eps made
*		system dependent
* *kob* 8/96 - check for a negative w/out a leading zero to prevent output
*              looking like "0-.xxx" - for ibm and hp
* *acm* 11/04  incoming value may be NaN (e.g. in a SHOW DATA/FULL if missing 
*              flag is NaN)
* *acm* 11/05  incoming value may be Inf (e.g. in a STAT, if data contains Inf)
* *acm*  1/06  When val is 0, log10(aval) is infinite.  Causes crash on IRIX;
*              take care of val=0 case individually.
* 

* calling arguments:
* val		- REAL*4 value to be converted
* digits	- IF >0 maximum number of significant digits desired
*	        - ELSE  maximum number of decimal places desired
* max_len	- longest allowable length for string to be returned
* str_len	- (returned) actual length of the string generated

* calling argument declarations:
	INTEGER		digits, max_len, str_len
	REAL		val

* internal variable declarations:
	LOGICAL		exp_fmt, negative, fixed_pt
	INTEGER		slen, sig_left, sig_full, nleft, nright, full,
     .			sig_max, full10, xtra, sig_needed, pow10,
     .			chk10, chk10_save, waste
	REAL		aval, l10
	CHARACTER*10	fmt

* local parameters:
	INTEGER		TM_CHECK_NAN, TM_CHECK_INF, max_int_len, sig_min
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
	TM_FMT   = ' '
	exp_fmt  = .FALSE.

* maximum length of output string (at least 1 character long)
	slen = MIN( LEN(TM_FMT), max_len )
	slen = MAX( 1, slen )

* If the value is NaN, set the string here
        IF ( TM_CHECK_NAN(aval) .EQ. 1   ) THEN
           IF (slen .GE. 3) THEN
              TM_FMT  = 'NaN'
              str_len = 3
           ELSE
              TM_FMT = '*'
              str_len = 1
           ENDIF
	   RETURN
        ENDIF

* If the value is NaN, or Inf, set the string here
        IF ( TM_CHECK_INF(val)  .EQ. 1   ) THEN
           IF (slen .GE. 3) THEN
              TM_FMT  = 'Inf'
              str_len = 3
           ELSE
              TM_FMT = '*'
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
* but at lease sig_min digits
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
	   GOTO 50
	ENDIF

* get number of digits remaining after trimming off right hand zeros
	full10 = 10 ** (sig_full-sig_min)
	DO 100 sig_needed = sig_min, sig_full
	   IF ( MOD(full, full10) .EQ. 0 ) GOTO 200
	   full10 = full10 / 10
 100	CONTINUE
	sig_needed = sig_full

* * * * CREATE A FORMAT STRING AND WRITE THE RESULT * * *
* EXACT ZERO - fixed point numbers that would be 0.000...
 200	IF ( fixed_pt .AND. sig_min-pow10 .GT. slen-waste ) THEN
	   GOTO 1000

* EXPONENTIAL STRING
* numbers too small for fixed point (eg. 0.0000025 represented as 0.0)
* or too big for the string length  (eg. 150000 can't fit in 4 spaces)
* or with integer parts big enuf to ovflow the integer size (eg.123456789.123)
* must be written in exponential format
	ELSEIF ( sig_min-pow10 .GT. slen-waste
     .	.OR. nleft .GT. slen 
     .	.OR. nleft .GT. max_int_len ) THEN
	   IF ( negative ) THEN
	      xtra = 6
	   ELSE
	      xtra = 5
	   ENDIF
	   IF ( slen .LE. xtra ) THEN
	      TM_FMT  = '******'
	      str_len = slen
	   ELSE
	      str_len = MIN( slen, sig_needed+xtra)
	      WRITE ( fmt, 3000 ) str_len, str_len-(xtra+1)
 3000	      FORMAT( '(1PG',I2,'.',I2,')' )
	      WRITE ( TM_FMT, fmt ) val
	   ENDIF

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
	   str_len = nleft
	   WRITE ( fmt, 3100 ) str_len
 3100	   FORMAT ( '(I',I2,')' )
	   WRITE ( TM_FMT, fmt ) NINT(val)

* FIXED POINT STRING
	ELSE
	   nright = MIN( sig_needed-pow10, slen-(nleft+1) )
	   str_len = nleft + nright + 1
	   WRITE ( fmt, 3200 ) str_len, nright
 3200	   FORMAT ( '(F',I2,'.',I2,')' )
	   WRITE ( TM_FMT, fmt ) val
	   IF (slen .GE. 3) THEN 
	       IF (TM_FMT(1:3) .EQ. ' -.') THEN
		  TM_FMT(1:3) = '-0.'	! 8/96 for aix&hp
	       ELSEIF (TM_FMT(1:1) .EQ. ' ') THEN
                  TM_FMT(1:1) = '0'
	       ENDIF	
	   ENDIF
	ENDIF
	RETURN

* return exact zero
 1000	TM_FMT  = '0'
	str_len = 1
	RETURN

      END
