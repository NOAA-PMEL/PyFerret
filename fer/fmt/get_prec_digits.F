	SUBROUTINE GET_PREC_DIGITS( lo_val, hi_val, small_val, places, right )

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
* determine the number of characters and the number of decimal places needed
* to represent the range of data from lo_val to hi_val with precision small_val

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 - 8/18/87
* revision 0.1 - 7/21/88  - eliminated LOG(0) error in big_10 calculation
* V580 *acm* 12/04 Fix bug 1129; formatting axis for small lon/lat ranges
*                  Note this changes some LIST output by removing a space.

* calling argument declarations:
	INTEGER		places, right
	REAL		small_val
	REAL*8		lo_val, hi_val

* internal variable declarations:
	INTEGER		big_10, left
	REAL		big

*  left	 - number of digits needed left of decimal
*  right - number of digits needed right of decimal

* decimal places to resolve small_val
	IF ( small_val .GE. 1.0 ) THEN
	   right = 0
	ELSE
	   right = 1 - INT( LOG10(small_val) )

* Change the rounding; if small_val is exactly a power of 10, 
* computtation just abvoe is incorrect. Say if small_val is 0.01, 
* then the computation of right says that 3 digits to the right 
* needed; should be 2.
           IF (INT( LOG10(small_val) ) .EQ. LOG10(small_val) )
     .        right = right - 1
	ENDIF

* integer places to represent the largest label
	big = MAX( ABS(hi_val), ABS(lo_val) )
	IF ( big .EQ. 0 ) THEN
	   left = 1
	ELSE
	   big_10  = INT( LOG10( 1.01*big ) )
	   IF ( big_10 .GT. 0 ) THEN
	      left = big_10 + 1
	   ELSE
	      left = 1			! 0.xxx
	   ENDIF
	   IF ( lo_val .LT. 0 ) left = left + 1	! minus sign
	ENDIF

* total places
	IF ( right .EQ. 0 ) THEN
	   places = left		! xxxx
	ELSE
	   places = left + 1 + right	! xx.xxx
	ENDIF

	RETURN
      END
