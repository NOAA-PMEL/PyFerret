	CHARACTER*(*) FUNCTION LEFT_REAL( val, fmt, len )

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
* create a left-justified real string from val with format fmt
* typical usage:
* i) declare length of function in calling routine 
*	eg. CHARACTER*8 LEFT_REAL
* ii) use as a character variable
*	eg. WRITE( 20,'(1X,A)') LEFT_REAL( x, 2, len )

* note: if value cannot fit within field '****' will be inserted

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written in transportable FORTRAN 77

* revision 0.0 - 7/13/87 modeled after LEFINT rev. 0.0

* calling argument declarations:
	REAL	val
	CHARACTER*(*) fmt
	INTEGER	len

* internal variable declarations:
	CHARACTER*16 buff16
	INTEGER	     first

* create right-justified string of up to 16 digits
	WRITE ( buff16, fmt ) val

* find left-most non-blank
	DO 100 first = 1, 15
	   IF ( buff16(first:first) .NE. ' ' ) GOTO 200
 100	CONTINUE
	first = 16

* insert left-justified string
 200	LEFT_REAL = buff16(first:16)

* measure its length
	DO 300 len = 0, 15
	   IF ( LEFT_REAL( len+1:len+1 ) .EQ. ' ' ) RETURN
 300	CONTINUE

* error - didn't fit
	LEFT_REAL = '****'
	len = 4
	RETURN

	END
