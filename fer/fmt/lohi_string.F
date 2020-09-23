	CHARACTER*(*) FUNCTION LOHI_STRING( lo,hi,idim,grid,
     .                                      to_word,ndec,slen )

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
* create a string describing a world coordinate range
* for example: 160E:140W or 10m (if lo depth = hi)

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 -  1/10/89 - extracted from LOHI_WORLD 0.3
* V200:  2/27/90 - allowed unspecified limits as "???"
* V230:  2/10/92 - SUN port: must use a temporary string, strbuf, instead of
*                  routine name LOHI_STRING internally.
*       *acm* 3/12 cleanup ifdefs and unnecessary include files
* V751 6/19 *acm* TM_FMT rewrite: formatting of lon/lat coordinate strings
* V761 8/20 *acm* Longer LOHI_WORLD, LOHI_STRING strings higher-precision outputs

        IMPLICIT NONE
	include	'ferret.parm'

* calling argument declarations
	INTEGER	idim, grid, ndec, slen
	CHARACTER*(*) to_word
	REAL*8	lo, hi

* internal variable declarations:
	LOGICAL need_more
	INTEGER	STR_SAME, sub_len, full_len
        CHARACTER*80 strbuf       ! needed for SUN FORTRAN

* initialize
	full_len = LEN( LOHI_STRING )
	strbuf = ' '

* always put in lower limit
	IF ( lo .NE. unspecified_val8 ) THEN
	   CALL TRANSLATE_TO_WORLD( lo, idim, grid, ndec, strbuf )
	ELSE
	   strbuf = '???'
	ENDIF
	CALL LEFT_JUST( strbuf, strbuf, slen )
	sub_len = slen

* put in upper limit if different from lower
	IF ( hi .NE. lo ) THEN
	   strbuf = strbuf(:slen) // to_word	! either " to " or ":"
	   slen = MIN( full_len, slen + LEN(to_word) + 1 )
	   CALL TRANSLATE_TO_WORLD( hi, idim, grid, ndec, strbuf(slen:) )

	   need_more = ( STR_SAME(strbuf(1:sub_len) , strbuf(slen:full_len)) .EQ. 0)

	   CALL LEFT_JUST( strbuf(slen:), strbuf(slen:), sub_len )
           slen = MIN( full_len, slen + sub_len - 1 )
	ENDIF
	
        LOHI_STRING = strbuf

	RETURN
	END
