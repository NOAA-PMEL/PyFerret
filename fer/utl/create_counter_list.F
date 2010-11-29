	SUBROUTINE CREATE_COUNTER_LIST( string, array, nmax, status )
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

* return a list of values for a counter (REPEAT/RANGE=string ) where
! string is the string, e.g. 10:1000:100

* ACM
* As in counter_var_context, without error checking since it has
* already been done in that routine.
* V554: *acm* 1/4

* calling argument declarations
	INTEGER		nmax, status
	REAL		array(nmax)
	CHARACTER*(*)	string

* internal variable declarations:

        INTEGER		TM_LENSTR, ATOM_POS,  slen, 
     .                  colon_pos, colon2_pos, lo_start, lo_end, hi_start, 
     .                  hi_end, dum, xlo, xhi, n
        LOGICAL         colon, colon2, fmat, use_subscripts
        REAL*8          ss_answer		

	include 'ferret.parm'
	include 'errmsg.parm'

* set the limits

        slen = TM_LENSTR(string)

* ... ":"
        lo_start = 1
        colon = .FALSE.
	colon_pos = ATOM_POS( string, ':' )
	lo_end	 = colon_pos - 1

*  ... second ":"

        colon2 = .FALSE.
	colon2_pos = ATOM_POS(string( colon_pos+1:slen ), ':' )
	colon2 = colon2_pos .NE. atom_not_found
	IF ( colon2 ) THEN
	   colon2_pos = colon2_pos + colon_pos
           hi_end     = colon2_pos - 1
	ELSE
	   hi_end     = slen
        ENDIF
	hi_start      = colon_pos + 1

* Call translate_limit to check syntax and set ss limits

        use_subscripts = .TRUE.
        dum = 1

	CALL TRANSLATE_LIMIT
     .		      (  string(lo_start:lo_end), x_dim,
     .			use_subscripts,
     .			ss_answer, fmat, dum, status )
        xlo = INT(ss_answer)

	CALL TRANSLATE_LIMIT
     .		      (  string(hi_start:hi_end), x_dim,
     .			use_subscripts,
     .			ss_answer, fmat, dum, status )
        xhi = INT(ss_answer)

        DO 100 n = xlo, xhi
           array(n) = n
  100   CONTINUE

* success
 1000	status = ferr_ok
	RETURN
        END
