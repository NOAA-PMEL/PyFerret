	SUBROUTINE CREATE_COUNTER_LIST( string, array, nmax, status )

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
