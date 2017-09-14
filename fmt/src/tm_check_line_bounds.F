	SUBROUTINE TM_CHECK_LINE_BOUNDS ( line, line_bnds, npts, iaxis, 
     .      reversed, vname, vlen, is_double, regular, ok)


        include 'tmap_dims.parm'
        include 'xtm_grid.cmn_text'
        external xgt_grid_data
c#include "gt_lib.parm"

* argument declarations

	LOGICAL TM_CHECK_BNDS, TM_CHECK_BNDS_CENTERED, 
     .		reversed, is_double, its_irregular, regular, ok
	REAL line(*), line_bnds(*)
	INTEGER npts, vlen, iaxis, bad_bndsid
	CHARACTER*(*) vname

* Local declarations	
	INTEGER i, irev
	REAL tmp8

* if the coordinates were reversed then the box bounds must be, too 
        IF ( reversed  .AND.  npts .GT. 1) THEN
	   irev = 2*npts
           DO 550 i = 1, npts
               tmp8 = line_bnds(i)
	       line_bnds(i) = line_bnds(irev)
	       line_bnds(irev) = tmp8
	       irev = irev - 1
 550       CONTINUE
        ENDIF

* check to see that each data point is contained inside its box,
* and if the bounds are contiguous, warning user if not. 7/27/2017

        IF (.NOT. TM_CHECK_BNDS(line, line_bnds, npts, iaxis, is_double, vname) ) THEN  

           ok = .FALSE.
      
        ELSE  ! Only check for centered bounds if the bounds are valid

* For axes specified with BOUNDS, check whether coordinates are also 
* centered in the boxes, and whether the size of the boxes are equal.
* If there is one point, use upper and lower bound as given in the file 
* to determine line_delta.

           IF (TM_CHECK_BNDS_CENTERED(line, line_bnds, npts, iaxis, is_double)) THEN
* It's actually regular - save it as such
              IF (.NOT. line_regular(iaxis)) THEN
                 line_regular(iaxis) = .TRUE.
                 line_start  (iaxis) = line(1)
                 IF (npts .GT. 1) THEN
                    line_delta  (iaxis) = 
     .                (line(npts) - line(1))/DBLE(npts-1)
                 ELSE
                    line_delta(iaxis) = line_bnds(2)-line_bnds(1)
                 ENDIF
              ENDIF
           ELSE
              line_regular(iaxis) = .FALSE.
           ENDIF
	   ok = .TRUE.
	
        ENDIF  ! valid bounds, also checked centered

* If valid, will convert these bounds to edges back in the calling routine

	RETURN
	END
