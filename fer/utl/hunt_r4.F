	SUBROUTINE HUNT_R4 (x1,lo_dim,hi_dim,x2,ndx_lo, frac)

*	Routine to determine nearest neighbor index (lower index)
*	on GRID1 for points on GRID2.

*	Taken from Numerical Recipes, Chapter 3, Section "Search With
*	Correlated Values"

* V450:  7/97 - new, cloned from double prec. version hunt.F
*             - also return "frac" - the interpolation factor for ndx_lo 

	integer		lo_dim,hi_dim,ndx_lo,ndx_mid,ndx_hi,inc

	real		x1(lo_dim:hi_dim),x2, frac

	logical		ascnd

********************************************************************************

*	TRUE IF ASCENDING GRID COORDINATES, FALSE OTHERWISE
	ascnd = x1(hi_dim) .gt. x1(lo_dim)
	
*	IF INITIAL GUESS IS NO GOOD, GO IMMEDIATELY TO BISECTION
	if (ndx_lo .lt. lo_dim .or. ndx_lo .gt. hi_dim) then
	   ndx_lo = lo_dim - 1
	   ndx_hi = hi_dim + 1
	   goto 3
	end if

	inc = 1

*	QUICKLY GET A GOOD ESTIMATE FOR LOCATION IN GRID1
	if (x2 .ge. x1(ndx_lo) .eqv. ascnd) then
*	   HUNT UP FOR PLACEMENT IN GRID
1	   ndx_hi = ndx_lo + inc
	   
	   if (ndx_hi .gt. hi_dim) then 
*	      HUNT OVER -- OFF HIGH END OF GRID
	      ndx_hi = hi_dim + 1
!	   else if (x2 .ge. x1(ndx_hi) .eqv. ascnd) then (ORIGINAL CODE)
	   else if (x2 .gt. x1(ndx_hi) .eqv. ascnd) then
	      ndx_lo = ndx_hi
	      inc    = inc + inc
	      goto 1
	   end if
	else
*	   HUNT DOWN FOR PLACEMENT
	   ndx_hi = ndx_lo
2	   ndx_lo = ndx_hi - inc

	   if (ndx_lo .lt. lo_dim) then
*	      FALL OFF LOW END OF GRID -- HUNT IS OVER
	      ndx_lo = lo_dim - 1
	   else if (x2 .lt. x1(ndx_lo) .eqv. ascnd) then
	      ndx_hi = ndx_lo
	      inc = inc + inc
	      goto 2
	   end if	
	end if

*	USE BISECTION TO DETERMINE EXACT LOCATION IN GRID
3	IF (ndx_hi - ndx_lo .eq. 1) THEN

* return result -- including interpolation fraction

	   IF (ndx_lo.GE.lo_dim .AND. ndx_lo.LT.hi_dim) THEN
	      IF (x2 .EQ. x1(ndx_lo)) THEN
	         frac = 1.0
	      ELSE
	         frac = (x1(ndx_hi)-x2) / (x1(ndx_hi)-x1(ndx_lo))
	      ENDIF
	   ELSE    
	      frac = -999.0
	   ENDIF
	   return
	ENDIF

	ndx_mid = (ndx_hi + ndx_lo)/2
	if (x2 .gt. x1(ndx_mid) .eqv. ascnd) then
	   ndx_lo = ndx_mid
	else
	   ndx_hi = ndx_mid
	end if

	goto 3


	end
