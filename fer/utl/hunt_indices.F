	SUBROUTINE HUNT_INDICES (src, ksmin, ksmax,
     .				 bad_src, targ,
     .				 indices, coefs, kdmin, kdmax,
     .				 kdlo_ok, kdhi_ok, ascnd )

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
*	Routine to determine nearest neighbor index (lower index)
*	on GRID1 for points on GRID2.

*	Taken from Numerical Recipes, Chapter 3, Section "Search With
*	Correlated Values"

*       Note that the SRC array may be either ascending or descending order
*       but the TARG array is always ascending

* variable naming:
*	's' refers to the source array -- the incoming list of values
*	'd' refers to the destination arrays -- the outgoing indices and coefs
*	min/max refer to the dimensioned limits of arrays
*	_ok refers to to limits within which the arrays contain valid values

* V690 *sh* 12/13 - derived from Ferret source routine HUNT_R4
*             compute an entire array of indices and coefficient (frac)
*             in a single call.  If exact match, insist on frac=1.0 case.
*             IF reversed (.not.ascnd) return the index and coef for the
*	      lower *coordinate value* point, not the lower index as before

	include 'ferret.parm'

	integer		ksmin, ksmax, kdmin, kdmax, indices(kdmin:kdmax),
     .			kslo_ok,kshi_ok,ndx_lo,ndx_mid,ndx_hi,inc, k,
     .			kdlo_ok, kdhi_ok

	real		src(ksmin:ksmax),
     .			targ(kdmin:kdmax), coefs(kdmin:kdmax),
     .			bad_src, frac

	logical		find_1st_ok, ascnd

*******************************************************************************
* initialize
	kdlo_ok = kdmin
	kdhi_ok = kdmax

*       FIND A CONTIGUOUS BLOCK OF VALID VALUES IN THE src ARRAY
	find_1st_ok = .TRUE.  ! here refers to the input values src
	DO 50 k = ksmin, ksmax
* ... skip invalid values at start of line
	   IF ( find_1st_ok ) THEN
	      IF (src(k) .EQ. bad_src) GOTO 50
	      find_1st_ok = .FALSE.
	      kslo_ok = k
* ... terminate if invalid point encountered after valid points
	   ELSEIF ( src(k) .EQ. bad_src ) THEN
	      kshi_ok = k-1
	      GOTO 60
	   ENDIF
 50	CONTINUE
	IF (find_1st_ok) THEN
	   kdlo_ok = unspecified_int4    !  no input values to hunt among
	   kdhi_ok = unspecified_int4
	   RETURN
	ELSE
	   kshi_ok = ksmax
	ENDIF

*       DEGENERATE CASE? (ONLY A SINGLE POINT TO 'hunt' AMONG)
 60	IF (kshi_ok .EQ. kslo_ok) THEN
	   DO 100 k = kdmin, kdmax
	      indices(k) = kslo_ok
	      IF (targ(k) .eq. src(kslo_ok)) THEN
	         coefs  (k) = 1.0
	         kdlo_ok = k
	         kdhi_ok = k
	      ELSE
	         coefs  (k) = bad_val4
	      ENDIF
 100	   CONTINUE
	   RETURN
	ENDIF

*	TRUE IF ASCENDING GRID COORDINATES, FALSE OTHERWISE
	ascnd = src(kshi_ok) .gt. src(kslo_ok)

* set initial guess for index from array src
	ndx_lo = kslo_ok
	find_1st_ok = .TRUE.  ! now refers to the output coefs

	DO 1000 k = kdmin, kdmax

! this should never happen, givin the initialization value
!*	IF INITIAL GUESS IS NO GOOD, GO IMMEDIATELY TO BISECTION
!	    if (ndx_lo .lt. kslo_ok .or. ndx_lo .gt. kshi_ok) then
!	       ndx_lo = kslo_ok - 1
!	       ndx_hi = kshi_ok + 1
!	       goto 3
!	    end if
 
 	    inc = 1

*	QUICKLY GET A GOOD ESTIMATE FOR LOCATION IN GRID1
	   if (targ(k) .ge. src(ndx_lo) .eqv. ascnd) then
*	   HUNT UP FOR PLACEMENT IN GRID
1	      ndx_hi = ndx_lo + inc
	      
	      if (ndx_hi .gt. kshi_ok) then 
*	      HUNT OVER -- OFF HIGH END OF GRID
	         ndx_hi = kshi_ok + 1
	      else if (targ(k) .gt. src(ndx_hi) .eqv. ascnd) then
	         ndx_lo = ndx_hi
	         inc    = inc + inc
	         goto 1
	      end if
	   else
*	   HUNT DOWN FOR PLACEMENT
	      ndx_hi = ndx_lo
2	      ndx_lo = ndx_hi - inc

	      if (ndx_lo .lt. kslo_ok) then
*	      FALL OFF LOW END OF GRID -- HUNT IS OVER
	         ndx_lo = kslo_ok - 1
	      else if (targ(k) .lt. src(ndx_lo) .eqv. ascnd) then
	         ndx_hi = ndx_lo
	         inc = inc + inc
	         goto 2
	      end if	
	   end if

*	USE BISECTION TO DETERMINE EXACT LOCATION IN GRID
3	   IF (ndx_hi - ndx_lo .eq. 1) THEN

* return result -- including interpolation fraction

	      IF (ndx_lo.GE.kslo_ok .AND. ndx_lo.LT.kshi_ok) THEN
	         IF (targ(k) .EQ. src(ndx_lo)) THEN
	            frac = 1.0
	         ELSEIF (targ(k) .EQ. src(ndx_hi)) THEN
	            ndx_lo = ndx_hi		! 11/13 addition
	            frac = 1.0
	         ELSEIF (.NOT.ascnd) THEN	! 12/13 adtn - reverse-ordered
	            frac = (src(ndx_lo)-targ(k)) / (src(ndx_lo)-src(ndx_hi))
	            ndx_lo = ndx_hi		! hi index is lo position
	         ELSE
	            frac = (src(ndx_hi)-targ(k)) / (src(ndx_hi)-src(ndx_lo))
	         ENDIF
	         find_1st_ok = .FALSE.
	      ELSE    
	         frac = bad_val4
* ... the "ok" limits can include only valid coefficient values
	         IF (find_1st_ok) THEN
	            kdlo_ok = k + 1
	         ELSE
	            IF (k .LE. kdhi_ok) kdhi_ok = k - 1
	         ENDIF
	      ENDIF

* store the result just found
	      indices(k) = ndx_lo
	      coefs(k) = frac

	      GOTO 900
	   ENDIF

	   ndx_mid = (ndx_hi + ndx_lo)/2
	   if (targ(k) .gt. src(ndx_mid) .eqv. ascnd) then
	      ndx_lo = ndx_mid
	   else
	      ndx_hi = ndx_mid
	   end if
   
   	   goto 3

* prepare for next hunt
 900	   IF     (ndx_lo .LT. kslo_ok) THEN
	      ndx_lo = kslo_ok
	   ELSEIF (ndx_lo .GT. kshi_ok) THEN
	      ndx_lo = kshi_ok
	   ENDIF

 1000	CONTINUE

	RETURN
	end
