	SUBROUTINE HUNT (x1,lo_dim,hi_dim,x2,ndx_lo)

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

*	J Davison JISAO/PMEL/NOAA
*	5.3.88


	integer		lo_dim,hi_dim,ndx_lo,ndx_mid,ndx_hi,inc

	real*8		x1(lo_dim:hi_dim),x2

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
3	if (ndx_hi - ndx_lo .eq. 1) return

	ndx_mid = (ndx_hi + ndx_lo)/2
	if (x2 .gt. x1(ndx_mid) .eqv. ascnd) then
	   ndx_lo = ndx_mid
	else
	   ndx_hi = ndx_mid
	end if

	goto 3


	end
