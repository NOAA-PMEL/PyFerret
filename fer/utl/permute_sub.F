	SUBROUTINE PERMUTE_SUB( src_cx, perm,
     .		src, slox, shix, sloy, shiy, sloz, shiz, slot, shit,
     .		dst, dlox, dhix, dloy, dhiy, dloz, dhiz, dlot, dhit )

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
* permute the 4D data from the source grid into the destination grid
* - 24 possible permutations  (!4)

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* V200: 10/16/89 - extracted from PERMUTE
*	 3/16/90 - fixed bug: "src(i,j,l,k)" --> "src(i,j,k,l)"
* V530: 9/00 *sh* - condensed code hugely to perform the same logic

#ifdef unix
	include 'ferret.parm'
	include	'xcontext.cmn'
	include	'xvariables.cmn'
#else
	INCLUDE 'FERRET_CMN:FERRET.PARM'
	INCLUDE	'FERRET_CMN:XCONTEXT.CMN'
	INCLUDE	'FERRET_CMN:XVARIABLES.CMN'
#endif

* calling argument declarations:
	INTEGER src_cx, perm(4),
     .		slox, shix, sloy, shiy, sloz, shiz, slot, shit,
     .		dlox, dhix, dloy, dhiy, dloz, dhiz, dlot, dhit
	REAL    src( slox:shix,sloy:shiy,sloz:shiz,slot:shit ),
     .		dst( dlox:dhix,dloy:dhiy,dloz:dhiz,dlot:dhit )

* internal variable declarations:
	INTEGER	loi, loj, lok, lol, hii, hij, hik, hil, i, j, k, l, idim
	INTEGER	lo(4), hi(4), ii(4)
	EQUIVALENCE
     .		(lo(1), loi), (hi(1), hii), (ii(1), i),
     .		(lo(2), loj), (hi(2), hij), (ii(2), j),
     .		(lo(3), lok), (hi(3), hik), (ii(3), k),
     .		(lo(4), lol), (hi(4), hil), (ii(4), l)

* --- end of introductory code ---

* initialize - save loop limits as unsubscripted variables
	DO 100 idim = 1, 4
	   lo(idim) = cx_lo_ss(src_cx,idim)
	   hi(idim) = cx_hi_ss(src_cx,idim)
 100	CONTINUE

	DO 200  l = lol, hil
	DO 200  k = lok, hik
	DO 200  j = loj, hij
	DO 200  i = loi, hii
	   dst(ii(perm(1)),ii(perm(2)),ii(perm(3)),ii(perm(4))) =
     .		src(i,j,k,l)
 200	CONTINUE

	RETURN
	END 
