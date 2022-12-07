	SUBROUTINE PERMUTE_PTR_SUB( src_cx, perm,
     .		src, slox, shix, sloy, shiy, sloz, shiz, 
     .		slot, shit, sloe, shie, slof, shif,
     .		dst, dlox, dhix, dloy, dhiy, dloz, dhiz, 
     .		dlot, dhit, dloe, dhie, dlof, dhif )

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
* note: can probably be done in a more condensed form using EQUIVALENCES and
* perm to set up the loop indices. PERMUTE_SUB, likewise.

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
*
* V533: 6/01 *sh*
*       *acm* 3/12 Add E and F dimensions (use nferdims in tmap_dims.parm)
* 11/2022 *acm* Code cleanup: Modernize DO-loop syntax. 

        IMPLICIT NONE
	include 'tmap_dims.parm'
	include 'ferret.parm'
	include	'xcontext.cmn'
	include	'xvariables.cmn'

* calling argument declarations:
	INTEGER src_cx, perm(nferdims),
     .		slox, shix, sloy, shiy, sloz, shiz, 
     .		slot, shit, sloe, shie, slof, shif,
     .		dlox, dhix, dloy, dhiy, dloz, dhiz, 
     .		dlot, dhit, dloe, dhie, dlof, dhif
	REAL*8  src( slox:shix,sloy:shiy,sloz:shiz,slot:shit,sloe:shie,slof:shif ),
     .		dst( dlox:dhix,dloy:dhiy,dloz:dhiz,dlot:dhit,dloe:dhie,dlof:dhif )

* internal variable declarations:
	INTEGER	loi, loj, lok, lol, lom, lon, 
     .		hii, hij, hik, hil, him, hin, 
     .		i, j, k, l, m, n, idim
	INTEGER	lo(nferdims), hi(nferdims), ii(nferdims)
	EQUIVALENCE
     .		(lo(1), loi), (hi(1), hii), (ii(1), i),
     .		(lo(2), loj), (hi(2), hij), (ii(2), j),
     .		(lo(3), lok), (hi(3), hik), (ii(3), k),
     .		(lo(4), lol), (hi(4), hil), (ii(4), l),
     .		(lo(5), lom), (hi(5), him), (ii(5), m),
     .		(lo(6), lon), (hi(6), hin), (ii(6), n)

* --- end of introductory code ---

* initialize - save loop limits as unsubscripted variables
	DO idim = 1, nferdims
	   lo(idim) = cx_lo_ss(src_cx,idim)
	   hi(idim) = cx_hi_ss(src_cx,idim)
	ENDDO

	DO n = lon, hin
	DO m = lom, him
	DO l = lol, hil
	DO k = lok, hik
	DO j = loj, hij
	DO i = loi, hii
	   CALL COPY_C_STRING( src(i,j,k,l,m,n),
     .		dst(ii(perm(1)),ii(perm(2)),ii(perm(3)),ii(perm(4)),
     .		ii(perm(5)),ii(perm(6))))
	ENDDO
	ENDDO
	ENDDO
	ENDDO
	ENDDO
	ENDDO

	RETURN
	END 
