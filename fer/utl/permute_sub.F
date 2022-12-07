	SUBROUTINE PERMUTE_SUB( src_cx, perm,
     .		src, slox, shix, sloy, shiy, sloz, shiz, 
     .		slot, shit, sloe, shie, slof, shif,
     .		dst, dlox, dhix, dloy, dhiy, dloz, dhiz, 
     .		dlot, dhit, dloe, dhie, dlof, dhif )

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
* permute the 4D data from the source grid into the destination grid
* - 24 possible permutations  (!4)

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* V200: 10/16/89 - extracted from PERMUTE
*	 3/16/90 - fixed bug: "src(i,j,l,k)" --> "src(i,j,k,l)"
*
* v541: 3/02 - *kob* need to replace ifdef 0 because IRIX preprocessor
*                    didn't like it
*       *acm* 3/12 Add E and F dimensions (use nferdims in tmap_dims.parm)
*	           restore code that was ifdef'd out due to a DEC compiler error
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

	REAL    src( slox:shix,sloy:shiy,sloz:shiz,slot:shit,sloe:shie,slof:shif ),
     .		dst( dlox:dhix,dloy:dhiy,dloz:dhiz,dlot:dhit,dloe:dhie,dlof:dhif )

* internal variable declarations:
	INTEGER	loi, loj, lok, lol, lom, lon, hii, hij, 
     .		hik, hil, him, hin, i, j, k, l, m, n, idim
	INTEGER	lo(nferdims), hi(nferdims), ii(nferdims)
	EQUIVALENCE
     .		(lo(1), loi), (hi(1), hii),
     .		(lo(2), loj), (hi(2), hij),
     .		(lo(3), lok), (hi(3), hik),
     .		(lo(4), lol), (hi(4), hil),
     .		(lo(5), lom), (hi(5), him),
     .		(lo(6), lon), (hi(6), hin)

* local parameter declarations
	INTEGER	i_ax, j_ax, k_ax, l_ax, m_ax, n_ax
	PARAMETER  (i_ax=1, j_ax=2, k_ax=3, l_ax=4, m_ax=5, n_ax=6)

* 1/00 *kob* -  the below code was used to upgrade permute_sub.F for
*		v5.30.  However, a bug in the DEC optimizer caused the
*		loop to be unrolled incorrectly, leading to improper
*		results.  It's in included here so as not to lose it, 
*		but stuck in an ifdef which shouldn't be accessed.
* 3/12 * resurrect this code. Much simpler than the replacement code.

	EQUIVALENCE
     .		(ii(1), i),
     .		(ii(2), j),
     .		(ii(3), k),
     .		(ii(4), l),
     .		(ii(5), m),
     .		(ii(6), n)

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
	   dst(ii(perm(1)),ii(perm(2)),ii(perm(3)),ii(perm(4)),
     .	       ii(perm(5)),ii(perm(6))) =
     .		src(i,j,k,l,m,n)
	ENDDO
	ENDDO
	ENDDO
	ENDDO
	ENDDO
	ENDDO

	RETURN
	END 
