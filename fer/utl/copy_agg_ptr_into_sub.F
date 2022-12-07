	SUBROUTINE COPY_AGG_PTR_INTO_SUB( aloc, cxlo, cxhi, agg_dim,
     .		src, slox, shix, sloy, shiy, sloz, shiz, 
     .		slot, shit, sloe, shie, slof, shif,
     .		dst, dlox, dhix, dloy, dhiy, dloz, dhiz, 
     .		dlot, dhit, dloe, dhie, dlof, dhif  )

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
* copy data from source grid into dstination grid using limits of source
* context (i.e. destination grid is ASSUMED) big enough to hold data
* Copy the aggregate member into Aggregate-axis location pointed to
* by index aloc.
*
* programmer - Ansley Manke
* NOAA/PMEL, Seattle, WA - Thermal Modeling and Analysis Program
*
* V680 5/12 *acm* 6D Ferret, changes for DEFINE DATA/AGGREGATE
* V695 2/15 *acm* 6D add aggregation in forecast direction
* 11/2022 *acm* Code cleanup: Modernize DO-loop syntax. 
*
        IMPLICIT NONE
	include 'ferret.parm'

* calling argument declarations:
	INTEGER aloc, cxlo(*), cxhi(*), agg_dim,
     .		slox, shix, sloy, shiy, sloz, shiz, 
     .		slot, shit, sloe, shie, slof, shif,
     .		dlox, dhix, dloy, dhiy, dloz, dhiz, 
     .		dlot, dhit, dloe, dhie, dlof, dhif
	REAL*8  src( slox:shix,sloy:shiy,sloz:shiz,slot:shit,sloe:shie,slof:shif ),
     .		dst( dlox:dhix,dloy:dhiy,dloz:dhiz,dlot:dhit,dloe:dhie,dlof:dhif )

* internal variable declarations:
	INTEGER	i, j, k, l, m, n

* --- end of introductory code ---

	IF (agg_dim .EQ. e_dim) THEN
	   m = cxlo(agg_dim)
           DO n = cxlo(6), cxhi(6)
	   DO l = cxlo(4), cxhi(4)
	   DO k = cxlo(3), cxhi(3)
	   DO j = cxlo(2), cxhi(2)
	   DO i = cxlo(1), cxhi(1)

	      CALL COPY_C_STRING( src(i,j,k,l,m,n), dst(i,j,k,l,aloc,n) )

	   ENDDO
	   ENDDO
	   ENDDO
	   ENDDO
	   ENDDO
	ENDIF

	IF (agg_dim .EQ. f_dim) THEN
	   n = cxlo(agg_dim)
           DO m = cxlo(5), cxhi(5)
	   DO l = cxlo(4), cxhi(4)
	   DO k = cxlo(3), cxhi(3)
	   DO j = cxlo(2), cxhi(2)
	   DO i = cxlo(1), cxhi(1)

	      CALL COPY_C_STRING( src(i,j,k,l,m,n), dst(i,j,k,l,m,aloc) )

	   ENDDO
	   ENDDO
	   ENDDO
	   ENDDO
	   ENDDO
	ENDIF

	RETURN
	END 
