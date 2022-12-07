	SUBROUTINE COPY_GRID_TRANS_SUB( md1, md2,
     .		src, slox, shix, sloy, shiy, sloz, shiz, 
     .		slot, shit, sloe, shie, slof, shif, bad_src,
     .		dst, dlox, dhix, dloy, dhiy, dloz, dhiz, 
     .		dlot, dhit, dloe, dhie, dlof, dhif, bad_dst )

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
* copy data from one grid to another transposing dim1 and dim2 components

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
*
* V200:	10/16/89 - from COPY_GRID_TRANS
*       *acm* 3/12 Add E and F dimensions (use nferdims in tmap_dims.parm)

        IMPLICIT NONE
	include 'ferret.parm'

* calling argument declarations:
* note: dst has dimensions transposed
	INTEGER md1, md2,
     .		slox, shix, sloy, shiy, sloz, shiz, 
     .		slot, shit, sloe, shie, slof, shif,
     .		dlox, dhix, dloy, dhiy, dloz, dhiz, 
     .		dlot, dhit, dloe, dhie, dlof, dhif
	REAL    src( slox:shix,sloy:shiy,sloz:shiz,slot:shit,sloe:shie,slof:shif ),
     .		dst( md2, md1 )
	REAL	bad_src, bad_dst

* internal variable declarations:
	INTEGER	ii, jj, i, j, k, l, m, n 

* --- end of introductory code ---

	ii = 1
	jj = 1
	DO n = dlof, dhif
	DO m = dloe, dhie
	DO l = dlot, dhit
	DO k = dloz, dhiz
	DO j = dloy, dhiy
	DO i = dlox, dhix

	   dst(jj,ii) = src(i,j,k,l,m,n)

	   ii = ii + 1
	   IF ( ii .GT. md1 ) THEN
	      ii = 1
	      jj = jj + 1
	   ENDIF

	ENDDO
	ENDDO
	ENDDO
	ENDDO
	ENDDO
	ENDDO

* check for differing bad data flags in an efficient way ...
	IF ( bad_src .NE. bad_dst ) THEN
	   DO ii = 1, md1
	   DO jj = 1, md2

	      IF ( dst(jj,ii) .EQ. bad_src ) dst(jj,ii) = bad_dst

	   ENDDO
	   ENDDO
	ENDIF

	RETURN
	END 
