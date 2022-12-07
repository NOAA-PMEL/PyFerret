	SUBROUTINE BAD_PTR_EDGES_SUB( cx,
     .		src, slox, shix, sloy, shiy, sloz, shiz, slot, shit, sloe, shie, slof, shif )


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
* context cx contains the limits of the valid data within array X
* fill the edges of array X with bad-data flags

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* V533: 6/01 - like BAD_EDGES_SUB but initialize null pointers
*       *acm* 3/12 Add E and F dimensions (use nferdims in tmap_dims.parm)
* 11/2022 *acm* Code cleanup: Modernize DO-loop syntax. 

        IMPLICIT NONE
	include 'tmap_dims.parm'
	include 'ferret.parm'
	include	'xcontext.cmn'
	include	'xvariables.cmn'
	include	'xmem_subsc.cmn'

* calling argument declarations:
	INTEGER cx, slox, shix, sloy, shiy, sloz, shiz, 
     .		slot, shit, sloe, shie, slof, shif
	REAL*8  src( slox:shix,sloy:shiy,sloz:shiz,slot:shit,sloe:shie,slof:shif )

* internal variable declarations:
	INTEGER	i, j, k, l, m, n, i_lo, j_lo, k_lo, l_lo, 
     .		m_lo, n_lo, i_hi, j_hi, k_hi, l_hi, m_hi, n_hi

* --- end of introductory code ---

* valid limits
	i_lo = cx_lo_s1(cx)
	j_lo = cx_lo_s2(cx)
	k_lo = cx_lo_s3(cx)
	l_lo = cx_lo_s4(cx)
	m_lo = cx_lo_s5(cx)
	n_lo = cx_lo_s6(cx)
	i_hi = cx_hi_s1(cx)
	j_hi = cx_hi_s2(cx)
	k_hi = cx_hi_s3(cx)
	l_hi = cx_hi_s4(cx)
	m_hi = cx_hi_s5(cx)
	n_hi = cx_hi_s6(cx)

* 3/93
* if a low valid limit exceeds a high limit then there must be no valid region
        IF ( i_lo .GT. i_hi
     .  .OR. j_lo .GT. j_hi
     .  .OR. k_lo .GT. k_hi
     .  .OR. l_lo .GT. l_hi
     .  .OR. m_lo .GT. m_hi
     .  .OR. n_lo .GT. n_hi ) THEN

* fill entire result with bad flags
	   DO i = slox, shix
	   DO j = sloy, shiy
	   DO k = sloz, shiz
	   DO l = slot, shit
	   DO m = sloe, shie
	   DO n = slof, shif
	      CALL SET_NULL_C_STRING(src(i,j,k,l,m,n))
 	   ENDDO
 	   ENDDO
 	   ENDDO
 	   ENDDO
 	   ENDDO
 	   ENDDO

           RETURN
        ENDIF

* fill in along X axis
	DO i = slox, i_lo-1
	DO j = j_lo, j_hi
	DO k = k_lo, k_hi
	DO l = l_lo, l_hi
	DO m = m_lo, m_hi
	DO n = n_lo, n_hi
	   CALL SET_NULL_C_STRING(src(i,j,k,l,m,n))
 	ENDDO
 	ENDDO
 	ENDDO
 	ENDDO
 	ENDDO
 	ENDDO

	DO i = i_hi+1, shix
	DO j = j_lo, j_hi
	DO k = k_lo, k_hi
	DO l = l_lo, l_hi
	DO m = m_lo, m_hi
	DO n = n_lo, n_hi
	   CALL SET_NULL_C_STRING(src(i,j,k,l,m,n))
 	ENDDO
 	ENDDO
 	ENDDO
 	ENDDO
 	ENDDO
 	ENDDO

* fill in along Y axis
	DO j = sloy, j_lo-1
	DO i = i_lo, i_hi
	DO k = k_lo, k_hi
	DO l = l_lo, l_hi
	DO m = m_lo, m_hi
	DO n = n_lo, n_hi
	   CALL SET_NULL_C_STRING(src(i,j,k,l,m,n))
 	ENDDO
 	ENDDO
 	ENDDO
 	ENDDO
 	ENDDO
 	ENDDO

	DO j = j_hi+1, shiy
	DO i = i_lo, i_hi
	DO k = k_lo, k_hi
	DO l = l_lo, l_hi
	DO m = m_lo, m_hi
	DO n = n_lo, n_hi
	   CALL SET_NULL_C_STRING(src(i,j,k,l,m,n))
 	ENDDO
 	ENDDO
 	ENDDO
 	ENDDO
 	ENDDO
 	ENDDO

* fill in along Z axis
	DO k = sloz, k_lo-1
	DO i = i_lo, i_hi
	DO j = j_lo, j_hi
	DO l = l_lo, l_hi
	DO m = m_lo, m_hi
	DO n = n_lo, n_hi
	   CALL SET_NULL_C_STRING(src(i,j,k,l,m,n))
 	ENDDO
 	ENDDO
 	ENDDO
 	ENDDO
 	ENDDO
 	ENDDO

	DO k = k_hi+1, shiz
	DO i = i_lo, i_hi
	DO j = j_lo, j_hi
	DO l = l_lo, l_hi
	DO m = m_lo, m_hi
	DO n = n_lo, n_hi
	   CALL SET_NULL_C_STRING(src(i,j,k,l,m,n))
 	ENDDO
 	ENDDO
 	ENDDO
 	ENDDO
 	ENDDO
 	ENDDO

* fill in along T axis
	DO l = slot, l_lo-1
	DO i = i_lo, i_hi
	DO j = j_lo, j_hi
	DO k = k_lo, k_hi
	DO m = m_lo, m_hi
	DO n = n_lo, n_hi
	   CALL SET_NULL_C_STRING(src(i,j,k,l,m,n))
 	ENDDO
 	ENDDO
 	ENDDO
 	ENDDO
 	ENDDO
 	ENDDO

	DO l = l_hi+1, shit
	DO i = i_lo, i_hi
	DO j = j_lo, j_hi
	DO k = k_lo, k_hi
	DO m = m_lo, m_hi
	DO n = n_lo, n_hi
	   CALL SET_NULL_C_STRING(src(i,j,k,l,m,n))
 	ENDDO
 	ENDDO
 	ENDDO
 	ENDDO
 	ENDDO
 	ENDDO

* fill in along E axis
	DO m = sloe, m_lo-1
	DO i = i_lo, i_hi
	DO j = j_lo, j_hi
	DO l = l_lo, l_hi
	DO k = k_lo, k_hi
	DO n = n_lo, n_hi
	   CALL SET_NULL_C_STRING(src(i,j,k,l,m,n))
 	ENDDO
 	ENDDO
 	ENDDO
 	ENDDO
 	ENDDO
 	ENDDO

	DO m = m_hi+1, shie
	DO i = i_lo, i_hi
	DO j = j_lo, j_hi
	DO k = k_lo, k_hi
	DO l = l_lo, l_hi
	DO n = n_lo, n_hi
	   CALL SET_NULL_C_STRING(src(i,j,k,l,m,n))
 	ENDDO
 	ENDDO
 	ENDDO
 	ENDDO
 	ENDDO
 	ENDDO

* fill in along F axis
	DO n = slof, n_lo-1
	DO i = i_lo, i_hi
	DO j = j_lo, j_hi
	DO k = k_lo, k_hi
	DO l = l_lo, l_hi
	DO m = m_lo, m_hi
	   CALL SET_NULL_C_STRING(src(i,j,k,l,m,n))
 	ENDDO
 	ENDDO
 	ENDDO
 	ENDDO
 	ENDDO
 	ENDDO

	DO n = n_hi+1, shif
	DO i = i_lo, i_hi
	DO j = j_lo, j_hi
	DO k = k_lo, k_hi
	DO l = l_lo, l_hi
	DO m = m_lo, m_hi
	   CALL SET_NULL_C_STRING(src(i,j,k,l,m,n))
 	ENDDO
 	ENDDO
 	ENDDO
 	ENDDO
 	ENDDO
 	ENDDO

	RETURN
	END 
