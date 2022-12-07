	SUBROUTINE COPY_PTR_INTO_MAPPED_SUB( cxs, axis, hilim,
     .		sorted_indices, index_map,
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
* copy string data from source grid into dstination grid using limits of source
* context (i.e. dstination grid is ASSUMED) big enough to hold data

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
*
* V533 *sh* 6/01
*       *acm* 3/12 Add E and F dimensions (use nferdims in tmap_dims.parm)
* 11/2022 *acm* Code cleanup: Modernize DO-loop syntax. 

        IMPLICIT NONE
	include 'tmap_dims.parm'
	include 'ferret.parm'
	include	'xcontext.cmn'

* calling argument declarations:
	INTEGER cxs, axis, hilim,
     .		slox, shix, sloy, shiy, sloz, shiz, 
     .		slot, shit, sloe, shie, slof, shif,
     .		dlox, dhix, dloy, dhiy, dloz, dhiz, 
     .		dlot, dhit, dloe, dhie, dlof, dhif 
	REAL	sorted_indices(hilim), index_map(hilim)
	REAL*8  src( slox:shix,sloy:shiy,sloz:shiz,slot:shit,sloe:shie,slof:shif ),
     .		dst( dlox:dhix,dloy:dhiy,dloz:dhiz,dlot:dhit,dloe:dhie,dlof:dhif )

* internal variable declarations:
	INTEGER	i, j, k, l, m, n, pointer, src_index, dst_index, isorted,
     .		src_index_2

* --- end of introductory code ---

* initialize
* ... skip missing value flags at the upper end of the sorted_indices
	DO pointer = hilim, 1, -1
	   IF (sorted_indices(pointer) .NE. bad_val4) GOTO 50
	ENDDO
	RETURN	! all of the indices are missing values


**** Mapped copy along X axis
 50	IF ( axis .EQ. x_dim ) THEN

* ... initialize
	   src_index = cx_hi_s1(cxs)
* ... if the input data has no range (-999:-999) then match on sorted_indices=1
	   IF (src_index .EQ. unspecified_int4) THEN
	      src_index_2 = 1
	   ELSE
	      src_index_2 = src_index
	   ENDIF

 100	   CONTINUE	! find where src_index belongs in the result
	   DO isorted = pointer, 1, -1

	      IF (src_index_2 .EQ. sorted_indices(isorted)) THEN

* ... copy a source point to the destination
	         dst_index = index_map(isorted)
	         DO n = cx_lo_s6(cxs), cx_hi_s6(cxs)
	         DO m = cx_lo_s5(cxs), cx_hi_s5(cxs)
	         DO l = cx_lo_s4(cxs), cx_hi_s4(cxs)
	         DO k = cx_lo_s3(cxs), cx_hi_s3(cxs)
	         DO j = cx_lo_s2(cxs), cx_hi_s2(cxs)
		    CALL COPY_C_STRING( src(src_index,j,k,l,m,n),
     .				     dst(dst_index,j,k,l,m,n) )
	         ENDDO
	         ENDDO
	         ENDDO
	         ENDDO
	         ENDDO

	      ELSEIF (src_index .GT. sorted_indices(isorted)) THEN

	         src_index   = src_index   - 1
	         src_index_2 = src_index_2 - 1
	         IF ( src_index .LT. cx_lo_s1(cxs) ) THEN
	            RETURN	! that's all this comonent is gonna supply
	         ELSE
	            pointer = isorted	! (just for efficiency)
	            GOTO 100
	         ENDIF

	      ENDIF

	   ENDDO

	ELSEIF ( axis .EQ. y_dim ) THEN
**** Mapped copy along Y axis

* ... initialize
	   src_index = cx_hi_s2(cxs)
* ... if the input data has no range (-999:-999) then match on sorted_indices=1
	   IF (src_index .EQ. unspecified_int4) THEN
	      src_index_2 = 1
	   ELSE
	      src_index_2 = src_index
	   ENDIF

 200	   CONTINUE	! find where src_index belongs in the result
	   DO isorted = pointer, 1, -1

	      IF (src_index_2 .EQ. sorted_indices(isorted)) THEN

* ... copy a source point to the destination
	         dst_index = index_map(isorted)
	         DO n = cx_lo_s6(cxs), cx_hi_s6(cxs)
	         DO m = cx_lo_s5(cxs), cx_hi_s5(cxs)
	         DO l = cx_lo_s4(cxs), cx_hi_s4(cxs)
	         DO k = cx_lo_s3(cxs), cx_hi_s3(cxs)
	         DO i = cx_lo_s1(cxs), cx_hi_s1(cxs)
		    CALL COPY_C_STRING( src(i,src_index,k,l,m,n),
     .				     dst(i,dst_index,k,l,m,n) )
	         ENDDO
	         ENDDO
	         ENDDO
	         ENDDO
	         ENDDO

	      ELSEIF (src_index .GT. sorted_indices(isorted)) THEN

	         src_index   = src_index   - 1
	         src_index_2 = src_index_2 - 1
	         IF ( src_index .LT. cx_lo_s2(cxs) ) THEN
	            RETURN	! that's all this comonent is gonna supply
	         ELSE
	            pointer = isorted	! (just for efficiency)
	            GOTO 200
	         ENDIF

	      ENDIF

	   ENDDO

	ELSEIF ( axis .EQ. z_dim ) THEN
**** Mapped copy along Z axis

* ... initialize
	   src_index = cx_hi_s3(cxs)
* ... if the input data has no range (-999:-999) then match on sorted_indices=1
	   IF (src_index .EQ. unspecified_int4) THEN
	      src_index_2 = 1
	   ELSE
	      src_index_2 = src_index
	   ENDIF

 300	   CONTINUE	! find where src_index belongs in the result
	   DO isorted = pointer, 1, -1

	      IF (src_index_2 .EQ. sorted_indices(isorted)) THEN

* ... copy a source point to the destination
	         dst_index = index_map(isorted)
	         DO n = cx_lo_s6(cxs), cx_hi_s6(cxs)
	         DO m = cx_lo_s5(cxs), cx_hi_s5(cxs)
	         DO l = cx_lo_s4(cxs), cx_hi_s4(cxs)
	         DO j = cx_lo_s2(cxs), cx_hi_s2(cxs)
	         DO i = cx_lo_s1(cxs), cx_hi_s1(cxs)
		    CALL COPY_C_STRING( src(i,j,src_index,l,m,n),
     .				     dst(i,j,dst_index,l,m,n) )
	         ENDDO
	         ENDDO
	         ENDDO
	         ENDDO
	         ENDDO

	      ELSEIF (src_index .GT. sorted_indices(isorted)) THEN

	         src_index   = src_index   - 1
	         src_index_2 = src_index_2 - 1
	         IF ( src_index .LT. cx_lo_s3(cxs) ) THEN
	            RETURN	! that's all this comonent is gonna supply
	         ELSE
	            pointer = isorted	! (just for efficiency)
	            GOTO 300
	         ENDIF

	      ENDIF

	   ENDDO

	ELSEIF ( axis .EQ. t_dim ) THEN
**** Mapped copy along T axis

* ... initialize
	   src_index = cx_hi_s4(cxs)
* ... if the input data has no range (-999:-999) then match on sorted_indices=1
	   IF (src_index .EQ. unspecified_int4) THEN
	      src_index_2 = 1
	   ELSE
	      src_index_2 = src_index
	   ENDIF

 400	   CONTINUE	! find where src_index belongs in the result
	   DO isorted = pointer, 1, -1

	      IF (src_index_2 .EQ. sorted_indices(isorted)) THEN

* ... copy a source point to the destination
	         dst_index = index_map(isorted)
	         DO n = cx_lo_s6(cxs), cx_hi_s6(cxs)
	         DO m = cx_lo_s5(cxs), cx_hi_s5(cxs)
	         DO k = cx_lo_s3(cxs), cx_hi_s3(cxs)
	         DO j = cx_lo_s2(cxs), cx_hi_s2(cxs)
	         DO i = cx_lo_s1(cxs), cx_hi_s1(cxs)
		    CALL COPY_C_STRING( src(i,j,k,src_index,m,n),
     .				     dst(i,j,k,dst_index,m,n) )
	         ENDDO
	         ENDDO
	         ENDDO
	         ENDDO
	         ENDDO

	      ELSEIF (src_index .GT. sorted_indices(isorted)) THEN

	         src_index   = src_index   - 1
	         src_index_2 = src_index_2 - 1
	         IF ( src_index .LT. cx_lo_s4(cxs) ) THEN
	            RETURN	! that's all this comonent is gonna supply
	         ELSE
	            pointer = isorted	! (just for efficiency)
	            GOTO 400
	         ENDIF

	      ENDIF

	   ENDDO

	ELSEIF ( axis .EQ. e_dim ) THEN
**** Mapped copy along E axis

* ... initialize
	   src_index = cx_hi_s5(cxs)
* ... if the input data has no range (-999:-999) then match on sorted_indices=1
	   IF (src_index .EQ. unspecified_int4) THEN
	      src_index_2 = 1
	   ELSE
	      src_index_2 = src_index
	   ENDIF

 500	   CONTINUE	! find where src_index belongs in the result
	   DO isorted = pointer, 1, -1

	      IF (src_index_2 .EQ. sorted_indices(isorted)) THEN

* ... copy a source point to the destination
	         dst_index = index_map(isorted)
	         DO n = cx_lo_s6(cxs), cx_hi_s6(cxs)
	         DO l = cx_lo_s4(cxs), cx_hi_s4(cxs)
	         DO k = cx_lo_s3(cxs), cx_hi_s3(cxs)
	         DO j = cx_lo_s2(cxs), cx_hi_s2(cxs)
	         DO i = cx_lo_s1(cxs), cx_hi_s1(cxs)
		    CALL COPY_C_STRING( src(i,j,k,l,src_index,n),
     .				     dst(i,j,k,l,dst_index,n) )
	         ENDDO
	         ENDDO
	         ENDDO
	         ENDDO
	         ENDDO

	      ELSEIF (src_index .GT. sorted_indices(isorted)) THEN

	         src_index   = src_index   - 1
	         src_index_2 = src_index_2 - 1
	         IF ( src_index .LT. cx_lo_s5(cxs) ) THEN
	            RETURN	! that's all this comonent is gonna supply
	         ELSE
	            pointer = isorted	! (just for efficiency)
	            GOTO 500
	         ENDIF

	      ENDIF

	   ENDDO

	ELSEIF ( axis .EQ. f_dim ) THEN
**** Mapped copy along F axis

* ... initialize
	   src_index = cx_hi_s6(cxs)
* ... if the input data has no range (-999:-999) then match on sorted_indices=1
	   IF (src_index .EQ. unspecified_int4) THEN
	      src_index_2 = 1
	   ELSE
	      src_index_2 = src_index
	   ENDIF

 600	   CONTINUE	! find where src_index belongs in the result
	   DO isorted = pointer, 1, -1
	      IF (src_index_2 .EQ. sorted_indices(isorted)) THEN

* ... copy a source point to the destination

	         dst_index = index_map(isorted)
	         DO m = cx_lo_s5(cxs), cx_hi_s5(cxs)
	         DO l = cx_lo_s4(cxs), cx_hi_s4(cxs)
	         DO k = cx_lo_s3(cxs), cx_hi_s3(cxs)
	         DO j = cx_lo_s2(cxs), cx_hi_s2(cxs)
	         DO i = cx_lo_s1(cxs), cx_hi_s1(cxs)
		    CALL COPY_C_STRING( src(i,j,k,l,m,src_index),
     .				     dst(i,j,k,l,m,dst_index) )
	         ENDDO
	         ENDDO
	         ENDDO
	         ENDDO
	         ENDDO

	      ELSEIF (src_index .GT. sorted_indices(isorted)) THEN

	         src_index   = src_index   - 1
	         src_index_2 = src_index_2 - 1
	         IF ( src_index .LT. cx_lo_s6(cxs) ) THEN
	            RETURN	! that's all this comonent is gonna supply
	         ELSE
	            pointer = isorted	! (just for efficiency)
	            GOTO 600
	         ENDIF

	      ENDIF

	   ENDDO

	ELSE
	   STOP 'COPY_INTO_MAPPED_SUB: axis!!'

	ENDIF

* success
* the index map has been completely traversed -- destination var is complete
	RETURN
	END 
