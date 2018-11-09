	SUBROUTINE FULL_SCAN_INDICES (src, ksmin, ksmax,
     .				 ks_start, ks_end,
     .				 bad_src, targ,
     .				 indices, coefs, kdmin, kdmax,
     .				 ascnd, has_valid )

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

*       Code is a delta on HUNT_INDICES.  That routine assumes src array
*       is sorted, so it can use a binary search.  This routine uses a
*       full scan to find nearest indices, and then continues the scan to make
*       sure the result is not multiple-valued (a result of non-monotonic src)

*       Note that the TARG array is always ascending

* variable naming:
*	's' refers to the source array -- the incoming list of values
*	'd' refers to the destination arrays -- the outgoing indices and coefs
*	min/max refer to the dimensioned limits of arrays
*	_ok refers to to limits within which the arrays contain valid values


* V698 *sh* 5/15 derived from Ferret source routine HUNT_INDICES

        IMPLICIT NONE
	include 'ferret.parm'

* calling argument declarations
	LOGICAL	ascnd, has_valid
	INTEGER	ksmin, ksmax, ks_start, ks_end, kdmin, kdmax,
     .		indices(kdmin:kdmax)
	REAL	src(ksmin:ksmax),
     .			targ(kdmin:kdmax), coefs(kdmin:kdmax),
     .			bad_src

* internal variable declarations
	INTEGER	ks, kd, slo, shi, sincr, nfound, ksfound, ksprev
	REAL	vt, v0, v1

********************************************
* initialize
	has_valid = .FALSE.
	ksfound = ks_start    ! any valid index may go with coef=bad_val4

* initialize loop direction
	IF (ascnd) THEN
	  sincr = 1
	  slo   = ks_start
	  shi   = ks_end
	ELSE
	  sincr = -1
	  slo   = ks_end
	  shi   = ks_start
	ENDIF

* outer loop - target axis points
	DO kd = kdmin, kdmax

* inner loop - a column of non-monotonic (semi-ordered) auxiliary coordinates
	   vt = targ(kd)
	   nfound = 0
	   v0 = src(slo)
	   DO ks = slo, shi, sincr

	      IF (ks .NE. slo) v0 = v1   ! preceding test point
	      v1 = src(ks)               ! new test point

* does our target exactly match this src value?
	      IF (vt .EQ. v1) THEN
	         ksfound = ks
	         nfound = nfound + 1
	         CYCLE
	      ENDIF

* does our target value lie between two valid src values? 
	      IF (v0 .NE. bad_src
     .      .AND. v1 .NE. bad_src ) THEN
* ... target lies between if sign of diff product is non-negative  
	         IF ( ((vt-v0)*(v1-vt)) .GT. 0.0) THEN
	            ksfound = ks
	            nfound = nfound + 1
	         ENDIF
	      ENDIF
	   ENDDO

* save result
	   indices(kd) = ksfound
	   coefs(kd) = bad_val4

	   IF (nfound .EQ. 1) THEN
	      ksprev = ksfound-sincr
	      IF (vt .EQ. src(ksfound) ) THEN
!	         indices(kd) = ksfound
	         coefs(kd) = 1.0
	         has_valid = .TRUE.
	      ELSEIF (src(ksfound) .GT. src(ksprev) ) THEN ! only ordered ok
	         indices(kd) = ksprev
	         coefs(kd) = (src(ksfound)-vt) / (src(ksfound)-src(ksprev))
	         has_valid = .TRUE.
	      ENDIF
	   ENDIF
	
	ENDDO     ! end outer loop      

	RETURN
	END

