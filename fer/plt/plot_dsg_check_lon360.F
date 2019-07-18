	SUBROUTINE PLOT_DSG_CHECK_LON360 (dset, cx, array, bad_flag, npts,
     .      nfeatures)

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
* Checking for longitude crossing the dateline or prime meridian within each
* trajectory, adding or subtracting 360 so that if possible the data does not 
* jump to new branch degrees within the path.

* programmer - Ansley Manke
* NOAA/PMEL, Seattle, WA - SDIG program
* 4/24/2018
* V74+ 4/18 *acm*  Handle missing-data in the longitudes
* V75+ 6/19 *acm*  simpler, more-robust corrections

	include 'tmap_dims.parm'
        include 'ferret.parm'
	include 'xprog_state.cmn'
	include 'xdset_info.cmn_text'
	include 'xdyn_linemem.cmn_text'

	INTEGER dset, cx, npts, nfeatures  ! if not dsg, nfeatures will be 0.

	REAL	array(*), bad_flag, lo, hi

* internal variable declarations:
        LOGICAL process_feature(nfeatures),  do_360
	INTEGER i, idim, irow_l, irow_f, nobs, ifeature, row_size_lm
	REAL	tol, add, del, a_add
	CHARACTER TM_FMT*8

* initialize

	IF (.NOT. grid_is_dsg) RETURN
	idim = 1
	CALL DSG_COORD_LIMS(dset, idim, lo, hi)
 
	CALL MAKE_DSG_FEATURE_MASK(dset, cx, process_feature, nfeatures)
                  
	tol = 160.
	add = 360. 

	do_360 = .FALSE.

* See if the delta-longitude is large, working within each feature.

	row_size_lm = dsg_loaded_lm(dsg_row_size_var(dset))

	irow_f = 1
	irow_l = 0

	DO ifeature = 1, nfeatures 
	   nobs = dsg_linemem(row_size_lm)%ptr(ifeature)  ! feature length
	   irow_l = irow_f + nobs - 1

	   IF (process_feature(ifeature)) THEN

*  Correct the big-delta by adding or subtracting 360 and check
*  that this keeps it all within range 

	      a_add = 0.
	      DO i = irow_f+1, irow_l
	         IF ( array(i)   .EQ. bad_flag ) CYCLE
	         array(i) = array(i) + a_add

	         IF ( array(i-1) .EQ. bad_flag ) CYCLE

	         del = array(i) - array(i-1)

	         IF (ABS(del) .GT. tol) THEN

		    IF (a_add .NE. 0) THEN
		       array(i) = array(i) - a_add
		       del = array(i) - array(i-1)
		       IF (ABS(del) .LT. tol) THEN
		          a_add = 0
			  CYCLE
		       ENDIF
		    ENDIF

	            IF (del .LT. 0) a_add = add
	            IF (del .GT. 0) a_add = -1*add
	            array(i) = array(i) + a_add
		    do_360 = .TRUE.
	         ENDIF
		 
	      ENDDO

	   ENDIF

	   irow_f = irow_l + 1
	ENDDO

        RETURN
        END
