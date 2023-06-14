	INTEGER FUNCTION DO_SHIFT_DSG (	idim, arg, dset, nfeatures,
     .					com, com_mr, com_cx,
     .					res, res_mr, res_cx,
     .					wkbuf	)

*
*
*  This software was developed by the Thermal Modeling and Analysis
*  Project(TMAP) of the National Oceanographic and Atmospheric
*  Administration''s (NOAA) Pacific Marine Environmental Lab(PMEL),
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
* shift the data from a component grid into the result grid in direction idim
* by amount ishift
* for data on Distributed Sampling Geometries grids

* programmer - Ansley Manke
* NOAA/PMEL, Seattle, WA - Science Data Integration Group
* 2/2019
* V76   1/20 *acm* Orientation of Point-type data set to e_dim
* 10/2022 *acm* Code cleanup: remove comment after # include statement

        IMPLICIT NONE
        include 'tmap_dims.parm'
#	include	"tmap_dset.parm" 
	include	'ferret.parm'
	include	'errmsg.parm'
	include	'interp_stack.parm'
	include 'xcontext.cmn'
	include 'xvariables.cmn'
	include	'xmem_subsc.cmn'
	include 'xprog_state.cmn'
	include 'xdyn_linemem.cmn_text'
	include 'xdset_info.cmn_text'
	include 'xdsg_context.cmn'

* calling argument declarations:
	INTEGER	idim, dset, com_mr, com_cx, res_mr, res_cx, nfeatures
	REAL	arg, wkbuf(*), 
     .          com   (*),
     .          res   (*)

* internal variable declarations:
	LOGICAL fmask(nfeatures)
	INTEGER ii, shf, i, lo, hi, grid, ifeature, iobs,
     .          flen, base, orientation, row_size_lm
	REAL	bad_com, bad_res, val


* initialize
	shf = INT(arg)
	bad_res = mr_bad_data( res_mr )

* initialize
	grid = mr_grid( com_mr )
	bad_com = mr_bad_data( com_mr )
	bad_res = mr_bad_data( res_mr )

* DSG-specific initialize

        CALL MAKE_DSG_FEATURE_MASK(dset, com_cx, fmask, nfeatures)
	row_size_lm  = dsg_loaded_lm(dsg_row_size_var(dset))
	orientation  = dsg_orientation(dset)

* SUM

* Handle z transforms for feature-type of trajectory-profile or timeseries-profile - within this routine
* treat the data type as profile.

	IF (orientation.EQ.pfeatureType_TrajectoryProfile .OR. 
     .	      orientation.EQ.pfeatureType_TimeseriesProfile) orientation = pfeatureType_Profile

* ****** PERFORM THE COMPUTATION *****
	IF (idim .EQ. orientation) THEN

* *** true ragged array processing:
* sum independently along each feature 
*     Work with all data in the feature. WW coordinate masking etc  
*     will be applied beforelisting or plotting

           base = 0    ! obs index at end of preceding feature

* ... loop over the features
	   ftrloop: DO ifeature = 1, nfeatures 

	      flen = dsg_linemem(row_size_lm)%ptr(ifeature)  ! feature length

* ... have the user's constraints removed this feature?
	      IF (.NOT.fmask(ifeature)) THEN
                 base = base + flen
                 CYCLE ftrloop
              ENDIF

	      
* compute limits within which calculation is possible

              lo = MAX(1, 1 - shf)
              hi = MIN(flen, flen - shf)

* ... loop over the observations within each feature
              iobs = base

* Single item in the feature, result is missing-value

	      IF (flen .EQ. 1) THEN 
	          iobs = iobs + 1 
		  res(iobs) = bad_res

		  base = base + flen
		  CYCLE ftrloop 

	      ENDIF

* ... loop over the observations within each feature, if obs is not masked 
*     then sum it.

	      iobs = base
	      DO i = lo, hi
	         iobs = iobs + 1   ! index in the contig ragged array

                 val = com(iobs+shf)
		 res(iobs) = val
		 IF (val .EQ. bad_com) res(iobs) = bad_res
		 
	      ENDDO

	      base = base + flen

	   ENDDO ftrloop

	ELSE  

* *** a simple list of instances:
* Work along the feature-number axis

	   lo = MAX(1, 1 - shf)
	   hi = MIN(nfeatures, nfeatures - shf)

	   DO ifeature = lo, hi

* ... have the user's constraints removed this feature?
	      IF (.NOT.fmask(ifeature)) CYCLE

* ... the input data value
              
              val = com(ifeature+shf)
	      res(ifeature) = val
	      IF (val.EQ.bad_com .OR. val.EQ.dsg_masked_out_flag) 
     .              res(ifeature) = bad_res
		 

           ENDDO


	ENDIF  ! obs vs feature

	
	RETURN
	END

