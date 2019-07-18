	SUBROUTINE SEQUENCE_DSG( grid, com, res, com_cx, nfeatures,
     .		 bad_com, bad_res, nres)

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
* Apply an XSEQUENCE-type operation for data on Distributed Sampling Geometries grids
* 
*
* Ansley Manke
* NOAA/PMEL, Seattle, WA - Science Data Integration Group
*  3/2019
*  5/2019  Correct the bad-flag handling

        IMPLICIT NONE
        include 'tmap_dims.parm'
	include	'ferret.parm'
	include	'xcontext.cmn'
	include	'xvariables.cmn'
	include	'xmem_subsc.cmn'
	include	'xdset_info.cmn_text'
	include 'xdyn_linemem.cmn_text'


* calling argument declarations
* Declare size of result so we can initialize it.
	INTEGER	grid, com_cx, nfeatures, nres
	REAL	com   ( * ),
     .          res   (nres), bad_com, bad_res

* internal variable declarations:
	LOGICAL fmask(nfeatures), process_obs(nres), obsvar
	INTEGER	TM_DSG_DSET_FROM_GRID, TM_DSG_NFEATURES, CX_DIM_LEN, 
     .		dset, idim, i, ndx, ifeature, iobs, flen, base, row_size_lm

* Initialize the result

	res = bad_res

* DSG-specific initialize

	dset = TM_DSG_DSET_FROM_GRID(grid)
	row_size_lm = dsg_loaded_lm(dsg_row_size_var(dset))

* Does the grid of com have a non-E axis?

	obsvar = .FALSE.
	DO idim = 1, nferdims 
	   IF (idim .EQ. e_dim) CYCLE
	   IF (CX_DIM_LEN(idim, com_cx) .GT. 1) obsvar = .TRUE.
	ENDDO

* get feature mask
        CALL MAKE_DSG_FEATURE_MASK(dset, com_cx, fmask, nfeatures)

* *** Flatten the data to a single list after applying the masks

	   IF (obsvar) THEN ! data along 'obs' axis

	      base = 0    ! obs index at end of preceding feature
	      ndx = 0

* ... loop over the features
	      DO ifeature = 1, nfeatures 

	      flen = dsg_linemem(row_size_lm)%ptr(ifeature)  ! feature length

* ... have the user's constraints removed this feature?
	      IF (.NOT.fmask(ifeature)) THEN
                 base = base + flen
                 CYCLE
	      ENDIF

* ... loop over the observations within the feature

* ... apply observation-level mask for this feature
	      CALL MAKE_DSG_OBS_MASK(dset, com_cx, ifeature, base, process_obs, flen)

	     iobs = base
	      DO i = 1, flen
	         iobs = iobs + 1
	         IF (process_obs(i)) THEN
	            ndx = ndx + 1
		    IF (com(iobs) .EQ. bad_com) THEN
		       res(ndx) = bad_res
		    ELSE
	               res(ndx) = com(iobs)
		    ENDIF
	         ENDIF
	      ENDDO
	      base = base + flen
	   ENDDO

	
	ELSE

* *** a simple list of instances, return the data where feature not masked

	   ndx = 0
	   DO ifeature = 1, nfeatures 

* ... have the user's constraints removed this feature?
	      IF (fmask(ifeature)) THEN
	         ndx = ndx + 1
		 IF (com(ifeature) .EQ. bad_com) THEN
	            res(ndx) = com(ifeature)
		 ELSE
		    res(ndx) = bad_res
		 ENDIF
	      ENDIF
	   ENDDO

	ENDIF

	RETURN
	END
