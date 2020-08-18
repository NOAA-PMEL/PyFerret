        SUBROUTINE CD_RELEASE_DSG_INSTANCE_VARS(dset)


*  This software was developed by the Thermal Modeling and Analysis
*  Project(TMAP) of the National Oceanographic and Atmospheric
*  Administration's (NOAA) Pacific Marine Environmental Lab(PMEL),
*  hereafter referred to as NOAA/PMEL/TMAP.
*.
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
* Release the memory occupied by DSG instance variables

* Programmer Steve Hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program

* v720 4/2017 *sh*
* V74+ 7/18 *acm* Handle case of rowsize variable that we have synthesized 
*                 when there's not one in the file. 


* include files
        include 'netcdf.inc'
        include 'tmap_errors.parm'
        include 'tmap_dims.parm'
#include "tmap_dset.parm"
#include "gt_lib.parm"
        include 'xbuild_grids.cmn'
        include 'xdset_info.cmn_text'

* argument declarations
        INTEGER  dset

* internal variable declarations
        INTEGER idim, ivar

* if dsg_row_size_var is negative, there is no rowsize varible, but 
* otherwise it's a valid single-feature dataset.

        ivar = dsg_row_size_var(dset)
        IF (ivar .NE. int4_init .AND. ivar.GT.0) THEN
           IF (dsg_loaded_lm(ivar) .NE. int4_init) CALL TM_FREE_LM_DYNMEM(dsg_loaded_lm(ivar))
           dsg_row_size_var(dset) = int4_init
           dsg_loaded_lm(ivar)    = int4_init
        ENDIF

        DO idim = 1, 4
          ivar = dsg_coord_var(idim,dset)
          IF (ivar .NE. int4_init) THEN
             IF (dsg_loaded_lm(ivar) .NE. int4_init) CALL TM_FREE_LM_DYNMEM(dsg_loaded_lm(ivar))
             dsg_coord_var(idim,dset) = int4_init
             dsg_loaded_lm(ivar)      = int4_init
          ENDIF
        ENDDO

* index var for timeseriesProfile or trajectoryprofile
	
        ivar = dsg_index_var(dset)
        IF (ivar .NE. int4_init .AND. ivar.GT.0) THEN
           IF (dsg_loaded_lm(ivar) .NE. int4_init) CALL TM_FREE_LM_DYNMEM(dsg_loaded_lm(ivar))
           dsg_index_var(dset) = int4_init
           dsg_loaded_lm(ivar)    = int4_init
        ENDIF

* id var for station or trajectory for timeseriesProfile or trajectoryProfile
	
        ivar = dsg_ftrset_var(dset)
        IF (ivar .NE. int4_init .AND. ivar.GT.0) THEN
           IF (dsg_loaded_lm(ivar) .NE. int4_init) CALL TM_FREE_LM_DYNMEM(dsg_loaded_lm(ivar))
           dsg_index_var(dset) = int4_init
           dsg_loaded_lm(ivar)    = int4_init
        ENDIF

* was a feature-mask set? deallocate that.
	IF (dsg_msk_lm(dset).NE.int4_init .AND. dsg_msk_lm(dset).NE.unspecified_int4) THEN
	   CALL FREE_LINE_DYNMEM( dsg_msk_lm(dset) )
	   CALL TM_DEALLO_DYN_LINE( dsg_msk_lm(dset) )
	   dsg_msk_lm(dset) = int4_init
	ENDIF

* was a index-mask set? deallocate that.
	IF (dsg_mask_ftrset_lm(dset).NE.int4_init .AND. dsg_mask_ftrset_lm(dset).NE.unspecified_int4) THEN
	   CALL FREE_LINE_DYNMEM( dsg_mask_ftrset_lm(dset) )
	   CALL TM_DEALLO_DYN_LINE( dsg_mask_ftrset_lm(dset) )
	   dsg_mask_ftrset_lm(dset) = int4_init
	ENDIF



* Reset a couple more settings regarding dsg datasets

        dsg_orientation(dset) = int4_init
        dsg_user_direction = int4_init

* always successful completion
        RETURN

        END
