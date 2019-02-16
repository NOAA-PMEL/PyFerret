        SUBROUTINE DEFINE_E_AGGREGATE_GRID(agg_dset, nvars, vname,
     .     fvar, iline, ref_grid, member_sets, nsets, status)

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
* Define ensemble axis and grid using that axis.
* For uvars, grid is set to unspecified_int4 when calling this.
* Will defer the grid check for later.

* Programmer Ansley Manke
* NOAA/PMEL, Seattle, WA - TMAP 
*
* V680 5/12 *acm* 6D Ferret, changes for DEFINE DATA/AGGREGATE
* V695+ 4/15 *acm* and *sh* -- added support for forecast aggregations
* V698 1/16 *sh* - support gappy and inconsistent length forecast members
* V698 3/16 *sh* - shift both lag axes and F-calendar to pts at low cell edge
* V710 *acm* 10/16 Ticket 2422: variables uvar_grid, uvar_data_type, uvar_aux_cat, 
*                  uvar_aux_var to be replaced with calls to LIST routines in NCF_Util.c 
* v71  12/16 *acm* Ticket 2158: working with true monthly time axes.
* v72   6/17 *acm* Variable type in aggregations should match what is in the member files
* v74  10/18 *kms* Split E and F aggregation routines

      IMPLICIT NONE
        include 'tmap_errors.parm'
#include "tmap_dset.parm"
        include 'tmap_dims.parm'
        include 'ferret.parm'
	include 'xdset_info.cmn_text'
	external xdset_info_data
	include 'xdset_info.cd_equiv'
        include 'xprog_state.cmn'
	include 'xtm_grid.cmn_text'  
	include 'errmsg.parm'
	include 'xvariables.cmn'
	include 'xcontext.cmn'
	include 'xfr_grid.cmn'
	include 'interp_stack.parm'
	include 'netcdf.inc'

* calling argument declarations:
        INTEGER agg_dset, nvars, fvar, iline, ref_grid,
     .          member_sets(*), nsets, status
        CHARACTER*(*) vname

* local variable declarations:
        LOGICAL ITS_FMRC_2DTIME, found_fvar, ef_ordered_time2d
        INTEGER STR_SAME, TM_LENSTR1,
     .          igrid, idim, agg_grid, nv, type, ii, iset, slen, ndim,
     .          vtype, i, varid, nv_dummy, igrid_dummy, vtype_dummy,
     .          num_indices, ef_time2d_dset, ef_time2d_grid, fvartype

* initialize
	ef_time2d_dset = ds_var_setnum(fvar)

* Get the file-variable type
        slen = TM_LENSTR1(vname)
	varid = 0
        CALL CD_GET_VAR_TYPE (ef_time2d_dset, varid, vname(:slen), 
     .                        fvartype, status)

* the 2D time variable in an E-aggregation of F-aggregations has special needs
* its grid should not have the agg axis 
        ef_ordered_time2d = ITS_FMRC_2DTIME(ef_time2d_dset, fvar)

* Define aggregate grid

      igrid = unspecified_int4
      IF (ref_grid .NE. unspecified_int4) THEN
	 CALL TM_ALLO_TMP_GRID (igrid, status)
         IF (status .NE. ferr_ok) GOTO 5000

         DO 700 idim = 1, nferdims
            grid_line(idim, igrid) = grid_line(idim,ref_grid)
 700     CONTINUE
	 IF (ef_ordered_time2d) THEN
	    ef_time2d_grid = ref_grid  ! grid of fvar (w/out agg axis)
	 ELSE
            grid_line(e_dim, igrid) = iline
	 ENDIF
* Make up a unique grid name for aggregate grid
         grid_name(igrid) = 'GEN'

	 agg_grid = igrid
      ENDIF

* ds_var slot, fill with aggregate variable in aggregate dataset.
      DO 720 nv = 1, maxvars
         IF ( STR_SAME(ds_var_code(nv),char_init) .EQ. 0 ) GOTO 725
 720  CONTINUE
*     No slots available
      CALL ERRMSG(ferr_prog_limit, status, 'too many variables', 5000)

 725	ds_var_setnum(nv)   = agg_dset
	CALL string_array_modify(ds_var_code_head, nv, 
     .         ds_var_code(fvar), LEN(ds_var_code(fvar)))
! ?? better: inherit these attibutes from members in SET_AGG_VAR_ATTS
	ds_var_title(nv)    = ds_var_title(fvar)
	IF ( ds_var_title(nv) .EQ. char_init128) 
     .       ds_var_title(nv)  = ds_var_code(nv)
	ds_var_titl_mod(nv) = ds_var_titl_mod(fvar)
	ds_var_units(nv)    = ds_var_units(fvar)
	ds_missing_flag(nv) = ds_missing_flag(fvar)	! was bad_val4
	ds_bad_flag(nv)     = ds_bad_flag(fvar)		! was bad_val4
	ds_var_type(nv)     = ds_var_type(fvar)
	cd_scale_factor(nv) = 1.
	cd_add_offset(nv)   = 0.
	cd_scaled(nv)       = .FALSE.

	IF (ref_grid .NE. unspecified_int4) THEN
	   ndim = 1
	   DO 730 idim = 1, nferdims
	      ds_grid_start(idim,nv)= ds_grid_start(idim,fvar)
	      ds_grid_end(idim,nv)  = ds_grid_end(idim,fvar)
	      ds_ordering(idim,nv)  = ds_ordering(idim,fvar)
	      IF (ds_ordering(ndim,fvar) .GT. 0) ndim = ndim + 1
 730	   CONTINUE
           ds_grid_start(e_dim,nv) = 1
           ds_grid_end(e_dim,nv) = nsets
           ds_ordering(ndim,nv) = e_dim  ! last dimension of the grid
	   ds_grid_number(nv) = agg_grid
	ENDIF

* Type should be what is in the original file
	IF (ds_var_type(nv) .EQ. ptype_string) THEN
	   type = NCCHAR
	ELSE
	   type = fvartype
	ENDIF

* Hook the aggregated variable into the linked list structures
	CALL CD_ADD_VAR( agg_dset, vname, nvars, 
     .         ds_var_title(nv), ds_var_units(nv), type, 
     .         .FALSE., ds_bad_flag(nv), status)

* Add aggregate-grid info to the lists in aggregate datset and aggregate-grid list.
* Mark user-variables with a grid number of -999.  Note if there is a LET/D variable
* that duplicates the name of a file var it is the file var that gets registered,
* even though the LET/D var has precedence during evaluation.  This allows for the
* possibility that the LET/D var may be cancelled.  But it also allows the possibility
* that the grid of the fvar is unacceptable -- a chance of a crash for a pathological
* dataset.
	dsets: DO ii = 1, nsets
	   iset = member_sets(ii)
* ... special hack for the TF_TIMES 2d time array when it is E-aggregated
	   IF (ef_ordered_time2d) THEN
	      vtype = cat_ef_time2d
	      CALL CD_ADD_AGG_VAR_INFO (agg_dset, vname, ii, vtype, 
     .                ef_time2d_dset, ef_time2d_grid, iline, fvar, status)
	      CYCLE dsets
	   ENDIF
	   found_fvar = .FALSE.
	   IF (iset .NE. pdset_dummy) THEN
* ... add this member data set as an fvar as part of this aggregated variable
	      DO nv = 1, maxvars
	         IF ( (ds_var_setnum(nv) .EQ. iset) .AND. 
     .                (STR_SAME(ds_var_code(nv), vname) .EQ. 0) ) THEN
	            igrid = ds_grid_number(nv)
		    vtype = cat_file_var
	            CALL CD_ADD_AGG_VAR_INFO (agg_dset, vname, ii, vtype, 
     .                iset, igrid, iline, nv, status)
	            found_fvar = .TRUE.
	            EXIT
	         ENDIF
	      ENDDO
	      IF (.NOT.found_fvar) THEN
* ... add this member data set as a LET/D as part of this aggregated variable
* ... optimize search using deleted_list_get_undel
!	         DO nv = 1, max_uvar      ! brute-force equivalent loop
		 CALL deleted_list_get_undel(uvar_num_items_head,
     .                              deleted_list_result(1),
     .                              max_uvar, num_indices )
	         DO i = 1, num_indices
	            nv = deleted_list_result(i)
	            IF ( uvar_num_items(nv) .EQ. uvar_deleted) CYCLE
	            IF ( (uvar_dset(nv) .EQ. iset) .AND. 
     .                   (STR_SAME(uvar_name_code(nv), vname) .EQ. 0) ) THEN
		       vtype = cat_user_var
		       CALL GET_SAVED_UVAR_GRID(nv, iset, igrid)
	               CALL CD_ADD_AGG_VAR_INFO (agg_dset, vname, ii, vtype, 
     .                   iset, igrid, iline, nv, status)
* ... bump the use count for this grid to protect it from being deleted if
*     the user changes the LET/D definition -- grid is needed during the
*     gather process (IS_STRIP & IS_AGGREGATE_GATHER).
*  ==> be sure to un-use (deallo) it when the aggregation gets canceled
*  ==> or if an error occurs prior to the complete initialization of this agg
	               CALL TM_USE_DYN_GRID(igrid)
	               EXIT
	            ENDIF
	         ENDDO
	      ENDIF
	   ELSE
* ... add the dummy data set as a part of this variable 
*     Note that the first member of the agg by common sense cannot be a dummy
	      CALL CD_ADD_AGG_VAR_INFO (agg_dset, vname, ii, vtype_dummy, 
     .             pdset_dummy, igrid_dummy, iline, nv_dummy, status)
	   ENDIF
* ... grab information from the first dataset to reuse with dummy datasets 
	   IF (ii .EQ. 1) THEN
*    ... OK_AS_AGG_VAR guarantees grid is always valid -- even for LET/D vars
              IF (igrid .EQ. unspecified_int4) 
     .           STOP 'DEFINE_E_AGGREGATE_GRID grid err'
	      nv_dummy = nv
	      igrid_dummy = igrid
	      vtype_dummy = cat_file_var
	   ENDIF
	ENDDO dsets

 5000   RETURN
	END
