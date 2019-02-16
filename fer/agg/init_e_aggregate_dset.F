        SUBROUTINE INIT_E_AGGREGATE_DSET(memb_dsetname_var,
     .     nsets, have_expr, dname, dpath, dtitle, agg_dset, 
     .     agg_quiet, agg_hide, status)
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

* initialize a new ensemble (E) or forecast (F) aggregated dataset
* Aggregation member variables may be a blend of file variables and LET/D vars
* Since the user may alter LET/D definitions after the aggregation has been
* defined, the following approach is taken to processing:
*  1) during DEFINE DATA/AGG the grids of all member variables are determined
*     This means that IS_UVAR_GRID is called for LET/D vars.
*     These grid numbers are stored in the aggregation LIST structures
*     The grids of the variables in the aggregated (parent) dataset are also
*     determined and stored in ds_grid_number as if they are file variables.
*     These grids are "static" as with file variables -- always valid. 
*  2) (TBD as of this writing!!!)==> the use counts of *all* member grids are
*     bumped, so that they cannot be deleted, even if LET/D definitions change
*  3) (TBD as of this writing!!!)==> at get_grid time all of the components
*     of the aggregation are checked to ensure they have a grid (IS_UVAR_GRID) 
*     -- allowing for last-minute LET/D var changes
*  4) during evaluation (in IS_STRIP and IS_AGGREGATE_GATHER) the variable
*     name and member dataset number are used to make a fresh determination of
*     the variable category and grid.  Error out if grid number has changed

* Programmer Ansley Manke
* NOAA/PMEL, Seattle, WA - TMAP 
*
* V680 5/12 *acm* 6D Ferret, changes for DEFINE DATA/AGGREGATE
* V685 4/13 *acm* Ability to define ensemble with /HIDDEN
*                 Get grid of user-vars, allow more combinations with user vars
*                 better error handling.
* V695 2/15 *acm* Aggregation in either E or F direction, as defined by agg_dim
* V695+ 5/15 *sh* Call GET_AXIS_SPEEDS (pulled out of CHECK_MEMBER_DSETS)
* V695+ 10/15 *sh* rename this routine to INIT_EF_AGGREGATE_DSET
* V698 12/15 *sh* reorganization of code in order to accommodate:
*		  checks of uvar grids delayed until eval
*	          allow for LET/D def changes after DEFINE/AGG (hopefully)
*		  allow dataset list passed as a Ferret string variable
*                 allow forecast datasets out of order (sort em)
*		  allow gaps in the forecast series (insert dummy datasets)
* V698 1/16  *sh* pass explicit path name (for "USE file.agg" compatibility)
*      2/16  *sh* automatically xfer attributes from member vars to agg vars
*      3/16  *sh* trap simple invalid commands
*                 start TBOXLO[L=1], not TBOX[L=1], of agg_lag_t axis at 0.0 
* V71 12/16 *acm* For ticket 2158: working with true monthly time axes.
* V741 *acm* 5/18 Promoting global attributes to a variable in ensembles
*                 Clarify the error message at label 5500.
* v74  10/18 *kms* Split E and F aggregation routines

* arguments:
*	memb_dsetname_var - if have_expr, the variable of names (input)
*	nsets		- number of datasets in the aggregation (input)
*       have_expr	- flag if filenames were listed or in an expr (input)
*	dname		- dataset name to assign to the aggregation (input)
*	dpath		- dataset path name to assign to aggregation (input)
*	dtitle		- dataset title (input)
*	agg_dset	- number assigned to this data set (output)
*       agg_quiet       - flag whether to report certain conditions (input)
*	agg_hide        - flag whether to hide the component datasets (input)
*	status		- return status (output)

      IMPLICIT NONE
	include 'netcdf.inc'
        include 'tmap_errors.parm'
#include "tmap_dset.parm"
        include 'tmap_dims.parm'
        include 'ferret.parm'
	include 'xdset_info.cmn_text'
	external xdset_info_data
	include 'xdset_info.cd_equiv'
	include 'xunits.cmn_text'
        include 'xprog_state.cmn'
	include 'xtm_grid.cmn_text'  
	include 'errmsg.parm'
	include 'xvariables.cmn'
	include 'xcontext.cmn'
	include 'xfr_grid.cmn'

* calling argument declarations:
	CHARACTER*(*)	dname, dpath, dtitle
        INTEGER nsets, agg_dset, status
	LOGICAL		have_expr, agg_quiet, agg_hide
	REAL		memb_dsetname_var(*)

* parameter definitions
        INTEGER pmaxvarsperset
        PARAMETER (pmaxvarsperset = 400)

* local variable declarations:
        LOGICAL OK_AS_AGG_VAR, match, has_uvars
        INTEGER TM_LENSTR1, STR_SAME, NCF_ADD_AGG_MEMBER,
     .          iset, dset1, ii, max_str_len, member_sets(nsets),
     .          nv, grid, nvars, nuvars, iline, slen, ref_fvar, 
     .          ref_fvars(pmaxvarsperset), num_indices, ignored
	CHARACTER TM_STRING*13
	CHARACTER*512 GET_STRING_ELEMENT_1D_X, memb_dsetname(nsets)
	CHARACTER*128 vname

* initialize
	max_str_len = 512
        DO iset = 1, nsets
	   member_sets(iset) = pdset_dummy
	ENDDO

* check to see if this new dataset name matches an already-open dataset name.
* if so, that is an error
	DO iset = 1, maxdsets
           match = ds_name(iset) .EQ. dname
           IF ( match ) GOTO 5100
	ENDDO

* set up array of dataset identifier strings
* these may be string dataset numers or names, pre-init'ed or not
	IF (have_expr) THEN
*  ... dataset list passed through a string variable
	  DO iset = 1, nsets
             memb_dsetname(iset) = GET_STRING_ELEMENT_1D_X(nsets,
     .                 memb_dsetname_var,iset, max_str_len, slen)
	  ENDDO
	ELSE
*  ... dataset names given by user on command line
	  DO iset = 1, nsets
	    memb_dsetname(iset) = cmnd_buff(item_start(iset):item_end(iset))
	  ENDDO
	ENDIF

* See if requested datasets are open and if not, try to open them.
        CALL CHECK_MEMBER_SETS(e_dim, nsets, memb_dsetname,
     .                         member_sets, status)
	IF (status .NE. ferr_ok) GOTO 9950

* Assign aggregate data set number, exiting if maximum number 
* of data sets are open
	DO iset = 1,maxdsets
	  IF (ds_name(iset) .EQ. char_init2048) THEN
	    agg_dset = iset
	    GOTO 220
	  ENDIF
	ENDDO
* ... too many data sets are open
        CALL TM_ERRMSG(merr_dsetlim, status, 'INIT_E_AGGREGATE_DSET',
     .    no_descfile, no_stepfile, 'MAX='//TM_STRING(DBLE(maxdsets)),
     .    no_errstring, *9950)
 220	CONTINUE

* Create datset. Will delete it later if there are no aggregate variables found
* Note: the pattern will the first dataset given.  They may get reordered.
        CALL CREATE_AGG_DSET(agg_dset, dname, dpath, dtitle,
     .         nsets, member_sets(1), e_dim, iline, status)
	IF (status .NE. ferr_ok) GOTO 5300

* perform initializations needed on a new dataset
* determine relative access speeds along each axis
	CALL GET_AXIS_SPEED( agg_dset )

* 1st member dset sets pattern for all to follow
	dset1 = member_sets(1)

*****
* find variables that are acceptable for aggregation using dset1 as the basis
* an aggregatable var must have at least member dset that is an fvar - ref_fvar
* note LET/D names that duplicate file variable names take precedence, so
*     check them first
	nuvars = 0
!	DO nv = 1, max_uvar      ! brute-force equivalent loop
	CALL deleted_list_get_undel(uvar_num_items_head,
     .                              deleted_list_result(1),
     .                              max_uvar, num_indices )
        DO ii = 1, num_indices
           nv = deleted_list_result(ii)
	   IF (uvar_dset(nv) .EQ. dset1) THEN
	      IF (OK_AS_AGG_VAR(nv, cat_user_var, dset1, member_sets,
     .           nsets, e_dim, agg_quiet, ref_fvar, status)  ) THEN
	         nuvars = nuvars + 1
	         ref_fvars(nuvars) = ref_fvar
	      ELSE
	         IF (status .NE. ferr_ok) GOTO 5000
	      ENDIF
	   ENDIF
	ENDDO
	nvars = nuvars

* now find the file variables
        DO nv = 1, maxvars
	   IF (ds_var_setnum(nv) .EQ. dset1) THEN
* ... skip it if it duplicates a name already found as a LET/D variable
	      DO ii = 1, nuvars
	         IF (STR_SAME(ds_var_code(ref_fvars(ii)),
     .			      ds_var_code(nv)           ) .EQ. 0) GOTO 300
	      ENDDO
	      IF (OK_AS_AGG_VAR(nv, cat_file_var, dset1, member_sets,
     .           nsets, e_dim, agg_quiet, ref_fvar, status)  ) THEN
	         nvars = nvars + 1
	         ref_fvars(nvars) = ref_fvar
	      ELSE
	         IF (status .NE. ferr_ok) GOTO 5000
	      ENDIF
	   ENDIF
 300	   CONTINUE
	ENDDO
*****

* If no variables from the datasets match, delete this dataset. 
	IF (nvars .EQ. 0) GOTO 5500

* register the list of variables found suitable for this aggregation
        DO nv = 1, nvars
           ref_fvar = ref_fvars(nv)
	   vname = ds_var_code(ref_fvar)
	   grid = ds_grid_number(ref_fvar)
           CALL DEFINE_E_AGGREGATE_GRID(agg_dset, nv, vname, ref_fvar,
     .                        iline, grid, member_sets, nsets, status)
	   IF (status .NE. ferr_ok) GOTO 5000

* transfer to the aggregated variable whatever special attributes it inherits
           CALL SET_AGG_VAR_ATTS(agg_dset, vname, nv, nsets, 
     .			         ref_fvar, member_sets, status)
	   IF (status .NE. ferr_ok) GOTO 9960
	ENDDO

* Promote global attributes to variables in the aggregated dataset. 

        CALL PROMOTE_AGG_GLOBAL_ATTS(agg_dset, nsets, member_sets, 
     .                               e_dim, iline, nvars, status)
	IF (status .NE. ferr_ok) GOTO 5600

* register member datasets to aggregate dataset
        DO iset = 1, nsets
	   status = NCF_ADD_AGG_MEMBER (agg_dset, iset, member_sets(iset)) 
	ENDDO
 
* remove duplicate temp grid definitions amd make grid names unique
        ignored = unspecified_int4
        CALL AGG_CLEAN_GRIDS(agg_dset, e_dim, iline, ignored)

* set use counts and remove temp status of agg grids and axes
	CALL TM_GARB_COL_GRIDS(agg_dset)

* set the official time axis of the dataset
        CALL CD_GET_TIME_AXIS(agg_dset, t_dim, ignored)
* (always successful)

* If user requested /HIDE, then mark the member datasets for hidden status
	IF (agg_hide) THEN
           DO iset = 1, nsets
	      ds_hide (member_sets(iset)) = .TRUE. 
	   ENDDO
	ENDIF

* successful completion
	RETURN

* error bail out
* remove all hidden member datasets of the failed aggregation
* ... this is a little brute force in that it may cancel a previously opened
* ... dataset included using DEFINE DATA/AGG/HIDE (have not checked carefully)
 5000   DO ii = 1, nsets
	   iset = member_sets(ii)
	   IF (iset .EQ. pdset_dummy
     .    .OR. iset .EQ. unspecified_int4) CYCLE
	   IF (ds_name(iset) .EQ. char_init2048) CYCLE ! already deleted
           IF (ds_hide (iset)) CALL CANCEL_ONE_DATA_SET(iset, ignored)
	ENDDO
 6000	RETURN

* Errors: too-many datasets open so aggregate dataset cannot be opened,
* or error on a member dataset.  Return no-dataset-number.
 9950	agg_dset = unspecified_int4
	status = ferr_TMAP_error
	GOTO 5000

* wipe out the partially-created aggregated dataset
 9960   CALL TM_PURGE_TMP_GRIDS(ignored)
        CALL TM_CLOSE_SET_COND_DEALLO(agg_dset, .FALSE., ignored)
	agg_dset = unspecified_int4
	GOTO 5000

 5100   CALL ERRMSG( ferr_invalid_command,status,
     .		'dataset name already in use: '
     .		//dname(:TM_LENSTR1(dname)), *5000 )
     
 5300	CALL ERRMSG( ferr_aggregate_error, status, ' ', *9950 )


 5500	CALL ERRMSG( ferr_aggregate_error, status,
     .		'No valid datasets or datasets share no variables '
     .		//'that can be aggregated. ', 
     .		*9960 )

 5600	CALL ERRMSG( ferr_aggregate_error, status,
     .		'Error setting global attributes. ', 
     .		*9960 )

	END

