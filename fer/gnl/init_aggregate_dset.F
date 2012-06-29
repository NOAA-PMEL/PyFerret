	SUBROUTINE INIT_AGGREGATE_DSET (dname, dtitle, agg_dset, 
     .     agg_quiet, status )
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
* Programmer Ansley Manke
* NOAA/PMEL, Seattle, WA - TMAP 
*
* V680 5/12 *acm* 6D Ferret, changes for DEFINE DATA/AGGREGATE

      IMPLICIT NONE
        include 'tmap_errors.parm'
#include "tmap_dset.parm"
        include 'tmap_dims.parm'
	include	'ferret.parm'
	include 'xdset_info.cmn_text'
	external xdset_info_data
	include 'xdset_info.cd_equiv'
        include 'xprog_state.cmn'
	include 'xtm_grid.cmn_text'  
	include 'errmsg.parm'
	include 'xvariables.cmn'
	include 'xcontext.cmn'
	include 'xfr_grid.cmn'

* calling argument declarations:
	CHARACTER*(*)	dname, dtitle
	INTEGER		agg_dset, status
	LOGICAL		agg_quiet

* local variable declarations:
	LOGICAL	match, has_uvars
	INTEGER	TM_LENSTR, TM_LENSTR1, STR_SAME, NCF_DELETE_DSET,
     .		NCF_ADD_AGG_MEMBER, NCF_GET_AGG_MEMBER_INFO, 
     .          iset, dset, item, ii,
     .		perm(nferdims), nsets, member_sets(maxdsets), 
     .		imatch, nv, grid, igrid, idim, ivar, nvars, iline, 
     .		slen, memb_var, memb_dset, memb_grid, 
     .		type
	CHARACTER vname*128
	CHARACTER TM_STRING*13



* check to see if this name matches an already-open dataset name.
* if so, thats an error

	DO 100 iset = 1, maxdsets
           match = ds_name(iset) .EQ. dname
           IF ( match ) GOTO 5100
  100	CONTINUE

* See if requested datasets are open and if not, try to open them.

	CALL CHECK_MEMBER_SETS (nsets, member_sets, status)
	IF (status .NE. ferr_ok .OR. nsets .EQ. 0) GOTO 9950

* Ok, all the member datasets are open and available

* Assign aggregate data set number, exiting if maximum number 
* of data sets are open

	DO 200 iset = 1,maxdsets
	  IF (ds_name(iset) .EQ. char_init1024) THEN
	    agg_dset = iset
	    GOTO 220
	  ENDIF
  200	CONTINUE

* Too many data sets are open
	CALL TM_ERRMSG ( merr_dsetlim, status, 'INIT_AGGREGATE_DSET',
     .			 no_descfile, no_stepfile,
     .			 'MAX='//TM_STRING(DBLE(maxdsets)),
     .			 no_errstring, *9950)
 220	CONTINUE

* Create datset. Will delete it later if there are no aggregate variables found.

	CALL CREATE_AGG_DSET (agg_dset, dname, dtitle, nsets, 
     .			 e_dim, iline, status)  
	IF (status .NE. ferr_ok) GOTO 5300

* See what variables are shared by all the datsets

	dset = member_sets(1)
	nvars = 0
	
	DO 800 ivar = 1, maxvars
           IF ( ds_var_setnum(ivar) .NE. dset ) GOTO 800
	   vname = ds_var_code(ivar)
	   imatch = 1
	   DO 400 ii = 2, nsets
	      iset = member_sets(ii)
              DO 300 nv = 1, maxvars
	         IF ( (ds_var_setnum(nv) .EQ. iset) .AND.
     .                (STR_SAME(ds_var_code(nv), vname) .EQ. 0) ) THEN
                    imatch = imatch + 1
		 ENDIF
  300	      CONTINUE

* See if a user-defined LET/D variable name matches
              DO 350 nv = 1, maxvars
	         IF ( (uvar_dset(nv) .EQ. iset) .AND.
     .                (STR_SAME(uvar_name_code(nv), vname) .EQ. 0) ) THEN
                    imatch = imatch + 1
		 ENDIF
  350	      CONTINUE

  400	   CONTINUE

           IF (imatch .NE. nsets) THEN
	      IF ( .NOT.agg_quiet ) 
     .	         CALL WARN('Exclude variable from aggregate. '//
     .	         'Does not appear in all member datasets: '//
     .	         vname)
	      GOTO 800 ! Variable not in all datasets
	   ENDIF

           grid = ds_grid_number(ivar)
	   IF (grid_line(e_dim, grid) .NE. 0) GOTO 5400

* Check that grids match for file variables. 
* NOTE if the variables in the member datasets share the same grid, and
*      there is a record axis, then they may have different timesteps but 
*      grid_line and ds_grid_start, ds_grid_end will have the same contents.
*      How to find that case? (check line_mem) Or just allow it?

           imatch = 1
	   has_uvars = .FALSE.
           DO 600 ii = 2, nsets
	      iset = member_sets(ii)
              DO 500 nv = 1, maxvars
	         IF ( (ds_var_setnum(nv) .EQ. iset) .AND. 
     .                (STR_SAME(ds_var_code(nv), vname) .EQ. 0) ) THEN
                    igrid = ds_grid_number(nv)
		    DO 550 idim = 1, nferdims
		       IF ( grid_line(idim, grid) .NE.  
     .                      grid_line(idim, igrid) ) GOTO 500
  550	            CONTINUE
                    imatch = imatch + 1
		 ENDIF
	         IF ( (uvar_dset(nv) .EQ. iset)   .AND. 
     .                (STR_SAME(uvar_name_code(nv), vname) .EQ. 0) ) THEN
		    imatch = imatch + 1  ! will check user-var grids later
		    has_uvars = .TRUE.
	         ENDIF
  500	      CONTINUE
  600	   CONTINUE

*  variable in all datsets but they dont have the same grid.
	   IF (imatch .NE. nsets) THEN
	      IF ( .NOT.agg_quiet ) 
     .	         CALL WARN('Exclude variable from aggregate. '//
     .	         'Grid differs in member datasets: '//
     .	         vname)
	      GOTO 800 
	   ENDIF
	      
* same grid for this variable in all datasets 
	   
           nvars = nvars + 1
	   IF (has_uvars) grid = unspecified_int4
	   CALL DEFINE_AGGREGATE_GRID (agg_dset, nvars, vname,
     .           ivar, iline, grid, e_dim, member_sets, nsets)
	   IF (status .NE. ferr_ok) GOTO 5000

  800   CONTINUE

* See if user-defined (LET/D=) variables from the first dataset are 
* shared by all the datsets

	dset = member_sets(1)
	DO 1800 ivar = 1, maxvars
           IF ( uvar_dset(ivar) .NE. dset ) GOTO 1800
	   vname = uvar_name_code(ivar)
	   imatch = 1
	   DO 1400 ii = 2, nsets
	      iset = member_sets(ii)
              DO 1300 nv = 1, maxvars
	         IF ( (ds_var_setnum(nv) .EQ. iset) .AND.
     .                (STR_SAME(ds_var_code(nv), vname) .EQ. 0) ) THEN
                    imatch = imatch + 1
		    grid = ds_grid_number(nv)

		 ENDIF
 1300	      CONTINUE

* See if a user-defined LET/D variable name matches
              DO 1350 nv = 1, maxvars
	         IF ( (uvar_dset(nv) .EQ. iset) .AND.
     .                (STR_SAME(uvar_name_code(nv), vname) .EQ. 0) ) THEN
                    imatch = imatch + 1
		 ENDIF
 1350	      CONTINUE

 1400	   CONTINUE

	   IF (imatch .NE. nsets) GOTO 1800 ! Variable not in all datasets

	   nvars = nvars + 1

* Do not define the grid for the variable when it has members that are
* user-variables 


* Cannot check grids for user-variables at this point. Just go ahead and
* add the variable to the datset.
* ??? Inherit uvar settings for the aggregate dataset

           DO 1720 nv = 1, max_uvar
	     IF ( uvar_num_items(nv) .EQ. uvar_deleted ) THEN
	        CALL deleted_list_modify(uvar_num_items_head, nv, uvar_on_hold)
	        GOTO 1725
	     ENDIF
 1720      CONTINUE
 1725      uvar_dset(nv)      = agg_dset
* copy name code and text into slot and change as needed to upper case
	   CALL string_array_modify_upcase(uvar_name_code_head, nv, 
     .             uvar_name_code(ivar), LEN(uvar_name_code(ivar)))
	   uvar_text(nv)      = uvar_text(ivar)
	   uvar_bad_data(nv)  = uvar_bad_data(ivar)
	   uvar_units(nv)     = uvar_units(ivar)
	   uvar_title(nv)     = uvar_title(ivar)
	   uvar_need_dset(nv) = uvar_need_dset(ivar)
	   uvar_data_type(nv,agg_dset) = 
     .        uvar_data_type(ivar,member_sets(1))
	   uvar_num_items(nv) = uvar_num_items(ivar)
	   DO 1730 item = 1, uvar_num_items(nv)
	      uvar_item_type(item,nv)  = uvar_item_type(item,ivar)
	      uvar_item_start(item,nv) = uvar_item_start(item,ivar)
	      uvar_item_end(item,nv)   = uvar_item_end(item,ivar)
 1730      CONTINUE

	   dset = member_sets(1)
	   uvar_grid(nv,agg_dset) = uvar_grid(ivar,dset) !  is -999 

           CALL CD_ADD_VAR( agg_dset, uvar_name_code(nv), nv, 
     .         uvar_title(nv), uvar_units(nv), uvar_data_type(nv,agg_dset), 
     .         .FALSE., uvar_bad_data(nv), status)

 1800	CONTINUE

* If no variables from the datasets match, delete this dataset. 

	if (nvars .EQ. 0) goto 5500

* Add aggregate members to aggregate dataset

	DO 2000 iset = 1, nsets
	   status = NCF_ADD_AGG_MEMBER (agg_dset, member_sets(iset)) 
 2000	CONTINUE

	IF (dtitle(1:2) .NE. char_init) THEN
	   ds_title(agg_dset) = dtitle
	ELSE
	   ds_title(agg_dset) = 'Ensemble'
	ENDIF
	ds_mod_title(agg_dset) = ' '
	ds_time_axis(agg_dset) = ds_time_axis(member_sets(1))
	ds_format(agg_dset) = ds_format(member_sets(1))

 5000   RETURN	

* Errors: too-many datasets open so aggregate dataset can't be opened,
* or error on a member dataset. 
* ?? If some member datasets were opened, close them??)
* Return no-dataset-number.

 9950	agg_dset = unspecified_int4
	status = ferr_TMAP_error
	GOTO 5000

 5100   CALL ERRMSG( ferr_syntax,status,'dataset name already in use:'
     .		//dname(:TM_LENSTR1(dname)), *5000 )
     
 5300	CALL ERRMSG( ferr_aggregate_error, status, ' ', *9950 )

 5400	CALL ERRMSG( ferr_TMAP_error, status,
     .		' Variables in member datasets already '
     .		//'have axis in the E direction', *9950 )

 5500	status = NCF_DELETE_DSET(agg_dset)
	CALL CANCEL_DATA_SET( agg_dset, status )
	agg_dset = unspecified_int4

	CALL ERRMSG( ferr_aggregate_error, status,
     .		'No valid datasets or datasets share no variables. ', 
     .		*9950 )
	END

