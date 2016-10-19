	SUBROUTINE DEFINE_U_AGGREGATE_VAR (agg_dset, iset, nvars, vname, 
     .         fvar, itsa_uvar, nsets, status)

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
* Save variable in the Union aggregation.  Check that the variable name
* is not already in the aggregation.

* Programmer Ansley Manke
* NOAA/PMEL, Seattle, WA - SDIG

* V7   5/12 *acm* Union aggregations: DEFINE DATA/AGG/U
* V710 *acm* 10/16 Ticket 2422: variable uvar_grids to be replaced with calls to 
*                  LIST routines in NCF_Util.c 
* V710 *acm* 10/16 Ticket 2422: Also removing need for uvar_data_type(uvar,dset). 


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
	include 'interp_stack.parm'
	include 'netcdf.inc'

* calling argument declarations:
	LOGICAL itsa_uvar
	INTEGER agg_dset, iset, nvars, fvar, nsets, status
	CHARACTER*(*) vname

* local variable declarations:
	INTEGER STR_SAME, TM_LENSTR1, DIR_LINE,
     .		big_cx, idim, nv, nu, type, slen, igrid, vtype, 
     .		i, num_indices, varid, iline, imemb, itype
	INTEGER igrid_test
	CHARACTER*8 TM_FMT, fset

* Is this variable already in the union aggregation? If so, do not add it.

      CALL CD_GET_VAR_ID (agg_dset, vname, varid, status)
      IF (varid .GT. 0) GOTO 5100

* ds_var slot, fill with next variable in Union dataset.
      IF (.NOT. itsa_uvar) THEN
	DO 720 nv = 1, maxvars
	      IF ( STR_SAME(ds_var_code(nv),char_init) .EQ. 0 ) GOTO 725
 720	CONTINUE
 725	ds_var_setnum(nv)   = agg_dset
	CALL string_array_modify(ds_var_code_head, nv, 
     .         ds_var_code(fvar), LEN(ds_var_code(fvar)))
! ?? better: inherit these attibutes from members in SET_AGG_VAR_ATTS
	ds_var_title(nv)    = ds_var_title(fvar)
	IF ( ds_var_title(nv) .EQ. char_init128) 
     .       ds_var_title(nv)  = ds_var_code(fvar)
	ds_var_titl_mod(nv) = ds_var_titl_mod(fvar)
	ds_var_units(nv)    = ds_var_units(fvar)
	ds_missing_flag(nv) = ds_missing_flag(fvar)	! was bad_val4
	ds_bad_flag(nv)     = ds_bad_flag(fvar)		! was bad_val4
	ds_var_type(nv)     = ds_var_type(fvar)
	cd_scale_factor(nv) = 1.
	cd_add_offset(nv)   = 0.
	cd_scaled(nv)       = .FALSE.

	DO 730 idim = 1, nferdims
	   ds_grid_start(idim,nv)= ds_grid_start(idim,fvar)
	   ds_grid_end(idim,nv)  = ds_grid_end(idim,fvar)
	   ds_ordering(idim,nv)  = ds_ordering(idim,fvar)
 730	CONTINUE
	ds_grid_number(nv)       = ds_grid_number(fvar)
	
* Type should be NCFLOAT or NCCHAR
	IF (ds_var_type(nv) .EQ. ptype_string) THEN
	   type = NCCHAR
	ELSE
	   type = NCFLOAT
	ENDIF

* Add the aggregated variable to the linked list structure
	CALL CD_ADD_VAR( agg_dset, vname, nvars, ds_var_title(nv), 
     .         ds_var_units(nv), type, .FALSE., ds_bad_flag(nv), status)

* Store the dataset number of this variable.

	   igrid = ds_grid_number(nv)
	   vtype = cat_file_var
	   imemb = nvars
	   iline = unspecified_int4
	   
	   CALL CD_GET_VAR_ID (iset, vname, varid, status)
	   CALL CD_ADD_AGG_VAR_INFO (agg_dset, vname, imemb, 
     .          vtype, iset, igrid, iline, fvar, status)

      ELSE ! done with file vars

* User Variables
* ... optimize search using deleted_list_get_undel

	CALL deleted_list_get_undel(uvar_num_items_head,
     .                              deleted_list_result(1),
     .                              max_uvar, num_indices )
	DO i = 1, num_indices
	   nu = deleted_list_result(i)
	   IF ( uvar_num_items(nu) .EQ. uvar_deleted) CYCLE
	   IF ( (uvar_dset(nu) .EQ. iset) .AND. 
     .         (STR_SAME(uvar_name_code(nu), vname) .EQ. 0) ) THEN
	      vtype = cat_user_var
	      
	      DO 740 nv = 1, maxvars
	      IF ( STR_SAME(ds_var_code(nv),char_init) .EQ. 0 ) EXIT ! from loop
 740	      CONTINUE
	      ds_var_setnum(nv)   = agg_dset
c	      ds_var_code(nv)     = uvar_name_code(nu)
	      CALL string_array_modify(ds_var_code_head, nv, 
     .                            uvar_name_code(nu), LEN(uvar_name_code(nu)))
	      ds_var_title(nv)    = uvar_title(nu)
	      IF ( ds_var_title(nv) .EQ. char_init128) 
     .            ds_var_title(nv)  = uvar_text(nu)    
	      IF ( ds_var_title(nv) .EQ. ' ') 
     .            ds_var_title(nv)  = uvar_text(nu)     
	      ds_var_titl_mod(nv) = ' '
	      ds_var_units(nv)    = uvar_units(nu)
	      ds_missing_flag(nv) = uvar_bad_data(nu)
	      ds_bad_flag(nv)     = uvar_bad_data(nu)

	      cd_scale_factor(nv) = 1.
	      cd_add_offset(nv)   = 0.
	      cd_scaled(nv)       = .FALSE.

	      CALL GET_UVAR_GRID( nu, iset, status )
	      CALL GET_SAVED_UVAR_GRID(nu, iset, igrid)

	      ds_grid_number(nv) = igrid
	      DO idim = 1, nferdims
	         iline = grid_line( idim, igrid )
	         ds_grid_start(idim,nv)= 1
	         ds_grid_end(idim,nv)  = line_dim(iline)
	         ds_ordering(idim,nv)  = DIR_LINE(line_direction(iline))
	      ENDDO
* Type  ??
	      
	      CALL GET_SAVED_UVAR_DTYPE(nu, iset, itype)
	      IF (itype .EQ. unspecified_int4) itype = ptype_unknown
	      IF (itype .NE. uvar_data_type(nu, iset)) THEN
	         CALL WARN("UVGRIDERR: define_u_aggregate_var")
	      ENDIF
	      ds_var_type(nv)  = itype
c*c REMOVE LATER, replace the ref to uvar_data_type(,) with a 
c*c call GET_SAVED_UVAR_DTYPE returning itype
	      ds_var_type(nv)  = uvar_data_type(nu, iset)

	      IF (ds_var_type(nv) .EQ. ptype_string) THEN
	         type = NCCHAR
	      ELSE
	         type = NCFLOAT
	      ENDIF

* Hook the aggregated variable into the linked list structures
	      CALL CD_ADD_VAR( agg_dset, vname, nvars, uvar_title(nu), 
     .         uvar_units(nu), type, .FALSE., uvar_bad_data(nu), status)

* Store the info of this variable. The member dataset stored is the 
* LET/D= datset where it was defined.

	      CALL GET_UVAR_GRID( nu, iset, status )
	      CALL GET_SAVED_UVAR_GRID(nu, iset, igrid)
	      vtype = cat_user_var
	      imemb = nvars
	      iline = unspecified_int4
	      
	      CALL CD_ADD_AGG_VAR_INFO (agg_dset, vname, imemb, 
     .          vtype, iset, igrid, iline, nu, status)

* ... bump the use count for this grid to protect it from being deleted if
*     the user changes the LET/D definition -- grid is needed during the
*     gather process (IS_STRIP & IS_AGGREGATE_GATHER).
*  ==> be sure to un-use (deallo) it when the aggregation gets canceled
*  ==> or if an error occurs prior to the complete initialization of this agg
	      CALL TM_USE_DYN_GRID(igrid)
	      EXIT
	   ENDIF
	ENDDO

	ENDIF  ! itsa_uvar 

 5000	RETURN
 5100	fset = TM_FMT(FLOAT(iset), 5, 8, slen)
	CALL WARN('Variable '//vname(:TM_LENSTR1(vname))//
     .		' from dataset '//fset(:slen)//
     .		' is already in the union dataset. Skipping it')
	nvars = nvars - 1
	status = ferr_ok
	RETURN
	END
