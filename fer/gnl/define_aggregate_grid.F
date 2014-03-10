	SUBROUTINE DEFINE_AGGREGATE_GRID (agg_dset, nvars, vname,
     .      fvar, iline, grid, agg_dim, member_sets, nsets)

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
* Define ensemble axis and grid using that axis. For uvars, grid is 
* set to unspecified_int4 when calling this. Will defer the grid check for later.

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
	include 'interp_stack.parm'
	include 'netcdf.inc'

* calling argument declarations:
	INTEGER agg_dset, nvars, fvar, iline, grid,
     .		agg_dim, member_sets(*), nsets
	CHARACTER*(*) vname

* local variable declarations:
	LOGICAL	its_dynamic, itsa_uvar
	INTEGER STR_SAME,
     .		big_cx, igrid, idim, gnum, nv, type, 
     .		ii, iset, slen, vtype, status

* Just for testing, get info back from the attribute-handling structure
	CHARACTER varname*128, name*128
	INTEGER vartype, nvdims, vdims(8), nvatts, all_outflag, varid
	LOGICAL coordvar

* Define aggregate grid

      igrid = unspecified_int4
      IF (grid .NE. unspecified_int4) THEN
         igrid = 0
         CALL TM_NEXT_TMP_GRID (igrid, 710)
  710    CALL TM_ALLO_TMP_GRID (igrid, status)
         IF (status .NE. ferr_ok) GOTO 5000

         DO 700 idim = 1, nferdims
            grid_line(idim, igrid) = grid_line(idim,grid)
  700    CONTINUE
         grid_line(agg_dim, igrid) = iline
         
* Make up a unique grid name for aggregate grid
           IF (agg_dim .EQ. 5) grid_name(igrid) = 'GEN'
           IF (agg_dim .EQ. 6) grid_name(igrid) = 'GFN'
           CALL TM_NEW_GRID_NAME( grid_name(igrid), name )
           grid_name(igrid) = name

	   gnum = igrid
      ENDIF

* ds_var slot, fill with aggregate variable in aggregate dataset.

      itsa_uvar = .FALSE.
      DO 720 nv = 1, maxvars
	      IF ( STR_SAME(ds_var_code(nv),char_init) .EQ. 0 ) GOTO 725
 720	CONTINUE
 725	ds_var_setnum(nv)   = agg_dset
	CALL string_array_modify(ds_var_code_head, nv, 
     .         ds_var_code(fvar), LEN(ds_var_code(fvar)))
	ds_var_title(nv)    = ds_var_title(fvar)
	IF ( ds_var_title(nv) .EQ. char_init128) 
     .       ds_var_title(nv)  = varname
	ds_var_titl_mod(nv) = ds_var_titl_mod(fvar)
	ds_var_units(nv)    = ds_var_units(fvar)
	ds_var_type(nv)     = ds_var_type(fvar)
	ds_missing_flag(nv) = bad_val4
	ds_bad_flag(nv)     = bad_val4
	cd_scale_factor(nv) = 1.
	cd_add_offset(nv)   = 0.
	cd_scaled(nv)       = .FALSE.

	IF (grid .NE. unspecified_int4) THEN
	   DO 730 idim = 1, nferdims
	      ds_grid_start(idim,nv)= ds_grid_start(idim,fvar)
	      ds_grid_end(idim,nv)  = ds_grid_end(idim,fvar)
	      ds_ordering(idim,nv)  = ds_ordering(idim,fvar)
 730	   CONTINUE
	   ds_ordering(agg_dim,nv) = agg_dim
	   ds_grid_start(agg_dim,nv) = 1
	   ds_grid_end(agg_dim,nv) = nsets
	   ds_grid_number(nv) = gnum
	ENDIF

* Type should be NCFLOAT or NCCHAR
	IF (ds_var_type(nv) .EQ. ptype_string) THEN
	   type = NCCHAR
	ELSE
	   type = NCFLOAT
	ENDIF
      
	CALL CD_ADD_VAR( agg_dset, vname, nvars, 
     .         ds_var_title(nv), ds_var_units(nv), type, 
     .         .FALSE., ds_bad_flag(nv), status)

* Add aggregate-grid info to the lists in aggregate datset and aggregate-grid list.
* Mark user-variables with a grid number of -999

	DO 750 ii = 1, nsets
	   iset = member_sets(ii)
	   DO 740 nv = 1, maxvars
	      IF ( (ds_var_setnum(nv) .EQ. iset) .AND. 
     .             (STR_SAME(ds_var_code(nv), vname) .EQ. 0) ) THEN
	         igrid = ds_grid_number(nv)
		 vtype = 1 ! file var
	         CALL CD_ADD_AGG_VAR_INFO (agg_dset, vname, ii, vtype, 
     .             iset, igrid, iline, nv, status)
	      ENDIF
	      IF ( (uvar_dset(nv) .EQ. iset) .AND. 
     .             (STR_SAME(uvar_name_code(nv), vname) .EQ. 0) ) THEN
	         igrid = unspecified_int4
		 vtype = 3 ! user var
	         CALL CD_ADD_AGG_VAR_INFO (agg_dset, vname, ii, vtype, 
     .             iset, igrid, iline, nv, status)
	      ENDIF
 740	   CONTINUE
 750	CONTINUE

 5000	RETURN
     
	END
