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
	INTEGER STR_SAME, TM_LENSTR1,
     .		big_cx, igrid, idim, gnum, nv, type, ii, iset, slen, 
     .		ndim, vtype, attype, attoutflag, i2, status

* Getting info back from the attribute-handling structure
	CHARACTER dimname*128, buff*128, tcoordname*128, name1*128
	INTEGER varid, tline, g2d

* initialize
      tline = unspecified_int4

* Define aggregate grid

      igrid = unspecified_int4
      IF (grid .NE. unspecified_int4) THEN
         igrid = 0
         CALL TM_NEXT_TMP_GRID (igrid, 710)
  710    CALL TM_ALLO_TMP_GRID (igrid, status)
         IF (status .NE. ferr_ok) GOTO 5000

         DO 700 idim = 1, nferdims
            grid_line(idim, igrid) = grid_line(idim,grid)
	    IF (idim .EQ. t_dim) tline = grid_line(idim, igrid)
  700    CONTINUE
         grid_line(agg_dim, igrid) = iline
         
* Make up a unique grid name for aggregate grid
         IF (agg_dim .EQ. t_dim) grid_name(igrid) = 'GEN'
         IF (agg_dim .EQ. f_dim) grid_name(igrid) = 'GFN'
         CALL TM_NEW_GRID_NAME( grid_name(igrid), buff )
         grid_name(igrid) = buff

	 gnum = igrid
      ENDIF

* If it's a Forecast aggregation, we will define a 2D time variable. Create its name.
* The GRIB and thredds FMRC use the same name as the t dimension. We'll add
* some underscores to the name to make it obviuos

      IF (agg_dim.EQ.f_dim .AND. tline.NE.unspecified_int4) tcoordname = 
     .  'T__'//line_name(tline)

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
     .       ds_var_title(nv)  = ds_var_code(nv)
	ds_var_titl_mod(nv) = ds_var_titl_mod(fvar)
	ds_var_units(nv)    = ds_var_units(fvar)
	ds_var_type(nv)     = ds_var_type(fvar)
	ds_missing_flag(nv) = bad_val4
	ds_bad_flag(nv)     = bad_val4
	cd_scale_factor(nv) = 1.
	cd_add_offset(nv)   = 0.
	cd_scaled(nv)       = .FALSE.

	IF (grid .NE. unspecified_int4) THEN
	   ndim = 1
	   DO 730 idim = 1, nferdims
	      ds_grid_start(idim,nv)= ds_grid_start(idim,fvar)
	      ds_grid_end(idim,nv)  = ds_grid_end(idim,fvar)
	      ds_ordering(idim,nv)  = ds_ordering(idim,fvar)
	      IF (ds_ordering(ndim,fvar) .GT. 0) ndim = ndim + 1
 730	   CONTINUE
	   ds_grid_start(agg_dim,nv)= 1
	   ds_grid_end(agg_dim,nv)  = nsets
	   ds_ordering(ndim,nv)     = agg_dim
           
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
        IF (nvars .EQ. 1) name1 = vname

* If this is a forecast aggregation, then the 2D time variable has been defined 
* (or will be, just below). Add a coordinates attribute to variables in the FMRC 
* dataset which now have a 2D forecast-time coordinate variable. Named by 
* prepending T__ to the time coordinate.
* Also add an attribute (name to be finalized), _ftime2dRef pointing to TIME2D
	      
	IF (agg_dim.EQ.f_dim .AND. grid_line(t_dim, gnum).NE.int4_init) THEN
	   buff = " "
	   slen = 0
	   DO 2200 idim = nferdims, 1, -1 
	      IF (grid_line(idim, gnum) .NE. int4_init) THEN
	         dimname = line_name(grid_line(idim, gnum))
		 IF (idim .EQ. t_dim) dimname = tcoordname
	         IF (slen.EQ.0) THEN
		    buff = dimname(:TM_LENSTR1(dimname) )
		 ELSE
 	            buff = buff(:slen)//' '//dimname(:TM_LENSTR1(dimname) )
                 ENDIF
		 slen = TM_LENSTR1(buff)
	      ENDIF
 2200	   CONTINUE
	   CALL CD_GET_VAR_ID (agg_dset, vname, varid, status)
	   slen = TM_LENSTR1(buff)
	   attoutflag = 1
	   attype = NCCHAR
	   CALL CD_PUT_NEW_ATTR (agg_dset, varid, 'coordinates', attype, 
     .             slen, attoutflag, buff, 0., status )

	   slen = TM_LENSTR1(tcoordname)
	   attoutflag = 1
	   attype = NCCHAR
	   CALL CD_PUT_NEW_ATTR (agg_dset, varid, '_ftime2dRef', attype, 
     .             slen, attoutflag, tcoordname, 0., status )

	ENDIF

* Add aggregate-grid info to the lists in aggregate datset and aggregate-grid list.
* Mark user-variables with a grid number of -999

	DO 750 ii = 1, nsets
	   iset = member_sets(ii)
	   DO 740 nv = 1, maxvars
	      IF ( (ds_var_setnum(nv) .EQ. iset) .AND. 
     .             (STR_SAME(ds_var_code(nv), vname) .EQ. 0) ) THEN
	         igrid = ds_grid_number(nv)
		 vtype = cat_file_var ! file var
	         CALL CD_ADD_AGG_VAR_INFO (agg_dset, vname, ii, vtype, 
     .             iset, igrid, iline, nv, status)
	      ENDIF
	      IF ( (uvar_dset(nv) .EQ. iset) .AND. 
     .             (STR_SAME(uvar_name_code(nv), vname) .EQ. 0) ) THEN
	         igrid = unspecified_int4
		 vtype = cat_user_var! user var
	         CALL CD_ADD_AGG_VAR_INFO (agg_dset, vname, ii, vtype, 
     .             iset, igrid, iline, nv, status)
	      ENDIF
 740	   CONTINUE
 750	CONTINUE

 
* Define 2D time variable for forecast datasets
	IF (agg_dim.EQ.f_dim .AND. tline.NE.unspecified_int4) THEN

	   CALL CD_GET_VAR_ID (agg_dset, tcoordname, varid, status)
	   IF (status .EQ. ferr_ok) GOTO 5000  ! already defined

	   igrid = 0
	   CALL TM_NEXT_TMP_GRID (igrid, 760)
  760	   CALL TM_ALLO_TMP_GRID (igrid, status)
	   IF (status .NE. ferr_ok) GOTO 5000
	   grid_name(igrid) = 'FTI'
	   g2d = igrid

* ds_var slot, define TIME2D

	   DO 770 nv = 1, maxvars
	      IF ( STR_SAME(ds_var_code(nv),char_init) .EQ. 0 ) GOTO 775
 770	   CONTINUE
 775	   ds_var_setnum(nv)   = agg_dset
	   CALL string_array_modify(ds_var_code_head, nv, 
     .         tcoordname, LEN(tcoordname))
	   ds_var_title(nv)    = 'Time coordinates for ForecastModelRunCollection'
	   ds_var_titl_mod(nv) = 'Defined by Ferret FMRC aggregation'
	   ds_var_units(nv)    = line_units(tline)
	   ds_var_type(nv)     = ptype_float
	   ds_missing_flag(nv) = bad_val4
	   ds_bad_flag(nv)     = bad_val4
	   cd_scale_factor(nv) = 1.
	   cd_add_offset(nv)   = 0.
	   cd_scaled(nv)       = .FALSE.

	   IF (g2d .NE. unspecified_int4) THEN
	      DO 780 idim = 1, nferdims
	         grid_line(idim, g2d) = 0
	         ds_grid_start(idim,nv) = 1
	         ds_grid_end(idim,nv) = 1
	         ds_ordering(idim,nv) = 0
 780	      CONTINUE

	      grid_line(t_dim, g2d) = grid_line(t_dim,gnum)
	      grid_line(f_dim, g2d) = iline

	      ds_grid_start(t_dim,nv) = 1
	      ds_grid_end(t_dim,nv) = line_dim(tline)
	      ds_ordering(1,nv) = t_dim

	      ds_grid_start(f_dim,nv) = 1
	      ds_grid_end(f_dim,nv) = nsets
	      ds_ordering(2,nv) = agg_dim

	      ds_grid_number(nv) = g2d
	   ENDIF

	   type = NCFLOAT
	   nvars = nvars + 1
      
	   CALL CD_ADD_VAR( agg_dset, tcoordname, nvars, 
     .         ds_var_title(nv), ds_var_units(nv), type, 
     .         .FALSE., ds_bad_flag(nv), status)

* Add the aggregate-grid info with CALL CD_ADD_AGG_VAR_INFO
* save the grid of the member datasets so we can get back the
* right time axis for each member, but set member dataset to 0
* because the 2d Time variable is not in the member sets.

	   DO 790 ii = 1, nsets
              iset = member_sets(ii)
	      DO 795 i2 = 1, maxvars
	         IF ( (ds_var_setnum(i2) .EQ. iset) .AND. 
     .                (STR_SAME(ds_var_code(i2), name1) .EQ. 0) ) 
     .            igrid = ds_grid_number(i2)
 795	      CONTINUE

	      iset = 0
	      vtype = cat_agg_f_var
	      CALL CD_ADD_AGG_VAR_INFO (agg_dset, tcoordname, ii, vtype, 
     .             iset, igrid, iline, nv, status)
 790	   CONTINUE

* Give the time2D variable a _CoordinateAxisType attribute

	   CALL CD_GET_VAR_ID (agg_dset, tcoordname, varid, status)
	   attype = NCCHAR
	   CALL CD_PUT_NEW_ATTR (agg_dset, varid, '_CoordinateAxisType', 
     .       attype, 4, attoutflag, 'Time', 0., status )

	ENDIF  ! Forecast - 2D time 

 5000	RETURN
     
	END
