	SUBROUTINE CDF_LIST_DSG ( dset, cdfid, fname, append, nvars,
     .			     nfeatures, nobs_in, longest_feature,
     .			     mr_list, cx_list, title,
     .			     out_type, quiet, do_coords, status )


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
*
* Write the indicated list of variables out in netCDF format where
* data is from a DSG datset, writing to a ragged-array DSG netCDF file.

* programmer - Ansley Manke
* NOAA/PMEL, Seattle, WA - Science Data Integration Group

* 10/2018
*  2/2019 add /APPENDing as an option, for LET/D variables.
*         TODO:
*         At present the routine does not check the grids but will let the 
*         subroutine calls report any errors.  When appending, masks etc
*         get set up, but it doesn't make any sense to apply them.  
*  6/2019 *acm* MASK_DSG_OBSVAR call has changed, for use by trajectory-plotting 
*  8/2019 *acm* Allow writing user-defined varaibles to DSG datasets
* V751 *acm* 11/19 new qualifier SAVE/NOCOORDS: subroutine calls change.
* V76  *acm*  4/20 Fix when setting instance_var flag. 
* V76  *acm*  4/20 Fix for data type, writing string data
* V76  *acm*  4/20 Additions for writing subsets of point datasets
* v7.6 *acm*  5/20 Issue 1876: can read string-typed data, write as char.
* v7.61 *acm* 8/20 If there was a USE/FEATURE= then write the data with
*                  both featuretype and cf_role as that feature type.
* V7.63 *acm* 9/20 DSG trajectoryProfile, TimeseriesProfile data saves not implemented.
*                  issue an error message.

        include 'netcdf.inc'
#include "netcdf_declareUnderscore.inc"
        include 'tmap_errors.parm'
        include 'cd_lib.parm'
        include 'cdf_tmap.parm'
        include 'tmap_dims.parm'
        include 'xtm_grid.cmn_text'
	include 'xdset_info.cmn_text'
	include 'xdset_info.cd_equiv'
	include 'xdyn_linemem.cmn_text'
	include 'ferret.parm'
	include 'errmsg.parm'
	include 'xcontext.cmn'
	include 'xvariables.cmn'
	include 'xprog_state.cmn'
	include 'xrevision.cmn'
	include 'xtoday.cmn'
	include 'xinterrupt.cmn'
        include 'netcdf4_settings.cmn'
#        include "tmap_dset.parm"

* calling argument declarations:
        LOGICAL         append, quiet, do_coords
	INTEGER		dset, cdfid, nvars, mr_list(nvars), cx_list(nvars),
     .			keepax_flag, nfeatures, nobs_in, longest_feature, status
        CHARACTER*(*)   fname, title, out_type

* internal variable declarations:

	INTEGER	  max_len
	PARAMETER (max_len = 2048)

        LOGICAL         TM_LEGAL_NAME, NC_GET_ATTRIB, TM_ITSA_DSG, 
     .			itsa_uvar, its_cdf, new, mode_up_in, flushed, 
     .			enhead,itsa_string, output_units, output_title,
     .                  output_history, get_ds, edges_flag,
     .                  got_it, got_title, got_units, got_history,
     .                  got_missing, got_fill, do_warn, coordvar,
     .                  process_feature(nfeatures), process_obs(nobs_in), 
     .                  has_mask, have_dimname,
     .                  instance_coord(4), relevant_coord(4),
     .                  instance_var(nvars), point_var(nvars),
     .                  instance_vars_only, point_vars_only, synthetic_rowsiz, 
     .                  ragged, no_id

	INTEGER		TM_LENSTR1, GET_MAX_C_STRING_LEN, STR_SAME,MGRID_SIZE, 
     .			CGRID_SIZE, DSG_WHATS_IT, STR_DNCASE, TM_DSG_DSET_FROM_GRID,
     .			cx, mr, grid,  ivar, nready, lbuff, cdfstat, 
     .                  idim, i, recdim, do_bounds, cat, var, cat1, var1, 
     .			iaxis, final_status, maxstrlen, 
     .			write_lo(nferdims), write_hi(nferdims), 
     .			lo_e(nferdims), hi_e(nferdims), 
     .			lo_o(nferdims), hi_o(nferdims), stride(nferdims),
     .                  dset_last, dset_num, attid, len, varid, vartype, 
     .                  nvdims,  nvatts, vdims(8), iatt, slen, alen, 
     .                  attype, attlen, attoutflag, all_outflag,
     .                  outtype, egrid, ogrid, maxlen, fvar_varid,
     .                  ifeature, ifeat, iobs, n_feat, n_obs, nobsf,
     .                  orientation, obsdimlen, featr_ivar, featr_mr, coord_lm(4), 
     .                  ptr, bufflen, line, gxlate, row_size_lm, clen, nblank,
     .                  slen1, slen2, idummy_line, base, fobs, count,
     .                  line_fcoord, line_obscoord, dummy, nmasked, dset_dsg,
     .                  dndims, dlen, dnv, ngatts, ftype, f_direc
	REAL*8		GET_LINE_COORD, vals(100), bad, val, scalefac, addoff

	CHARACTER	SANITARY_VAR_CODE*128, TM_FMT*48,
     .                  CD_DSG_FEATURENAME*20, CD_AXIS_NAME*128, 
     .                  varcode*128, varname*128, buff*2048, 
     .                  buff1*128, buff2*128, obsdimname*128, 
     .                  ftrname*20, attname*128, out_typ_in*20

* local parameter declarations:

* For error messages
	CHARACTER*9 typnam(6)
	DATA typnam/'NC_BYTE', 'NC_CHAR', 'NC_SHORT', 'NC_INT', 
     .            'NC_FLOAT', 'NC_DOUBLE'/
	
* initialize
	ivar = 1  ! just house-keeping -- make sure all is init'ed
	flushed = .FALSE. ! on error, whether OK variables were written
	final_status = ferr_ok  ! innocent unless proven guilty
	do_warn = .NOT.quiet
	got_missing = .FALSE.
	got_fill = .FALSE.
	recdim = no_dim
	enhead = .FALSE.
	edges_flag = .FALSE.
	do_bounds  = -1
	itsa_uvar = .FALSE.
	itsa_string = .FALSE.
	keepax_flag = 0
	maxlen = 128
	its_cdf = .TRUE.  ! later, write data to cdf from other sources...
	
	out_typ_in = out_type
	mode_up_in =  mode_upcase_output
	mode_upcase_output = .FALSE.

	egrid = mgrid_buff
	ogrid = mgrid_buff
	line_fcoord = mnormal
	line_obscoord = mnormal

* Some sanity checking

	ivar = 1
	cx  = cx_list(ivar)
	grid	= cx_grid( cx )
	grid_is_dsg = TM_ITSA_DSG( grid )

* For handling user-vars and expressions based in another non-dsg dataset
	dset_dsg = TM_DSG_DSET_FROM_GRID( grid )  

	dset_num = cx_data_set( cx_list(ivar) )
	dset_num = dset_dsg

	orientation = dsg_orientation(dset_num)
	IF (orientation.EQ.pfeatureType_TrajectoryProfile .OR. 
     .	       orientation.EQ.pfeatureType_TimeseriesProfile ) GOTO 5700

* The variables must all be from the same DSG dataset.

* are there any observation vars?  Or is this all instance vars?
	point_vars_only    = .TRUE.
	instance_vars_only = .TRUE.
	DO ivar = 1, nvars
	   cx  = cx_list(ivar)
           IF (cx_data_set( cx ).GT.pdset_irrelevant .AND. cx_data_set(cx).NE.dset) GOTO 4200  ! error message for inconsistent datasets

	   grid	= cx_grid( cx )
	   instance_var(ivar) = DSG_WHATS_IT(grid) .EQ. pdsg_instance
	   point_var(ivar) = CGRID_SIZE(cx) .EQ. 1
	   IF (.NOT.point_var(ivar)   ) point_vars_only    = .FALSE.
	   IF (.NOT.instance_var(ivar)) instance_vars_only = .FALSE.

	ENDDO

	ivar = 1
	cx  = cx_list(ivar)
	grid	= cx_grid( cx )

* The varid of the feature-id variable
 
	varcode = SANITARY_VAR_CODE(cat_file_var, dsg_feature_var(dset) )
	CALL CD_GET_VAR_ID  (dset, varcode, fvar_varid, status)
	buff1 = varcode

* which dimensions are relevant to show

	have_dimname = .FALSE.

	gxlate = dsg_xlate_grid(dset)
	DO idim = 1, 4
	   line = grid_line(idim, gxlate)
	   relevant_coord(idim) =  line.NE.mnormal .AND. .NOT.point_vars_only
	   IF (relevant_coord(idim)) THEN
	      ivar = dsg_coord_var(idim,dset)
	      coord_lm(idim) = dsg_loaded_lm(ivar)  ! line-memory table indices
	      instance_coord(idim) = line_dim(line) .EQ. nfeatures

* get obs dimension name ! test this with NCEI files that have vars on (trajectory,obs) axis
* with trajectory length = 1

	      IF (.NOT. instance_coord(idim) .AND. .NOT. have_dimname) THEN
	        
	         varcode = SANITARY_VAR_CODE( cat_file_var, ivar )
		 CALL CD_GET_VAR_ID (dset, varcode, varid, status)

		 CALL CD_GET_VAR_DIMS_INFO (dset, varid, varcode, nvdims, vdims, status)
	         CALL CD_GET_DS_DIMS (dset, vdims(1), obsdimname, i, status)
	         have_dimname = .TRUE.
	      ENDIF

	   ENDIF

	ENDDO

* For new files, the feature-id var has been loaded, after the data variables requested 
* in the LIST command. Find it and keep track its order as loaded.

	featr_ivar = var_name_nonexist

	DO ivar = 1, nvars
	   cx  = cx_list(ivar)
	   mr  = mr_list(ivar)
	   cat = cx_category(cx)
	   var = cx_variable(cx)
	   varcode = SANITARY_VAR_CODE( cat, var )
	   CALL CD_GET_VAR_ID (dset, varcode, varid, status)

	   IF (varid .EQ. fvar_varid) THEN
	      featr_ivar = ivar
	      featr_mr = mr
	      EXIT
	   ENDIF
	ENDDO

	ivar = 1
	cx   = cx_list(ivar)
	grid = cx_grid( cx )

* Create a mask showing which features to list.
	cx  = cx_list(ivar)
	CALL MAKE_DSG_FEATURE_MASK(dset, cx, process_feature, nfeatures)

* Find the number of features to list after masking, and the count
* of obs to write after masking.

* If the incoming file has no rowsize and if there is just one feature, we treat 
* the dataset as a 1-D DSG file and we have synthezised a RowSize for internal use.
* In this case skip writing a RowSize.
* also skip RowSize if it's a point-type dataset
 
	varid = dsg_row_size_varid(dset)
	synthetic_rowsiz = (varid .EQ. unspecified_int4) 

	IF (synthetic_rowsiz) THEN

	   n_feat = 1
	   n_obs = nobs_in
	   IF (orientation .EQ. e_dim) THEN
	      
	      n_feat = 0
	      DO ifeature = 1, nfeatures
	        IF (process_feature(ifeature)) n_feat = n_feat + 1
	      ENDDO
	      n_obs = n_feat
	   ENDIF

	ELSE
	   row_size_lm = dsg_loaded_lm(dsg_row_size_var(dset))

	   n_feat = 0
	   base = 0     ! obs index at end of preceding feature
	   n_obs = 0     
	   DO ifeature = 1, nfeatures
	     nobsf = dsg_linemem(row_size_lm)%ptr(ifeature)  ! feature length
	     IF (process_feature(ifeature)) THEN

* ... get observation-level mask for this feature
	        CALL MAKE_DSG_OBS_MASK(dset, cx, ifeature, base, process_obs, nobsf)
	        fobs = 0
	        DO iobs = 1, nobsf
	           IF (process_obs(iobs)) fobs = fobs + 1
	        ENDDO
	        IF (fobs .GT. 0) THEN
	           n_obs = n_obs + fobs
		   n_feat = n_feat + 1
	        ELSE
	           process_feature(ifeature) = .FALSE.
	        ENDIF
	    ENDIF
	     base = base + nobsf
	   ENDDO

	ENDIF

* Empty request is an error
	IF (n_feat .EQ. 0) GOTO 4300  ! 

* Sanity check: names and size of dimensions of appended variable against file.
* (should check the rowsize coords too...)

	IF (append) THEN

	   status = NF_INQ ( cdfid, dndims, dnv, ngatts, recdim )
	   recdim = no_dim

	   CALL CD_DSG_GET_FEATURETYPE (cdfid, ftype, f_direc, ragged, do_warn)
	   IF (orientation.NE.f_direc .OR. .NOT.ragged) GOTO 4400


	   DO i = 1, dndims
	      status = NF_INQ_DIMNAME (cdfid, i, varname)
	      IF (STR_SAME(buff1, varname) .EQ. 0) THEN
	         status =  NF_INQ_DIMLEN(cdfid, i, dlen)
		 IF (status .EQ. NF_NOERR) THEN
		    idim = 5
		    IF (dlen .NE. n_feat) GOTO 4500
		 ENDIF
	      ENDIF

	      IF (STR_SAME(obsdimname, varname) .EQ. 0) THEN
	         status =  NF_INQ_DIMLEN(cdfid, i, dlen)
		 IF (status .EQ. NF_NOERR) THEN
		    idim = orientation
		    IF (dlen .NE. n_obs) GOTO 4500
		 ENDIF
	      ENDIF
	   ENDDO 
	ENDIF

* write the featureType global attribute

	ftrname = CD_DSG_FEATURENAME(dsg_orientation(dset))
	slen = TM_LENSTR1( ftrname )
	attname = 'featureType'

*-*-*-*-*-*-*-*-*-*-*-*-
	IF (.NOT. append) CALL CD_WRITE_ATTRIB(cdfid, pcdf_global,
     .              attname, ftrname(:slen), .FALSE., status )

* ... Define the feature-length variables specific to DSG files:  
*     feature-id variable, rowSize

* ... Define 1-D storage for masked data in the feature-direction (E-dir).
*     The masked data  in the feature-direction (E-dir) will be stored 
*     in lineedg(line_fcoord)

	CALL TM_ALLO_TMP_LINE(line_fcoord, status)
	CALL GET_LINE_DYNMEM (n_feat, line_fcoord, status)

	line_dim(line_fcoord)  = n_feat
	line_regular(line_fcoord)  = .FALSE.
	line_dim_only(line_fcoord) = .TRUE.

* get feature dimension name

	CALL CD_GET_VAR_DIMS_INFO (dset, fvar_varid, varcode, nvdims, vdims, status)
	CALL CD_GET_VAR_TYPE (dset, fvar_varid, varcode, vartype, status)
	IF (vartype .EQ. NF_CHAR) THEN 
	   CALL CD_GET_DS_DIMS (dset, vdims(2), line_name(line_fcoord), i, status)
	ELSE
	   CALL CD_GET_DS_DIMS (dset, vdims(1), line_name(line_fcoord), i, status)
	ENDIF
	line_name_orig(line_fcoord) = line_name(line_fcoord)

* ... And a grid for feature variables

	egrid = 0
	CALL TM_ALLO_TMP_GRID (egrid, status)
	IF (status .NE. ferr_ok) GOTO 5000

	DO idim = 1, nferdims
	   grid_line(idim, egrid) = mnormal
	ENDDO
	grid_line(e_dim, egrid) = line_fcoord

* ... Assign coordinates to the feature-axis for writing

	DO count = 1, n_feat
	  val = count
	  CALL PUT_LINE_COORD(linemem(line_fcoord)%ptr, count, val)
	ENDDO

* ... Define the output obs dimension, with the number of masked obs
*     The masked data  in the obs direction (obs-dir) will be stored 
*     in the lineedg(line_obscoord)

	CALL TM_ALLO_TMP_LINE(line_obscoord, status)
	CALL GET_LINE_DYNMEM (n_obs, line_obscoord, status)

	line_name(line_obscoord) = obsdimname
	line_name_orig(line_obscoord) = obsdimname
	line_dim(line_obscoord)  = n_obs
	line_regular(line_obscoord) = .FALSE.
	line_dim_only(line_obscoord)  = .TRUE.

* ... And a grid for obs variables

c	IF (.NOT. instance_vars_only) THEN
	ogrid = 0
	CALL TM_ALLO_TMP_GRID (ogrid, status)
	IF (status .NE. ferr_ok) GOTO 5000

	DO idim = 1, nferdims
	   grid_line(idim, ogrid) = mnormal
	ENDDO

	grid_line(orientation, ogrid) = line_obscoord

	DO idim = 1, nferdims
	   lo_o(idim) = unspecified_int4
	   hi_o(idim) = unspecified_int4
	ENDDO
	lo_o(orientation) = 1
	hi_o(orientation) = n_obs

* ... Assign coordinates for the obs-axis of masked data

	count = 0
	base = 0     ! obs index at end of preceding feature
	DO ifeature = 1, nfeatures
	  IF (synthetic_rowsiz) THEN
	     nobsf = n_obs
	     IF (orientation .EQ. e_dim) nobsf = 1
	  ELSE
	     nobsf = dsg_linemem(row_size_lm)%ptr(ifeature)  ! feature length
	  ENDIF
	  IF (process_feature(ifeature)) THEN

* ... get observation-level mask for this feature
	     CALL MAKE_DSG_OBS_MASK(dset, cx, ifeature, base, process_obs, nobsf)

*    indices of obs written in this feature after masking

	     DO iobs = 1, nobsf
	        IF (process_obs(iobs)) THEN
	           count = count + 1       ! coordinate index on output feature axis
		   val = count
	           CALL PUT_LINE_COORD(linemem(line_obscoord)%ptr, count, val)
		ENDIF
	     ENDDO

	  ENDIF

	  base = base + nobsf
	ENDDO

* ... when appending, do not set up the DSG feature, rowSize or coordinate variables

	IF (.NOT. append) THEN ! *-*-*-*-*-*-*-*-*-*-*-*-

* ... Define the feature-id variable

* ... Get the feature-id variable name and type.  If it's string, get the 
*     string-dimension lengthin maxstrlen, and buff2 is the string-dimension name.

	CALL CD_GET_VAR_INFO (dset, fvar_varid, varcode, vartype, nvdims, 
     .            vdims, nvatts, coordvar, all_outflag, status)
	IF (vartype .EQ. NF_CHAR) THEN
	   CALL CD_GET_DS_DIMS (dset, vdims(1), buff2, maxstrlen, status)

	   out_type = 'DFLT'
	   itsa_string = .TRUE.
	   
	ENDIF

* If the variable in the dataset is type string, synthesize a name for the string-length dimension.

	IF (vartype .EQ. nf_string) THEN
	
	   CALL CD_GET_DS_DIMS (dset, vdims(1), buff2, maxstrlen, status)
	   buff2 = TM_FMT(FLOAT(maxstrlen), 5, 12, slen)
	   buff2 = 'STRING1_'//buff2(:slen)
	ENDIF

	outtype = vartype
	
	CALL CDF_SET_OUT_TYPE (dset, fvar_varid, itsa_uvar, 
     .                itsa_string, out_type, outtype, status)

	grid = egrid

	DO idim = 1, nferdims
	   lo_e(idim) = unspecified_int4
	   hi_e(idim) = unspecified_int4
	ENDDO
	lo_e(e_dim) = 1
	hi_e(e_dim) = n_feat

	attname = 'cf_role'
	got_it = NC_GET_ATTRIB (
     .		  	     dset, fvar_varid, attname, .FALSE., ftrname,
     .		  	     maxlen, attlen, attoutflag, buff1, vals)

	no_id = (orientation.EQ.e_dim .AND. .NOT.got_it)   ! ?? or just .NOT.got_it ??
	
	IF (.NOT. no_id) THEN

	   CALL CD_MAKE_VAR( cdfid, dset, varcode, outtype, maxstrlen,
     .               grid, lo_e, hi_e, recdim, enhead, new, 
     .               edges_flag, do_bounds, mode_upcase_output, 
     .               keepax_flag, date_fmt_out,
     .               netcdf4_type, xchunk_size, 
     .               ychunk_size, zchunk_size, tchunk_size, 
     .               deflate_lev, shuffle_flag, endian_code, buff2,
     .               do_coords, status )

* ... Set attributes for feature-id variable

	   attname = 'long_name'
	   got_it = NC_GET_ATTRIB (
     .		  	     dset, fvar_varid, attname, .FALSE., ftrname,
     .		  	     maxlen, attlen, attoutflag, buff1, vals)

	   IF (status .NE. ferr_ok) buff1 = ftrname(:slen)//' ID'
	   lbuff = TM_LENSTR1(buff1)
	
	   CALL CD_WRITE_ATTRIB(cdfid, varcode, attname,
     .                              buff1(:lbuff), .FALSE., status )
     
	   attname = 'cf_role'
	   got_it = NC_GET_ATTRIB (
     .		  	     dset, fvar_varid, attname, .FALSE., ftrname,
     .		  	     maxlen, attlen, attoutflag, buff1, vals)

	   IF (status .NE. ferr_ok) THEN
	      status = STR_DNCASE(buff1, ftrname)
	      slen = TM_LENSTR1(buff1)
 	      buff1 = ftrname(:slen)//'_id'

	   ELSE
	   
* Have they done USE/FEATURE= xx  to change the feature-type?  If so 
* change the cf_role attribute and add a comment to the global attributes

	      buff = ''
	      buff2 = ftrname(:TM_LENSTR1(ftrname))//'_id'
	      IF (STR_SAME(buff1, buff2) .NE. 0) THEN
	         buff = 'Written from '//buff1(:TM_LENSTR1(buff1)-3)// 
     .                 ' file treated as Feature-Type = '//
     .                 ftrname(:TM_LENSTR1(ftrname)) 
	         buff1 = buff2
	         status = STR_DNCASE(buff1, buff2)

	      ENDIF

	   ENDIF

	   lbuff = TM_LENSTR1(buff1)
	   CALL CD_WRITE_ATTRIB(cdfid, varcode, attname,
     .                              buff1(:lbuff), .FALSE., status )

	   lbuff = TM_LENSTR1(buff)
	   IF (lbuff .GT. 1) THEN
	      attname = 'comment'
	      slen = TM_LENSTR1(buff)
	      CALL CD_WRITE_ATTRIB(cdfid, pcdf_global,
     .              attname, buff(:slen), .TRUE., status )
	   ENDIF

	ENDIF  ! no_od


* ... Now define the rowSize var. It has the same output grid.
*     The way things are stored, dsg_row_size_var(dset) contains the line-memory 
*     location of the data, and dsg_row_size_varid the varid of rowSize in datast dset.
	
	varid = dsg_row_size_varid(dset)

	IF (.NOT. synthetic_rowsiz) THEN
	   CALL CD_GET_VAR_VARNAME (dset, varid, varcode, status)

	   outtype = nf_int
	   CALL CD_MAKE_VAR( cdfid, dset, varcode, outtype, maxstrlen,
     .               grid, lo_e, hi_e, recdim, enhead, new, 
     .               edges_flag, do_bounds, mode_upcase_output, 
     .               keepax_flag, date_fmt_out,
     .               netcdf4_type, xchunk_size, 
     .               ychunk_size, zchunk_size, tchunk_size, 
     .               deflate_lev, shuffle_flag, endian_code, buff2,
     .               do_coords, status )

* ... Set attributes for rowSize variable

	   attname = 'long_name'
	   buff1 = 
     .	      'Number of Observations for this '//ftrname(:TM_LENSTR1(ftrname))
	   lbuff = TM_LENSTR1(buff1)
	
	   CALL CD_WRITE_ATTRIB(cdfid, varcode, attname,
     .                              buff1(:lbuff), .FALSE., status )
     
	   attname = 'sample_dimension'
	   buff1 = line_name(line_obscoord)
	   lbuff = TM_LENSTR1(buff1)
	   CALL CD_WRITE_ATTRIB(cdfid, varcode, attname,
     .                              buff1(:lbuff), .FALSE., status )

	ENDIF  ! synthesized rowsize in incoming file


	bad = bad_val4  ! default bad value for remaining variables
	itsa_uvar = .FALSE.
	itsa_string = .FALSE.

* ... Define world-coordinate variables in x, y, z, t directions 

	DO idim = 1, 4
	   IF ( relevant_coord(idim) ) THEN

	      line = grid_line(idim, gxlate)  

*  Use the coordinate name and type from the incoming file
	      varcode = line_name_orig (line)  

	      dummy = 0
	      CALL CD_GET_VAR_TYPE (dset, dummy, varcode, vartype, status)
	      dummy = -1
	      outtype = vartype
	      CALL CDF_SET_OUT_TYPE (dset, dummy, itsa_uvar, 
     .                itsa_string, out_type, outtype, status)

	      IF (instance_coord(idim)) THEN
* ... feature coordinate

		 grid = egrid
	         CALL CD_MAKE_VAR( cdfid, dset, varcode, outtype, maxstrlen,
     .               egrid, lo_e, hi_e, recdim, enhead, new, 
     .               edges_flag, do_bounds, mode_upcase_output, 
     .               keepax_flag, date_fmt_out,
     .               netcdf4_type, xchunk_size, 
     .               ychunk_size, zchunk_size, tchunk_size, 
     .               deflate_lev, shuffle_flag, endian_code, buff2,
     .               do_coords, status )

	      ELSE
* ... obs coordinate

		 grid = ogrid
	         CALL CD_MAKE_VAR( cdfid, dset, varcode, outtype, maxstrlen,
     .               grid, lo_o, hi_o, recdim, enhead, new, 
     .               edges_flag, do_bounds, mode_upcase_output, 
     .               keepax_flag, date_fmt_out,
     .               netcdf4_type, xchunk_size, 
     .               ychunk_size, zchunk_size, tchunk_size, 
     .               deflate_lev, shuffle_flag, endian_code, buff2,
     .               do_coords, status )

	      ENDIF

* ... attributes for the coordinate variable
* Write the attributes from a dataset variable as requested by the settings 
* The title, units, and history may be updated later.

	      IF (all_outflag .GT. 0) 
     .          CALL CDF_LIST_DSETVAR_ATTS (dset, itsa_uvar, its_cdf, 
     .           varid, varcode, bad, outtype, cdfid, do_warn, 
     .           scalefac, addoff, got_title, output_title, got_history, 
     .           output_history, got_units, output_units, status)

	      IF (status .NE. ferr_ok) GOTO 5400

* Write variable title, units, title modifier, history if not already done

              CALL CDF_LIST_DEFAULT_ATTS (dset, varid, varcode, cx_none, cdfid, 
     .          outtype, all_outflag, itsa_uvar, got_title, output_title, 
     .          got_history, output_history,  got_units, output_units, status )	
	      IF (status .NE. ferr_ok) GOTO 5400   

	   ENDIF
	ENDDO  ! coordinate variables


	ENDIF   ! IF (.NOT. append) THEN  *-*-*-*-*-*-*-*-*-*-*-*-

*  Loop to create all of the variables
*  (defer writing of binary data until end) 

        dset_last = dset

	out_type = out_typ_in  ! restore incoming data-type requested
	DO 100 ivar = 1, nvars

	   IF (ivar .EQ. featr_ivar) CYCLE  ! already set up the feature-id variable

	   cx  = cx_list(ivar)
	   mr  = mr_list(ivar)
	   cat = cx_category(cx)
	   var = cx_variable(cx)
	   varcode = SANITARY_VAR_CODE( cat, var )
	   got_title = .FALSE.
	   got_units = .FALSE.
	   got_history = .FALSE.
	   output_title = .TRUE.  ! output the var long_name by default
	   output_units = .FALSE.
	   output_history = .TRUE.
	   itsa_string = cx_type(cx) .EQ. ptype_string

	   bad = mr_bad_data(mr)


* Check that they haven't asked to write the feature-id variable, rowsize, 
* or a coordinate. Those are already done.

	   CALL CD_GET_VAR_VARNAME (dset, fvar_varid, varname, status)
	   IF (STR_SAME(varcode, varname) .EQ. 0) CYCLE

	   IF (.NOT.synthetic_rowsiz) THEN
	      CALL CD_GET_VAR_VARNAME (dset, dsg_row_size_varid(dset), varname, status)
	      IF (STR_SAME(varcode, varname) .EQ. 0) CYCLE
	   ENDIF
	   
	   DO idim = 1, 4
	      IF ( relevant_coord(idim) ) THEN
	         line = grid_line(idim, gxlate)  
	         varname = line_name_orig (line)
	         IF (STR_SAME(varcode, varname) .EQ. 0) GOTO 100
	      ENDIF
	   ENDDO

* is it in the linked-list for dset. If so get attr info, including original 
* upper/lowercase form of the name. If mode upcase_output is set, then upcase 
* the variable name.
* If it is a LET/D= user-var its linked list info is with that datset.
 
* For user-defined variables, with double-precision Ferret, write as double-precision
* unless the user has requested otherwise.

	   itsa_uvar = cat .EQ. cat_user_var

	   dset = cx_data_set(cx)
	   IF (dset .EQ. pdset_irrelevant) dset = pdset_uvars
	   IF (itsa_uvar .AND. dset .LE. pdset_irrelevant)  dset = pdset_uvars

	   CALL CD_GET_VAR_ID (dset, varcode, varid, status)
	   IF (status .NE. ferr_ok) THEN
	      IF (itsa_uvar .AND. dset.GE.pdset_irrelevant) THEN
	         dset = pdset_uvars
	         CALL CD_GET_VAR_ID (dset, varcode, varid, status)
	         IF (status .NE. ferr_ok) dset = dset_last
	      ENDIF
	   ENDIF

	   vartype = 0
	   outtype = 0

	   IF (status .EQ. ferr_ok) 
     .       CALL CD_GET_VAR_INFO (dset, varid, varname, vartype, nvdims, 
     .            vdims, nvatts, coordvar, all_outflag, status)

	   varcode = varname
	   IF (mode_upcase_output) CALL STR_UPCASE( varcode, varname)

	   CALL CDF_SET_OUT_TYPE (dset, varid, itsa_uvar, 
     .                itsa_string, out_type, outtype, status)
	   IF (status .NE. ferr_ok) GOTO 5500

	   IF (cat .EQ. cat_pseudo_var)  GOTO 5600

* allowed name?
	   IF (cat .NE. cat_file_var) THEN
	      IF ( .NOT.TM_LEGAL_NAME(varcode) ) GOTO 5200
	      CALL FIND_VAR_NAME ( pdset_irrelevant, varcode, cat1, var1 )
	      IF (  var1 .NE. munknown_var_name
     .      .AND. cat1 .EQ. cat_pseudo_var    ) GOTO 5200
	   ENDIF

* if a string variable, then find max len
* If the len comes back as 0 the variable contains just null strings.
* Set maxstrlen to 1, else we cant define the variable.
	   IF (itsa_string) THEN
	      maxstrlen = GET_MAX_C_STRING_LEN(
     .				  mr_c_pointer(mr), MGRID_SIZE(mr))
	      IF (maxstrlen .LE. 0) maxstrlen = 1
	   ELSE
	      maxstrlen = 0
	   ENDIF

* create the variable and its grid (coordinate output gets deferred)

	   IF (varid .GT. 0) THEN
              CALL  CD_GET_VAR_VARNAME (dset, varid, varname, status)
              varcode = varname
              IF (mode_upcase_output) CALL STR_UPCASE( varcode, varname)
	   ENDIF

* If this is a user var the axes are based on the defining dset (if any).
	   dset_num = dset
	   IF (dset.EQ.pdset_uvars .AND. cx_data_set( cx ).GT.0)
     .        dset_num = cx_data_set( cx ) 

* Use the max string length from the originating dataset.
* buff2 is dimension name for string-length.
	   buff2 = " "
	   IF (vartype .EQ. NF_CHAR .AND. dset .GE. 1) THEN
	     buff2 = ""
	     IF (dset_num.GE.1 .AND.(.NOT.itsa_uvar)) THEN
	        CALL CD_GET_DS_DIMS (dset_num, vdims(1), buff2, len, status)
	        IF (maxstrlen .LE. len) maxstrlen = len
	     ENDIF
	   ENDIF

	   IF (instance_var(ivar) ) THEN
	      grid = egrid 
	      CALL CD_MAKE_VAR( cdfid, dset_dsg, varcode, outtype, maxstrlen,
     .               grid, lo_e, hi_e, recdim, enhead, new, 
     .               edges_flag, do_bounds, mode_upcase_output, 
     .               keepax_flag, date_fmt_out,
     .               netcdf4_type, xchunk_size, 
     .               ychunk_size, zchunk_size, tchunk_size, 
     .               deflate_lev, shuffle_flag, endian_code, buff2,
     .               do_coords, status )

	   ELSE
	      grid = ogrid
	      CALL CD_MAKE_VAR( cdfid, dset_dsg, varcode, outtype, maxstrlen,
     .               grid, lo_o, hi_o, recdim, enhead, new, 
     .               edges_flag, do_bounds, mode_upcase_output, 
     .               keepax_flag, date_fmt_out,
     .               netcdf4_type, xchunk_size, 
     .               ychunk_size, zchunk_size, tchunk_size, 
     .               deflate_lev, shuffle_flag, endian_code, buff2,
     .               do_coords, status )
	   ENDIF

	   IF ( status .NE. merr_ok ) GOTO 5800

* if its a newly-created variable write attributes
	   IF ( new ) THEN

* If the variable is to be written scaled, set the scale and offset
*  values that will be written as attributes and sent to CD_WRITE_VAR. 

	      IF (varid .GT. 0) CALL CDF_SET_SCALE (dset, varid, 
     .        do_warn, scalefac, addoff, ferr_ok, status)

* Write the attributes from a dataset variable as requested by the settings 
* The title, units, and history may be updated later.
* (If a file variable came in with an "axis" attribute, the DSG file won't be 
*  correct - axis atts are only on coordinates - so don't write that attrib.)
	      
              CALL CD_GET_VAR_ATT_ID (dset, varid, "axis", attid, status)
	      IF (attid.GT.0 .AND. dset.GT.pdset_irrelevant) CALL CD_SET_ATT_FLAG(dset, varid, "axis", 0, status)

	      IF (all_outflag.GT.0 .AND. varid.GT.0) 
     .          CALL CDF_LIST_DSETVAR_ATTS (dset, itsa_uvar, its_cdf, 
     .           varid, varcode, bad, outtype, cdfid, do_warn,  
     .           scalefac, addoff, got_title, output_title, got_history, 
     .           output_history, got_units, output_units, status)

	      IF (status .NE. ferr_ok) GOTO 5400

* Write variable title, units, title modifier, history if not already done
	      CALL CDF_LIST_DEFAULT_ATTS (dset, varid, varcode, cx, cdfid, 
     .          outtype, all_outflag, itsa_uvar, got_title, output_title, 
     .          got_history, output_history,  got_units, output_units, status )	
	      IF (status .NE. ferr_ok) GOTO 5400   

	   ENDIF  ! writing attributes of file or user variables

 100    CONTINUE  ! loop over nvars

	nready = nvars

	dset = dset_dsg

*****
* Finally, write all of the coordinates and data values -- in netCDF DATA mode
* Note that this block of code is also executed following an error to
* ensure that all deferred coordinates and "ready" variables are flushed
 400	flushed = .TRUE.	! errors from here to exit may leave corrupted file
*
* ...  write the deferred coordinates
*
	IF (.NOT. append) THEN
	   CALL CD_WRITE_DEFER_COORD( cdfid, status )
	   IF ( status .NE. merr_OK ) GOTO 5800
	ENDIF

* ...
* ...
* ... Now write all the data

* ... when appending, do not write the DSG feature, rowSize or coordinate variables

	IF (.NOT. append) THEN  !*-*-*-*-*-*-*-*-*-*-*-*-

*     DSG-specific variables
	
* ... The masked feature ID's
* ... Mask the feature-id variable for writing
	
*	which mr goes with feature-var
	
	IF (.NOT. no_id) THEn
	   CALL MASK_DSG_FVAR ( nfeatures, process_feature, dset, 
     .              memry(featr_mr)%ptr, lineedg(line_fcoord)%ptr )

	   DO idim = 1, nferdims
	       write_lo(idim) = unspecified_int4
	       write_hi(idim) = unspecified_int4
	   ENDDO
	   write_lo(e_dim) = 1
	   write_hi(e_dim) = n_feat

	   CALL CD_GET_VAR_VARNAME (dset, fvar_varid, varcode, status)

           grid = egrid
	   CALL CD_WRITE_DSG_DATA (cdfid, varcode, grid, 
     .                 write_lo, write_hi, lineedg(line_fcoord)%ptr, 
     .                 mode_upcase_output, bad, status )

	ENDIF  ! no_id

* ... Compute the output obs len to write using RowSizes of masked data 
*     values; mask may select only part of a given feature

	IF (.NOT. synthetic_rowsiz) THEN
	   row_size_lm = dsg_loaded_lm(dsg_row_size_var(dset))

	   count = 0
	   base = 0     ! obs index at end of preceding feature  
	   DO ifeature = 1, nfeatures
	      nobsf = dsg_linemem(row_size_lm)%ptr(ifeature)  ! feature length
	      IF (process_feature(ifeature)) THEN
	         count = count + 1

* ... get observation-level mask for this feature
*      and compute the number of obs from features after masking

	         CALL MAKE_DSG_OBS_MASK(dset, cx, ifeature, base, process_obs, nobsf)
	         fobs = 0
	         DO iobs = 1, nobsf
	            IF (process_obs(iobs)) fobs = fobs + 1
	         ENDDO

	         val = fobs
	         CALL PUT_LINE_COORD(lineedg(line_fcoord)%ptr, count, val)
	      ENDIF
	      base = base + nobsf
	   ENDDO
	
	   CALL CD_GET_VAR_VARNAME (dset, dsg_row_size_varid(dset), varcode, status)

* ... Same grid as the feature-id variable
           grid = egrid
	   CALL CD_WRITE_DSG_DATA (cdfid, varcode, grid, 
     .                 write_lo, write_hi, lineedg(line_fcoord)%ptr, 
     .                 mode_upcase_output, bad, status )

	ENDIF  ! synthetic_rowsiz


* ... Now the coordinate variables with mask applied

	DO idim = 1, 4
	   IF ( relevant_coord(idim) ) THEN

	      line = grid_line(idim, gxlate)  
	      varcode = line_name_orig (line)  

* ... instance coordinates
	      IF (instance_coord(idim)) THEN

		 grid = egrid
		 DO i = 1, nferdims
	            write_lo(i) = unspecified_int4
		    write_hi(i) = unspecified_int4
		 ENDDO
		 write_lo(e_dim) = 1
		 write_hi(e_dim) = n_feat

		 CALL MASK_DSG_FVAR (nfeatures, process_feature, dset, 
     .              dsg_linemem(coord_lm(idim))%ptr, lineedg(line_fcoord)%ptr)

		 CALL CD_WRITE_DSG_DATA (cdfid, varcode, grid, 
     .                 write_lo, write_hi, lineedg(line_fcoord)%ptr, 
     .                 mode_upcase_output, bad, status )

* ... obs coordinates.  nmasked returned by mask_dsg_obsvar is the same as n_obs.
	      ELSE

		 grid = ogrid
		 DO i = 1, nferdims
	            write_lo(i) = unspecified_int4
		    write_hi(i) = unspecified_int4
		 ENDDO

		 write_lo(orientation) = 1
		 write_hi(orientation) = n_obs 
		 
		 IF (synthetic_rowsiz) THEN
		    CALL CD_WRITE_DSG_DATA (cdfid, varcode, grid, 
     .                 write_lo, write_hi, dsg_linemem(coord_lm(idim))%ptr, 
     .                 mode_upcase_output, bad, status )
		 ELSE
                   CALL MASK_DSG_OBSVAR (nfeatures, process_feature, process_obs,
     .               dsg_linemem(row_size_lm)%ptr, dset, cx,
     .               dsg_linemem(coord_lm(idim))%ptr, lineedg(line_obscoord)%ptr, 
     .               nmasked)

                   CALL CD_WRITE_DSG_DATA (cdfid, varcode, grid, 
     .                 write_lo, write_hi, lineedg(line_obscoord)%ptr, 
     .                 mode_upcase_output, bad, status )
		 ENDIF

	      ENDIF

	   ENDIF  ! relevant_coord

	ENDDO     ! idim


	ENDIF   ! IF (.NOT. append) THEN *-*-*-*-*-*-*-*-*-*-*-*-


*
* write the variables
*
	DO 500 ivar = 1, nready

	   IF (ivar .EQ. featr_ivar) CYCLE  ! already wrote up the feature-id variable

	   cx  = cx_list(ivar)
	   mr  = mr_list(ivar)
	   grid	= cx_grid( cx )
	   dset_num = cx_data_set( cx )
	   cat = cx_category(cx)
	   var = cx_variable(cx)
	   bad = mr_bad_data(mr)
	   varcode = SANITARY_VAR_CODE( cat, var )
	   itsa_uvar = cat .EQ. cat_user_var

	   dset = dset_dsg

* Check that they haven't asked to write the feature-variable, rowsize, 
* or a coordinate. Those are already done.

           CALL CD_GET_VAR_VARNAME (dset, fvar_varid, varname, status)
	   IF (STR_SAME(varcode, varname) .EQ. 0) CYCLE

	   IF (dset .GT. pdset_irrelevant .AND. .NOT.synthetic_rowsiz) THEN
	      CALL CD_GET_VAR_VARNAME (dset, dsg_row_size_varid(dset), varname, status)
	      IF (STR_SAME(varcode, varname) .EQ. 0) CYCLE
	   ENDIF

	   DO idim = 1, 4
	      IF ( relevant_coord(idim) ) THEN
	         line = grid_line(idim, gxlate)  
	         varname = line_name_orig (line)
	         IF (STR_SAME(varcode, varname) .EQ. 0) GOTO 500 
	      ENDIF
	   ENDDO


* For user-defined variables, with double-precision Ferret, write as double-precision
* unless the user has requested otherwise.  Use the same upper- or lower-case spelling
* as when the variable was created above.

	   itsa_uvar = cat .EQ. cat_user_var

	   dset = cx_data_set(cx)
	   IF (dset .EQ. pdset_irrelevant) dset = pdset_uvars
	   IF (itsa_uvar .AND. dset .LE. pdset_irrelevant)  dset = pdset_uvars

	   CALL CD_GET_VAR_ID (dset, varcode, varid, status)
	   IF (status .NE. ferr_ok) THEN
	      IF (itsa_uvar .AND. dset.GE.pdset_irrelevant) THEN
	         dset = pdset_uvars
	         CALL CD_GET_VAR_ID (dset, varcode, varid, status)
	         IF (status .NE. ferr_ok) dset = dset_last
	      ENDIF
	   ENDIF

	   vartype = 0
	   outtype = 0
 
	   IF (status .EQ. ferr_ok) 
     .       CALL CD_GET_VAR_INFO (dset, varid, varname, vartype, nvdims, 
     .            vdims, nvatts, coordvar, all_outflag, status)
     
	   varcode = varname
	   IF (mode_upcase_output) CALL STR_UPCASE( varcode, varname)

* check for interrupts - dont check again until entire variable is written
	   IF (interrupted) CALL ERRMSG(ferr_interrupt,status,' ',*5800)


* write the data for this variable

	   IF (instance_var(ivar) ) THEN
	      grid = egrid
	      
	      DO i = 1, nferdims
	         write_lo(i) = unspecified_int4
	         write_hi(i) = unspecified_int4
	      ENDDO
	      write_lo(e_dim) = 1
	      write_hi(e_dim) = n_feat

	      CALL MASK_DSG_FVAR ( nfeatures, process_feature, dset_dsg, 
     .              memry(mr)%ptr, lineedg(line_fcoord)%ptr )

	      CALL CD_WRITE_DSG_DATA (cdfid, varcode, grid, 
     .                 write_lo, write_hi, lineedg(line_fcoord)%ptr, 
     .                 mode_upcase_output, bad, status )

	   ELSE
	      grid = ogrid
	      DO i = 1, nferdims
	         write_lo(i) = unspecified_int4
	         write_hi(i) = unspecified_int4
	      ENDDO
	      write_lo(orientation) = 1
	      write_hi(orientation) = n_obs

	      IF (synthetic_rowsiz) THEN
		 val = n_obs  ! stand-in for rowsize
		 CALL MASK_DSG_OBSVAR ( nfeatures, process_feature, process_obs,
     .               val, dset_dsg, cx, 
     .               memry(mr)%ptr, lineedg(line_obscoord)%ptr, nmasked )
	      ELSE
	        CALL MASK_DSG_OBSVAR ( nfeatures, process_feature, process_obs,
     .               dsg_linemem(row_size_lm)%ptr, dset_dsg, cx, 
     .               memry(mr)%ptr, lineedg(line_obscoord)%ptr, nmasked )
	      ENDIF

	      CALL CD_WRITE_DSG_DATA (cdfid, varcode, grid, 
     .                 write_lo, write_hi, lineedg(line_obscoord)%ptr, 
     .                 mode_upcase_output, bad, status )

	   ENDIF

	 IF ( status .NE. merr_ok ) GOTO 5800
 500	CONTINUE

* close the file

        cdfstat = NF_CLOSE(cdfid)
        IF ( cdfstat .NE. NF_NOERR ) CALL TM_ERRMSG
     .     ( cdfstat+pcdferr, status, 'CDF_LIST', unspecified_int4,
     .     no_varid, 'could not close CDF output file: ',
     .     fname, *5000 )


* final completion -- maybe after an error
	status = final_status

	mode_upcase_output = mode_up_in
 5000	CONTINUE

* Deallocate the temporary grids and axes used (as in RESET_DSG)
	IF (egrid .NE. mgrid_buff) CALL TM_DEALLO_DYN_GRID( egrid )
	IF (ogrid .NE. mgrid_buff) CALL TM_DEALLO_DYN_GRID( ogrid )

	IF (line_fcoord .NE. mnormal) CALL FREE_LINE_DYNMEM (line_fcoord)
	IF (line_obscoord .NE. mnormal) CALL FREE_LINE_DYNMEM (line_obscoord)

	RETURN

* error exit(s)

 4200	CALL ERRMSG ( ferr_invalid_command, status,
     .        'Writing to DSG file: variables must all be from the same DSG dataset: '//varcode,
     .        *5000 )
 4300	CALL ERRMSG ( ferr_invalid_command, status,
     .        'Writing to DSG file: Subset or mask results in no data: '//varcode,
     .        *5000 )

 4400	CALL ERRMSG ( ferr_invalid_command, status,
     .        'Appending DSG variable to file that is not DSG or different featureType',
     .        *5000 )

 4500	CALL ERRMSG ( ferr_invalid_command, status,
     .        'Appending to DSG file: Size of '//varname(:TM_LENSTR1(varname))//
     .        ' dimension does not match dimension in file: '//varcode,
     .        *5000 )

 5100    CALL ERRMSG ( ferr_invalid_command, status,
     .                 'Writing to DSG file but feature-type is not set: '//varcode,
     .                 *5800 )

 5200    CALL ERRMSG ( ferr_syntax, status,
     .                 'illegal output variable name: '//varcode,
     .                 *5210 )

 5210    CALL TM_NOTE(
     .    'Name must use letters and digits beginning with a letter',
     .                 err_lun )
         CALL TM_NOTE( 'X,Y,Z,Y,I,J,K,L,XBOX,... are reserved names',
     .                 err_lun )
         CALL TM_NOTE( 'Use the LET command to define a legal name',
     .                 err_lun )
         GOTO 5800
 5220    CALL ERRMSG ( ferr_syntax, status,
     .                 'error writing variable: '//varcode,
     .                 *5800 )

 5400    CALL ERRMSG ( ferr_TMAP_error, status, ' ', *5000 )

 5500    CALL ERRMSG ( ferr_TMAP_error, status,
     .                 'unable to set variable type for output: '//varcode,
     .                 *5800 )

 5600    CALL ERRMSG ( ferr_invalid_command, status,
     .                 'Writing psedo-variables to DSG files is not implemented: '//varcode,
     .                 *5800 )
 5700    CALL ERRMSG ( ferr_invalid_command, status,
     .        'Writing data from TrajectoryProfile and TimeseriesProfile datasets'//pCR//
     .        ' to DSG files is not implemented. Try starting with USE/FEATURE=profile',
     .                 *5900 )

 5800   CALL ERRMSG ( ferr_TMAP_error, final_status, ' ', *5810 )
 5810	nready = ivar - 1
	IF (.NOT.flushed) GOTO 400 ! yea ... a bit of spagetti code ... 

	cdfstat = NF_CLOSE(cdfid)
	GOTO 5000
	
 5900	RETURN
	END

