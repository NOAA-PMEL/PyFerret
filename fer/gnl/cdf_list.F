	SUBROUTINE CDF_LIST ( fname, append, crecdim, nvars,
     .			      mr_list, cx_list, title, enhead, mode,
     .			      clobber, edges_flag, do_bounds, 
     .			      keepax_flag, out_type, quiet, as_global,
     .			      usr_maxstrlen, do_coords, status )

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
* write the indicated list of variables out in netCDF format

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program

* V230:  6/92
*       8/18/92 - added "_FillValue" attribute to pre-fill missing gaps      
*       11/4/92 - expanded user variable name length
* V320: 5/94 - array "memory" as a calling argument
*       8/94 - added qualifiers /ILIMITS=, /JLIMITS, etc.
*	11/94 - output "title" and "long_name_mod" in addition to global
*		and variable "history" attributes ("history" ignored on input)
*	 1/95 - bug introduced 8/94: error branch to 5300 causes crash 
*     1/25/95 - added argument "enhead" for enhanced heading control
* V411: 9/95 - corrected wrong type declaration of ISUBSCRIPT (discovered
*		only on IBM port)
* V450: 11/96 - added argument clobber to allow deletion of file
*		added argument "mode" and code to support NetCDF cache suitable
*		for DODS. Since DODS communications are stateless a cached
*		version of a remote file which contains all of the metadata
*		and coordinates is sufficient to prepare Ferret to read
*		remote DODS data. Only ncopn need be called then on the
*		remote file.
* Linux Port - 6/97 - reorder if def logic when setting cd_data_type to 
*                     BYTE or INTEGER*1
*		    - create equivalences so that all CD_WRITEVAL calls have
*			the same arg daat types
* kob*   9/97 - replaced call to ISUBSCRIPT w/ call to ISUBSCR_CX in order
*               to have access to context information - needed for negative
*               time step processing
* V500 *kob* 3/99- up SANITARY_VAR_CODE and local variable varcode to 64 chars
*             and VAR_UNITS to 64
* V512; *sh* 7/00 - allow SAVE/ILIMITS=n:n (no range) - new arg to EQUAL_RANGE
* V530; *sh* 12/00 - defer writing of axis coordinates and variable data values
*		to a single consolidated point at the end -- to alleviate
*		performance problems due to switching netCDF modes
* V530 *acm* 3/01 pass calendar ID to EQUAL_RANGE
*            4/16/01 properly declare TM_GET_CALENDAR_ID as integer
* V533 *sh* 6/01 - added support for netCDF output string variables
* V600 *acm* 8/05 - Allow more digits in xrevision number
* V552 *acm* 4/03- up SANITARY_VAR_CODE and local variable varcode to 128 chars
* V600 *acm* 6/06  Attribute control; get information for output from  
*                  linked-list structure.
* V600  4/06 *acm* - additional declarations found by Solaris compiler
* V601 10/06 *acm* Fix bug, bug 1454 if the user has set title or units, the users
*                  value is written, previously had always only got these from the
*                  attribute structure, so the new values were not saved.
* V601 10/06 *acm* Fix bug 1460; for value of string attributes, buff is 2048, 
*                  so set buff_size to match. Also, when looping through attributes
*                  (DO 200 loop) only get attribute values from the linked list structure
*                  if they are to be written to the output file.
* V601 10/06 *acm* fix bug 1451, cd_write_var needs the value of mode_upcase_output.
* V602  2/07 *acm* Fix bug 1492, changing attributes of coordinate variables 
* V604  7/07 *acm* Fix bug 1520, if a var has only a _FillValue attribute, and not a 
*                  missing_value one, then neither was written to the output file.
*                  Both should be written.
* V612 8/08  *acm* Fix bug 1587: so that correct bounds are written for axes of a 
*                  user-defined variable, send the dataset number of the underlying 
*                  dataset to cd_write_var
* V62  *acm*  2/09 - Fix bug 1634: allow SAVE/KEEP_AXISNAMES so that child axis names
*                    arent written. They can get too long for OPeNDAP library calls
* V62  *acm*  3/09 - Allow for CANCEL ATT/OUTPUT var.missing_flag and var._FillValue
*                   the special code for these made it impossible to turn off writing
*                   these attributes.
* V641 *acm*  1/10 -Fix bug 1715: packing data on output (bug is in ordering of args
*                   in call to NC_GET_ATTRIB for add_offset, also fix up the output 
*                   missing and FillValue flags which werent correct.)
* V65  *acm*  1/10  Data from some datasets get a mismatch between _FillValue and 
*                   data on output. Fix this.
*       *acm* 3/12  Add E and F dimensions (use nferdims in tmap_dims.parm)
*       *acm* 4/12  The record axis must be the last (slowest varying) axis. If a 
*                   time axis is written, and an E or F axis is written, then write 
*                   in /RIGID mode, with no record axis.
* V680  5/12 *acm* 6D Ferret, changes for DEFINE DATA/AGGREGATE
* V683  8/12 *acm*  New SET LIST/OUTTYPE and SAVE/OUTTYPE 
* V683 12/12 *acm*  Fix ticket 2004: appending data when writing packed variable
* V685 *acm*  3/13  List the full path or url in the history att for variable.
* V685 6/13 *acm* new qualifier SAVE/NOBOUNDS
* V685 6/13 *acm* Dont add SAVE/NOCOORDS
* v685 6/13 *acm*  bug 2081. For string vars, if the len comes back as 0 the variable contains 
*                  just null strings. Set maxstrlen to 1, else we can't define the variable.
* V686 12/13*acm* For string variables from datasets, keep the string-dimension name and length
*                 from the original dataset on output.
* V686 1/14 *acm* Fix ticket 2136: Appending a string variable to a dataset.
* V687 3/14 *acm* Correct the writing of missing and fill attributes when writing a scaled
*                 user-defined variable.	
* V687 3/14 *acm* Fix ticket 1750. Write Ferret-internal axis names to cdf files, unless
*                 the user has said /KEEP_AXISNAMES
* V694 *acm* 3/15 Ticket 2240. Dont send scale, attribute, and bad-val to CD_WRITE_VAR. That
*                 routine gets those attributes per variable from the dataset being written.
* V695 *acm* 7/15 If they said LIST/QUIET, don't do all the warnings and NOTEs
* V698 *acm* 1/16 Allow for Epic CDF files, ds_type = 'ECDF'
* V698  2/16 *acm* For ticket 2352: variables defined with LET/D are
*                  added to that dataset in the attribute structure
* V7   *acm*  6/12 Add Union aggregations: DEFINE DATA/AGG/U, ds_type 'UNI'
* V702 1/17 *sh* for trac enhancement #2369 -- dynamic memory management 
* v73+ *acm* 12/2017 Issue # 1846, writing user-variables with scale_factor, add_offset
* V74  *acm* 2/18 Issue # 1861, handling missing- and _FillValue with scale and offset
*                 write correct attributes to netCDF output
* V741 *acm* 6/18 V7.40, Issue 1332: write global attributes, and a fix for scalar numeric
*                 attributes 6/6/18.
* V743 *acm* 7/18 Issue 1883: var with only a _FillValue which is NaN got no
*                 _FillValue or missing_value attribute written
* v745 *acm* 12/18  Issue 1909, option to write dates in ISO8601 format
* V751 *acm* 6/19 Remove unused declarations
* V751 *acm* 7/19 Ticket 1946: File with user-variable strings truncates strings appended later.
*		  - Allow for the user to set a length for string-length dimension
* V751 *acm*11/19 new qualifier SAVE/NOCOORDS
* V7.6 *acm* 2/20 When save/asglobal, do not update the conventions or history
* v7.6 *acm* 5/20 Issue 1876: can read string-typed data, write as char.
* V761 *acm* 7/20 Issue 1981:in ISUBSCR_CX distinguish rounding for single 
*                 point vs range of points


        IMPLICIT NONE
        include 'netcdf.inc'
#include "netcdf_declareUnderscore.inc"
        include 'tmap_errors.parm'
        include 'cd_lib.parm'
        include 'cdf_tmap.parm'
        include 'tmap_dims.parm'
        include 'xtm_grid.cmn_text'
	include 'xdset_info.cmn_text'	! for DODS caching
	include 'xdset_info.cd_equiv'	! for DODS caching
	include 'ferret.parm'
	include 'errmsg.parm'
	include 'xcontext.cmn'
	include 'xvariables.cmn'
	include 'xprog_state.cmn'
	include 'xrevision.cmn'
	include 'xtoday.cmn'
	include 'xinterrupt.cmn'
        include 'netcdf4_settings.cmn'

* calling argument declarations:
        LOGICAL         append, enhead, clobber, do_coords, quiet
	INTEGER		nvars, mr_list(nvars), cx_list(nvars), mode,
     .			edges_flag, keepax_flag, do_bounds, usr_maxstrlen, 
     .			status
        CHARACTER*(*)   fname, title, out_type
        CHARACTER*1     crecdim

* internal variable declarations:

	INTEGER	   max_len
	PARAMETER (max_len = 2048)

        LOGICAL         TM_LEGAL_NAME, MATCH_NAME, TM_HAS_STRING,
     .			itsa_uvar, need_doc, new, formatted, flushed,
     .			itsa_string, output_units, output_title,
     .                  output_history, get_ds, its_nc, as_global, 
     .                  hist_and_conv
	INTEGER		TM_LENSTR1, ISUBSCR_CX, GET_MAX_C_STRING_LEN,
     .			MGRID_SIZE, STR_SAME, 
     .			cx, mr, dset, grid, cdfid, ivar, nready,
     .                  lbuff, cdfstat, idim, i, loc, errloc,
     .                  recdim, cat, var, cat1, var1, iaxis,
     .			final_status, type, maxstrlen, 
     .			write_lo(nferdims), write_hi(nferdims), 
     .			make_lo(nferdims), make_hi(nferdims),
     .                  dset_last, dset_num, attid, len, ndims_size, wlen

* *kob* 3/99- up SANITARY_VAR_CODE and local variable varcode to 64 chars
*             and VAR_UNITS to 64
* V552 *acm* 4/03- up SANITARY_VAR_CODE and local variable varcode to 128 chars

	CHARACTER	SANITARY_VAR_CODE*128, VAR_TITLE*140,
     .                  VAR_TITLE_MOD*140, VAR_UNITS*64, CX_DIM_STR*48,
     .                  TM_FMT*48, GET_STRING_ELEMENT*(max_len),
     .                  varcode*128, varname*128, buff*2048, revnum*12, 
     .                  c1*1, aname*128, short*128

	REAL*8		user_lo, user_hi, delta, loest_ww, hiest_ww,
     .                  miss_in, fill_in, r_scale, r_off, oneval

* local parameter declarations:
        LOGICAL    already_warned
	DATA    already_warned       / .FALSE. /

* For error messages
      CHARACTER*9 typnam(6)
      DATA typnam/'NC_BYTE', 'NC_CHAR', 'NC_SHORT', 'NC_INT', 
     .            'NC_FLOAT', 'NC_DOUBLE'/

* special equivalence (not in tmap_format/xdset_info.cd_equiv because 
* "BYTE" isnt supported on DECstation)
* reorder loop logic because linux cpp didnt like it *kob* 10/96

#   ifdef sun
      BYTE cd_data_type(maxvars)
#   else
      INTEGER*1 cd_data_type(maxvars)
#   endif

	EQUIVALENCE ( ds_precision, cd_data_type )

* internal equivalence so to match data types for F90
#   ifdef sun
      BYTE ibuff1
#   else
      INTEGER*1 ibuff1
#   endif

	REAL*4		rbuff4
	REAL		vals(100), bad
        REAL*8          dvals(100), avals(0:99), scalefac, addoff
	INTEGER*4	ibuff4
	EQUIVALENCE (rbuff4, ibuff4), (rbuff4, ibuff1)

* local parameter declarations

        INTEGER         TM_GET_CALENDAR_ID, TM_CHECK_NAN, DSG_WHATS_IT, 
     .                  TM_DSG_DSET_FROM_GRID, TM_DSG_NFEATURES,
     .                  buff_size, sslimits0, 
     .			wwlimits0, cal_id, tax, varid, vartype, nvdims, 
     .                  nvatts, vdims(8), iatt, slen, alen, istat, 
     .                  attype, attlen, attoutflag, all_outflag,
     .                  outtype, no_missing_flag, no_fill_flag, 
     .                  varoutflag, nfeatures, longest_feature, nobs
	REAL		little_val, big_val
        LOGICAL         full_precision, DODS_cache, range_not_rqd,
     .                  got_it, got_title, got_history, got_units, its_cdf,
     .                  got_missing, got_fill, do_warn, coordvar, user_var, has_dsg

        PARAMETER     ( buff_size = 2048,
     .			sslimits0 = 1+23-1,
     .			wwlimits0 = 1+29-1,
     .                  full_precision = .TRUE.,
     .			range_not_rqd = .FALSE. )
	CHARACTER	cal_name*32 , attname*128, buff1*128, buff2*128

* initialize
	ivar = 1  ! just house-keeping -- make sure all is init'ed
	cx  = cx_list(1)
	dset = cx_data_set( cx )

	 flushed = .FALSE. ! on error, whether OK variables were written
	 final_status = ferr_ok  ! innocent unless proven guilty
	 DODS_cache = mode .EQ. pcdf_mode_cache
         do_warn = .NOT.quiet
	 got_missing = .FALSE.
	 got_fill = .FALSE.

         CALL CD_DEFER_COORD_WRITE(0,0,0,0,0,pcd_defer_init)

* open/create the file
         CALL CD_OPEN_OUT( fname, append, cdfid, clobber, 
     .                     netcdf4_type, status )
         IF ( status .NE. merr_ok ) GOTO 5400

* History and conventions attributes updated, but not if we're just
* writing a global attribute.

* update history attribute in the file
        revnum = TM_FMT(revision_level, 5, 12, slen)

	hist_and_conv = .TRUE.
	IF (.NOT.clobber .AND. as_global) hist_and_conv = .FALSE.
	IF (hist_and_conv) THEN

           CALL CD_STAMP_OUT( append, cdfid,
     .       program_name(:len_program_name)//' V'//revnum(:slen)//' '//
     .       progname_mod(:len_progname_mod)//' '//today_date, status )
           IF ( status .NE. merr_ok ) GOTO 5800

* update Conventions attribute in the file

           buff1 = 'CF-1.6'

           CALL CD_CONVENTIONS_OUT( append, cdfid, buff1, status )
           IF ( status .NE. merr_ok ) GOTO 5800
	ENDIF

* optional title for data set
	IF ( title .NE. ' ' ) THEN
	   CALL CD_WRITE_ATTRIB(cdfid, pcdf_global, 'title',
     .			title(:TM_LENSTR1(title)), .FALSE., status )
	   IF ( status .NE. merr_ok ) GOTO 5800
	ENDIF

* Did they set a string-length?

* DODS URL saved as a global attribute (11/96)
	IF ( DODS_cache ) THEN
! note that routine CDF_PREP_CACHE is assumore med to have checked the validity
!      of the data set as a DODS data set
           dset = cx_data_set(cx_list(1))
	   CALL CD_WRITE_ATTRIB(cdfid, pcdf_global, 'DODS_URL',
     .			ds_des_name(dset), .FALSE., status )
           IF ( status .NE. merr_ok ) GOTO 5800
	ENDIF

* Other global attributes, if they have been marked for output.
* Mark global attrs for output with SET ATT/OUT=ALL .

        varid = 0      
	cx  = cx_list(1)
        dset = cx_data_set( cx )
	its_cdf = .FALSE.

        IF (dset .GT. pdset_irrelevant) THEN
           its_cdf = (ds_type(dset).EQ.'CDF' .OR. ds_type(dset).EQ.'ECDF' 
     .        .OR. ds_type(dset).EQ.' MC' .OR. ds_type(dset).EQ.'ENS' 
     .        .OR. ds_type(dset).EQ.'FCT' .OR. ds_type(dset).EQ.'UNI') 
           IF (its_cdf) CALL CDF_LIST_GLOBAL_ATTS (dset, cdfid, status)
	ENDIF

* Are variables from a DSG dataset?  Further checking for consistency within
* the CDF_LIST_DSG routine

	has_dsg = .FALSE.
	IF (.NOT.as_global) THEN
	   DO ivar = 1, nvars
              cx = cx_list(ivar)
              grid = cx_grid(cx)
	      IF (DSG_WHATS_IT(grid) .NE. pdsg_not_dsg) THEN 
                 dset = TM_DSG_DSET_FROM_GRID( grid )
                 nfeatures = TM_DSG_NFEATURES( grid )
                 CALL DSG_COORD_LIMS(dset, pdsg_row_size, little_val, big_val)
                 longest_feature = big_val
                 has_dsg = .TRUE.
                 EXIT
	      ENDIF
	   ENDDO
	ENDIF  ! its_cdf

* can append a variable if its grid matches what's in the file using
* non-dsg-code, below (?)

	IF (has_dsg) THEN
	  nobs = dsg_obs_dimlen(dset)
	  CALL CDF_LIST_DSG ( dset, cdfid, fname, append, nvars,
     .			     nfeatures, nobs, longest_feature,
     .			     mr_list, cx_list, title,
     .			     out_type, quiet, do_coords, status )
	   GOTO 5990  ! any error messages have been issued 
	ENDIF

        dset_last = dset
        scalefac = 1.
        addoff = 0.

*  Loop to create all of the variables and their metadata
*  (defer writing of binary data until end) 
*
       DO 100 ivar = 1, nvars
	   cx  = cx_list(ivar)
	   mr  = mr_list(ivar)
	   grid	= cx_grid( cx )
           dset = cx_data_set( cx )
           IF (dset_last .NE. dset) THEN
              IF (dset .GE. 1) THEN
                 its_cdf = (ds_type(dset) .EQ. 'CDF' .OR. 
     .               ds_type(dset) .EQ. ' MC' .OR. 
     .               ds_type(dset) .EQ. 'ENS' .OR. 
     .               ds_type(dset) .EQ. 'FCT' .OR. 
     .               ds_type(dset) .EQ. 'UNI') 
                 IF (its_cdf) CALL CDF_LIST_GLOBAL_ATTS (dset, cdfid, status)
              ENDIF
              dset_last = dset
           ENDIF

           cat = cx_category(cx)
           var = cx_variable(cx)
           varcode = SANITARY_VAR_CODE( cat, var )
           got_title = .FALSE.
           got_history = .FALSE.
           got_units = .FALSE.
           output_title = .FALSE.
           output_title = .TRUE.  ! output the var long_name by default
           output_units = .FALSE.
           output_history = .TRUE.
	   itsa_string = cx_type(cx) .EQ. ptype_string

* is it in the linked-list for dset. If so get attr info, including original 
* upper/lowercase form of the name. If mode upcase_output is set, then upcase 
* the variable name.
* If it is a LET/D= user-var its linked list info is with that datset.
 
* For user-defined variables, with double-precision Ferret, write as double-precision
* unless the user has requested otherwise.

           itsa_uvar = cat .EQ. cat_user_var
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

           CALL CDF_SET_OUT_TYPE (dset, varid, itsa_uvar, 
     .                itsa_string, out_type, outtype, status)
           IF (status .NE. ferr_ok) GOTO 5500

           IF ( cat .EQ. cat_pseudo_var )THEN
	      varcode = varcode(:TM_LENSTR1(varcode))//"_"
	      IF (varcode(1:1) .EQ. "_") varcode = varcode(2:)
	   ENDIF

	   tax = grid_line(t_dim,grid)
           cal_name = line_cal_name(tax)
           cal_id = TM_GET_CALENDAR_ID ( cal_name )

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
* Ticket 1946
* For file variables we use the string-length dimension from the input file.
* For user-variables allow the user to set a length. A file created with 
* initial strings of shorter length then truncated strings appended later.

	   IF (itsa_string) THEN
	      maxstrlen = GET_MAX_C_STRING_LEN(
     .				   mr_c_pointer(mr), MGRID_SIZE(mr))
	      IF (maxstrlen .LE. 0) maxstrlen = 1

	      IF (usr_maxstrlen.GT.0 .AND. append .AND. do_warn) THEN 
	         CALL WARN ('SAVE/STRLEN= is ignored when appending to the file')
		 usr_maxstrlen = 0
	      ENDIF

* Here, for user-variables, we COULD arbitrarily set maxstrlen to something larger than 
* the maxstrlen computed for this write, when the user hasn't set a value with /STRLEN, 
* maybe only when the grid has a time axis, e.g. IF(tax.NE.mnormal), so that appends 
* with longer strings would work.  See ticket 1946.

	      IF (itsa_uvar .AND. .NOT.append) THEN
	         IF (usr_maxstrlen.GT.0 .AND. usr_maxstrlen.LT.maxstrlen) GOTO 5600
		 IF (usr_maxstrlen.GT.0) maxstrlen = usr_maxstrlen
	      ENDIF
	   ELSE
	      maxstrlen = 0
	   ENDIF

* get slab limits to use for writing the values of this variable

           ndims_size = 0
	   wlen = 0

           DO 50 idim = 1, nferdims
* ... limits within which to write variable data
              write_lo(idim) = cx_lo_ss(cx,idim)
              write_hi(idim) = cx_hi_ss(cx,idim)

	      IF (write_hi(idim) - write_lo(idim) .GT. 1) THEN
	         ndims_size = ndims_size + 1
	         wlen = write_hi(idim) - write_lo(idim) + 1
	      ENDIF

* get slab limits to use for creating this NetCDF variable
* (elaborate checking if user manually specifies creation limits)
* ... limits within which to create NetCDF variable
              make_lo(idim)  = write_lo(idim)
              make_hi(idim)  = write_hi(idim)
	      IF ( make_lo(idim) .NE. unspecified_int4 ) THEN
	         iaxis = grid_line(idim,grid)
* ... creation limits manually specified by world coordinates?
	         loc = qual_given(wwlimits0+idim)
                 IF ( loc .GT. 0 ) THEN
		    errloc = loc
	            CALL EQUAL_RANGE_T(
     .			cmnd_buff(qual_start(loc):qual_end(loc)), idim,
     .                  grid, user_lo, user_hi, delta, cal_id, status )
                    IF ( status .NE. ferr_OK ) GOTO 5800
	            CALL GRID_WORLD_EXTREMES( loest_ww, hiest_ww,
     .			grid, idim )
	            IF ( .NOT.line_modulo(iaxis) 
     .		   .AND. ( user_lo .LT. loest_ww
     .		      .OR. user_hi .GT. hiest_ww ) ) GOTO 5300
	            make_lo(idim) = ISUBSCR_CX
     .			( user_lo, grid, idim, cx, round_up )
	            make_hi(idim) = ISUBSCR_CX 
     .			( user_hi, grid, idim, cx, round_dn )
	            IF (user_hi .EQ. user_lo) make_hi(idim) = ISUBSCR_CX 
     .			( user_hi, grid, idim, cx, round_sngl )
	            IF ( .NOT.line_modulo(iaxis)  
     .		   .AND. ( make_hi(idim) .GT. line_dim(iaxis)
     .	              .OR. make_lo(idim) .LT. 1 )	   )	GOTO 5300
	            IF ( make_lo(idim) .GT. write_lo(idim)
     .		    .OR. make_hi(idim) .LT. write_hi(idim) ) 	GOTO 5310
	         ENDIF
* ... creation limits manually specified by subscript?
	         loc = qual_given(sslimits0+idim)
                 IF ( loc .GT. 0 ) THEN
		    errloc = loc
	            CALL EQUAL_RANGE(
     .			cmnd_buff(qual_start(loc):qual_end(loc)),
     .                  idim, user_lo, user_hi, delta, formatted,
     .			range_not_rqd, cal_id, status )
                    IF ( status .NE. ferr_OK ) GOTO 5990
	            make_lo(idim) = user_lo
	            make_hi(idim) = user_hi
	            IF ( .NOT.line_modulo(iaxis) 
     .	           .AND. ( make_hi(idim) .GT. line_dim(iaxis)
     .	              .OR. make_lo(idim) .LT. 1 )	      ) GOTO 5300
	            IF ( make_lo(idim) .GT. write_lo(idim)
     .		    .OR. make_hi(idim) .LT. write_hi(idim) ) 	GOTO 5310
	         ENDIF
	      ENDIF

* ... if a time axis is written, and an E or F axis is written,
*     then write in /RIGID mode, with no record axis.
	      IF (idim .GT. t_dim .AND.
     .	          make_lo(t_dim) .NE. unspecified_int4 .AND.
     .	          make_lo(idim) .NE. unspecified_int4) THEN
                 crecdim = 'N'
		 IF (.NOT.already_warned .AND. do_warn) CALL WARN (
     .	          'Saving E or F axis with time axis uses /RIGID in time')
                 already_warned = .TRUE.
	      ENDIF

 50        CONTINUE
 
* If writing variable as a global variable, only write 1-D variables.
           IF (as_global .AND. ndims_size .GT. 1) GOTO 5320
	   IF (as_global) CYCLE

* determine the netCDF "record dimension" to use
           IF ( write_lo(t_dim) .EQ. unspecified_int4 ) THEN
              recdim = 0
           ELSEIF ( crecdim .EQ. 'T' ) THEN
              recdim = t_dim
           ELSE
              recdim = 0
           ENDIF

* convert the data type into netCDF parameter value
           IF (DODS_cache .AND. cx_type(cx).EQ.unspecified_int4) THEN
	      cx_type(cx) = nf_double
	   ENDIF

	   IF (cx_type(cx) .EQ. ptype_float) THEN
	      type = nf_float
	   ELSEIF  (cx_type(cx) .EQ. ptype_string) THEN
	      type = nf_char
	   ELSEIF  (cx_type(cx) .EQ. ptype_double) THEN
	      type = nf_double
	   ELSEIF  (cx_type(cx) .EQ. ptype_int4) THEN
	      type = nf_int
	   ELSEIF  (cx_type(cx) .EQ. ptype_int2) THEN
	      type = nf_short
	   ELSEIF  (cx_type(cx) .EQ. ptype_int1) THEN
	      type = nf_byte
	   ELSE
	      CALL WARN('Internal err: unknown data type, using FLOAT')
	      type = nf_float
	   ENDIF
           IF (type .NE. 0 .AND. varid .GT. 0) type = outtype
           IF (dset .EQ. pdset_uvars) type = outtype
           IF (itsa_uvar) type = outtype 
           IF (type .EQ. 0) outtype = vartype

* create the variable and its grid (coordinate output gets deferred)

* If this is a user var the axes are based on the defining dset (if any).
           dset_num = dset
           IF (dset.EQ.pdset_uvars .AND. cx_data_set( cx ).GT.0)
     .        dset_num = cx_data_set( cx )  

* When its a netCDF/OPeNDAP file variable use the max string length from the originating dataset.
* If the variable in the dataset is type string, synthesize a name for the string-length dimension.
           buff2 = " "
           IF (type .EQ. NF_CHAR .AND. dset .GE. 1) THEN
	     buff2 = ""
	     its_nc = ( .NOT. TM_HAS_STRING(ds_type(dset),'EZ') )
             IF (dset_num.GE.1 .AND.(.NOT.itsa_uvar) .AND.its_nc) THEN
                CALL CD_GET_DS_DIMS (dset_num, vdims(1), buff2, len, status)
                IF (maxstrlen .LE. len) maxstrlen = len
             ENDIF
	     CALL CD_GET_VAR_TYPE (dset, varid, varname, vartype, status)
	     IF (vartype .EQ. nf_string) THEN
                CALL CD_GET_DS_DIMS (dset_num, vdims(1), buff2, len, status)
	        buff2 = TM_FMT(FLOAT(maxstrlen), 5, 12, slen)
		buff2 = 'STRING1_'//buff2(:slen)
             ENDIF
           ENDIF

	   IF (varid .GT. 0) THEN
              CALL  CD_GET_VAR_VARNAME (dset, varid, varname, status)
              varcode = varname
              IF (mode_upcase_output) CALL STR_UPCASE( varcode, varname)
	   ENDIF

* If they set a larger string-len, use it
           IF (usr_maxstrlen .GT. 0) THEN 
	      IF (usr_maxstrlen .LT. maxstrlen .AND. do_warn) CALL WARN 
     .        ('/STRLEN setting shorter than max string length. Strings will be truncated.'
     .        //pCR//'Variable: '//varcode )
	      maxstrlen = MAX(maxstrlen, usr_maxstrlen)  
           ENDIF

	   CALL CD_MAKE_VAR( cdfid, dset_num, varcode, type, maxstrlen,
     .               grid, make_lo, make_hi, recdim, enhead, new, 
     .               edges_flag, do_bounds, mode_upcase_output, 
     .               keepax_flag, date_fmt_out, 
     .               netcdf4_type, xchunk_size, 
     .               ychunk_size, zchunk_size, tchunk_size, 
     .               deflate_lev, shuffle_flag, endian_code, buff2,
     .               do_coords, status )

           IF ( status .NE. merr_ok ) GOTO 5800
	   
* if its a newly-created variable write attributes
           IF ( new ) THEN

* If the variable is to be written scaled, set the scale and offset
*  values that will be written as attributes and sent to CD_WRITE_VAR. 

	      IF (varid .GT. 0) CALL CDF_SET_SCALE (dset, varid, 
     .        do_warn, scalefac, addoff, ferr_ok, status)
	      
* Write the attributes from a dataset variable as requested by the settings 
* The title, units, and history may be updated later.

	      bad = mr_bad_data(mr)

	      IF (varid.GT.0 .AND. all_outflag.GT.0) THEN
	         CALL CDF_LIST_DSETVAR_ATTS (dset, itsa_uvar, its_cdf, 
     .             varid, varcode, bad, outtype, cdfid, do_warn,  
     .             scalefac, addoff, got_title, output_title, got_history, 
     .             output_history, got_units, output_units, status)

	         IF (status .NE. ferr_ok) GOTO 5400
	      ENDIF
	      
* Write variable title, units, title modifier, history if not already done

              CALL CDF_LIST_DEFAULT_ATTS (dset, varid, varcode, cx, cdfid, 
     .          outtype, all_outflag, itsa_uvar, got_title, output_title, 
     .          got_history, output_history,  got_units, output_units, status )	
	      IF (status .NE. ferr_ok) GOTO 5400   

* NetCDF variable IDs - additional information needed for DODS caching (11/96)
	      IF ( DODS_cache ) THEN
	         ibuff4 = cd_varid(var)		! for F90 data type checks
	         CALL CD_WRITE_ATTVAL(cdfid, varcode,
     .                     'netcdf_var_id',rbuff4, 1, nf_int, status )
                 IF ( status .NE. merr_ok ) GOTO 5800
	         ibuff1 = cd_data_type(var)	! for F90 data type checks
	         CALL CD_WRITE_ATTVAL(cdfid, varcode,
     .                  'netcdf_data_type', rbuff4, 1, nf_byte, status )
                 IF ( status .NE. merr_ok ) GOTO 5800
	      ENDIF

           ENDIF  ! writing attributes of file or user variables

* Write attributes for psuedo-variables
           IF (cat .EQ. cat_pseudo_var) THEN  

              IF (.NOT.got_missing) THEN
 	         CALL CD_WRITE_ATTVAL_DP(cdfid, varcode,
     .          'missing_value', bad, 1, type, status )
                 IF ( status .NE. merr_ok ) GOTO 5800
              ENDIF

              IF (.NOT.got_fill) THEN
                 CALL CD_WRITE_ATTVAL_DP(cdfid, varcode,
     .          '_FillValue', bad, 1, type, status )
                IF ( status .NE. merr_ok ) GOTO 5800
	      ENDIF

              buff = VAR_TITLE( cx )
              lbuff = TM_LENSTR1( buff )
                 CALL CD_WRITE_ATTRIB(cdfid, varcode, 'long_name',
     .                              buff(:lbuff), .FALSE., status )
              IF ( status .NE. merr_ok ) GOTO 5800

* ...other variable modifiers
              c1 = VAR_TITLE_MOD( cx )
              IF ( c1 .NE. ' ' ) THEN
                 IF ( lbuff .GT. 1 ) THEN
                    buff = buff(:lbuff)//', '
                    lbuff = MIN( 140, lbuff+2)
                 ENDIF
                 buff = buff(:lbuff)//VAR_TITLE_MOD( cx )
                 lbuff = TM_LENSTR1( buff )
              ENDIF
* ... write it to the file
              IF ( lbuff .GE. 2 ) THEN
                 CALL CD_WRITE_ATTRIB( cdfid, varcode, 'long_name_mod', 
     .                              buff(2:lbuff), .FALSE., status )
                 IF ( status .NE. merr_ok ) GOTO 5800
              ENDIF
           ENDIF  ! pseudo-variables

 100    CONTINUE  ! loop over nvars

	nready = nvars

*****
* Finally, write all of the coordinates and data values -- in netCDF DATA mode
* Note that this block of code is also executed following an error to
* ensure that all deferred coordinates and "ready" variables are flushed
 400	flushed = .TRUE.	! errors from here to exit may leave corrupted file
*
* write the deferred coordinates
*
	IF (.NOT.as_global .AND. do_coords) CALL CD_WRITE_DEFER_COORD( cdfid, status )
	IF ( status .NE. merr_OK ) GOTO 5800
*
* write the variables
*
        DO 500 ivar = 1, nready
	   cx  = cx_list(ivar)
	   mr  = mr_list(ivar)
	   grid	= cx_grid( cx )
           dset = cx_data_set( cx )
           cat = cx_category(cx)
           var = cx_variable(cx)
           varcode = SANITARY_VAR_CODE( cat, var )

           itsa_uvar = cat .EQ. cat_user_var
           CALL CD_GET_VAR_ID (dset, varcode, varid, status)  ! is it in the linked-list for dset
           IF (status .NE. ferr_ok .AND. itsa_uvar ) THEN
              dset = pdset_uvars    ! is it a user var based on a var in dset
              CALL CD_GET_VAR_ID (dset, varcode, varid, status) 
           ENDIF

* Get varcode into original upper/lowercase form. If mode upcase_output is set, 
* then upcase the variable name.

           IF (status .EQ. ferr_ok) THEN
              CALL CD_GET_VAR_INFO (dset, 
     .                  varid, varname, vartype,  nvdims, vdims,
     .                  nvatts, coordvar, all_outflag, status) 
              varcode = varname
              IF (mode_upcase_output) CALL STR_UPCASE( varcode, varname)
           ENDIF
     
           IF ( cat .EQ. cat_pseudo_var )THEN
	      varcode = varcode(:TM_LENSTR1(varcode))//"_"
	      IF (varcode(1:1) .EQ. "_") varcode = varcode(2:)
	   ENDIF

* check for interrupts - dont check again until entire variable is written
           IF (interrupted) CALL ERRMSG(ferr_interrupt,status,' ',*5800)

	   IF (as_global) THEN
	      CALL CD_GET_VAR_TYPE (dset, varid, varname, vartype, status)

	      IF (outtype .NE. 0) vartype = outtype
	      IF (itsa_uvar .AND. vartype.EQ.0) THEN  ! for user vars type not known in attrib structure
                  vartype = 6      ! netcdf nf_double
                  IF (itsa_string) vartype = 2
               ENDIF

	      IF (vartype .EQ. NF_CHAR) THEN 

	         IF (wlen .GT. 1) GOTO 5325
		 buff = GET_STRING_ELEMENT(
     .				mr_lo_s1(mr), mr_hi_s1(mr),
     .				mr_lo_s2(mr), mr_hi_s2(mr),
     .				mr_lo_s3(mr), mr_hi_s3(mr),
     .				mr_lo_s4(mr), mr_hi_s4(mr),
     .				mr_lo_s5(mr), mr_hi_s5(mr),
     .				mr_lo_s6(mr), mr_hi_s6(mr),
     .				memry(mr)%ptr,
     .				cx_lo_s1(cx), cx_lo_s2(cx),
     .				cx_lo_s3(cx), cx_lo_s4(cx),
     .				cx_lo_s5(cx), cx_lo_s6(cx),
     .				max_len, len)

	         CALL CD_WRITE_ATTRIB(cdfid, pcdf_global, varcode,
     .			buff(:len), .FALSE., status )
                 IF ( status .NE. merr_ok ) GOTO 5800

	      ELSE

	         IF (wlen .GT. 100) GOTO 5320
		 IF (wlen .EQ. 0) wlen = 1
		 avals(:wlen-1) = memry(mr)%ptr
		 CALL CD_WRITE_ATTVAL_DP(
     .                    cdfid, pcdf_global, varcode, avals(0), wlen, 
     .                    vartype, status )
	      ENDIF

	   ELSE  ! not as-global,write regular variable

* get slab limits to use for writing the values of this variable

	      DO 450 idim = 1, nferdims
                 write_lo(idim) = cx_lo_ss(cx,idim)
                 write_hi(idim) = cx_hi_ss(cx,idim)
 450          CONTINUE

* write the data for this variable
              user_var = cat .EQ. cat_user_var
	      CALL CD_WRITE_VAR (cdfid, dset_num, varcode, grid, 
     .                        write_lo, write_hi, memry(mr)%ptr, 
     .                        edges_flag, do_bounds, mode_upcase_output, 
     .                        keepax_flag, user_var, ds_missing_flag(ivar), 
     .                        do_coords, status )

	   ENDIF

           IF ( status .NE. merr_ok ) GOTO 5800
 500	CONTINUE

* close the file
        cdfstat = NF_CLOSE(cdfid)
        IF ( cdfstat .NE. NF_NOERR ) CALL TM_ERRMSG
     .     ( cdfstat+pcdferr, status, 'CDF_LIST', unspecified_int4,
     .     no_varid, 'could not close CDF output file: ',
     .     fname, *5990 )


* final completion -- maybe after an error
	status = final_status
	RETURN

* error exit(s)
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

 5300    CALL ERRMSG ( ferr_invalid_command, status,
     .                 'Exceeds grid limits: '//
     .			cmnd_buff(qual_start(errloc):qual_end(errloc))
     .			//pCR//'Variable: '//varcode, *5800 )

 5310    CALL ERRMSG ( ferr_invalid_command, status,
     .                 'Does not encompass data to be written: '//
     .			cmnd_buff(qual_start(errloc):qual_end(errloc))
     .			//pCR//'Variable: '//varcode, *5800 )

 5320    CALL ERRMSG ( ferr_invalid_command, status,
     .                'Global attribute must be a 1D variable length < 100'
     .			//pCR//'Variable: '//varcode, *5800 )

 5325    CALL ERRMSG ( ferr_invalid_command, status,
     .                'Multi-valued Global string attributes not implemented'
     .			//pCR//'Variable: '//varcode, *5800 )

 5400    CALL ERRMSG ( ferr_TMAP_error, status, ' ', *5990 )

 5500    CALL ERRMSG ( ferr_TMAP_error, status,
     .                 'unable to set variable type for output: '//varcode,
     .                 *5800 )
     
* User-specified string-length is too short (aname used as dummy string here)

 5600	buff = TM_FMT(FLOAT(maxstrlen), 5, 12, slen)
 	aname = TM_FMT(FLOAT(usr_maxstrlen), 5, 12, alen)
	CALL ERRMSG ( ferr_invalid_command, status,
     .          'Strings are length '//buff(:slen)//
     .		': longer than /STRLEN setting of '//aname(:alen) 
     .		//pCR//'Variable: '//varcode, *5800 )

 5800   CALL ERRMSG ( ferr_TMAP_error, final_status, ' ', *5810 )
 5810	nready = ivar - 1
	IF (.NOT.flushed) GOTO 400 ! yea ... a bit of spagetti code ... 
        cdfstat = NF_CLOSE(cdfid)
 5990   RETURN
	END

