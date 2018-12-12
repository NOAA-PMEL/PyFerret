	SUBROUTINE CDF_LIST ( fname, append, crecdim, nvars,
     .			      mr_list, cx_list, title, enhead, mode,
     .			      clobber, edges_flag, do_bounds, 
     .			      keepax_flag, out_type, quiet, as_global,
     .			      status )

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
        LOGICAL         append, enhead, clobber, quiet
	INTEGER		nvars, mr_list(nvars), cx_list(nvars), mode,
     .			edges_flag, keepax_flag, do_bounds, status
        CHARACTER*(*)   fname, title, out_type
        CHARACTER*1     crecdim

* internal variable declarations:

	INTEGER	   max_len
	PARAMETER (max_len = 2048)

        LOGICAL         TM_LEGAL_NAME, MATCH_NAME, TM_HAS_STRING,
     .			itsa_uvar, need_doc, new, formatted, flushed,
     .			itsa_string, output_units, output_title,
     .                  output_history, get_ds, its_nc, as_global
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
     .                  TM_FMT*12, GET_STRING_ELEMENT*(max_len),
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

        INTEGER         TM_GET_CALENDAR_ID, TM_CHECK_NAN,
     .                  buff_size, sslimits0, 
     .			wwlimits0, cal_id, tax, varid, vartype, nvdims, 
     .                  nvatts, vdims(8), iatt, slen, alen, istat, 
     .                  attype, attlen, attoutflag, all_outflag,
     .                  outtype, no_missing_flag, no_fill_flag, 
     .                  varoutflag
        LOGICAL         NC_GET_ATTRIB, NC_GET_ATTRIB_DP, 
     .                  full_precision, DODS_cache, range_not_rqd,
     .                  got_it, got_title, got_units, got_history, its_cdf,
     .                  got_missing, got_fill, do_warn, coordvar, user_var

        PARAMETER     ( buff_size = 2048,
     .			sslimits0 = 1+23-1,
     .			wwlimits0 = 1+29-1,
     .                  full_precision = .TRUE.,
     .			range_not_rqd = .FALSE. )
	CHARACTER	cal_name*32 , attname*128, buff1*128, buff2*128

* initialize
	 ivar = 1  ! just house-keeping -- make sure all is init'ed
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

* update history attribute in the file
        revnum = TM_FMT(revision_level, 5, 12, slen)

        CALL CD_STAMP_OUT( append, cdfid,
     .       program_name(:len_program_name)//' V'//revnum(:slen)//' '//
     .       progname_mod(:len_progname_mod)//' '//today_date, status )
        IF ( status .NE. merr_ok ) GOTO 5800

* update Conventions attribute in the file
        buff1 = 'CF-1.6'

        CALL CD_CONVENTIONS_OUT( append, cdfid, buff1, status )
        IF ( status .NE. merr_ok ) GOTO 5800


* optional title for data set
	IF ( title .NE. ' ' ) THEN
	   CALL CD_WRITE_ATTRIB(cdfid, pcdf_global, 'title',
     .			title(:TM_LENSTR1(title)), .FALSE., status )
           IF ( status .NE. merr_ok ) GOTO 5800
	ENDIF


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

        IF (dset .GT. pdset_irrelevant) THEN
           its_cdf = (ds_type(dset).EQ.'CDF' .OR. ds_type(dset).EQ.'ECDF' 
     .        .OR. ds_type(dset).EQ.' MC' .OR. ds_type(dset).EQ.'ENS' 
     .        .OR. ds_type(dset).EQ.'FCT' .OR. ds_type(dset).EQ.'UNI') 
           IF (its_cdf) CALL CDF_LIST_GLOBAL_ATTS (dset, cdfid, status)
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
           got_units = .FALSE.
           got_history = .FALSE.
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

           IF (status .EQ. ferr_ok) THEN
              CALL CD_GET_VAR_INFO (dset, varid, varname, vartype, nvdims, 
     .            vdims, nvatts, coordvar, all_outflag, status)
              varcode = varname
              IF (mode_upcase_output) CALL STR_UPCASE( varcode, varname)
              IF (itsa_uvar) THEN  ! for user vars type not known in attrib structure
                 vartype = 6       ! netcdf nf_double
                 IF (itsa_string) vartype = 2
              ENDIF
              outtype = vartype

* If there is an OUTTYPE attribute, that came from a SET VAR/OUTTYPE.
* It will be superceded by a SET LIST/OUTTYPE or a SAVE/OUTTYPE setting,
* applied just below.
	      CALL CD_GET_VAR_OUTTYPE (dset, varid, outtype, status)

* If its just an expression it doesnt have a varid in the attribute structure
* for user-vars. Still set the output type.

           ELSE
              vartype = 6  ! netcdf nf_double
              IF (itsa_string) vartype = 2
           ENDIF
   
	   IF (outtype .EQ. 0) outtype = vartype

* Setting made with SAVE/OUTTYPE or SET LIST/OUTTYPE
* SAVE/OUTTYPE takes precedence over a SET LIST/OUTTYPE
* out_type comes in from a SAVE/OUTTYPE  Use it if its not DFLT
* list_outtype is from SET LIST/OUTTYPE  IF out_type is DFLT, use list_outtype

	   IF ( out_type(1:3) .EQ. 'DFL') out_type = list_outtype

	   IF (out_type(1:3) .EQ. 'DOU') outtype = nf_double
	   IF (out_type(1:3) .EQ. 'FLO') outtype = nf_float
	   IF (out_type(1:3) .EQ. 'INT') outtype = nf_int
	   IF (out_type(1:3) .EQ. 'SHO') outtype = nf_short
	   IF (out_type(1:3) .EQ. 'BYT') outtype = nf_byte

* When the user requests higher precision output from a lower precision file 
* variable, e.g. converting FLOAT to DOUBLE, write an attribute explaining 
* that it contains data converted in this way.

	   IF (dset .GT. 0 .AND. outtype.GT.vartype) THEN
              CALL CD_GET_VAR_INFO (dset, varid, varname, vartype, 
     .                  nvdims, vdims, nvatts, coordvar, varoutflag, 
     .                  status) 

              attname = 'Ferret_Precision_Note'
              CALL CD_GET_VAR_ATT_ID (dset, varid, attname, attid, status)
	      IF (status.EQ.ferr_ok) 
     .             CALL CD_DELETE_ATTRIBUTE(dset, varid, attname, status)
	      slen = TM_LENSTR1(typnam(vartype))
              alen = TM_LENSTR1(typnam(outtype))
              CALL CD_PUT_NEW_ATTR (dset, varid, 
     .              attname, 2, 61, 1, 
     .         'This variable written using Ferret was converted from '//
     .              typnam(vartype)(4:slen)//' to '//
     .              typnam(outtype)(4:alen), 1., status )
              IF (status .NE. ferr_ok) GOTO 5990

	   ENDIF

           IF ( cat .EQ. cat_pseudo_var )THEN
	      varcode = varcode(:TM_LENSTR1(varcode))//"_"
	      IF (varcode(1:1) .EQ. "_") varcode = varcode(2:TM_LENSTR1(varcode))
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
	   IF (itsa_string) THEN
	      maxstrlen = GET_MAX_C_STRING_LEN(
     .				   mr_c_pointer(mr), MGRID_SIZE(mr))
	      IF (maxstrlen .LE. 0) maxstrlen = 1
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
           buff2 = " "
           IF (type .EQ. NF_CHAR .AND. dset .GE. 1) THEN
	     buff2 = ""
	     its_nc = ( .NOT. TM_HAS_STRING(ds_type(dset),'EZ') )
             IF (dset_num.GE.1 .AND.(.NOT.itsa_uvar) .AND.its_nc) THEN
                CALL CD_GET_DS_DIMS (dset_num, vdims(1), buff2, len, status)
                IF (maxstrlen .LE. len) maxstrlen = len
             ENDIF
           ENDIF

           IF (.NOT. as_global) THEN
           CALL CD_MAKE_VAR( cdfid, dset_num, varcode, type, maxstrlen,
     .               grid, make_lo, make_hi, recdim, enhead, new, 
     .               edges_flag, do_bounds, mode_upcase_output, 
     .               keepax_flag, date_fmt_out,
     .               netcdf4_type, xchunk_size, 
     .               ychunk_size, zchunk_size, tchunk_size, 
     .               deflate_lev, shuffle_flag, endian_code, buff2,
     .               status )

           IF ( status .NE. merr_ok ) GOTO 5800

	   
* If the variable is to be written scaled, set the scale and offset
*  values that will be written as attributes and sent to CD_WRITE_VAR. 


           IF (varid .GT. 0) THEN
	      
              scalefac = 1.
              attname = 'scale_factor'
              CALL CD_GET_VAR_ATT_ID (dset, varid, attname, attid, 
     .                  istat)
              IF (attid .GT. 0) CALL CD_GET_VAR_ATT_INFO (dset, 
     .                  varid, attid, aname, attype, attlen, attoutflag, 
     .                  istat )

              IF (istat .EQ. ferr_ok .AND. attoutflag.EQ.1) THEN
                 IF (attype .EQ. nf_double) THEN
                    got_it = NC_GET_ATTRIB_DP (dset, varid,
     .                  aname, do_warn, varcode, attlen, 
     .                  attoutflag, dvals)
                    scalefac = dvals(1)

                 ELSE
                    got_it = NC_GET_ATTRIB (dset, varid, aname,
     .                  do_warn, varcode, buff_size, attlen, 
     .                  attoutflag, buff, vals)
                    scalefac = vals(1)
                 ENDIF
              ENDIF

              addoff = 0.
              attname = 'add_offset'
              CALL CD_GET_VAR_ATT_ID (dset, varid, attname, attid, 
     .                  istat)
              IF (attid .GT. 0) CALL CD_GET_VAR_ATT_INFO (dset, 
     .                  varid, attid, aname, attype, attlen, attoutflag, 
     .                  istat )

              IF (istat .EQ. ferr_ok .AND. attoutflag.EQ.1) THEN
                 IF (attype .EQ. nf_double) THEN
                    got_it = NC_GET_ATTRIB_DP (dset, varid,
     .                  aname, do_warn, varcode, attlen, 
     .                  attoutflag, dvals)
                    addoff = dvals(1)

                 ELSE
                    got_it = NC_GET_ATTRIB (dset, varid, aname,
     .                  do_warn, varcode, buff_size, attlen, 
     .                  attoutflag, buff, vals)
                    addoff = vals(1)
                 ENDIF
              ENDIF

           ENDIF  ! setting scalefac and addoff

           ENDIF  ! as_global

* if its a newly-created variable write attributes
           IF (( new  .AND. all_outflag .GT. 0) .AND. .NOT. as_global) THEN

* Write missing value and fill flag here

              IF (varid .GT. 0) THEN

* Output the bad flag used by Ferret (and _FillValue which has been made  
* the same as missing_flag)

                 bad = mr_bad_data(mr)
                 dvals(1) = bad
                 attlen = 1

* Get attype for missing flag

                 istat = ferr_ok
                 attoutflag = 0
                 no_missing_flag = 1
                 no_fill_flag = 0  ! Write _fillValue unless specifically told not to
                 attname = 'missing_value'
                 CALL CD_GET_VAR_ATT_ID (dset, varid, attname, attid, 
     .                  istat)
                 IF (attid .GT. 0) THEN
                    no_missing_flag = 0
                    CALL CD_GET_VAR_ATT_INFO (dset, varid, 
     .                attid, aname, attype, attlen, attoutflag, istat )
                    IF (attoutflag .EQ. 0) no_missing_flag = 1
		    IF (attoutflag .NE. 0) got_it = NC_GET_ATTRIB_DP 
     .                (dset, varid, attname, do_warn, varcode, attlen, 
     .                attoutflag, miss_in)

                 ELSE
                   attname = '_FillValue'
                   CALL CD_GET_VAR_ATT_ID (dset, varid, attname, attid, 
     .                  istat)
                   IF (attid .GT. 0) THEN
                      no_fill_flag = 0
                      CALL CD_GET_VAR_ATT_INFO (dset, varid, 
     .                attid, aname, attype, attlen, attoutflag, istat )
                      IF (attoutflag .EQ. 0) no_fill_flag = 1
                   ENDIF
                 ENDIF
                 IF (no_missing_flag .EQ. 1) THEN
                   attname = '_FillValue'
                   CALL CD_GET_VAR_ATT_ID (dset, varid, attname, attid, 
     .                  istat)
                   IF (attid .GT. 0) 
     .                CALL CD_GET_VAR_ATT_INFO (dset, varid, 
     .                attid, aname, attype, attlen, attoutflag, istat )
                    IF (attoutflag .EQ. 0) no_fill_flag = 1
                 ENDIF

* If the data is numeric, check the missing flag, getting the original one, 
* and write with the output type

                 IF (istat .EQ. ferr_ok .AND. outtype .NE. nf_char) THEN

* If un-scaling on output, write the original missing flag.
                    IF (scalefac .NE. 1.D0  .OR. addoff .NE. 0.D0) THEN
                       dvals(1) = miss_in
                       vals(1) = dvals(1)

* Otherwise convert missing flag to requested output type for the variable
                    ELSE
                       IF (attype .NE. outtype .AND. istat .EQ. ferr_ok)  THEN
                          alen = TM_LENSTR1(typnam(attype))
                          slen = TM_LENSTR1(typnam(outtype))
                          IF ((.NOT.itsa_uvar) .AND. (dset .GT. 0 .AND. do_warn)) 
     .                      CALL WARN (
     .                      'Converting data type of missing_value '//
     .                       typnam(attype)(1:alen)//
     .                      ' to match output type of variable '//
     .                      typnam(outtype)(1:slen) )
                       ENDIF
                    ENDIF
                    attype = outtype
   
                    IF (attoutflag .EQ. 1) THEN
                       IF (no_missing_flag.EQ.0) THEN
		          CALL CD_WRITE_ATTVAL_DP(
     .                    cdfid, varcode, 'missing_value', dvals, attlen, 
     .                    attype, status )
                          IF (status .NE. ferr_ok) GOTO 5400
			  got_missing = .TRUE.

			  IF (dset .EQ. pdset_uvars) THEN 
		             CALL CD_WRITE_ATTVAL_DP(
     .                       cdfid, varcode, '_FillValue', dvals, attlen, 
     .                       attype, status )
                             IF (status .NE. ferr_ok) GOTO 5400
			     got_fill = .TRUE.
			  ENDIF
                       ENDIF

* Ferret doesn't keep the values corresponding to two flags in the data that it stores.
* If there was a _FillValue in the input file, and its the bad-flag, write it.
* Issue #1883: If there is a _FillValue and it's NaN, it will be different from the
* bad-flag. We do want to write it.

		       IF (.NOT.got_fill ) THEN 
		       attname = '_FillValue'
		       CALL CD_GET_VAR_ATT_ID (dset, varid, attname, attid, 
     .                     istat)
                       IF (attid .GT. 0) 
     .                    CALL CD_GET_VAR_ATT_INFO (dset, varid, 
     .                   attid, aname, attype, attlen, attoutflag, istat )
                          IF (attoutflag .EQ. 0) no_fill_flag = 1
		       got_it = .FALSE.
		       IF (attoutflag .NE. 0) got_it = NC_GET_ATTRIB_DP  !.AND. .NOT.got_missin
     .                     (dset, varid, attname, do_warn, varcode, attlen, 
     .                     attoutflag, fill_in)
                          IF ( TM_CHECK_NAN(fill_in) .EQ. 1   ) fill_in = bad
			  IF (got_it .AND. (fill_in.EQ.bad .OR. fill_in.EQ.miss_in) ) THEN
			     dvals(1) = fill_in
                             vals(1) = dvals(1)
                             attype = outtype
			  ELSE
			     IF (its_cdf .AND. .NOT.itsa_uvar) no_fill_flag = 1
			  ENDIF
		       ENDIF
                       IF (no_fill_flag .EQ. 0) THEN
		          CALL CD_WRITE_ATTVAL_DP(
     .                     cdfid, varcode, '_FillValue', dvals, attlen, 
     .                     attype, status )
		          got_fill = .TRUE.
                       ENDIF

                    ENDIF
                 ENDIF ! istat 


* Get varcode into original upper/lowercase form. If mode upcase_output is set, 
* then upcase the variable name.

                 CALL CD_GET_VAR_INFO (dset, varid, varname, vartype, 
     .                   nvdims, vdims, nvatts, coordvar, all_outflag, 
     .                   status) 
                 varcode = varname
                 IF (mode_upcase_output) CALL STR_UPCASE( varcode, 
     .                 varname)

                 DO 200 iatt = 1, nvatts
                    CALL CD_GET_VAR_ATT_NAME( dset, varid, iatt, 
     .                                        attname, status)
                    CALL CD_GET_VAR_ATT_INFO (dset, varid, 
     .                   iatt, attname, attype, attlen, attoutflag, istat )
                         
                    slen = TM_LENSTR1(attname)
                    IF ( MATCH_NAME (attname,  slen,
     .                               'MISSING_VALUE', 13 ) .OR. 
     .                   MATCH_NAME (attname,  slen,
     .                               '_FILLVALUE', 10 ) ) THEN
                       slen = 0   ! done already

                    ELSE IF ( MATCH_NAME (attname,  slen,
     .                               'SCALE_FACTOR', 12 ) .OR. 
     .                   MATCH_NAME (attname,  slen,
     .                               'ADD_OFFSET', 10 ) ) THEN
                       got_it = NC_GET_ATTRIB_DP (dset, varid, attname,
     .                    do_warn, varcode, attlen, attoutflag, dvals)
                       IF (attoutflag  .EQ. 1) 
     .                    CALL CD_WRITE_ATTVAL_DP (cdfid, varcode,
     .                          attname, dvals, attlen, attype, status )
                          IF (status .NE. ferr_ok) GOTO 5400
                    ELSE

                       attype = 0
                       attlen = 0
                       CALL CD_GET_VAR_ATT_INFO (dset, varid, iatt,
     .                    aname, attype, attlen, attoutflag, istat )
     
                       IF (attype .NE. nf_char .AND.
     .                     attlen .GT. 100) THEN
                          attlen = 100
                          alen = TM_LENSTR1(aname)
                          IF (do_warn) CALL WARN 
     .                ('writing only first 100 elements of attribute'//
     .                  aname(1:alen))
                       ENDIF

                       IF (attype .EQ. nf_char .AND. attoutflag.EQ.1) THEN
                          got_it = NC_GET_ATTRIB ( dset, varid, aname, 
     .                       do_warn, varcode, buff_size, attlen,
     .                       attoutflag, buff, vals)
                          lbuff = TM_LENSTR1(buff)
                          CALL CD_WRITE_ATTRIB(cdfid, varcode,
     .                         attname, buff(:lbuff), .FALSE., status )
                       ELSE IF (attoutflag.EQ.1) THEN
                          got_it = NC_GET_ATTRIB ( dset, varid, aname, 
     .                       do_warn, varcode, buff_size, attlen,
     .                       attoutflag, buff, vals)
                          CALL CD_WRITE_ATTVAL_DP (cdfid, varcode,
     .                          attname, vals, attlen, attype, status )
                          IF (status .NE. ferr_ok) GOTO 5400

                       ENDIF

                       IF (MATCH_NAME (attname,  slen,
     .                           'LONG_NAME', 9 )) THEN
                          got_title = .TRUE.
                          output_title = (attoutflag.EQ.1)
                       ENDIF

                       IF (MATCH_NAME (attname,  slen,
     .                           'UNITS', 5 )) THEN
                          got_units = .TRUE.
                          output_units = (attoutflag.EQ.1)
                       ENDIF

                       IF (MATCH_NAME (attname,  slen,
     .                           'HISTORY',  7 )) THEN
                          got_history = .TRUE.
                          output_history = (attoutflag.EQ.1)
                       ENDIF

                       IF (MATCH_NAME (attname,  slen,
     .                           'SCALE_FACTOR', 12 )) THEN
                          got_it = NC_GET_ATTRIB ( dset, varid, attname, 
     .                       do_warn, varcode, buff_size, attlen,
     .                       attoutflag, buff, vals)
                       ENDIF

                       IF (MATCH_NAME (attname,  slen,
     .                           'ADD_OFFSET', 10 )) THEN
                          got_it = NC_GET_ATTRIB ( dset, varid, attname, 
     .                       do_warn, varcode, buff_size, attlen,
     .                       attoutflag, buff, vals)
                       ENDIF

                    ENDIF

 200             CONTINUE
! done getting attributes from linked-list structure

* If its just an expression, write missing_value and _FillValue
* attributes with default Ferret values.

              ELSE ! varid GT 0

		 attname = 'missing_value'
		 vals(1) = bad_val4
		 attlen = 1
		 attype = outtype
		 CALL CD_WRITE_ATTVAL_DP(cdfid, varcode,
     .                          attname, vals, attlen, attype, status )
		 got_missing = .TRUE.
		 attname = '_FillValue'
		 vals(1) = bad_val4
		 attlen = 1
		 attype = outtype
		 CALL CD_WRITE_ATTVAL_DP(cdfid, varcode,
     .                          attname, vals, attlen, attype, status )
		 got_fill = .TRUE.

              ENDIF ! varid GT 0

* variable title
* May already have this from Ferret variables from attribute structure, or if there
* is no long_name attribute, get it by means of VAR_TITLE.

              IF (.NOT. got_title) THEN
                 buff = VAR_TITLE( cx ) !!
                 lbuff = TM_LENSTR1( buff )

                 IF ( buff .NE. ' ' .AND. all_outflag .NE. 0 .AND.
     .                output_title ) THEN
                    CALL CD_WRITE_ATTRIB(cdfid, varcode, 'long_name',
     .                              buff(:lbuff), .FALSE., status )
                    IF ( status .NE. merr_ok ) GOTO 5800
                    got_title = .TRUE.
                 ENDIF

              ENDIF

* modifier to variable title
* ..."compressing" transforms and limits not given in definitions
	      buff = ' '
	      lbuff = 1
              DO 60 idim = 1, nferdims
                 need_doc =  cx_trans(idim,cx) .GT. trans_compress_code
                 IF ( itsa_uvar ) need_doc = need_doc
     .                    .OR. uvar_given(idim,var) .GT. uvlim_needed
                 IF ( need_doc ) THEN
                    IF ( lbuff .GT. 1 ) THEN      ! append a comma ?
                       buff = buff(:lbuff)//', '
                       lbuff = MIN( 140, lbuff+2)
                    ENDIF
                    buff = buff(:lbuff)//
     .                     CX_DIM_STR(idim, cx, ':',full_precision,i)
                    lbuff = TM_LENSTR1( buff )
                 ENDIF
 60           CONTINUE
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
              IF ( lbuff .GE. 2 .AND. all_outflag .NE. 0) THEN

* ..... unless the user has created a long_name_mod attribute, or asked that writing
*       it be turned off. See if long_name_mod attribute has been created.

                 attoutflag = 1
                 slen = TM_LENSTR1(varcode)
                 buff2 = varcode(1:slen)//'.long_name_mod'
                 IF (varid .GT. 0) THEN
                    CALL BREAK_VARATTNAME (buff2, dset, 
     .               buff1, attname, varid, .FALSE., status)

                    IF (status .EQ. ferr_ok) THEN

                       CALL CD_GET_VAR_ATT_ID (dset, varid,
     .                     'long_name_mod', attid, istat)
                       IF (attid .GT. 0) CALL CD_GET_VAR_ATT_INFO (dset, 
     .                     varid, attid, aname, attype, attlen, 
     .                     attoutflag, istat )

                       IF (attoutflag .EQ. 1) got_it = NC_GET_ATTRIB 
     .                     ( dset, varid, attname, .FALSE., varcode, 
     .                      buff_size, attlen, attoutflag, 
     .                      buff(2:), vals)
                       lbuff = TM_LENSTR1( buff )
                    ENDIF
                 ENDIF
                 status = ferr_ok

                 IF (all_outflag .NE. 0 .AND. attoutflag .NE. 0)
     .              CALL CD_WRITE_ATTRIB(cdfid, varcode, 'long_name_mod',  
     .                buff(2:lbuff), .FALSE., status )

                 IF ( status .NE. merr_ok ) GOTO 5800
              ENDIF

* data set of origin
              IF ( dset .NE. pdset_irrelevant .AND. dset.NE.pdset_uvars) THEN
       
* ..... unless the user has created a history attribute, or asked that writing
*       it be turned off. See if there is a history attribute.
*       If the history attribute is just "from dsetname", or is not present,
*       then write the dataset and its path or URL.

                 attoutflag = 1
                 IF (varid .GT. 0) THEN
                    slen = TM_LENSTR1(varcode)
                    buff2 = varcode(1:slen)//'.history'
                    CALL BREAK_VARATTNAME (buff2, dset, buff1, attname,
     .                   varid, .FALSE., status)

                    IF (status .EQ. ferr_ok) THEN
                       CALL CD_GET_VAR_ATT_ID (dset, varid, 'history', 
     .                     attid, istat)
                       IF (attid .GT. 0) CALL CD_GET_VAR_ATT_INFO (dset, 
     .                     varid, attid, aname, attype, attlen, 
     .                     attoutflag, istat )

                       IF (attoutflag .EQ. 1) got_it = NC_GET_ATTRIB (
     .		  	     dset, varid, attname, .FALSE., varcode,
     .		  	     buff_size, attlen, attoutflag, 
     .                       buff, vals)
                       lbuff = TM_LENSTR1( buff )
                       IF (buff .EQ. ' ') THEN
                          buff = 'From '
                          CALL GET_DSET_NAME( dset, buff(6:), lbuff )
			  IF (buff(6:7) .EQ. './') 
     .                      CALL GET_SHORT_DSET_NAME( dset, buff(6:), lbuff )
                          lbuff = TM_LENSTR1( buff )
                       ELSE
		          get_ds = .FALSE.
		          IF (STR_SAME(buff(1:5), 'From ') .EQ. 0) THEN
			    CALL GET_SHORT_DSET_NAME( dset, short, lbuff )
			    IF (STR_SAME(buff(6:attlen), short) .EQ. 0) 
     .                        get_ds = .TRUE.
                          ENDIF
                       ENDIF
                    ENDIF

                    IF (status .NE. ferr_ok .OR. get_ds) THEN 
                       buff = 'From '
                       CALL GET_DSET_NAME( dset, buff(6:), lbuff )
		       IF (buff(6:7) .EQ. './') 
     .                     CALL GET_SHORT_DSET_NAME( dset, buff(6:), lbuff )
                       lbuff = TM_LENSTR1( buff )
                    ENDIF
                 
                 ELSE
                    buff = 'From '
                    CALL GET_DSET_NAME( cx_data_set(cx), buff(6:), lbuff)
                    IF (buff(6:7) .EQ. './') CALL GET_SHORT_DSET_NAME
     .                  ( cx_data_set(cx), buff(6:), lbuff )
                    lbuff = TM_LENSTR1( buff )
                 ENDIF ! varid .gt. 0


                 status = ferr_ok

                 IF (all_outflag .NE. 0 .AND. attoutflag .NE. 0) THEN
                    lbuff = TM_LENSTR1( buff )
                    CALL CD_WRITE_ATTRIB(cdfid, varcode, 'history', 
     .                              buff(:lbuff), .FALSE., status )
                 ENDIF
                 IF ( status .NE. merr_ok ) GOTO 5800

              ELSEIF ( cx_data_set(cx) .NE. pdset_irrelevant .AND.
     .                 cx_data_set(cx) .NE. pdset_uvars) THEN

* Write history attribute containing the data set of origin for user variable, 
* if the user has not created a history attribute for the variable.

                 IF (.NOT. got_history .AND. all_outflag .NE. 0 .AND. 
     .             output_history ) THEN

                    attoutflag = 1
                    buff = 'From '
                    CALL GET_DSET_NAME(cx_data_set(cx), buff(6:), lbuff)
                    IF (buff(6:7) .EQ. './') CALL GET_SHORT_DSET_NAME
     .                  ( cx_data_set(cx), buff(6:), lbuff )
                    lbuff = TM_LENSTR1( buff )
                    status = ferr_ok

                    IF (all_outflag .NE. 0 .AND. attoutflag .NE. 0)
     .                 CALL CD_WRITE_ATTRIB(cdfid, varcode, 'history', 
     .                              buff(:lbuff), .FALSE., status )
                    IF ( status .NE. merr_ok ) GOTO 5800
                 ENDIF
              ENDIF

* units
              buff = VAR_UNITS( cx )
              lbuff = TM_LENSTR1( buff )

              IF ( buff .NE. ' ' .AND. all_outflag .NE. 0 .AND. 
     .             output_units ) THEN
                 CALL CD_WRITE_ATTRIB(cdfid, varcode, 'units',
     .                              buff(:lbuff), .FALSE., status )
                 IF ( status .NE. merr_ok ) GOTO 5800
              ENDIF

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
           IF (cat .EQ. cat_pseudo_var .AND. .NOT.as_global) THEN  

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
	IF (.NOT.as_global) CALL CD_WRITE_DEFER_COORD( cdfid, status )
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
	      IF (varcode(1:1) .EQ. "_") varcode = varcode(2:TM_LENSTR1(varcode))
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
     .                        keepax_flag, user_var, ds_missing_flag(ivar), status )
	   ENDIF

           IF ( status .NE. merr_ok ) GOTO 5800
 500	CONTINUE

* close the file
#ifdef usingDODSf2cUnderscore 
        cdfstat = NF_CLOSE_(cdfid)
#else
        cdfstat = NF_CLOSE(cdfid)
#endif
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

 5800   CALL ERRMSG ( ferr_TMAP_error, final_status, ' ', *5810 )
 5810	nready = ivar - 1
	IF (.NOT.flushed) GOTO 400 ! yea ... a bit of spagetti code ... 
#ifdef usingDODSf2cUnderscore 
        cdfstat = NF_CLOSE_(cdfid)
#else
        cdfstat = NF_CLOSE(cdfid)
#endif
 5990   RETURN
	END

