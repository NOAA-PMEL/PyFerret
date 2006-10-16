	SUBROUTINE CDF_LIST ( memory, fname, append, crecdim, nvars,
     .			      mr_list, cx_list, title, enhead, mode,
     .			      clobber, edges_flag, do_bounds, status )

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

        include 'netcdf.decl'
#include "netcdf.inc"
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

* calling argument declarations:
        LOGICAL         append, enhead, clobber, do_bounds
	INTEGER		nvars, mr_list(nvars), cx_list(nvars), mode,
     .			edges_flag, status
	REAL		memory( mem_blk_size, max_mem_blks )
        CHARACTER*(*)   fname, title
        CHARACTER*1     crecdim

* internal variable declarations:
        LOGICAL         TM_LEGAL_NAME, MATCH_NAME, 
     .			itsa_uvar, need_doc, new, formatted, flushed,
     .			itsa_string, output_units, output_title
	INTEGER		TM_LENSTR1, ISUBSCR_CX, GET_MAX_C_STRING_LEN,
     .			MGRID_SIZE,
     .			cx, mr, dset, grid, cdfid, ivar, nready,
     .                  lbuff, cdfstat, idim, i, loc, errloc,
     .                  recdim, cat, var, cat1, var1, iaxis,
     .			final_status, type, maxstrlen, 
     .			write_lo(4), write_hi(4), make_lo(4), make_hi(4),
     .                  dset_last, attid

* *kob* 3/99- up SANITARY_VAR_CODE and local variable varcode to 64 chars
*             and VAR_UNITS to 64
* V552 *acm* 4/03- up SANITARY_VAR_CODE and local variable varcode to 128 chars

	CHARACTER	SANITARY_VAR_CODE*128, VAR_TITLE*140,
     .                  VAR_TITLE_MOD*140, VAR_UNITS*64, CX_DIM_STR*48,
     .                  TM_FMT*12,
     .                  varcode*128, varname*128, buff*2048, revnum*12, 
     .                  c1*1, aname*128

	REAL*8		user_lo, user_hi, delta, loest_ww, hiest_ww

* For error messages
      CHARACTER*9 typnam(6)
      DATA typnam/'NC_BYTE', 'NC_CHAR', 'NC_SHORT', 'NC_INT', 
     .            'NC_FLOAT', 'NC_DOUBLE'/

* special equivalence (not in tmap_format/xdset_info.cd_equiv because 
* "BYTE" isnt supported on DECstation)
* reorder loop logic because linux cpp didnt like it *kob* 10/96

#ifdef unix
#   ifdef sun
      BYTE cd_data_type(maxvars)
#   else
      INTEGER*1 cd_data_type(maxvars)
#   endif
#else
      BYTE cd_data_type(maxvars)
#endif

	EQUIVALENCE ( ds_precision, cd_data_type )

* internal equivalence so to match data types for F90
#ifdef unix
#   ifdef sun
      BYTE ibuff1
#   else
      INTEGER*1 ibuff1
#   endif
#else
      BYTE ibuff1
#endif
	REAL*4		rbuff4, vals(100), bad
        REAL*8          dvals(100), scalefac, addoff
	INTEGER*4	ibuff4
	EQUIVALENCE (rbuff4, ibuff4), (rbuff4, ibuff1)

* local parameter declarations

        INTEGER         TM_GET_CALENDAR_ID, 
     .                  buff_size, sslimits0, 
     .			wwlimits0, cal_id, tax, varid, vartype, nvdims, 
     .                  nvatts, vdims(8), iatt, slen, alen, istat, 
     .                  attype, attlen, attoutflag, all_outflag,
     .                  outtype
        LOGICAL         NC_GET_ATTRIB, NC_GET_ATTRIB_DP,
     .                  full_precision, DODS_cache, range_not_rqd,
     .                  got_it, got_title, got_units, do_warn, coordvar

        PARAMETER     ( buff_size = 140,
     .			sslimits0 = 1+19-1,
     .			wwlimits0 = 1+23-1,
     .                  full_precision = .TRUE.,
     .			range_not_rqd = .FALSE. )
	CHARACTER	cal_name*32 , attname*128, buff1*128, buff2*128

* initialize
	 ivar = 1  ! just house-keeping -- make sure all is init'ed
	 flushed = .FALSE. ! on error, whether OK variables were written
	 final_status = ferr_ok  ! innocent unless proven guilty
	 DODS_cache = mode .EQ. pcdf_mode_cache
         do_warn = .TRUE.

         CALL CD_DEFER_COORD_WRITE(0,0,0,0,0,pcd_defer_init)

* open/create the file
         CALL CD_OPEN_OUT( fname, append, cdfid, clobber, status )
         IF ( status .NE. merr_ok ) GOTO 5400

* update history attribute in the file
        revnum = TM_FMT(revision_level, 5, 12, slen)

        CALL CD_STAMP_OUT( append, cdfid,
     .       program_name(:len_program_name)//' V'//revnum(:slen)//' '//
     .       progname_mod(:len_progname_mod)//' '//today_date, status )
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
        IF (dset .GE. 1) THEN
          IF (ds_type(dset) .EQ. 'CDF' .OR. ds_type(dset) .EQ. ' MC')
     .       CALL CDF_LIST_GLOBAL_ATTS (dset, cdfid, status)
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
                 IF (ds_type(dset) .EQ. 'CDF' .OR. 
     .               ds_type(dset) .EQ. ' MC')
     .          CALL CDF_LIST_GLOBAL_ATTS (dset, cdfid, status)
              ENDIF
              dset_last = dset
           ENDIF
           cat = cx_category(cx)
           var = cx_variable(cx)
           varcode = SANITARY_VAR_CODE( cat, var )
           got_title = .FALSE.
           got_units = .FALSE.
           output_title = .FALSE.
           output_units = .FALSE.
	   itsa_string = cx_type(cx) .EQ. ptype_string

* is it in the linked-list for dset. If so get attr info, including original 
* upper/lowercase form of the name. If mode upcase_output is set, then upcase 
* the variable name.

           itsa_uvar = cat .EQ. cat_user_var
           IF (itsa_uvar .OR. dset .EQ. pdset_irrelevant) dset = -1
           CALL CD_GET_VAR_ID (dset, varcode, varid, status)

           vartype = 0
           IF (status .EQ. ferr_ok) THEN
              CALL CD_GET_VAR_INFO (dset, varid, varname, vartype, nvdims, 
     .            vdims, nvatts, coordvar, all_outflag, status)
              varcode = varname
              IF (mode_upcase_output) CALL STR_UPCASE( varcode, varname)
              IF (dset .EQ. -1) THEN  !for user vars type not known in attrib structure
                 vartype = 5
                 IF (itsa_string) vartype = 2
              ENDIF
              CALL CD_GET_VAR_OUTTYPE (dset, varid, outtype, status) 
              IF (outtype .EQ. 0) outtype = vartype
           ENDIF

           IF ( cat .EQ. cat_pseudo_var ) varcode =
     .                            varcode(:TM_LENSTR1(varcode))//"_"

           tax = grid_line(t_dim,grid)
           cal_name = line_cal_name(tax)
           cal_id = TM_GET_CALENDAR_ID ( cal_name )

* allowed name?
           IF ( .NOT.TM_LEGAL_NAME(varcode) ) GOTO 5200
           CALL FIND_VAR_NAME ( pdset_irrelevant, varcode, cat1, var1 )
           IF (  var1 .NE. munknown_var_name
     .     .AND. cat1 .EQ. cat_pseudo_var    ) GOTO 5200

* if a string variable, then find max len
	   IF (itsa_string) THEN
	      maxstrlen = GET_MAX_C_STRING_LEN(
     .				   mr_c_pointer(mr), MGRID_SIZE(mr))
	   ELSE
	      maxstrlen = 0
	   ENDIF

* get slab limits to use for writing the values of this variable
           DO 50 idim = 1, 4
* ... limits within which to write variable data
              write_lo(idim) = cx_lo_ss(cx,idim)
              write_hi(idim) = cx_hi_ss(cx,idim)
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
 50        CONTINUE

* determine the netCDF "record dimension" to use
           IF ( write_lo(t_dim) .EQ. unspecified_int4 ) THEN
              recdim = 0
           ELSEIF ( crecdim .EQ. 'T' ) THEN
              recdim = t_dim
           ELSE
              recdim = 0
           ENDIF

* convert the data type into netCDF parameter value
	   IF (cx_type(cx) .EQ. ptype_float) THEN
	      type = NCFLOAT
	   ELSEIF  (cx_type(cx) .EQ. ptype_string) THEN
	      type = NCCHAR
	   ELSEIF  (cx_type(cx) .EQ. ptype_double) THEN
	      type = NCDOUBLE
	   ELSEIF  (cx_type(cx) .EQ. ptype_int4) THEN
	      type = NCLONG
	   ELSEIF  (cx_type(cx) .EQ. ptype_int2) THEN
	      type = NCSHORT
	   ELSEIF  (cx_type(cx) .EQ. ptype_int1) THEN
	      type = NCBYTE
	   ELSE
	      CALL WARN('Internal err: unknown data type')
	      type = NCFLOAT
	   ENDIF
           IF (type .NE. 0 .AND. varid .GT. 0) type = outtype
           IF (type .EQ. 0) outtype = vartype

* create the variable and its grid (coordinate output gets deferred)
           CALL CD_MAKE_VAR( cdfid, dset, varcode, type, maxstrlen,
     .               grid, make_lo, make_hi, recdim, enhead, new, 
     .               edges_flag, do_bounds, mode_upcase_output, status )

           IF ( status .NE. merr_ok ) GOTO 5800

* if its a newly-created variable write attributes
           IF ( new  .AND. all_outflag .GT. 0) THEN

* Write missing value and fill flag here; if writing the scale and
* offset attributes, then un-scale the missing and fill flags, using
* potentially double precision scale and offset from the linked-list
* structure.

              IF (varid .GT. 0) THEN
                 scalefac = 1.
                 attname = 'scale_factor'
                 CALL CD_GET_VAR_ATT_ID (dset, varid, attname, attid, 
     .                  istat)
                 IF (attid .GT. 0) CALL CD_GET_VAR_ATT_INFO (dset, 
     .              varid, attid, aname, attype, attlen, attoutflag, 
     .              istat )

                 IF (istat .EQ. ferr_ok .AND. attoutflag.EQ.1) THEN
                    IF (attype .EQ. nf_double) THEN
                       got_it = NC_GET_ATTRIB_DP (dset, varid,
     .                     aname, do_warn, varcode, attlen, 
     .                     attoutflag, dvals)
                       scalefac = dvals(1)

                    ELSE
                       got_it = NC_GET_ATTRIB (dset, varid, aname,
     .                    do_warn, varcode, buff_size, attlen, 
     .                    attoutflag, buff, vals)
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
     .                     aname, do_warn, varcode, attlen, 
     .                     attoutflag, dvals)
                      addoff = dvals(1)

                    ELSE
                       got_it = NC_GET_ATTRIB (dset, varid, aname,
     .                    do_warn, varcode, attlen, buff_size, 
     .                    attoutflag, buff, vals)
                      addoff = vals(1)
                    ENDIF
                 ENDIF

* Output the bad flag used by Ferret (and _FillValue which has been made  
* the same as missing_flag)

                 bad = mr_bad_data(mr)
                 dvals(1) = bad
                 attlen = 1

* Get attype for missing flag

                 istat = ferr_ok
                 attoutflag = 0
                 attname = 'missing_value'
                 CALL CD_GET_VAR_ATT_ID (dset, varid, attname, attid, 
     .                  istat)
                 IF (attid .GT. 0) CALL CD_GET_VAR_ATT_INFO (dset, varid, 
     .                attid, aname, attype, attlen, attoutflag, istat )

* If the data is numeric, see if the missing flag needs to be rescaled, 
* or converted to the output type

                 IF (istat .EQ. ferr_ok .AND. outtype .NE. NCCHAR) THEN

* If un-scaling on output, will also un-scale the missing flag.
                    IF (scalefac .NE. 1.D0  .OR. addoff .NE. 0.D0) THEN
                       dvals(1) = dvals(1)/scalefac - addoff
                       vals(1) = dvals(1)

* Otherwise convert missing flag to requested output type for the variable
                    ELSE
                       IF (attype .NE. outtype .AND. istat .EQ. ferr_ok)  THEN
                          alen = TM_LENSTR1(typnam(attype))
                          slen = TM_LENSTR1(typnam(outtype))
                          CALL WARN (
     .                      'Converting data type of missing_value '//
     .                       typnam(attype)(1:alen)//
     .                      ' to match output type of variable '//
     .                      typnam(outtype)(1:slen) )
                       ENDIF
                       attype = outtype
                    ENDIF
   
                    IF (attoutflag .EQ. 1) THEN
                       CALL CD_WRITE_ATTVAL_DP(cdfid, varcode, 
     .                   'missing_value', dvals, attlen, attype, status )
                       IF (status .NE. ferr_ok) GOTO 5400
                       CALL CD_WRITE_ATTVAL_DP(cdfid, varcode, 
     .                   '_FillValue', dvals, attlen, attype, status )

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
                    slen = TM_LENSTR1(attname)
                    IF ( MATCH_NAME (attname,  slen,
     .                               'MISSING_VALUE', 13 ) .OR. 
     .                   MATCH_NAME (attname,  slen,
     .                               '_FILLVALUE', 10 ) ) THEN
                       slen = 0   ! done already

                    ELSE IF ( MATCH_NAME (attname,  slen,
     .                               'SCALE_FACTOR', 13 ) .OR. 
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
                          CALL WARN 
     .                ('writing only first 100 elements of attribute'//
     .                  aname(1:alen))
                       ENDIF

                       got_it = NC_GET_ATTRIB ( dset, varid, aname, 
     .                       do_warn, varcode, buff_size, attlen,
     .                       attoutflag, buff, vals)

                       IF (attype .EQ. NCCHAR .AND. attoutflag.EQ.1) THEN
                          lbuff = TM_LENSTR1(buff)
                          CALL CD_WRITE_ATTRIB(cdfid, varcode,
     .                         attname, buff(:lbuff), .FALSE., status )
                       ELSE IF (attoutflag.EQ.1) THEN
                          CALL CD_WRITE_ATTVAL(cdfid, varcode,
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

              ENDIF ! varid GT 0

* variable title
* First get from Ferret variables by means of VAR_TITLE, and
* only if this fails from attribute structure.

              got_title = .FALSE.
              buff = VAR_TITLE( cx ) !!
              lbuff = TM_LENSTR1( buff )

              IF ( buff .NE. ' ' .AND. all_outflag .NE. 0 .AND.
     .             output_title ) THEN
                 CALL CD_WRITE_ATTRIB(cdfid, varcode, 'long_name',
     .                              buff(:lbuff), .FALSE., status )
                 IF ( status .NE. merr_ok ) GOTO 5800
                 got_title = .TRUE.
              ENDIF

              IF (.NOT. got_title) THEN
                 
                 attoutflag = 1
                 IF (varid .GT. 0) THEN
                    CALL CD_GET_VAR_ATT_ID (dset, varid, 'long_name', 
     .                     attid, istat)
                    IF (attid .GT. 0) THEN
                       CALL CD_GET_VAR_ATT_INFO (dset, 
     .                     varid, attid, aname, attype, attlen, 
     .                     attoutflag, istat )
                    ELSE
                       attoutflag = 1
                       buff = varcode
                    ENDIF
                 ENDIF
                 lbuff = TM_LENSTR1( buff )

                 IF (attoutflag .EQ. 1) 
     .               CALL CD_WRITE_ATTRIB(cdfid, varcode, 'long_name',
     .                              buff(:lbuff), .FALSE., status )
                 IF ( status .NE. merr_ok ) GOTO 5800
              ENDIF

* modifier to variable title
* ..."compressing" transforms and limits not given in definitions
	      buff = ' '
	      lbuff = 1
              DO 60 idim = 1, 4
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
              IF ( dset .NE. pdset_irrelevant .AND. dset.NE.-1) THEN
       
* ..... unless the user has created a history attribute, or asked that writing
*       it be turned off. See if there is a history attribute

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
                          CALL GET_SHORT_DSET_NAME( dset, buff(6:), lbuff )
                          lbuff = TM_LENSTR1( buff )
                       ENDIF

                    ELSE
                       buff = 'From '
                       CALL GET_SHORT_DSET_NAME( dset, buff(6:), lbuff )
                       lbuff = TM_LENSTR1( buff )
                    ENDIF
                 
                 ELSE
                    buff = 'From '
                    CALL GET_SHORT_DSET_NAME( cx_data_set(cx), 
     .                        buff(6:), lbuff)
                    lbuff = TM_LENSTR1( buff )
                 ENDIF ! varid .gt. 0


                 status = ferr_ok

                 IF (all_outflag .NE. 0 .AND. attoutflag .NE. 0)
     .               CALL CD_WRITE_ATTRIB(cdfid, varcode, 'history', 
     .                              buff(:lbuff), .FALSE., status )
                 IF ( status .NE. merr_ok ) GOTO 5800

              ELSEIF ( cx_data_set(cx) .NE. pdset_irrelevant .AND.
     .                 cx_data_set(cx) .NE. -1) THEN
* data set of origin for user variable

                 attoutflag = 1
                 buff = 'From '
                 CALL GET_SHORT_DSET_NAME(cx_data_set(cx), buff(6:), lbuff)
                 lbuff = TM_LENSTR1( buff )
                 status = ferr_ok

                 IF (all_outflag .NE. 0 .AND. attoutflag .NE. 0)
     .               CALL CD_WRITE_ATTRIB(cdfid, varcode, 'history', 
     .                              buff(:lbuff), .FALSE., status )
                 IF ( status .NE. merr_ok ) GOTO 5800

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
     .                     'netcdf_var_id',rbuff4, 1, NCLONG, status )
                 IF ( status .NE. merr_ok ) GOTO 5800
	         ibuff1 = cd_data_type(var)	! for F90 data type checks
	         CALL CD_WRITE_ATTVAL(cdfid, varcode,
     .                  'netcdf_data_type', rbuff4, 1, NCBYTE, status )
                 IF ( status .NE. merr_ok ) GOTO 5800
	      ENDIF

           ENDIF

* Write attributes for psuedo-variables
           IF (cat .EQ. cat_pseudo_var) THEN  

	      CALL CD_WRITE_ATTVAL(cdfid, varcode,
     .          'missing_value', bad, 1, type, status )
              IF ( status .NE. merr_ok ) GOTO 5800

              CALL CD_WRITE_ATTVAL(cdfid, varcode,
     .          '_FillValue', bad, 1, type, status )
              IF ( status .NE. merr_ok ) GOTO 5800

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

 100    CONTINUE
	nready = nvars

*****
* Finally, write all of the coordinates and data values -- in netCDF DATA mode
* Note that this block of code is also executed following an error to
* ensure that all deferred coordinates and "ready" variables are flushed
 400	flushed = .TRUE.	! errors from here to exit may leave corrupted file
*
* write the deferred coordinates
*
	CALL CD_WRITE_DEFER_COORD( cdfid, status )
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
           
           CALL CD_GET_VAR_ID (dset, varcode, varid, status)  ! is it in the linked-list for dset
           IF (status .NE. ferr_ok .AND. itsa_uvar ) THEN
              dset = -1    ! is it a user var based on a var in dset
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

           IF ( cat .EQ. cat_pseudo_var ) varcode =
     .                            varcode(:TM_LENSTR1(varcode))//"_"

* get slab limits to use for writing the values of this variable
           DO 450 idim = 1, 4
              write_lo(idim) = cx_lo_ss(cx,idim)
              write_hi(idim) = cx_hi_ss(cx,idim)
 450        CONTINUE

* check for interrupts - dont check again until entire variable is written
           IF (interrupted) CALL ERRMSG(ferr_interrupt,status,' ',*5800)

* write the data for this variable

           CALL CD_WRITE_VAR ( memory, cdfid, dset, varcode, grid, write_lo, 
     .                        write_hi, memory(1,mr_blk1(mr)), edges_flag,
     .                        do_bounds, scalefac, addoff, bad, status )
           IF ( status .NE. merr_ok ) GOTO 5800
 500	CONTINUE

* close the file
        CALL NCCLOS(cdfid, cdfstat)
        IF ( cdfstat .NE. NCNOERR ) CALL TM_ERRMSG
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

 5300    CALL ERRMSG ( ferr_invalid_command, status,
     .                 'Exceeds grid limits: '//
     .			cmnd_buff(qual_start(errloc):qual_end(errloc))
     .			//pCR//'Variable: '//varcode, *5800 )

 5310    CALL ERRMSG ( ferr_invalid_command, status,
     .                 'Does not encompass data to be written: '//
     .			cmnd_buff(qual_start(errloc):qual_end(errloc))
     .			//pCR//'Variable: '//varcode, *5800 )

 5400    CALL ERRMSG ( ferr_TMAP_error, status, ' ', *5990 )

 5800   CALL ERRMSG ( ferr_TMAP_error, final_status, ' ', *5810 )
 5810	nready = ivar - 1
	IF (.NOT.flushed) GOTO 400 ! yea ... a bit of spagetti code ... 
	CALL NCCLOS(cdfid, cdfstat)
 5990   RETURN
	END

