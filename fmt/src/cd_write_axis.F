	SUBROUTINE CD_WRITE_AXIS ( cdfid, dset, grid, idim, recdim,
     .                           loss, hiss, ax_type,  edges_flag, 
     .                           do_bounds, mode_upcase_output,  
     .                           keepax_flag, name_from_grid, dim_only, 
     .				 date_fmt_out, status )


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
* Write an axis definition out to the current output netCDF file
* May be a full "parent" definition, a sub_axis, or a "child" (a sub-axis
* which points to the full axis)

* Programmer Steve Hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program

* revision 0.0 - 3/92
*   10/92: for performance reasons the "point_spacing" attribute will not be
*          maintained on the record axis (see cd_rec_synch)
*        - and CD_SET_MODE used to put into define mode before writing attribs
*   12/92: write axes in double precision (esp. for time axis issues) 
*    3/93: check error status after creating dimension
*    4/93: check "line_direction" in determining if time_origin is needed
*    1/95: change logical argument into integer "ax_type"
*    2/95: write NOAA COOP-compliant files: "days since 1992-02-28",
*	   "degrees_east", "degrees_north"
* 10/96 *kob* - Linux port - had to move variable assignments off of
*		declaration line.
* v4.91 *kob* 12/97 - added maxchildlen paramater for the max length of 
*                     a child axis
* v500 *kob*  3/99 - change lengths of CD_CHILDAX_NAME, CD_AXIS_NAME,
*                    name to 64 from 24 to match longer line_name
* V530 *sh* 12/00 - avoid performance problems of switching between netCDF
*	DEFINE and DATA mode by deferring actual coordinate writes until later
* v530 *acm* 1/01  calendar axes: write the calendar name if non-Gregorian
* v541 *sh* 2/02 - added support for line_modulo_len
* V550 *ACM* 12/02 Write AXIS attribute to output files.
* V570 *acm*  5/04 write bounds attribute and bounds 2xN coordinates to NetCDF files
* V581 *acm* 6/05 - Fix for bug 1271, add flag line_shift_origin
*            and restore original t0 on output
* V600 *acm* 2/06 - Fix for bug 1394, send do_err flag to TM_DATE_REFORMAT
* V600 *acm* 6/06 attribute control.
* V601 *acm* 9/06 Fix bug 1440: Remove stray "IF (attid .GT. 0) " before call to 
*                 CD_GET_VAR_ATT_INFO just after DO 200
* V602 *acm*11/06 Fix bug 1471; new routine CD_AXIS_OUTNAME
* V602 *acm*12/06 Correct the setting of got_calendar 
* V602  2/07 *acm* Fix bug 1492, changing attributes of coordinate variables 
* V61+  3/08 *acm* (After fixing bug 1534), check for attoutflag on bounds 
*                  attribute before writing bounds attribute and values.
* V61_ 5/08 *acm*  Fix bug1574: If the axis is a user-defined axis, get its
*                  information from the attribute structure; previously only
*                  did this for user variables.
* V62  *acm*  2/09 - Fix bug 1634: allow SAVE/KEEP_AXISNAMES so that child axis names
*                    arent written. They can get too long for OPeNDAP library calls
* V65  *acm* 1/10  Change all netcdf2-style calls and parameter names to 
*                  netcdf3-style calls, which are also consistent with netcdf4.
* V65  *acm* 1/10  Add #ifdef usingDODSf2cUnderscore, for 32-bit linux build
*                  linked with OPeNDAP library.
* V67  *acm* 3/11  Fix bug 1750. Revert to older behavior: if grid uses same axis
*                  name but different contents, write both to file with the name
*                  Ferret uses internally.
* V674 2/12 *acm* 6D Ferret: use nferdims rather than 4 for dimension indexing
* V683 9/12 *acm* Ticket 1968, If there is a _FillValue attributes on axes, refuse to write it.
* V685 5/13 *acm* When appending, if dimension of that name in the file but not a coord var,
*                 just check the length. See ticket 2075.
* V685 5/13 *acm* Allow specifying data type for axis
* V685 6/13 *acm* new qualifier SAVE/NOBOUNDS
* V685 6/13 *acm* Dont add SAVE/NOCOORDS
* V686 1/14 *acm* When appending to a file with a dimension but no coord var, to not write
*                 a NOTE. They just clutter things up.
* V687 3/14 *acm* Fix ticket 1750. Write Ferret-internal axis names to cdf files, unless
*                 the user has said /KEEP_AXISNAMES
* V69+ 4/14 *acm* After the above fix, need a tweak to be able to append to files having 
*                 dimensions that are not coordinate variables
* V69+ 7/14 *acm* Further fix to 1750. Ticket 2182: Keep case of axis name consistent when
*                 creating and writing axis bounds.
* V69+ 7/14 *acm* Ticket 1863: When output-flag was reset on an attribute, wasn't being applied
*                 if the variable wasn't associated with the dataset where that axis lives.
* v694 1/15 *acm* for ticket 2227: if a dimension from a nc file is not also a 
*                 1-D coordinate var, don't write the axis Ferret creates
* V697 12/15 *acm* Fix ticket 2336: consistent application of CAN MODE UPCASE
* V698  2/16 *acm* Check status of queries before continuing
* V698  4/16 *acm* Fix ticket 2241: If file has dimension but not a coordinate 
*                  variable, do not write the coordinate variable, ONLY IF writing 
*                  the full-span in that direction.
*  V698  4/16 *acm* When checking for dimension-only, this applies to file axes only.
*  V698  5/16 *acm* Fix ticket 2415: write attributes using the requested data type
*                   Routine CD_WRITE_ATTVAL_DP converts to the specified type.
* V698  4/16 *acm* Account for the calendar inside TM_UNIT_ID
* V701  7/16 *kms* Compare ds_name to char_init2048 instead of char_init1024
* V710  9/16 *acm* Ticket 2459: If axis is already in the file with a different
*                  direction write axis with a unique name for this direction.
*                  Use ww_dim_name from xtext_info.cmn
* V71  3/16 *acm* Ticket 2158: working with true monthly time axes. Write these axes
*                  as irregularly spaced daily axes.
* 1/26/2017 *acm* Ticket 1480. Do not write a modulo attribute if the requested 
*                 region is largerthan the length of the modulo axis.
* 10/6/2017 *acm* See fix in cd_get_line_direction.F, where axis or dim named 
*                 TIME but with no attributes was not marked in the T direction.
*                 An axis made from a dimension-only is not written out as the 
*                 record axis.
* V74  *acm*  7/2018 Issue # 1882, save/nobounds on monthly axis
* V7.44 11/18 *acm* Issue 1905: add int64 as a netcdf data type, recognize 
*                   other newer types 
* v745 *acm* 12/18 Issue 1909, option to write dates in ISO8601 format
* v75 *acm*   5/19 Issue 1933, recompute T0 datestring length after it
*                  possibly was reformatted
* v751 *acm*  6/19 Allow for longer strings as attribute values
* V751 *acm*  8/19 Issue 1947: If axis attr doesn't match grid direction, don't write it.
* V751 *acm* 11/19 *acm* new qualifier SAVE/NOCOORDS
* V7.61 *acm* 7/20 See issue 334. If the test in cd_get_1_axis has decided the axis is
*                  irregular, write attrib as point_spacing="uneven"
*
* argument definitions:
*     dset    - dset (if netcdf file, for getting name and attribute 
*               info from linked-list structure  
*     cdfid   - netCDF id number for already opened CDF file
*     grid    - TMAP grid id
*     idim    - axis orientation (1-nferdims)
*     recdim  - is this to be the CDF record axis ? (input)
*     loss    - lo subscript (child axis, only)
*     hiss    - hi subscript (child axis, only)
*     status  - return status

* include files

        include 'netcdf.inc'
	include 'tmap_errors.parm'
	include 'tmap_dims.parm'
#include "tmap_dset.parm"
	include 'cdf_tmap.parm'
#include "gt_lib.parm"
        include 'cd_lib.parm'
        include 'xdset_info.cmn_text'
        external xdset_info_data
        include 'xtm_grid.cmn_text'
        external xgt_grid_data
        include 'xunits.cmn_text'
	include 'calendar.decl'
	include 'calendar.cmn'	
	include 'xio.cmn_text'
        external xio_data
	include 'xtext_info.cmn'  ! with ww_dim_name

! Declare netcdf function names. 
! The ones without trailing underscore are declared in netcdf.inc
#include "netcdf_declareUnderscore.inc"

* argument declarations
      LOGICAL recdim, mode_upcase_output, name_from_grid, dim_only
      INTEGER cdfid, dset, grid, idim, loss, hiss, ax_type, 
     .        edges_flag, keepax_flag, do_bounds, date_fmt_out, status

* local variable declarations:
* *kob* explicit definitions not allowed on decl line in linux...
*       moved it to parameter line 10/96
      LOGICAL TM_ABSTRACT_AXIS, TM_DATE_OK, MATCH_NAME, NC_GET_ATTRIB,
     .       ITSA_TRUEMONTH_AXIS,
     .       abstract, do_warn, got_modulo, got_calendar, got_units, 
     .       got_positive, got_spacing, got_axis, got_cartesian, got_it,
     .       coordvar, do_off, do_scale, true_month, is_regular,
     .       write_mod, dim_only_request
      PARAMETER (do_warn = .TRUE. )
      INTEGER TM_LENSTR1, CD_AXLEN, STR_SAME, STR_DNCASE, TM_LENSTR,
     .        dimid, iaxis, axid, cdfstat, lo, hi,
     .        nlen, elen, axlen, slen, edgid, bndid, npts, dim2d(2)
      INTEGER TM_GET_CALENDAR_ID, CD_WRITE_BNDSDIM, TM_UNIT_ID, TM_UNITS_CAL,
     .        cal_id, varid, llen, vartype, nvdims, nvatts, iatt, 
     .        attype, attlen,attoutflag, istat, maxlen, vdims(8), 
     .        newid, all_outflag, attid, dset_num, outtype, axis_packed, 
     .        dlen, vlen, iset, dim_status
      REAL vals(100)

!      LOGICAL condensed
!      REAL rlen
* *kob* 3/99 - change lengths of CD_CHILDAX_NAME, CD_AXIS_NAME,
*                    name to 64 from 24 to match longer line_name
* *acm* 6/06 - change axis name lengths to 128
      CHARACTER CD_CHILDAX_NAME*128, CD_AXIS_NAME*128, CD_AXIS_OUTNAME*128, 
     .       TM_FMT*48, name*128, name_in*128, units*48, datebuff*40, 
     .       buff*512, buff2*512, attname*128, dimstr*12, varstr*12, 
     .       name_bnds*140, axis_dir*1


!      REAL*8 val8_cdf, val8_tmap
* 12/97
      INTEGER maxchildlen
      PARAMETER (maxchildlen = 24)

* initialize
      iaxis = grid_line(idim, grid)
      dset_num = dset
      true_month = ITSA_TRUEMONTH_AXIS(iaxis)
      is_regular = line_regular(iaxis) .AND. .NOT.true_month 
      cal_id = 1

      dim_only_request = dim_only

      IF (true_month) THEN
         cal_id = TM_GET_CALENDAR_ID ( line_cal_name(iaxis) )
         IF (cal_id .NE. d360) THEN
            is_regular = .FALSE.
            IF (do_bounds .GE. 0) do_bounds = 1
         ENDIF
      ENDIF

      IF (.NOT.mode_upcase_output) keepax_flag = 1

      name = CD_AXIS_NAME( dset_num, grid, idim, nlen )
      IF (keepax_flag .EQ. 1) name = CD_AXIS_OUTNAME( dset_num, grid, idim, nlen )
      name_in = name
      
      name_from_grid = .FALSE.

* If the datset is user-variable or unknown, see if the axis is from a dataset. The
* OUTFLAG information for writing attributes is in the datset linked-list structure.

      IF (dset .LE. 0) THEN
         buff2  = name
         DO 50 iset = 1, maxdsets
	    IF (ds_name(iset) .NE. char_init2048) THEN
	       CALL CD_GET_VAR_ID (iset, buff2, varid, status)
	       IF (status .EQ. merr_ok) THEN
	          dset_num = iset
		  GOTO 60
	       ENDIF
	    ENDIF
   50 	CONTINUE
   60 	CONTINUE
      ENDIF

      buff2 = ' '

      npts = line_dim(iaxis)
      abstract = TM_ABSTRACT_AXIS( iaxis )

* If mode upcase_output is set, then upcase both name and name_in which is 
* being stored for later use.

      IF (mode_upcase_output) THEN
         CALL STR_UPCASE( name_in, name)
         CALL STR_UPCASE( name, name_in)

* If not mode upcase_output, then keep the incoming case. 
      ELSE 
         name = CD_AXIS_NAME( dset_num, grid, idim, nlen )
         IF (keepax_flag .EQ. 1) name = CD_AXIS_OUTNAME( dset_num, grid, idim, nlen )

         name_in = CD_AXIS_OUTNAME( dset_num, grid, idim, llen )

	 IF (STR_SAME(name, name_in) .EQ. 0) THEN 
	    name = name_in
	    nlen = llen
	 ENDIF
      ENDIF

* determine lo/hi limits of the axis to be written
      IF ( abstract ) THEN
         lo = loss
         hi = hiss
      ELSEIF ( ax_type .EQ. pcdf_full_span ) THEN
         lo = 1
         hi = line_dim(iaxis)
      ELSE
         lo = loss
         hi = hiss
      ENDIF

* dimension and size for this axis
      axlen = hi - lo + 1
      IF ( abstract .OR. ax_type.NE.pcdf_full_span ) THEN
         IF (keepax_flag .EQ. 0) 
     .       name = CD_CHILDAX_NAME( name, lo, hi, nlen )
* *kob* 12/9/97 - use new paramater maxchildlen in length check
         nlen = MIN(nlen,maxchildlen)
      ENDIF

* Is it just a dimension, not a coordinate variable

      buff = CD_AXIS_OUTNAME( dset_num, grid, idim, llen )

* Is it already in the file we're writing to?
* If it's there as a coordinate variable then dim_only is false.
* dim_status = 0 if dimension does not yet written
*            = 1 if dimension exists but there is not a coordinate variable
*            = 2 if dimension and coordinate variable are already in the file
      
      CALL CD_DIM_STATUS (cdfid, buff, llen, dim_status)
      IF (dim_only_request .AND. dim_status.LT.2) THEN
         dim_only = dim_only_request
      ELSE
         IF (dim_status .EQ. 2) dim_only = .FALSE.
      ENDIF
      
      cdfstat = NF_INQ_DIMID( cdfid, name(:nlen), dimid)

* ... check the axis attribute for direction. If the name has already
*     been used writing a different dimension, write this axis with a new name.

      IF ( cdfstat .EQ. NF_NOERR .AND. TM_LENSTR(name) .GT. 0) 
     .   CALL CD_AXNAME_TO_WRITE (cdfid, idim, name, nlen, cdfstat)

* already defined ? (limited consistency checking)
      IF ( cdfstat .EQ. NF_NOERR ) THEN

	 IF (.NOT. recdim) THEN 
            slen = CD_AXLEN( cdfid, dimid, status )
            IF ( status .NE. merr_ok ) GOTO 5900
            IF ( keepax_flag .NE. 1 .AND. .NOT.abstract) THEN

* ... check the length. 
               IF ( axlen .NE. slen ) THEN
	          IF ( cdfstat .NE. NF_NOERR ) GOTO 5300

* use the basic name straight from the grid
                  name = line_name(iaxis)
                  nlen = TM_LENSTR1( name )
                  name_in = name   ! ??

* If mode upcase_output is set, then upcase name

                  IF (mode_upcase_output) THEN
                     CALL STR_UPCASE( buff, name)
                  ENDIF

                  npts = line_dim(iaxis)
                  abstract = TM_ABSTRACT_AXIS( iaxis )

                  bndid = -1

* determine lo/hi limits of the axis to be written
                  IF ( abstract ) THEN
                     lo = loss
                     hi = hiss
                  ELSEIF ( ax_type .EQ. pcdf_full_span ) THEN
                     lo = 1
                     hi = line_dim(iaxis)
                  ELSE
                     lo = loss
                     hi = hiss
                  ENDIF

* dimension and size for this axis
                  axlen = hi - lo + 1

* Yes, use the basic name straight from the grid
	          name_from_grid = .TRUE.
                  
	       ENDIF
            ELSE 
               IF (axlen .NE. slen ) GOTO 5100
               GOTO 1000
            ENDIF
	 ENDIF
	 IF (.NOT. name_From_grid) GOTO 1000
      ENDIF

* Check that the axis type is set. User may have reset the type and then
* asked for SET AX/OUTTYPE=INTYPE.  If so need to re-get the orig type.
* If its a static line, set to double, and store that in line_dattype. If
* its an axis from a netcdf dataset, get the type from initializing the dataset.

      outtype = line_dattype(iaxis)

      CALL CD_GET_VAR_ID (dset_num, line_name(iaxis), varid, status)
      IF (varid .GT. 0) CALL CD_GET_VAR_INFO (dset_num, varid, buff, vartype, 
     .            nvdims, vdims, nvatts, coordvar, all_outflag, status)

      IF (outtype.LT.NF_BYTE .OR.  outtype.GT.NF_DOUBLE) THEN
         IF (iaxis .LE. max_lines) THEN
	    outtype = NF_DOUBLE  ! static lines - user-defined or Ferret internal
	    line_dattype(iaxis) = NF_DOUBLE
	 ELSE
	    outtype = vartype
         ENDIF
      ENDIF

* For now anyway write an axis that came in as INT64 using NF_DOUBLE instead.
* The NF_DEF_VAR below fails if the type is set to int64. 
* The unsigned integer types are defined but also not currently allowed by 
* NF_DEF_VAR, so switch to the signed versions.

      IF (outtype .EQ. NF_INT64 .OR. 
     .    outtype .EQ. NF_UINT64) outtype = NF_DOUBLE  
      IF (outtype .EQ. NF_USHORT) outtype = NF_SHORT  
      IF (outtype .EQ. NF_UINT) outtype = NF_INT  

* If axis came in packed we are not going to re-pack it on output (for now).
* write it as double and add a note.

      axis_packed = 0
      IF (varid .GT. 0) THEN
         attname = 'add_offset'
         CALL CD_GET_VAR_ATT_ID (dset_num, varid, attname, attid, status)
         IF (attid .GT. 0) axis_packed = 1
         attname = 'scale_factor'
         CALL CD_GET_VAR_ATT_ID (dset, varid, attname, attid, status)
         IF (attid .GT. 0) axis_packed = 1

         IF (axis_packed .EQ. 1) THEN	    
            outtype = NF_DOUBLE
         ENDIF
      ENDIF

      npts = line_dim(iaxis)
      abstract = TM_ABSTRACT_AXIS( iaxis )

      bndid = -1


* put CDF into define mode so axis can be defined
      CALL CD_SET_MODE(cdfid, pcd_mode_define, status)
      IF ( status .NE. merr_ok ) RETURN

* define the dimension (if its just a dimension, will not write as a record axis)
      IF ( recdim .AND. .NOT.line_dim_only(iaxis) .AND. .NOT.dim_only) THEN
         cdfstat = NF_DEF_DIM( cdfid, name(:nlen), NF_UNLIMITED, dimid )
      ELSE
         cdfstat = NF_DEF_DIM( cdfid, name(:nlen), axlen, dimid )
      ENDIF
      
* checking. debugging.  npts should be 0, no records written yet.
         cdfstat = NF_INQ_DIM ( cdfid, dimid, buff, npts )

      IF ( cdfstat .NE. NF_NOERR ) GOTO 5200

* If the dimension was a dimension only (not a 1-D coordinate variable) in the 
* source datset, just create the dimension in the output file. Don't write out
* the basic axis that Ferret defined. (If Ferret synthesized the variable, do write
* the axis.)

** ticket 2241: if it's a subspan axis, write it out...

      IF (line_dim_only(iaxis) .AND. dset.GT.0 .AND.
     .    ax_type.EQ.pcdf_full_span) GOTO 1000

      IF (dim_only) GOTO 1000

* create the coordinate variable (12/92 -> double precision)
      cdfstat = NF_DEF_VAR(cdfid, name(:nlen), outtype, 1, dimid, axid)
      IF ( cdfstat .NE. NF_NOERR ) GOTO 5200

* check: can we get the coord var id back??
         cdfstat = NF_INQ_VARID( cdfid, name(:nlen), newid )

* assign its attributes
      IF (ax_type .EQ. pcdf_child_span) THEN
         cdfstat = NF_PUT_ATT_TEXT (cdfid, axid, 'child_axis', 1, ' ')
      ENDIF
!      IF ( condensed ) THEN
!         cdfstat = NF_PUT_ATT_DOUBLE (cdfid, axid, 'true_size',
!     .         1, npts)
!      ENDIF

      got_modulo = .FALSE.
      got_calendar = .FALSE.
      got_units = .FALSE.
      got_positive = .FALSE.
      got_spacing = .FALSE.
      got_axis = .FALSE.
      got_cartesian = .FALSE.
      
      write_mod = .TRUE.

! get whatever attributes are available in the linked-list structure.
! use the name of the parent axis.

      CALL CD_GET_VAR_ID (dset_num, name_in, varid, status)

* If the axis is a user-defined axis, we still dont have its info. Look in 
* the pseudo-dataset of user-defined coordinate variables.

      IF (varid .EQ. 0 .AND. .NOT.abstract) THEN
         dset_num = -2
         CALL CD_GET_VAR_ID (dset_num, name_in, varid, status)
      ENDIF
      
      IF (varid .GT. 0) THEN

* Get axis name into original upper/lowercase form, and get nvatts.

         CALL CD_GET_VAR_INFO (dset_num, varid, buff, vartype, 
     .                   nvdims, vdims, nvatts, coordvar, 
     .                   all_outflag, status) 

* If mode upcase_output is set, then upcase the name.

         name_in = buff
         IF (mode_upcase_output) CALL STR_UPCASE( name_in, buff)

         DO 200 iatt = 1, nvatts
            attoutflag = 0
            CALL CD_GET_VAR_ATT_INFO (dset_num, varid, 
     .               iatt, attname, attype, attlen, attoutflag, status )

            maxlen = 512
            got_it = NC_GET_ATTRIB ( dset_num, varid, attname, 
     .               do_warn, name_in, maxlen, attlen, attoutflag, 
     .               buff, vals)

* If the user has not asked for bounds and if its a regular axis then
*  do not write bounds attribute. (The bounds themselves arent written.)
            IF ( (STR_SAME(attname, 'bounds') .EQ. 0) .AND. 
     .           (is_regular .AND. do_bounds.LE.0) ) 
     .               attoutflag = 0

* If they asked for /NOBOUNDS, dont write bounds even if the axis is irregular.
            IF ( STR_SAME(attname, 'bounds') .EQ. 0) THEN
	       name_bnds = name(:nlen)//'_bnds'
               IF (name_bnds.NE.buff) buff = name_bnds
	       IF ( do_bounds.LT.0) attoutflag = 0
            ENDIF

* If there is a _FillValue attributes on axes, refuse to write it.

            IF (MATCH_NAME (attname,  TM_LENSTR1(attname), 
     .           '_FILLVALUE', 10) ) attoutflag = 0
		
* If there is an axis attribute check that it matches the direction in the grid
* and skip writing it if not.

            IF (attoutflag.EQ.1 .AND. MATCH_NAME (attname,  TM_LENSTR1(attname), 'axis', 4) ) THEN
	       IF ( .NOT. MATCH_NAME (buff, 1, ww_dim_name(idim), 1) ) THEN
	          attoutflag = 0
		  got_axis = .TRUE.
	       ENDIF
            ENDIF
 
* Write the attribute, but skip this for the point-spacing attr on record axis.

            IF ( MATCH_NAME (attname,  TM_LENSTR1(attname), 
     .           'MODULO', 6) ) THEN

* Ticket 1480. For a modulo axis, if the requested region is longer than the 
* axis length, do not write the modulo attribute. Issue a note.

	       IF (hiss - loss .GE. line_dim(iaxis)) THEN
                  CALL WARN ('Requested region on modulo axis '//name(:nlen)//
     .               ' is larger than modulo length. Modulo attribute not written.')
                  attoutflag = 0
		  write_mod = .FALSE.
	       ENDIF

               IF (attype .NE. NF_CHAR .AND. vals(1) .EQ. 0.) THEN
! Write line_modulo_len 
                  vals(1) = line_modulo_len(iaxis)
                  IF (attoutflag .EQ. 1) attype = NF_FLOAT
               ENDIF
            ENDIF

            IF ( recdim  .AND. 
     .           MATCH_NAME (attname,  TM_LENSTR1(attname), 
     .           'POINT_SPACING', 13) ) THEN
               got_spacing = .TRUE.
	       IF (true_month) buff = 'uneven' 
	       IF (.NOT. line_regular(iaxis))  buff = 'uneven' 
            ELSE
               IF (attype .EQ. NF_CHAR .AND. attoutflag .EQ. 1) THEN
		  IF (MATCH_NAME (attname,  5, 'UNITS', 5) ) THEN
                     buff2 = buff
                     status = STR_DNCASE(buff, buff2)

		  ENDIF
		   
		  IF (MATCH_NAME (attname,
     .                 TM_LENSTR1(attname), 'POINT_SPACING', 13)  .AND. 
     .               .NOT.line_regular(iaxis))  buff = 'uneven' 
 
                  llen = TM_LENSTR1(buff)
                  CALL CD_WRITE_ATTRIB (cdfid, name(:nlen), attname,
     .                               buff(:llen), .FALSE., status )
	       ELSE IF (attoutflag .EQ. 1) THEN
                  CALL CD_WRITE_ATTVAL_DP(cdfid, name(:nlen), attname,
     .                                 vals, attlen, attype, status )
               ENDIF
            ENDIF

            slen = TM_LENSTR1(attname)
            IF (MATCH_NAME (attname,  slen, 'UNITS', 5 ) .AND. 
     .          attoutflag .EQ. 1)
     .                got_units = .TRUE.
            IF (MATCH_NAME (attname,  slen, 'CALENDAR', 8 ) .AND. 
     .          attoutflag .EQ. 1)
     .                got_calendar = .TRUE.
            IF (MATCH_NAME (attname,  slen, 'MODULO', 6 ) .AND. 
     .          attoutflag .EQ. 1)
     .                got_modulo = .TRUE.
            IF (MATCH_NAME (attname,  slen, 'POSITIVE', 8 ) .AND. 
     .          attoutflag .EQ. 1)
     .                got_positive = .TRUE.
            IF (MATCH_NAME (attname,  slen, 'POINT_SPACING', 13) .AND. 
     .          attoutflag .EQ. 1)
     .                got_spacing = .TRUE.
            IF (MATCH_NAME (attname,  slen, 'AXIS', 4 ) .AND. 
     .          attoutflag .EQ. 1)
     .                got_axis = .TRUE.
            IF (MATCH_NAME (attname,  slen, 'CARTESIAN_AXIS', 14 ) .AND. 
     .          attoutflag .EQ. 1)
     .                got_cartesian = .TRUE.

 200     CONTINUE
      ENDIF

* write modulo, t0, etc as attributes if they exist in Ferret 
* variables line_modulo etc. This is consistent with previous Ferret behavior.

      attoutflag = 1  

* May want to add since t0 

      IF ( line_direction(iaxis) .EQ. 'TI' ) THEN
         cal_id = TM_GET_CALENDAR_ID (line_cal_name(iaxis))
         IF (TM_DATE_OK( line_T0(iaxis),cal_id ) ) got_units = .FALSE.
      ENDIF

      IF ( line_units(iaxis) .NE. ' ' .AND. (.NOT.got_units) ) THEN
	 units = line_units(iaxis)	 

	 IF ( line_direction(iaxis) .EQ. 'WE'
     .	.AND. line_unit_code(iaxis) .EQ. pun_degrees ) THEN
	    units = 'degrees_east'
	 ELSEIF ( line_direction(iaxis) .EQ. 'SN'
     .	.AND. line_unit_code(iaxis) .EQ. pun_degrees ) THEN
	    units = 'degrees_north'
	ELSEIF ( line_direction(iaxis) .EQ. 'TI'
     . .AND. TM_DATE_OK( line_T0(iaxis),cal_id )    ) THEN
            buff = line_units(iaxis)
	    slen = TM_LENSTR1( buff )
	    units = buff(:slen)//' since '

* get original units. If a user-defined true-month axis, will write in units of days.
	    IF (true_month) THEN  
	       IF (buff2(1:8).EQ.'TRUMONTH' .OR. 
     .             TM_LENSTR1(buff2).LE.1) buff2 = 'days'
	       slen = INDEX(buff2, ' since') - 1
	       IF (slen .LE. 0) slen = TM_LENSTR1 (buff2)
	       units = buff2(:slen)//' since'
	    ENDIF

            IF (line_shift_origin(iaxis)) THEN
               datebuff = '01-JAN-0001 00:00:00'
            ELSE
 	       datebuff = line_T0(iaxis)
            ENDIF
            CALL TM_DATE_REFORMAT(datebuff,pdate_vax,cal_id,
     .				  units(slen+8:),pdate_numslashnum,
     .				  .TRUE., status )

            buff2 = units
            status = STR_DNCASE(units, buff2)

	ENDIF
        slen = TM_LENSTR1( units )

        attname = 'units' 
        IF (varid .GT. 0) THEN
           CALL CD_GET_VAR_ATT_ID (dset_num, varid, attname, attid, 
     .                 status)
           IF (attid .GT. 0) CALL CD_GET_VAR_ATT_INFO (dset_num, varid, 
     .         attid, attname,  attype, attlen, attoutflag, istat )
        ELSE
           attoutflag = 1
        ENDIF

        IF (attoutflag .EQ. 1) 
     .     cdfstat = NF_PUT_ATT_TEXT( cdfid, axid, attname,
     .        	       slen, units(:slen) )

      ENDIF
      IF ( idim .EQ. t_dim .AND. (.NOT.got_calendar) ) THEN                  ! 4/93 update
        cal_id = TM_GET_CALENDAR_ID (line_cal_name(iaxis))
	IF ( line_direction(iaxis) .EQ. 'TI'
     . .AND. TM_DATE_OK( line_T0(iaxis), cal_id )    ) THEN

            IF (line_shift_origin(iaxis)) THEN
               datebuff = '01-JAN-0001 00:00:00'
            ELSE
 	       datebuff = line_T0(iaxis)
            ENDIF

            slen = TM_LENSTR1( datebuff )
            
            attname = 'time_origin' 
            IF (varid .GT. 0) THEN
               CALL CD_GET_VAR_ATT_ID (dset_num, varid, attname, attid, 
     .                 status)
               IF (attid .GT. 0) CALL CD_GET_VAR_ATT_INFO (dset_num, varid, 
     .             attid, attname, attype, attlen, attoutflag, istat )
            ELSE 
               attoutflag = 1
            ENDIF

* If they have set MODE DATE_FORMAT:ISO then convert the date string
            CALL TM_DATE_REFORMAT(datebuff, pdate_vax, cal_id,
     .                            datebuff, date_fmt_out,
     .                            .TRUE., status )

            slen = TM_LENSTR1 (datebuff)
	    IF (attoutflag .EQ. 1) 
     .         cdfstat = NF_PUT_ATT_TEXT( cdfid, axid, attname,
     .        	       slen, datebuff(:slen) )
	ENDIF

*  Write the calendar name

        IF (cal_id .NE. 1) THEN
          slen = TM_LENSTR1( line_cal_name(iaxis) )

          attname = 'calendar' 
          IF (varid .GT. 0) THEN
             CALL CD_GET_VAR_ATT_ID (dset_num, varid, attname, attid, status)
             IF (attid .GT. 0) CALL CD_GET_VAR_ATT_INFO (dset_num, varid, 
     .                attid, attname, attype, attlen, attoutflag, istat )
          ELSE
             attoutflag = 1
          ENDIF

          IF (attoutflag .EQ. 1) 
     .      cdfstat = NF_PUT_ATT_TEXT( cdfid, axid, attname,
     .        	       slen, line_cal_name(iaxis)(:slen))
        ENDIF
      ENDIF

* Ticket 1480. For a modulo axis, if the requested region is longer than the 
* axis length, do not write the modulo attribute. 

      IF ( line_modulo(iaxis) .AND. (.NOT.got_modulo) ) THEN
	 IF (line_modulo_len(iaxis) .EQ. 0.0D0 ) THEN
           attname = 'modulo' 
           IF (varid .GT. 0) THEN
              CALL CD_GET_VAR_ATT_ID(dset_num, varid, attname, attid,status)
              IF (attid .GT. 0) CALL CD_GET_VAR_ATT_INFO (dset_num, varid, 
     .              attid, attname, attype, attlen, attoutflag, istat )
           ELSE
              attoutflag = 1
           ENDIF
           IF (attoutflag .EQ. 1 .AND. write_mod)
     .       cdfstat = NF_PUT_ATT_TEXT( cdfid, axid, attname,
     .        	       1, ' ' )
	 ELSE
           attname = 'modulo' 
           IF (varid .GT. 0) THEN
              CALL CD_GET_VAR_ATT_ID(dset_num, varid, attname, attid,status)
              IF (attid .GT. 0) CALL CD_GET_VAR_ATT_INFO (dset_num, varid, 
     .               attid, attname, attype, attlen, attoutflag, istat )
           ELSE
              attoutflag = 1
           ENDIF
           IF (attoutflag .EQ. 1 .AND. write_mod) 
     .      cdfstat = NF_PUT_ATT_DOUBLE( cdfid, axid, attname,
     .        	       outtype, 1, line_modulo_len(iaxis) )
	 ENDIF

* Check if user wants to write modulo=false to the file
      ELSE IF (got_modulo .AND. (varid .GT. 0) .AND.
     .         .NOT.line_modulo(iaxis)  ) THEN 
            attname = 'modulo'
            CALL CD_GET_VAR_ATT_ID (dset_num, varid, attname, attid, status)
            IF (attid .GT. 0) CALL CD_GET_VAR_ATT_INFO (dset_num, varid, 
     .              attid, attname, attype, attlen, attoutflag, istat )

            maxlen = 512
            got_it = NC_GET_ATTRIB ( dset_num, varid, attname, 
     .               do_warn, name_in, maxlen, attlen, attoutflag, 
     .               buff, vals)

            IF (attype .EQ. NF_CHAR .AND. attoutflag .EQ. 1) THEN
               llen = TM_LENSTR1(buff)
               CALL CD_WRITE_ATTRIB(cdfid, name(:nlen), attname,
     .                               'false', .FALSE., status )
            ELSE IF (attoutflag .EQ. 1) THEN
               CALL CD_WRITE_ATTVAL_DP(cdfid, name(:nlen), attname,
     .                              vals, attlen, attype, status )
            ENDIF


      ENDIF

      IF ( idim.EQ.z_dim .AND. line_direction(iaxis).EQ.'UD' .AND.
     .     (.NOT.got_positive) ) THEN
           attname = 'positive' 
           IF (varid .GT. 0) THEN
              CALL CD_GET_VAR_ATT_ID(dset_num, varid, attname, attid,status)
              IF (attid .GT. 0) CALL CD_GET_VAR_ATT_INFO (dset_num, varid, 
     .              attid, attname, attype, attlen, attoutflag, istat )
           ELSE
              attoutflag = 1
           ENDIF
           IF (attoutflag .EQ. 1) 
     .      cdfstat = NF_PUT_ATT_TEXT( cdfid, axid, attname,
     .        	       4, 'down' )
      ENDIF

* record axis (time, usually) doesn't get coordinate spacing or any
* coordinates inserted at creation time
      IF ( recdim ) THEN

         CONTINUE
      ELSE
        IF (.NOT. got_spacing) THEN
        IF ( is_regular ) THEN
           attname = 'point_spacing' 
           IF (varid .GT. 0) THEN
              CALL CD_GET_VAR_ATT_ID(dset_num, varid, attname, attid,status)
              IF (attid .GT. 0) CALL CD_GET_VAR_ATT_INFO (dset_num, varid, 
     .              attid, attname, attype, attlen, attoutflag, istat )
           ELSE
              attoutflag = 1
           ENDIF
           IF (attoutflag .EQ. 1) 
     .      cdfstat = NF_PUT_ATT_TEXT( cdfid, axid, attname,
     .        	       4, 'even' )
        ELSE
* sub-span axis may not have irregular spacing just cuz parent does
* 12/00 NO. ... regular appearing coordinates are not necessarily regular --
*       for example 2 points extracted from an irregular axis will always
*	appear to be regular, but their box limits are not necessarily so
           IF ( ax_type .EQ. pcdf_full_span
     .     .OR. ax_type .EQ. pcdf_sub_span ) THEN
              attname = 'point_spacing' 
              IF (varid .GT. 0) THEN
                 CALL CD_GET_VAR_ATT_ID (dset_num, varid, attname, 
     .                  attid, status)
                 IF (attid .GT. 0) CALL CD_GET_VAR_ATT_INFO (dset_num, varid, 
     .               attid, attname, attype, attlen, attoutflag, istat )
              ELSE 
                 attoutflag = 1
              ENDIF
              IF (attoutflag .EQ. 1)
     .      cdfstat = NF_PUT_ATT_TEXT( cdfid, axid, attname,
     .        	       6, 'uneven' )
           ENDIF
        ENDIF
        ENDIF  ! got_spacing

* set up for deferred writing of coordinate values (12/00)
	CALL CD_DEFER_COORD_WRITE(axid, grid, idim, lo, hi,
     .				  pcd_defer_points)

      ENDIF

* Write AXIS attribute

      IF (.NOT. got_axis) THEN

           attname = 'axis' 
           IF (varid .GT. 0) THEN
              CALL CD_GET_VAR_ATT_ID (dset_num, varid, attname, attid,
     .                  status)
              IF (attid .GT. 0) CALL CD_GET_VAR_ATT_INFO (dset_num, varid,
     .             attid, attname, attype, attlen, attoutflag, istat )
           ELSE
              attoutflag = 1
           ENDIF
            
          IF (attoutflag .EQ. 1) cdfstat = NF_PUT_ATT_TEXT
     .        	     ( cdfid, axid, attname, 1, ww_dim_name(idim) )
       ENDIF

* If the axis is a know geophysical direction write the standard name 
* attribute for it.

      attname = 'standard_name'
      IF ( line_direction(iaxis) .EQ. 'WE') THEN
         buff = 'longitude'
	 llen = TM_LENSTR1(buff)
         CALL CD_WRITE_ATTRIB (cdfid, name(:nlen), attname,
     .        buff(:llen), .FALSE., status )
      ENDIF
      IF ( line_direction(iaxis) .EQ. 'SN') THEN
         buff = 'latitude'
	 llen = TM_LENSTR1(buff)
         CALL CD_WRITE_ATTRIB (cdfid, name(:nlen), attname,
     .        buff(:llen), .FALSE., status )
      ENDIF
      IF ( line_direction(iaxis) .EQ. 'UD') THEN
         buff = 'depth'
	 llen = TM_LENSTR1(buff)
         CALL CD_WRITE_ATTRIB (cdfid, name(:nlen), attname,
     .        buff(:llen), .FALSE., status )
      ENDIF
      IF ( line_direction(iaxis) .EQ. 'DU') THEN
         buff = 'altitude'
	 llen = TM_LENSTR1(buff)
         CALL CD_WRITE_ATTRIB (cdfid, name(:nlen), attname,
     .        buff(:llen), .FALSE., status )
      ENDIF
      IF ( line_direction(iaxis) .EQ. 'TI') THEN
         buff = 'time'
	 llen = TM_LENSTR1(buff)
         CALL CD_WRITE_ATTRIB (cdfid, name(:nlen), attname,
     .        buff(:llen), .FALSE., status )
      ENDIF

* If the axis came in scaled, and the users asked for scaling on output
* then return a note.

       IF (varid .GT. 0) THEN
          do_scale = .FALSE.
	  do_off = .FALSE.
          attname = 'scale_factor' 
          CALL CD_GET_VAR_ATT_ID (dset_num, varid, attname, attid,
     .               status)
          IF (attid .GT. 0) THEN
	     CALL CD_GET_VAR_ATT_INFO (dset_num, varid,
     .          attid, attname, attype, attlen, attoutflag, istat )
             IF (attoutflag .EQ. 1) do_scale = .TRUE.
          ENDIF

          attname = 'add_offset' 
          CALL CD_GET_VAR_ATT_ID (dset_num, varid, attname, attid,
     .               status)
          IF (attid .GT. 0) THEN
             CALL CD_GET_VAR_ATT_INFO (dset_num, varid,
     .          attid, attname, attype, attlen, attoutflag, istat )
             IF (attoutflag .EQ. 1) do_off = .TRUE.
          ENDIF
 
          IF (do_scale .OR. do_off .OR. (axis_packed .EQ. 1) )  
     .    CALL WARN(name(1:nlen)//
     .      ': Writing scaled coordinates not implemented. Suggest '//
     .      'using nco operators on output file to pack coord values')
       ENDIF
            
* does it need axis bounds definitions ?

* Has the user explicitly asked not to write bounds attribute for the variable? 
* If so mark this with the value attoutflag = 2

      attname = 'bounds' 
      attoutflag = 0
      IF (varid .GT. 0) THEN
         CALL CD_GET_VAR_ATT_ID(dset_num, varid, attname, attid, status)
         IF (attid .GT. 0) THEN
            CALL CD_GET_VAR_ATT_INFO (dset_num, varid, 
     .      attid, attname, attype, attlen, attoutflag, istat )
            IF (attoutflag .EQ. 0) attoutflag = 2
         ENDIF
      ENDIF

* If they said /NOBOUNDS on the SAVE command, do not write bounds
* even if the axis is irregular.

      IF (do_bounds .EQ. -1) attoutflag = 2

      IF (  (ax_type.EQ.pcdf_full_span .OR. ax_type.EQ.pcdf_sub_span)
     .  .AND. ( .NOT.is_regular .OR. do_bounds.EQ.1) 
     .  .AND. ( edges_flag .EQ. 0) 
     .  .AND. ( attoutflag .NE. 2) ) THEN

* bounds attribute may already be set up.  If so use it, otherwise define bounds.
         got_it = NC_GET_ATTRIB ( dset_num, varid, attname, 
     .               do_warn, name_in, maxlen, attlen, attoutflag, 
     .               buff, vals)
         IF (got_it) THEN
            name = name_bnds
            elen = TM_LENSTR1(name)
	 ELSE
	    name = name(:nlen)//'_bnds'
	    elen = nlen + 5
	 ENDIF

* ... define the first dimension of bounds variable (what about record axis??)

         dim2d(2) = dimid    ! Dimension already defined above for coordinates

         IF ( cdfstat .NE. NF_NOERR ) GOTO 5200

* ... second dimension of bounds coordinate variable is 2
         dim2d(1) = CD_WRITE_BNDSDIM(cdfid, status)

* ... create the bounds variable, if it has not been created.
         IF (bndid .EQ. -1) cdfstat = NF_DEF_VAR(cdfid, 
     .    name(:elen), outtype, 2, dim2d, bndid)

         IF ( cdfstat .NE. NF_NOERR ) GOTO 5220

* ... point to it from axis coordinates definition

         attname = 'bounds' 
         IF (varid .GT. 0) THEN
            CALL CD_GET_VAR_ATT_ID (dset_num, varid, attname, attid, status)
            IF (attid .GT. 0) THEN
               CALL CD_GET_VAR_ATT_INFO (dset_num, varid, 
     .              attid, attname, attype, attlen, attoutflag, istat)
            ELSE
               attoutflag = 1
            ENDIF 
         ELSE
            attoutflag = 1
         ENDIF

         IF (attoutflag .EQ. 1) THEN
            cdfstat = NF_PUT_ATT_TEXT (cdfid, axid, attname,
     .                elen, name(:elen))

* set up for deferred writing of bounds coordinate values (12/00)
* Record axis and bounds are written when variable written.

           IF ( .NOT.recdim ) 
     .       CALL CD_DEFER_COORD_WRITE(bndid, grid, idim, lo, hi,
     .				  pcd_defer_bounds)
         ENDIF

      ENDIF

* does it need axis box edge definitions ?  
* edges_flag = 1 if requested with /EDGES qualifier
* edges_flag = 2 if requested with /RIGID qualifier

      IF (  (ax_type.EQ.pcdf_full_span .OR. ax_type.EQ.pcdf_sub_span)
     . .AND. ((edges_flag.EQ.1) .OR.
     .        (edges_flag.EQ.2 .AND.
     .         .NOT.is_regular .AND. 
     .         .NOT.recdim)) ) THEN

         name = name(:nlen)//'edges'
         elen = nlen + 5
         cdfstat = NF_DEF_DIM( cdfid, name(:elen), axlen+1, dimid )
         cdfstat = NF_DEF_VAR(cdfid, name(:elen), outtype, 1, 
     .           dimid, edgid)
         attname = 'edges' 
         IF (varid .GT. 0) THEN
            CALL CD_GET_VAR_ATT_ID (dset_num, varid, attname, attid, status)
            IF (attid .GT. 0) THEN
               CALL CD_GET_VAR_ATT_INFO (dset_num, varid, 
     .              attid, attname, attype, attlen, attoutflag, istat)
               IF (edges_flag .GT. 0) attoutflag = 1
            ELSE
               attoutflag = 1
            ENDIF 
         ELSE
            attoutflag = 1
         ENDIF
         IF (attoutflag .EQ. 1) THEN
            cdfstat = NF_PUT_ATT_TEXT( cdfid, edgid, attname, 1, ' ' )
* ... point to it from axis coordinates definition
            cdfstat = NF_PUT_ATT_TEXT( cdfid, axid, attname, elen,
     .                name(:elen) )

* set up for deferref writing of edge coordinate values (12/00)
	    CALL CD_DEFER_COORD_WRITE(edgid, grid, idim, lo, hi,
     .				  pcd_defer_edges)
         ENDIF

      ENDIF

* successful completion
 1000 status = merr_ok

      RETURN

* error messages
 5100 CONTINUE
      IF (keepax_flag .EQ. 1) THEN
         CALL WARN ("Writing a subset of data with original axis names")
         CALL WARN 
     . ("  Axis was already written with a different set of coordinates")
      ENDIF
      CALL TM_ERRMSG
     . ( merr_linepredef, status, 'CD_WRITE_AXIS', unspecified_int4,
     .   unspecified_int4,
     .  'axis '//name(:nlen)//' doesnt match CDF file',
     .  no_errstring, *5900 )

 5200 CALL TM_ERRMSG
     .     ( cdfstat+pcdferr, status, 'CD_WRITE_AXIS', cdfid,
     .       unspecified_int4, 'Failed creating coord variable '
     .       //name(:nlen), no_errstring, *5900 )

 5220 CALL TM_ERRMSG
     .     ( cdfstat+pcdferr, status, 'CD_WRITE_AXIS', cdfid,
     .       unspecified_int4, 
     .       'Failed creating bounds for coord variable '
     .       //name(:nlen), no_errstring, *5900 )

 5300 dimstr = TM_FMT(DBLE(slen), 0, 12, dlen)
      varstr = TM_FMT(DBLE(axlen), 0, 12, vlen)
      CALL TM_ERRMSG
     . ( merr_linepredef, status, 'CD_WRITE_AXIS', unspecified_int4,
     .   unspecified_int4,
     .  'Length of file dimension '//name(:nlen)//' is '//dimstr(:dlen)
     .  //'. Variable axis length '//varstr(:vlen)
     .  //' must match.',
     .  no_errstring, *5900 )

* error exit
 5900   RETURN
	END
