	SUBROUTINE CD_WRITE_AXIS ( cdfid, dset, grid, idim, recdim,
     .                           loss, hiss, ax_type,  edges_flag, 
     .                           do_bounds, mode_upcase_output, status )


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

* argument definitions:
*     dset    - dset (if netcdf file, for getting name and attribute 
*               info from linked-list structure  
*     cdfid   - netCDF id number for already opened CDF file
*     grid    - TMAP grid id
*     idim    - axis orientation (1-4)
*     recdim  - is this to be the CDF record axis ? (input)
*     loss    - lo subscript (child axis, only)
*     hiss    - hi subscript (child axis, only)
*     status  - return status

* include files

        include 'netcdf.decl'
#include "netcdf.inc"
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


* argument declarations
      LOGICAL recdim, do_bounds, mode_upcase_output
      INTEGER cdfid, dset, grid, idim, loss, hiss, ax_type, 
     .        edges_flag, status

* local variable declarations:
* *kob* explicit definitions not allowed on decl line in linux...
*       moved it to parameter line 10/96
      LOGICAL TM_ABSTRACT_AXIS, TM_DATE_OK, MATCH_NAME, NC_GET_ATTRIB,
     .       abstract, do_warn, got_modulo, got_calendar, got_units, 
     .       got_positive, got_spacing, got_axis, got_cartesian, got_it,
     .       coordvar
      PARAMETER (do_warn = .TRUE. )
      INTEGER TM_LENSTR1, CD_AXLEN,
     .        dimid, iaxis, axid, cdfstat, lo, hi,
     .        nlen, elen, axlen, slen, edgid, bndid, npts, dim2d(2)
      INTEGER TM_GET_CALENDAR_ID, CD_WRITE_BNDSDIM, cal_id, varid,
     .        llen, vartype, nvdims, nvatts, iatt, attype, attlen,
     .        attoutflag, istat, maxlen, vdims(8), newid, all_outflag,
     .        attid
      REAL vals(10)

!      LOGICAL condensed
!      REAL rlen
* *kob* 3/99 - change lengths of CD_CHILDAX_NAME, CD_AXIS_NAME,
*                    name to 64 from 24 to match longer line_name
* *acm* 6/06 - change axis name lengths to 128
      CHARACTER CD_CHILDAX_NAME*64, CD_AXIS_OUTNAME*64,
     .       name*128, name_in*128, units*48, datebuff*40, 
     .       buff*140, attname*128

!      REAL*8 val8_cdf, val8_tmap
* 12/97
      INTEGER maxchildlen
      PARAMETER (maxchildlen = 24)

* initialize
      iaxis = grid_line(idim, grid)
      name = CD_AXIS_OUTNAME( dset, grid, idim, nlen )

      name_in = name

* If mode upcase_output is set, then upcase both name and name_in which is 
* being stored for later use.

      IF (mode_upcase_output) THEN
         CALL STR_UPCASE( name_in, name)
         CALL STR_UPCASE( name, name_in)
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
!      condensed = line_regular(iaxis) .AND. (hi-lo).GT.3000

* dimension and size for this axis
      axlen = hi - lo + 1
      IF ( abstract .OR. ax_type.NE.pcdf_full_span ) THEN
         name = CD_CHILDAX_NAME( name, lo, hi, nlen )
* *kob* 12/9/97 - use new paramater maxchildlen in length check
         nlen = MIN(nlen,maxchildlen)
      ENDIF
      dimid = NCDID( cdfid, name(:nlen), cdfstat )

* already defined ? (limited consistency checking)
      IF ( cdfstat .EQ. NCNOERR ) THEN
* ... is there a coordinate variable by the same name ?
         axid = NCVID( cdfid, name(:nlen), cdfstat )

         IF ( cdfstat .NE. NCNOERR ) GOTO 5300
         IF ( .NOT.recdim ) THEN
* ... check the length
            slen = CD_AXLEN( cdfid, dimid, status )
            IF ( status .NE. merr_ok ) GOTO 5900
            IF ( axlen .NE. slen ) GOTO 5100
!* ... check the first value on the axis for consistency
! (check fails with "Attribute not found" in CD_RD_R8_1 - not debugged *sh*)
! (also note: this may require a mode change 10/92)
!            CALL NCVINQ(cdfid, axid, name, vartype, nvdims, vdims,
!     .                 nvatts,cdfstat)
!            CALL CD_RD_R8_1( cdfid, axid,   1, vartype, name(:nlen),
!     .                        val8_cdf, status )
!            IF ( status .NE. merr_ok ) GOTO 5900
!            val8_tmap = TM_WORLD( lo, grid, idim, box_middle )
!            IF ( SNGL(val8_cdf) .NE. SNGL(val8_tmap) ) GOTO 5100
         ENDIF
         GOTO 1000
      ENDIF

* put CDF into define mode so axis can be defined
      CALL CD_SET_MODE(cdfid, pcd_mode_define, status)
      IF ( status .NE. merr_ok ) RETURN

* define the dimension
      IF ( recdim ) THEN
         dimid = NCDDEF( cdfid, name(:nlen), NCUNLIM, cdfstat )
!      ELSEIF ( condensed ) THEN
!         dimid = NCDDEF( cdfid, name(:nlen), 2, cdfstat )
      ELSE
         dimid = NCDDEF( cdfid, name(:nlen), axlen, cdfstat )
      ENDIF
      IF ( cdfstat .NE. NCNOERR ) GOTO 5200

* create the coordinate variable (12/92 -> double precision)
      axid = NCVDEF(cdfid, name(:nlen), NCDOUBLE, 1, dimid, cdfstat)
      IF ( cdfstat .NE. NCNOERR ) GOTO 5200

* check: can we get the coord var id back??
         newid = NCVID( cdfid, name(:nlen), cdfstat )

* assign its attributes
      IF (ax_type .EQ. pcdf_child_span) THEN
         CALL NCAPTC( cdfid, axid, 'child_axis', NCCHAR, 1, ' ',
     .        cdfstat )
      ENDIF
!      IF ( condensed ) THEN
!         CALL NCAPT( cdfid, axid, 'true_size', NCLONG, 1, npts,
!     .        cdfstat )
!      ENDIF

      got_modulo = .FALSE.
      got_calendar = .FALSE.
      got_units = .FALSE.
      got_positive = .FALSE.
      got_spacing = .FALSE.
      got_axis = .FALSE.
      got_cartesian = .FALSE.

! get whatever attributes are available in the linked-list structure.
! use the name of the parent axis.

      CALL CD_GET_VAR_ID (dset, name_in, varid, status)
      IF (varid .GT. 0) THEN

* Get axis name into original upper/lowercase form, and get nvatts.

         CALL CD_GET_VAR_INFO (dset, varid, buff, vartype, 
     .                   nvdims, vdims, nvatts, coordvar, 
     .                   all_outflag, status) 

* If mode upcase_output is set, then upcase the name.

         name_in = buff
         IF (mode_upcase_output) CALL STR_UPCASE( name_in, buff)

         DO 200 iatt = 1, nvatts
            CALL CD_GET_VAR_ATT_NAME( dset, varid, iatt, 
     .                                attname, status)
            CALL CD_GET_VAR_ATT_INFO (dset, varid, 
     .               iatt, attname, attype, attlen, attoutflag, status )

            maxlen = 140
            got_it = NC_GET_ATTRIB ( dset, varid, attname, 
     .               do_warn, name_in, maxlen, attlen, attoutflag, 
     .               buff, vals)

            IF (attype .EQ. NCCHAR .AND. attoutflag .EQ. 1) THEN
               llen = TM_LENSTR1(buff)
               CALL CD_WRITE_ATTRIB (cdfid, name(:nlen), attname,
     .                               buff(:llen), .FALSE., status )
            ELSE IF (attoutflag .EQ. 1) THEN
               CALL CD_WRITE_ATTVAL(cdfid, name(:nlen), attname,
     .                              vals, attlen, attype, status )
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
     .                got_positive = .TRUE.
            IF (MATCH_NAME (attname,  slen, 'AXIS', 4 ) .AND. 
     .          attoutflag .EQ. 1)
     .                got_axis = .TRUE.
            IF (MATCH_NAME (attname,  slen, 'CARTESIAN_AXIS', 14 ) .AND. 
     .          attoutflag .EQ. 1)
     .                got_cartesian = .TRUE.

 200     CONTINUE
      ENDIF

* write modulo, point_spacing etc as attributes if they exist in Ferret 
* variables line_modulo etc. This is consistent with previous Ferret behavior.

      attoutflag = 1  

      IF ( line_units(iaxis) .NE. ' ' .AND. (.NOT.got_units) ) THEN
	 units = line_units(iaxis)	 

         IF ( line_direction(iaxis) .EQ. 'TI') 
     .        cal_id = TM_GET_CALENDAR_ID (line_cal_name(iaxis))

	 IF ( line_direction(iaxis) .EQ. 'WE'
     .	.AND. line_unit_code(iaxis) .EQ. pun_degrees ) THEN
	    units = 'degrees_east'
	 ELSEIF ( line_direction(iaxis) .EQ. 'SN'
     .	.AND. line_unit_code(iaxis) .EQ. pun_degrees ) THEN
	    units = 'degrees_north'
	ELSEIF ( line_direction(iaxis) .EQ. 'TI'
     . .AND. TM_DATE_OK( line_T0(iaxis),cal_id )    ) THEN
            slen = TM_LENSTR1( line_units(iaxis) )
	    units = line_units(iaxis)(:slen)//' since '
            IF (line_shift_origin(iaxis)) THEN
               datebuff = '01-JAN-0001 00:00:00'
            ELSE
 	       datebuff = line_T0(iaxis)
            ENDIF
            CALL TM_DATE_REFORMAT(datebuff,pdate_vax,cal_id,
     .				  units(slen+8:),pdate_numslashnum,
     .				  .TRUE., status )

	    IF ( status .NE. merr_ok ) GOTO 5900
	ENDIF
        slen = TM_LENSTR1( units )

        attname = 'units' 
        IF (varid .GT. 0) THEN
           CALL CD_GET_VAR_ATT_ID (dset, varid, attname, attid, 
     .                 status)
           IF (attid .GT. 0) CALL CD_GET_VAR_ATT_INFO (dset, varid, 
     .         attid, attname,  attype, attlen, attoutflag, istat )
        ELSE
           attoutflag = 1
        ENDIF

        IF (attoutflag .EQ. 1) 
     .     CALL NCAPTC( cdfid, axid, attname, NCCHAR, slen,
     .        	      units(:slen), cdfstat )

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
               CALL CD_GET_VAR_ATT_ID (dset, varid, attname, attid, 
     .                 status)
               IF (attid .GT. 0) CALL CD_GET_VAR_ATT_INFO (dset, varid, 
     .             attid, attname, attype, attlen, attoutflag, istat )
            ELSE 
               attoutflag = 1
            ENDIF

            IF (attoutflag .EQ. 1) 
     .         CALL NCAPTC( cdfid, axid, attname, NCCHAR, slen,
     .           datebuff(:slen), cdfstat )
	ENDIF

!  Write the calendar name

        IF (cal_id .NE. 1) THEN
          slen = TM_LENSTR1( line_cal_name(iaxis) )

          attname = 'calendar' 
          IF (varid .GT. 0) THEN
             CALL CD_GET_VAR_ATT_ID (dset, varid, attname, attid, status)
             IF (attid .GT. 0) CALL CD_GET_VAR_ATT_INFO (dset, varid, 
     .                attid, attname, attype, attlen, attoutflag, istat )
          ELSE
             attoutflag = 1
          ENDIF

          IF (attoutflag .EQ. 1) 
     .      CALL NCAPTC( cdfid, axid, attname, NCCHAR, slen,
     .        	      line_cal_name(iaxis)(:slen), cdfstat )
        ENDIF
      ENDIF


      IF ( line_modulo(iaxis) .AND. (.NOT.got_calendar) ) THEN
	 IF (line_modulo_len(iaxis) .EQ. 0.0D0 ) THEN
           attname = 'modulo' 
           IF (varid .GT. 0) THEN
              CALL CD_GET_VAR_ATT_ID(dset, varid, attname, attid,status)
              IF (attid .GT. 0) CALL CD_GET_VAR_ATT_INFO (dset, varid, 
     .              attid, attname, attype, attlen, attoutflag, istat )
           ELSE
              attoutflag = 1
           ENDIF
           IF (attoutflag .EQ. 1) 
     .       CALL NCAPTC( cdfid, axid, attname, NCCHAR, 1, ' ', cdfstat )
	 ELSE
           attname = 'modulo' 
           IF (varid .GT. 0) THEN
              CALL CD_GET_VAR_ATT_ID(dset, varid, attname, attid,status)
              IF (attid .GT. 0) CALL CD_GET_VAR_ATT_INFO (dset, varid, 
     .               attid, attname, attype, attlen, attoutflag, istat )
           ELSE
              attoutflag = 1
           ENDIF
           IF (attoutflag .EQ. 1) 
     .       CALL NCAPT( cdfid, axid, attname, NCDOUBLE, 1,
     .		       line_modulo_len(iaxis), cdfstat )
	 ENDIF

      ELSE IF (got_modulo .AND. varid .GT. 0) THEN  !user wants to write modulo=false to the file
            attname = 'modulo'
            CALL CD_GET_VAR_ATT_ID (dset, varid, attname, attid, status)
            IF (attid .GT. 0) CALL CD_GET_VAR_ATT_INFO (dset, varid, 
     .              attid, attname, attype, attlen, attoutflag, istat )

            maxlen = 140
            got_it = NC_GET_ATTRIB ( dset, varid, attname, 
     .               do_warn, name_in, maxlen, attlen, attoutflag, 
     .               buff, vals)

            IF (attype .EQ. NCCHAR .AND. attoutflag .EQ. 1) THEN
               llen = TM_LENSTR1(buff)
               CALL CD_WRITE_ATTRIB(cdfid, name(:nlen), attname,
     .                               'false', .FALSE., status )
            ELSE IF (attoutflag .EQ. 1) THEN

               CALL CD_WRITE_ATTVAL(cdfid, name(:nlen), attname,
     .                              vals, attlen, attype, status )
            ENDIF


      ENDIF

      IF ( idim.EQ.z_dim .AND. line_direction(iaxis).EQ.'UD' .AND.
     .     (.NOT.got_positive) ) THEN
           attname = 'positive' 
           IF (varid .GT. 0) THEN
              CALL CD_GET_VAR_ATT_ID(dset, varid, attname, attid,status)
              IF (attid .GT. 0) CALL CD_GET_VAR_ATT_INFO (dset, varid, 
     .              attid, attname, attype, attlen, attoutflag, istat )
           ELSE
              attoutflag = 1
           ENDIF
           IF (attoutflag .EQ. 1) 
     .       CALL NCAPTC( cdfid, axid, attname, NCCHAR, 4,
     .                'down', cdfstat )
      ENDIF

* record axis (time, usually) doesn't get coordinate spacing or any
* coordinates inserted at creation time
      IF ( recdim ) THEN

         CONTINUE
      ELSE
        IF ( line_regular(iaxis) .AND. (.NOT.got_spacing) ) THEN
           attname = 'point_spacing' 
           IF (varid .GT. 0) THEN
              CALL CD_GET_VAR_ATT_ID(dset, varid, attname, attid,status)
              IF (attid .GT. 0) CALL CD_GET_VAR_ATT_INFO (dset, varid, 
     .              attid, attname, attype, attlen, attoutflag, istat )
           ELSE
              attoutflag = 1
           ENDIF
           IF (attoutflag .EQ. 1) 
     .       CALL NCAPTC( cdfid, axid, attname, NCCHAR, 4,
     .             'even', cdfstat )
        ELSE
* sub-span axis may not have irregular spacing just cuz parent does
* 12/00 NO. ... regular appearing coordinates are not necessarily regular --
*       for example 2 points extracted from an irregular axis will always
*	appear to be regular, but their box limits are not necessarily so
           IF ( ax_type .EQ. pcdf_full_span
     .     .OR. ax_type .EQ. pcdf_sub_span ) THEN
              attname = 'point_spacing' 
              IF (varid .GT. 0) THEN
                 CALL CD_GET_VAR_ATT_ID (dset, varid, attname, 
     .                  attid, status)
                 IF (attid .GT. 0) CALL CD_GET_VAR_ATT_INFO (dset, varid, 
     .               attid, attname, attype, attlen, attoutflag, istat )
              ELSE 
                 attoutflag = 1
              ENDIF
              IF (attoutflag .EQ. 1) 
     .          CALL NCAPTC( cdfid, axid, attname, NCCHAR, 6,
     .             'uneven', cdfstat )
           ENDIF
        ENDIF

* set up for deferred writing of coordinate values (12/00)
	CALL CD_DEFER_COORD_WRITE(axid, grid, idim, lo, hi,
     .				  pcd_defer_points)

      ENDIF

* Write AXIS attribute

      IF (.NOT. got_axis) THEN

           attname = 'axis' 
           IF (varid .GT. 0) THEN
              CALL CD_GET_VAR_ATT_ID (dset, varid, attname, attid,
     .                  status)
              IF (attid .GT. 0) CALL CD_GET_VAR_ATT_INFO (dset, varid,
     .             attid, attname, attype, attlen, attoutflag, istat )
           ELSE
              attoutflag = 1
           ENDIF
            
          slen = TM_LENSTR1( 'X' )
          IF (idim .eq. x_dim) THEN
             IF (attoutflag .EQ. 1) CALL NCAPTC( cdfid, axid, 
     .        	      attname, NCCHAR, slen, 'X', cdfstat )
          ENDIF
          IF (idim .eq. y_dim) THEN
             IF (attoutflag .EQ. 1) CALL NCAPTC( cdfid, axid, 
     .        	      attname, NCCHAR, slen, 'Y', cdfstat )
          ENDIF
          IF (idim .eq. z_dim) THEN
             IF (attoutflag .EQ. 1) CALL NCAPTC( cdfid, axid, 
     .        	      attname, NCCHAR, slen, 'Z', cdfstat )
          ENDIF
          IF (idim .eq. t_dim) THEN
             IF (attoutflag .EQ. 1) CALL NCAPTC( cdfid, axid, 
     .        	      attname, NCCHAR, slen, 'T', cdfstat )
          ENDIF
       ENDIF

* does it need axis bounds definitions ?

      IF (  (ax_type.EQ.pcdf_full_span .OR. ax_type.EQ.pcdf_sub_span)
     .  .AND. ( .NOT.line_regular(iaxis) .OR. do_bounds) 
     .  .AND. ( edges_flag .EQ. 0) ) THEN

         name = name(:nlen)//'_bnds'
         elen = nlen + 5

* ... define the first dimension of bounds variable (what about record axis??)

         dim2d(2) = dimid    ! Dimension already defined above for coordinates

         IF ( cdfstat .NE. NCNOERR ) GOTO 5200

* ... second dimension of bounds coordinate variable is 2
         dim2d(1) = CD_WRITE_BNDSDIM(cdfid, status)

* ... create the bounds variable, if it has not been created.
         IF (bndid .EQ. -1) 
     .    bndid = NCVDEF(cdfid, name(:elen), NCDOUBLE, 2, dim2d, cdfstat)
         IF ( cdfstat .NE. NCNOERR ) GOTO 5220

* ... point to it from axis coordinates definition

         attname = 'bounds' 
         IF (varid .GT. 0) THEN
            CALL CD_GET_VAR_ATT_ID (dset, varid, attname, attid, status)
            IF (attid .GT. 0) CALL CD_GET_VAR_ATT_INFO (dset, varid, 
     .              attid, attname, attype, attlen, attoutflag, istat)
         ELSE
            attoutflag = 1
         ENDIF

         IF (attoutflag .EQ. 1) THEN
            CALL NCAPTC( cdfid, axid, attname, NCCHAR, elen,
     .                name(:elen), cdfstat )

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
     .         .NOT.line_regular(iaxis) .AND. 
     .         .NOT.recdim)) ) THEN

         name = name(:nlen)//'edges'
         elen = nlen + 5
         dimid = NCDDEF( cdfid, name(:elen), axlen+1, cdfstat )
         edgid = NCVDEF(cdfid, name(:elen), NCDOUBLE, 1, dimid, cdfstat)
         attname = 'edges' 
         IF (varid .GT. 0) THEN
            CALL CD_GET_VAR_ATT_ID (dset, varid, attname, attid, status)
            IF (attid .GT. 0) CALL CD_GET_VAR_ATT_INFO (dset, varid, 
     .              attid, attname, attype, attlen, attoutflag, istat )
         ELSE
            attoutflag = 1
         ENDIF
         IF (attoutflag .EQ. 1) THEN
            CALL NCAPTC( cdfid, edgid, attname, NCCHAR, 1, ' ', cdfstat )
* ... point to it from axis coordinates definition
            CALL NCAPTC( cdfid, axid, attname, NCCHAR, elen,
     .                name(:elen), cdfstat )

* set up for deferref writing of edge coordinate values (12/00)
	    CALL CD_DEFER_COORD_WRITE(edgid, grid, idim, lo, hi,
     .				  pcd_defer_edges)
         ENDIF

      ENDIF

* successful completion
 1000 status = merr_ok
      RETURN

* error messages
 5100 CALL TM_ERRMSG
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

 5300 CALL TM_ERRMSG
     . ( merr_linepredef, status, 'CD_WRITE_AXIS', unspecified_int4,
     .   unspecified_int4,
     .  'Bad CDF file: dimension '//name(:nlen)
     .  //' exists without coordinates',
     .  no_errstring, *5900 )

* error exit
 5900   RETURN
	END
