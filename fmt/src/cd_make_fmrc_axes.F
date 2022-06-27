        SUBROUTINE CD_MAKE_FMRC_AXES(dset, varid, vname, vartype,
     .                tstart, tend, delta, fevery, tevery, nT, nF,
     .                axlagt, axcalt, axlagf, axcalf, iline, status)

*  This software was developed by the Thermal Modeling and Analysis
*  Project(TMAP) of the National Oceanographic and Atmospheric
*  Administration's (NOAA) Pacific Marine Environmental Lab(PMEL),
*  hereafter referred to as NOAA/PMEL/TMAP.
*.
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

* this routine synthesizes the T axis calendar time and F axis lead time
* axes that are needed to support FMRC visualization and analysis

* V695+ *acm* and *sh*
* V698 2/16 *sh* - if identical pre-existing axes exist, re-use them
* V698 3/16 *sh* - F-lag axis starts at 1/2 avg_lag (low cell bound at zero)
* V698  4/16 *acm* new TM_UNITS_CAL to account for calendar in units id
* V71 12/16 *acm* Ticket 2158: working with true monthly time axes. fmrc axis is
*                 not defined as true month.
* V72 6/17 *acm* For trac enhancement #767 -- dynamic coordinate storage
* v767 6/22 *acm* See issue 112: Detect whether dimension becomes too 
*                 large for 4-byte integer indexing

* include files
        include 'netcdf.inc'
        include 'tmap_errors.parm'
#include "gt_lib.parm"
        include 'tmap_dims.parm'
#include "tmap_dset.parm"
        include 'xdset_info.cmn_text'

        include 'xtm_grid.cmn_text'
        external xgt_grid_data
        include 'xbuild_grids.cmn'

        external xdset_info_data
        include 'xdset_info.cd_equiv'
        include 'xio.cmn_text'
        include 'xunits.cmn_text'
        external xunits_data
c        include 'calendar.decl'
c        include 'calendar.cmn'

* calling argument declarations
        REAL*8 tstart, tend, delta
        INTEGER dset, varid, vartype, fevery, tevery,
     .          nT, nF, axlagt, axcalt, axlagf, axcalf, iline, status
        CHARACTER*(*) vname

* internal variable declarations
        INTEGER bufflen
        PARAMETER (bufflen = 256)
        CHARACTER TM_STRING*13, buff*256, dbuf*3, vupcase*128

        LOGICAL TM_DFPEQ_TOL, TM_FPEQ_EPS, NC_GET_ATTRIB, ITSA_TRUEMONTH_AXIS,
     .          coordvar, got_it, do_warn, ok_cal, yyyymmdd_time, got_torg,
     .          true_month
        INTEGER STR_UPCASE, TM_LENSTR1, TM_GET_CALENDAR_ID, TM_UNIT_ID,
     .          TM_UNITS_CAL, TM_FIND_LIKE_DYN_LINE, STR_SAME,
     .          i, l, n,  nvdim, vdims(8), nvatts,
     .          outflag, next_line_storage, pre_exist,
     .          maxlen, attlen, attoutflag, cal_id, since_T0, slen, vlen,
     .          units, istat, nlag, nleadtimes, indx
        REAL    val, testindex


* initialize things
        do_warn = .FALSE.

*********  FIRST THE 1D CALENDAR TIME AXIS OF THE FMRC  *************
* Define new axis

        CALL TM_ALLO_TMP_LINE (axcalt, status)

* Get units, calendar, origin
* Get the calendar name.

      line_cal_name(axcalt) = 'GREGORIAN'   ! default
      maxlen = bufflen
      vlen = TM_LENSTR1(vname)
      got_it = NC_GET_ATTRIB( dset, varid, 'calendar', do_warn,
     .                        vname(:vlen), maxlen, attlen, attoutflag,
     .                        buff, val)
      IF (got_it) then
         CALL CD_GET_CALENDAR_NAME(buff, ok_cal)
         IF (ok_cal) THEN
            line_cal_name(axcalt) = buff
         ELSE
            slen = TM_LENSTR1(buff)
            CALL TM_NOTE('calendar attribute for 2D Time variable "'//
     .                   vname(:vlen)//'" is not recognized: '//
     .                   buff(1:slen), lunit_errors)
            CALL TM_NOTE('Assuming Gregorian calendar', lunit_errors)
         ENDIF
      ENDIF
      cal_id = TM_GET_CALENDAR_ID (line_cal_name(axcalt))

* optional units for axis (may contain "tunits since T0_date" 2/95)
      line_units(axcalt) = ' '   ! default
      maxlen = bufflen
      got_it = NC_GET_ATTRIB( dset, varid, 'units', do_warn,
     .                        vname(:vlen), maxlen, attlen, attoutflag,
     .                        buff, val)
      since_T0 = MAX( INDEX(buff,'since'), INDEX(buff,'SINCE') )
      yyyymmdd_time = buff(1:15) .EQ. 'yyyymmddhhmmss'

      IF ( since_T0 .GT. 2 ) THEN
         line_units(axcalt) = buff(:since_T0-1)  ! just the "tunits" part

         istat = STR_UPCASE( dbuf, buff(1:3) )

         slen = TM_LENSTR1(buff)
         IF ( slen .GT. since_T0 + 5) THEN

            got_torg = .TRUE.
            CALL TM_DATE_REFORMAT(buff(since_T0+6:),pdate_numslashnum,
     .                            cal_id, line_t0(axcalt),
     .                            pdate_vax, .FALSE., status )

c fix for bug 1394
c If the part after since is not a date/time, just keep the whole
c string as the units.  i.e., "days_since_event".

            IF ( status .NE. merr_ok ) THEN
               line_units(axcalt) = buff(1:slen)
               got_torg = .FALSE.
               since_t0 = 0
               status = merr_ok
            ENDIF
         ELSE
            since_t0 = 0        ! flag that no T0 was present
         ENDIF
      ELSEIF ( yyyymmdd_time ) THEN
         line_units(axcalt) = 'days'
         line_t0(axcalt) = ' ' ! to be determined when coords are read
         got_torg = .TRUE.
      ELSE
         IF (got_it) line_units(axcalt) = buff
         got_torg = .FALSE.
      ENDIF

* ... decode the units

      units = TM_UNIT_ID( line_units(axcalt) )

* Named calendar with unit=month or year gets length according to calendar's year length.

      true_month = ITSA_TRUEMONTH_AXIS(axcalt)
      IF (true_month) THEN
         true_month = .FALSE.
         units = pun_day
         line_units(axcalt) = un_name(units) ! make it days, NOT true-month
      ENDIF

      units = TM_UNITS_CAL( units, cal_id, true_month )

      line_unit_code(axcalt) = units
      IF  ( units.EQ.0 .AND. line_units(axcalt) .NE. ' ' ) THEN
         CALL TM_NOTE('Units on axis "'//vname(:vlen)//
     .      '" are not recognized: '//line_units(axcalt), lunit_errors)
         CALL TM_NOTE('They will not be convertible:', lunit_errors)
      ENDIF
      IF ( units .GE. 0 ) THEN
         line_tunit(axcalt) = real4_init
      ELSE
         line_tunit(axcalt) = un_convert(line_unit_code(axcalt))
      ENDIF

* The units must describe calendar time

      ! check that this has mapped to the right TF orientation
      IF (line_tunit(axcalt) .NE. real4_init) THEN
         line_direction(axcalt) = 'TI'
      ELSE
         GOTO 5400
      ENDIF

      line_regular(axcalt)   = .TRUE.
      line_start(axcalt)     = tstart
      line_delta(axcalt)     = delta

      testindex = (tend - tstart + delta)/delta
      IF (testindex .GT. arbitrary_large_int4) GOTO 5500
      line_dim(axcalt)       = NINT(testindex)
      line_subsc1 (axcalt)   = unspecified_int4

* Save the axis name (upper case)
      CALL TM_NEW_LINE_NAME('TF_CAL_T', buff)
      line_name(axcalt)      = buff
      line_name_orig(axcalt) = buff
      line_dattype(axcalt)   = nf_double

*********  THE SUPERLINE FORECAST INITIALIZATION TIME AXIS  *************
! per Andrew W., this axis has been shifted 1/2. cell to the left
! so that the F coordinate lies on the low limit (initialization time) of
! the 1st time coordinate from each forecast
      CALL TM_ALLO_TMP_LINE (axcalf, status)
      CALL TM_NEW_LINE_NAME('TF_CAL_F', buff)
      line_name(axcalf)      = buff
      line_name_orig(axcalf) = buff
      line_regular(axcalf)   = .TRUE.
      line_start(axcalf)     = tstart - 0.5 * delta
      line_delta(axcalf)     = delta
      
      testindex = nF * fevery
      IF (testindex .GT. arbitrary_large_int4) GOTO 5500
      line_dim(axcalf)       = testindex

* inherit time units from the calendar time axis just created
      line_cal_name(axcalf)  = line_cal_name(axcalt)
      line_unit_code(axcalf) = line_unit_code(axcalt)
      line_units(axcalf)     = line_units(axcalt)
      line_tunit(axcalf)     = line_tunit(axcalt)
      line_t0(axcalf)        = line_t0(axcalt)
      line_direction(axcalf) = 'FI'
      line_dattype(axcalf)   = nf_double

*********  THE ACTUAL FORECAST INITIALIZATION TIME AXIS  *************
! per Andrew W., this axis has been shifted 1/2. cell to the left
! so that the F coordinate lies on the low limit (initialization time) of
! the 1st time coordinate from each forecast
      line_regular(iline)    = .TRUE.
      line_start(iline)      = tstart - 0.5 * delta
      line_delta(iline)      = delta * fevery
      testindex              = FLOAT(nF)
      IF (testindex .GT. arbitrary_large_int4) GOTO 5500
      line_dim(iline)        = nF

* inherit time units from the calendar time axis just created
      line_cal_name(iline)   = line_cal_name(axcalt)
      line_unit_code(iline)  = line_unit_code(axcalt)
      line_units(iline)      = line_units(axcalt)
      line_tunit(iline)      = line_tunit(axcalt)
      line_t0(iline)         = line_t0(axcalt)
      line_direction(iline)  = 'FI'
      line_dattype(iline)    = nf_double

*********  THE FORECAST LEAD TIME AXIS OF THE FMRC  *************
      nleadtimes = nT * tevery
      CALL TM_NEW_LINE_NAME('TF_LAG_F', buff)
      CALL TM_MAKE_BASIC_AXIS(buff, delta/2., delta,
     .                        nleadtimes, axlagf, status)

* inherit time units from the calendar time axis just created
      line_cal_name(axlagf)  = line_cal_name(axcalt)
      line_unit_code(axlagf) = line_unit_code(axcalt)
      line_units(axlagf)     = line_units(axcalt)
      line_tunit(axlagf)     = line_tunit(axcalt)
      line_t0(axlagf)        = line_t0(axcalt)
      line_direction(axlagf) = 'FF'
      line_dattype(axlagf)   = nf_double

*********  THE SUPERLINE TIME LAG AXIS OF THE FMRC  *************
      nleadtimes = nT * tevery
      CALL TM_NEW_LINE_NAME('TF_LAG_T', buff)
      CALL TM_MAKE_BASIC_AXIS(buff, delta/2., delta,
     .                        nleadtimes, axlagt, status)

* inherit time units from the calendar time axis just created
      line_cal_name(axlagt)  = line_cal_name(axcalt)
      line_unit_code(axlagt) = line_unit_code(axcalt)
      line_units(axlagt)     = line_units(axcalt)
      line_tunit(axlagt)     = line_tunit(axcalt)
      line_t0(axlagt)        = line_t0(axcalt)
      line_direction(axlagt) = 'TT'
      line_dattype(axlagt)   = nf_double

********************
* If axes are an exact duplicate of a pre-existing axis, then use that instead.
* (Shared axes are needed in order to E-aggregate multiple FMRCs)
      pre_exist = TM_FIND_LIKE_DYN_LINE( axlagt )
      IF (pre_exist .NE. unspecified_int4) THEN
         CALL TM_DEALLO_DYN_LINE(axlagt)
         axlagt = pre_exist
      ENDIF
      pre_exist = TM_FIND_LIKE_DYN_LINE( axcalf )
      IF (pre_exist .NE. unspecified_int4) THEN
         CALL TM_DEALLO_DYN_LINE(axcalf)
         axcalf = pre_exist
      ENDIF
      pre_exist = TM_FIND_LIKE_DYN_LINE( axcalt )
      IF (pre_exist .NE. unspecified_int4) THEN
         CALL TM_DEALLO_DYN_LINE(axcalt)
         axcalt = pre_exist
      ENDIF
      pre_exist = TM_FIND_LIKE_DYN_LINE( axlagf )
      IF (pre_exist .NE. unspecified_int4) THEN
         CALL TM_DEALLO_DYN_LINE(axlagf)
         axlagf = pre_exist
      ENDIF

* successful completion
        status = merr_ok

 5000   RETURN

 5400   CALL TM_ERRMSG(merr_notimeinfo, status, 'CD_MAKE_FMRC_AXES',
     .     no_descfile, no_stepfile, 'unrecognizable time encoding',
     .     no_errstring, *5000)

 5500   CALL TM_ERRMSG(merr_dimtoolarge, status, 'CD_MAKE_FMRC_AXES',
     .     no_descfile, no_stepfile, 'Long dimensions not implemented',
     .     no_errstring, *5000)
        RETURN
        END
