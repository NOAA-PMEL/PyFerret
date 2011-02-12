        SUBROUTINE CD_GET_1_AXIS(dset, cdfid, ivar, iaxis, its_epic,
     .                           reversed, tregular, status)


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
*
* Load the common block XGRID with a single axis definition from a
* netCDF file

* Programmer Steve Hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program

* revision 0.0 - 2/92
*   8/19/92: search for "LAT" in axis units was setup wrong
*  10/20/92: a more systematic approach to identifying the axis orientation
*  11/30/92:  changed Z axis orientation "UP" to "DU"
* 12/92: use TM_FPEQ_EPS to test regularity of axes
*  1/93: special error message for npts=0 (for UNLIMITED time axis in CDL)
*  2/93: allow zero length axis and out of order axes by substituting dummies
*  3/93: added EPIC support
*  4/93: bug fix: set default units if time_origin is read
* 10/93: EPIC longitudes now use "degree_east".  Allow west-encoded longitudes
*        for any data set using units like "degree west" or "positive=west".
*  8/94: for out of order coordinates 1) issue a max # of messages and 2)
*        allow strictly DECREASING coordinates by multiplying by -1
* 12/94: millibar and decibar (and all pressure) units should be UD 
*        allow "height" as a DU axis name
*  2/95: interpret time units like "hours since 1982-12-01 12:52:00"
*  2/95: note reverse-ordered coordinate axes in array "reversed"
*        and reverse the coordinates as saved
*        Also allow CDC-style yyyymmdd time format
* 10/95: if a longitude axis has 360 degrees then default to MODULO
* 11/95: recognize units "degrees_north" as a latitude
*  4/96: bug: millibar and decibar units identified by wrong constants
*  8/97: bug: X axis with "y" in axis name is mis-identified 
* 5/98 *kob* : mod so that CDC datafiles w/ two day offsets are compatible
*              w/ ferret.  
* V500 3/99 *sh*: instead of erroring out, simply return iaxis=0 for an
*       invalid coordinate axis -- to be handled by calling program
* V500 4/99 *kob*:  increase vname and vupcase from 24 to 64 chars 
* V510 *sh* 3/00 - added tregular argument
*                - convert to dynamic grids and axes
* V530 *sh* 1/01 - automatically flag as modulo any time axis that ends before
*                  year 0002
* V530 *acm* 12/00 attribute for different calendar types
*      *acm*  7/12/2001  correct for conflicts between prev two versions,
*                        somehow checked into CVS w/o proper merging
* V533 *acm*  8/30/01  change calendar name 360 to 360_day
*      *acm* 10/01  remove VMS includes
* V550 *sh*  2/02 - added support for line_modulo_len
*      *sh* 11/02 - automatically detect subspan modulo longitude and time axes
* V550 *acm*12/02 - AXIS and CARTESIAN_AXIS attributes, and modifications to
*                   order of precedence for axis orientation.
* V550 *acm* 1/03 - Account for differences in behavior of CD_GET_ATTVALC
*                   under OSF and PC.
* V552 *acm* 1/03 - error checking on value of calendar attribute.
* V552 *acm* 5/03 - increase vname and vupcase to 128 chars 
* V552 *acm* 6/03 - If AXIS attribute is set, put XX or YY in line_direction
*                   at first, then determine if the axis is longitude/latitude
* V552 *acm* 6/03 - calculate secsperyear using double precision arithmetic.
* V553 *acm* 7/03 - New code to read files with time axis encodings of:
*                      units: "days since -4713-01-01 00:00:00"
*                   The code detects the -4713 and replaces the time origin
*                   with 1-jan-1901.  Then subtract the correct number of days
*                   from the time coordinates: 2415386.  This is a file made
*                   by Matlab, which uses this "Julian" astromical date
*                   convention (different from our Julian calendar).  The 
*                   simple conversion is that May 23,1968 equals
*                   2440000 days.
* V554  *acm 12/03 Restore the last-ditch determination of line_direction
*                        using the starting letter of the axis.
* V570 *acm* 4/04  Add the calendar "all_leap", 366 days every year.
*                  also the name "365_DAY" as an equivalent to "NOLEAP" 
* V570 *acm* 5/04  Allow "bounds" attribute for defining an irregular axis.  Store
*                  all irregular axis intervals as Nx2 bounds rather than Nx1 edges.
*      *acm* 6/04  for compiler on rh7, need to pass dimensions of array into cd_rd_r8_bnds 
*                  as scalar integers not array elements.
*            6/18/04 Change order of operations so axes with edges are read the same as 
*                  in previous versions.  Also treat 1-point axes having bounds correctly
* V580 *acm* 10/04 Add a comment only, describing the hack to deal with CDC files starting 
*                  in 1-1-1 00:00:00
* V580 *acm* 11/04 Fix bug 1042 If bounds attribute has a problem found by TM_CHECK_BNDS_ATTRIB,
*                  just skip applying the bounds. The edges have been defined by putting them
*                  midway between coordinates; use this definition. If there is only one point 
*                  on the axis, then set it to be a regular axis.  
* V580 *acm* 11/04 further fixes for when there is just one point on an axis having bounds
*                  first_delta is meaningless and should not be used.
* V581 *acm*  1/05 Fixes for irreg axes with bounds: tm_check_bnds_centered now checks
*                  whether the coordinates are centered in the bound boxes, and whether
*                  the boxes are equal sized. Also fix def of d2e for reading axis bounds.
* V581 *acm*  2/05 Fix bug 1179: axis_dir .EQ. 'z' needs to be done as a case-insensitve
*                  comparison, in section testing positive=down.
* V581 *acm*  5/05 Fix bug 1231: check 'TT and 'TI' when applying tregular
* V581 *acm*  5/05 Fix bug 1232: if tregular, then dont try to use bounds
* V581*acm*  6/05  Fix for bug 1271, add flag line_shift_origin marking hack to shift axes 
*                   starting on 1-1-0001 (CDC files)
* V600 *acm* 10/05 TM_CHECK_BNDS_ATTRIB and TM_CHEC_EDGES_ATTRIB should not have 
*                  npts as an argument; npts from axis length was being reset
*                  as npts bounds or edges, which is npts+1
* V600 *acm* 11/05 Fix bug 1363: As is already done for years, named calendar with unit=month 
*                  gets assigned the length of month in that calendar.
* V600 *acm*  2/06 Fix bug 1394: if the time units contain a "since" but the remainder of
*                  the string is not a date, keep the whole units string.
* V600 *acm* 3/06  Fix bug 1400: Change in the call to TM_CHECK_BNDS
* V600 *acm* 8/05  attribute control. 
*                  Replace NCVID with CD_GET_VAR_ID
*                  Replace NCVINQ for variables, with CD_GET_VAR_INFO
*                  Change call to CD_GET_ATTRIB to NC_GET_ATTRIB - get attrib from 
*                  linked list structure
*                  CD_GET_ATTVAL replaced by NC_GET_ATTRIB, which now returns  
*                  strings or real values according to attrib type
* V600 *acm* 6/06  use TM_FPEQ_EPS to compare coordinates.
* V601 *acm* 9/19  Fix bug 1443; MAXLEN of bounds name should be NF_MAX_NAME from netcdf.inc
* V601 *acm* 9/19  Fix bug 1434; check bounds for regularly-spaced axes 
* V602 *acm* 1/07  Fix bug 1483; checking for irregularly spaced axis failed if the delta-coordinates
*                  on the axis varied by orders of magnitude.
* V602  2/07 *acm* Check upper and lowercase versions of axis (coordinate_axis) attribute
* V604  6/07 *acm* Further fixes for finely-spaced axes (see bug 1483); compute only one epsilion23
*                  for checking regularity of axes. For checking axis length of 360, use TM_FPEQ
*                  to allow for numerical accuracy of bounds for axes of exactly length360.
* V61   1/08 *acm* Fix bug 1559: Sanity-checking the modulo length can run into precision
*                  problems, in particular if the axis coordinates in the file are single-precision.
* V612  8/08 *acm* If the file has a point-spacing=uneven attribute, mark the axis as irreg
*                  and do not do all the testing for regular axis.
* V62  10/08 *acm* Add PROLEPTIC_GREGORIAN to the calendars recognized; same as GREGORIAN
* V62   2/09 *acm* Save the original upper/lowercase spelling of axis names in line_name_orig,
*                  for use on output of user-defined variables when MODE UPCASE_OUTPUT is cancelled.
* V63   7/09 *acm* Fix bug 1676 Reversed axis with EDGES 
* V65  *acm* 1/10  Make sure parameters values are consistent with 
*                              whats in netcdf.inc from netcdf4.
*      *kms* 5/10  Fix scaling/offset bug: using iaxis instead of i inside loops 334 and 336
*
* V67 *acm* 1/11   WE, SN, not XX, YY in line_direction

* argument definitions:
*       cdfid    - netCDF id number for already opened CDF file  (to go away with new attr. hanling)
*       dset     - dataset number for already opened CDF file
*       ivar     - netCDF variable id number
*       iaxis    - returned pointer to the axis created
*       its_epic - flags an EPIC netCDF data set
*       reversed    - logical array to record which axes have reversed coords
*       status   - return status

* include files
        include 'netcdf.decl'
#include "netcdf.inc"

        include 'tmap_errors.parm'
        include 'tmap_dims.parm'
#include "tmap_dset.parm"
#include "gt_lib.parm"
        include 'xbuild_grids.cmn'
        include 'xdset_info.cmn_text'
        external xdset_info_data
        include 'xio.cmn_text'
        external xio_data
        include 'xtm_grid.cmn_text'
        external xgt_grid_data
        include 'xunits.cmn_text'
        external xunits_data
        include 'calendar.decl'
        include 'calendar.cmn'
        include 'errmsg.parm'

* argument declarations
        LOGICAL  its_epic, reversed(max_lines:line_ceiling), tregular
        INTEGER  cdfid, dset, ivar, iaxis, status

* local parameter definitions:
      INTEGER     str_eq
      PARAMETER ( str_eq = 0 )
      REAL epsilon_23
      REAL*8 depsilon_46, depsilon, dlast_coord, dfirs_coord,
     .       dfirst_delta, ddelta

* local variable declarations:
      LOGICAL NC_GET_ATTRIB, CD_GET_ATTVAL, TM_CHECK_BOX_LOC,
     .        TM_HAS_STRING, TM_FPEQ_EPS, TM_DFPEQ, CD_GET_ATTVALC, 
     .        CD_GET_ATTVAL_L, TM_CHECK_BNDS, TM_CHECK_BNDS_CENTERED, 
     .        TM_FPEQ, TM_DFPEQ_TOL, NC_GET_ATTRIB_DP,
     .        got_it, got_torg, do_warn,
     .        epic_time, west_lon, yyyymmdd_time, 
     .        got_mod, mod_tf, setmodfalse, ok_cal, ok,
     .        got_edges, got_bnds, coordvar, its_irregular, 
     .        do_scale, do_offset, scale_is_dp

      INTEGER TM_UNIT_ID, TM_LENSTR1,TM_GET_CALENDAR_ID,      
     .        STR_CASE_BLIND_COMPARE, STR_UPCASE, 
     .        soft_err, 
     .        ipt1, ipte, vartype, vback, npts, vlen, cdfstat, istat,
     .        nvdims, vdims(8), nvatts, i, units, npts_cdf, elen, 
     .        edgid, spacing_atlen, epic_t2var, since_T0,
     .        cal_id, year, month, day, hour, minute, second, slen, nc,
     .        blen, bndid, startpts(2), numpts(2), next_line_storage,
     .        d1s, d1e, d2s, d2e, len_cal, attoutflag, maxlen,
     .        all_outflag, dir, attlen, iedge, irev, attype, iatt, ibx1

      REAL    rbuff, delta, first_delta, val, epsilon, lm_len, 
     .        scale, offset
      REAL*8  TM_LONW2LONE, TM_WW_AXLEN, tmp8, axwwlen , rdelta, 
     .        dscale, doffset
      CHARACTER*12 TM_LEFINT
      CHARACTER vname*128, ename*128, vupcase*128, words(8)*24,
     .        spacing_attr*14, bname*128
      CHARACTER*13 TM_STRING
      CHARACTER*1 axis_dir
      CHARACTER*50 allcals
      CHARACTER*3 dbuf
      CHARACTER*2 dcode

      INTEGER tt  ! nice short name
      EQUIVALENCE (tt,c)

      INTEGER bufflen
      PARAMETER (bufflen = 256)
      CHARACTER*256 buff
      REAL*8   TM_SECS_FROM_BC, first, secs2firststep, secs1590,
     .         t0_secs, last_coord, firs_coord, secsperyear, secs2start,
     .         days_neg4713, days_1901_may68, secs_to_1901, secs_to_1968 

* parameter declarations
        REAL*8          psecs_per_day
        PARAMETER     ( psecs_per_day = 60.*60.*24. )

* initialize for EPIC
      do_warn = .NOT.its_epic     ! EPIC files can get away with anything
      west_lon = .FALSE.          ! assume longitudes are positive east
      iaxis = 0

* gen; initialization
 
      got_torg = .FALSE.
      days_neg4713 = 0.D0
      next_line_storage = next_line_mem_pos

* (From Wikipedia on precision:)
* Single precision, called "float" in the C language family, and "real" 
* or "real*4" in Fortran. This is a binary format that occupies 32 bits (4 bytes) 
* and its significand has a precision of 24 bits (about 7 decimal digits).
* Double precision, called "double" in the C language family, and "double precision" 
* or "real*8" in Fortran. This is a binary format that occupies 64 bits (8 bytes) 
* and its significand has a precision of 53 bits (about 16 decimal digits).

      epsilon_23 =  2.**(-23) 
      depsilon_46 =  2.**(-52)
      depsilon_46 =  2.**(-46) 

* get the vital statistics of the axis 

      CALL CD_GET_VAR_INFO (dset, ivar, vname, vartype, 
     .            nvdims, vdims, nvatts, coordvar, all_outflag, status)

      IF (status .NE. merr_ok) GOTO 5100
      vlen = TM_LENSTR1(vname)
      istat = STR_UPCASE( vupcase, vname )
!      IF ( nvdims .NE. 1 ) CALL TM_ERRMSG
!     .     ( cdfstat+pcdferrmax, status, 'CD_GET_1_AXIS', cdfid, ivar,
!     .     vname(:vlen), 'not a 1D variable', *5900 )

* get axis length:

      CALL CD_GET_DS_DIMS (dset, vdims(1), buff, npts_cdf, status)
      IF (status .NE. ferr_ok) GOTO 5100

! removed 3/93          IF ( npts_cdf .EQ. 0 ) GOTO 5150
      npts = npts_cdf      ! differ only if "true_size" is used

* grab a temporary dynamic axis slot for it
      CALL TM_ALLO_TMP_LINE(iaxis, status)
      IF (status .NE. merr_ok) GOTO 5900

* Are there scale_factor and/or add_offset attributes?  If so need to unpack
* the coordinate values

      scale = 1.
      dscale = 1.D0
      offset = 0.
      doffset = 0.D0
      maxlen = bufflen
      do_scale = NC_GET_ATTRIB(dset, ivar, 'scale_factor',
     .                       do_warn, vname(:vlen), maxlen, attlen, 
     .                       attoutflag, buff, scale)
     
      scale_is_dp = .FALSE.

      IF ( do_scale ) THEN
         CALL CD_GET_VAR_ATT_ID (dset, ivar, 'scale_factor', iatt, 
     .              status)
         IF (iatt .GT. 0) CALL CD_GET_VAR_ATT_INFO (dset, ivar, 
     .              iatt, buff, attype, attlen, attoutflag, istat )

         IF (attype .EQ. nf_double) THEN
             got_it = NC_GET_ATTRIB_DP(dset, ivar, 'scale_factor',
     .                       do_warn, vname(:vlen), attlen, 
     .                       attoutflag, dscale)
             scale_is_dp = .TRUE.
	 ENDIF
      ENDIF

      maxlen = bufflen
      do_offset = NC_GET_ATTRIB(dset, ivar, 'add_offset',
     .                       do_warn, vname(:vlen), maxlen, attlen, 
     .                       attoutflag, buff, offset)

      IF ( do_offset ) THEN
         CALL CD_GET_VAR_ATT_ID (dset, ivar, 'add_offset', iatt, 
     .              status)
         IF (iatt .GT. 0) CALL CD_GET_VAR_ATT_INFO (dset, ivar, 
     .              iatt, buff, attype, attlen, attoutflag, istat )

         IF (attype .EQ. nf_double) THEN
            got_it = NC_GET_ATTRIB_DP(dset, ivar, 'add_offset',
     .                       do_warn, vname(:vlen), attlen, 
     .                       attoutflag, doffset)
            IF (do_scale .AND. (.NOT. scale_is_dp))  GOTO 5400
     	 ELSE
	    IF (do_scale .AND. scale_is_dp)  GOTO 5400
     	 ENDIF

      ENDIF
      do_scale = do_scale .OR. do_offset
 
* optional point spacing parameter: even, uneven, disordered
      IF ( its_epic ) THEN
         spacing_attr = 'type'          ! type = "EVEN" or ="E"
         spacing_atlen = 4
      ELSE
         spacing_attr = 'point_spacing' ! point_spacing = "even"
         spacing_atlen = 13
      ENDIF

      maxlen = bufflen
      got_it = NC_GET_ATTRIB(dset, ivar, spacing_attr(1:spacing_atlen),
     .                       do_warn, vname(:vlen), maxlen, attlen, 
     .                       attoutflag, buff, val)

      line_regular(iaxis) = .FALSE.
      its_irregular = .FALSE.
      IF ( got_it ) THEN
         line_regular(iaxis) = buff(1:1).EQ.'e' .OR. buff(1:1).EQ.'E'
         its_irregular = .NOT.line_regular(iaxis)
      ELSE
         line_regular(iaxis) = .FALSE. ! default
      ENDIF

* Bounds or edges attribute?

      maxlen = 128
      got_bnds = NC_GET_ATTRIB( dset, ivar, 'bounds',
     .                       do_warn, vname(:vlen), maxlen, 
     .                       attlen, attoutflag, bname, val)
     
      got_edges = NC_GET_ATTRIB( dset, ivar, 'edges',
     .                       do_warn, vname(:vlen), maxlen,
     .                       attlen, attoutflag, ename, val)

* optional "true_size" parameter for 2 point representations of even axes
      IF ( line_regular(iaxis) ) THEN

         got_it = NC_GET_ATTRIB( dset, ivar, 'true_size', 
     .                          do_warn, vname(:vlen), maxlen, 
     .                          attlen, attoutflag, buff, val)
         IF ( got_it ) THEN
            IF ( npts_cdf .NE. 2 ) THEN
               soft_err = 1
               GOTO 5000
            ENDIF
            npts = val
         ENDIF
      ENDIF
      line_dim( iaxis ) = npts

      IF (npts .LT. 2) line_regular(iaxis) = .TRUE.

* *ACM get the calendar name.

      line_cal_name(iaxis) = 'GREGORIAN'   ! default
      maxlen = bufflen
      got_it = NC_GET_ATTRIB( dset, ivar, 'calendar', do_warn,
     .                        vname(:vlen), maxlen, attlen, attoutflag, 
     .                        buff, val)
      IF (got_it) then
         istat = STR_UPCASE (buff, buff)

         IF (buff(1:3) .EQ. '360') buff = '360_DAY' 
         IF (buff .EQ. 'STANDARD') buff = 'GREGORIAN' 
         IF (buff .EQ. 'PROLEPTIC_GREGORIAN') buff = 'GREGORIAN' 
         IF (buff(1:6) .EQ. 'COMMON') buff = 'NOLEAP' 
         IF (buff(1:3) .EQ. '365') buff = 'NOLEAP' 
         IF (buff(1:3) .EQ. '366') buff = 'ALL_LEAP' 

         ok_cal = .FALSE.
         nc = 1
         allcals = ' '

         DO 100 i = 1, mxcals
  111       FORMAT (a9,' ')
            slen = TM_LENSTR1( allowed_calendars(i)(1:9) )
            write (allcals(nc:nc+9), 111) allowed_calendars(i)
            nc = nc + 10

            IF (buff(1:3) .EQ. allowed_calendars(i)(1:3) ) THEN
               ok_cal = .TRUE.
               line_cal_name(iaxis) = allowed_calendars(i)
            ENDIF
  100    CONTINUE
         IF (.NOT. ok_cal) THEN
            slen = TM_LENSTR1(buff)
            CALL TM_NOTE('calendar attribute on axis "'//vupcase(:vlen)//
     .                '" is not recognized: '//buff(1:slen), tt)
            CALL TM_NOTE('Valid calendars are '//allcals, tt)
            CALL TM_NOTE(
     .             'A dummy axis of subscripts will be used', tt)
            GOTO 1000
         ENDIF
      ENDIF
      cal_id = TM_GET_CALENDAR_ID (line_cal_name(iaxis))

* optional units for axis (may contain "tunits since T0_date" 2/95)
      line_units(iaxis) = ' '   ! default
      maxlen = bufflen
      got_it = NC_GET_ATTRIB( dset, ivar, 'units', do_warn,
     .                        vname(:vlen), maxlen, attlen, attoutflag, 
     .                        buff, val)
      since_T0 = MAX( INDEX(buff,'since'), INDEX(buff,'SINCE') )
      yyyymmdd_time = buff(1:15) .EQ. 'yyyymmddhhmmss'

      IF ( since_T0 .GT. 2 ) THEN
         line_units(iaxis) = buff(:since_T0-1)  ! just the "tunits" part 

         istat = STR_UPCASE( dbuf, buff(1:3) )

         slen = TM_LENSTR1(buff)
         IF ( slen .GT. since_T0 + 5) THEN
            IF (buff(since_T0+6:since_T0+10) .EQ. '-4713' .AND.
     .          dbuf .EQ. 'DAY')  THEN

c  This is the simple conversion; but we will reset the origin to 1901.
c               buff = 'DAYS since 1968-05-23 00:00:00'
c               days_neg4713 = 2440000.D0

               buff = 'DAYS since 1901-01-15 00:00:00'
               secs_to_1901 = TM_SECS_FROM_BC 
     .                        (GREGORIAN, 1901,1,15,0,0,0,status)
               secs_to_1968 = TM_SECS_FROM_BC 
     .                        (GREGORIAN, 1968,5,23,0,0,0,status)
               days_1901_may68 = (secs_to_1968 - secs_to_1901)/
     .                             psecs_per_day
               days_neg4713 = 2440000.D0 - days_1901_may68

            ENDIF
            got_torg = .TRUE.
            CALL TM_DATE_REFORMAT(buff(since_T0+6:),pdate_numslashnum,
     .                            cal_id, line_t0(iaxis), 
     .                            pdate_vax, .FALSE., status )

c fix for bug 1394
c If the part after since is not a date/time, just keep the whole
c string as the units.  i.e., "days_since_event".

c           IF ( status .NE. merr_ok ) GOTO 1000        ! 3/99 soft error 

            IF ( status .NE. merr_ok ) THEN   
               line_units(iaxis) = buff(1:slen)
               got_torg = .FALSE.
               since_t0 = 0
               status = merr_ok
            ENDIF
         ELSE
            since_t0 = 0        ! flag that no T0 was present
         ENDIF
      ELSEIF ( yyyymmdd_time ) THEN
         line_units(iaxis) = 'days'
         line_t0(iaxis) = ' ' ! to be determined when coords are read
         got_torg = .TRUE.
      ELSE
         IF (got_it) line_units(iaxis) = buff
         got_torg = .FALSE.
      ENDIF

* EPIC time axes are encoded as 2 integers in variables TIME and TIME2
      IF ( its_epic .AND.  vupcase .EQ. 'TIME' ) THEN
         CALL CD_GET_VAR_ID ( dset, 'time2', epic_t2var, status)
         epic_time  = epic_t2var .NE. atom_not_found
      ELSE
         epic_time  = .FALSE.
      ENDIF
      IF ( epic_time ) line_units(iaxis) = 'Days'     ! arbitrary

* ... decode the units

      units = TM_UNIT_ID( line_units(iaxis) )

* Named calendar with unit=year gets length of year in that calendar.

      IF (units .EQ. -6 .OR. units .EQ. -10) THEN
         IF (cal_id .EQ. d360) units = -9
         IF (cal_id .EQ. julian) units = -12
	 IF (cal_id .EQ. noleap) units = -11
	 IF (cal_id .EQ. all_leap) units = -8
      ENDIF

* Named calendar with unit=month gets length of month in that calendar.

      IF (units .EQ. -5) THEN
         IF (cal_id .EQ. d360) units = -14
         IF (cal_id .EQ. julian) units = -15
	 IF (cal_id .EQ. noleap) units = -13
      ENDIF

      line_unit_code(iaxis) = units
      IF  ( units.EQ.0 .AND. line_units(iaxis) .NE. ' ' ) THEN
         CALL TM_NOTE('Units on axis "'//vname(:vlen)//
     .                '" are not recognized: '//line_units(iaxis), tt)
         CALL TM_NOTE('They will not be convertible:', tt)
      ENDIF
      IF ( units .GE. 0 ) THEN
         line_tunit(iaxis) = real4_init
      ELSE
         line_tunit(iaxis) = un_convert(line_unit_code(iaxis))
      ENDIF

* optional "modulo" flag (and optionally the associated length)
      
      rbuff = 0.

*  **ACM** Note that under OSF (and PC? ) CD_GET_ATTVALC returns TRUE 
*      when the value of the attribute is, say, "F".  Under Solaris and 
*      Linux, returns FALSE  Reads a value from a character buffer if 
*      the value of the modulo attribute is type character. For solaris, 
*      linux, if the value of the attribue is non-numeric string, returns 
*      false, and rbuff=0. For OSF, returns true and rbuff = 0 or -1.

      line_modulo(iaxis) = CD_GET_ATTVALC( dset, ivar, 'modulo',
     .                             do_warn, vname(:vlen), rbuff, 0.0 )
      line_modulo_len(iaxis) = rbuff
      IF (line_modulo_len(iaxis) .LT. 0.) line_modulo_len(iaxis) = 0.

* See if there is a logical modulo attrib, with yes, true, false, etc.

      got_mod = .FALSE.
      mod_tf = .TRUE.

      IF (.NOT. line_modulo(iaxis) .OR. 
     .     line_modulo_len(iaxis) .LE. 0.) THEN
         got_mod = CD_GET_ATTVAL_L( dset, ivar, 'modulo',
     .                             do_warn, vname(:vlen),mod_tf)
         IF (got_mod) line_modulo(iaxis) = mod_tf

      ENDIF
      setmodfalse = (.NOT.mod_tf  .AND. got_mod)

* optional time_origin and time_unit values for time axis
* (time origin could have been set by the units "... since time_origin")
      maxlen = 20
      IF (.NOT.got_torg) got_torg = NC_GET_ATTRIB( dset, ivar,
     .          'time_origin', do_warn, vname(:vlen), maxlen, attlen, 
     .          attoutflag, line_t0(iaxis), val)
      IF ( .NOT.got_torg ) THEN
         IF ( epic_time ) THEN
            line_t0(iaxis) = '15-JAN-1901' ! arbitrary
         ELSEIF (since_t0 .LT. 2) THEN
            line_t0(iaxis) = char_init20
         ENDIF
      ENDIF

* a user can impose non-standard time units by setting the time_unit attribute
* (not fully tested, I suspect - *sh* 4/93)
      maxlen = bufflen
      got_it = NC_GET_ATTRIB( dset,ivar,'time_unit',
     .           do_warn, vname(:vlen), maxlen, attlen, attoutflag, 
     .           buff, line_tunit(iaxis))

*!      got_it = NC_GET_ATTRIB( dset,ivar,'month_lengths',do_warn,

* if a time_origin was given then some units are required (4/93)
* default to "seconds" if no units were specified
      IF ( got_torg .AND. line_tunit(iaxis).EQ.real4_init ) THEN
           line_units(iaxis) = 'seconds'
           line_unit_code(iaxis) = TM_UNIT_ID( line_units(iaxis) )
           line_tunit(iaxis) = un_convert(line_unit_code(iaxis))
      ENDIF

* try to determine the orientation and direction of the axis

      line_direction(iaxis) = 'NA'

* ... First check for an AXIS attribute

      axis_dir = char_init01
      maxlen = 1
      got_it = NC_GET_ATTRIB(dset, ivar, 'axis',
     .             do_warn, vname(:vlen), maxlen, attlen, 
     .             attoutflag, axis_dir, val)
      IF (got_it) THEN
         CALL CD_GET_VAR_ATT_ID (dset, ivar, 'axis', iatt, status)
         CALL CD_GET_VAR_ATT_INFO (dset, ivar, iatt, buff, attype, 
     .     attlen, attoutflag, status )
         IF (attype .NE. NCCHAR) got_it = .FALSE.
      ENDIF
      IF (got_it) THEN
         istat = STR_UPCASE( axis_dir, axis_dir )
         IF (axis_dir.EQ.'T' .OR. axis_dir.EQ.'L' .OR.
     .             axis_dir.EQ.'t' .or. axis_dir.eq.'l') THEN 
            line_direction(iaxis) = 'TI'  ! time
          ELSE IF (axis_dir.EQ.'X' .OR. axis_dir.EQ.'I' .OR.
     .             axis_dir.eq.'x' .or. axis_dir.eq.'i') THEN
            line_direction(iaxis) = 'XX'
         ELSE IF (axis_dir.EQ.'Y' .OR. axis_dir.EQ.'J' .OR.
     .            axis_dir.eq.'y' .or. axis_dir.eq.'j') THEN
            line_direction(iaxis) = 'YY'
         ELSE IF (axis_dir.EQ.'Z' .OR. axis_dir.EQ.'K' .OR.
     .            axis_dir.eq.'z' .or. axis_dir.eq.'k') THEN
            line_direction(iaxis) = 'DU'
         ENDIF
      ELSE
         axis_dir = char_init01
      ENDIF


* ... check for a CARTESIAN_AXIS attribute, if still undetermined

      IF (line_direction(iaxis) .EQ. 'NA') THEN

         maxlen = 1
         got_it = NC_GET_ATTRIB(dset, ivar, 'cartesian_axis',
     .              do_warn, vname(:vlen), maxlen, attlen, 
     .              attoutflag, axis_dir, val)
         IF (got_it) THEN
            CALL CD_GET_VAR_ATT_ID (dset, ivar, 'cartesian_axis', iatt, status)
            CALL CD_GET_VAR_ATT_INFO (dset, ivar, iatt, buff, attype, 
     .     attlen, attoutflag, status )
            IF (attype .NE. NCCHAR) got_it = .FALSE.
         ENDIF

         IF (got_it) THEN
            IF (axis_dir.EQ.'T' .OR. axis_dir.EQ.'L' .OR.
     .          axis_dir.EQ.'t' .or. axis_dir.eq.'l') THEN 
               line_direction(iaxis) = 'TI'  ! time
            ELSE IF (axis_dir.EQ.'X' .OR. axis_dir.EQ.'I' .OR.
     .               axis_dir.eq.'x' .or. axis_dir.eq.'i') THEN
            line_direction(iaxis) = 'WE'
            ELSE IF (axis_dir.EQ.'Y' .OR. axis_dir.EQ.'J' .OR.
     .               axis_dir.eq.'y' .or. axis_dir.eq.'j') THEN
             line_direction(iaxis) = 'SN'
            ELSE IF (axis_dir.EQ.'Z' .OR. axis_dir.EQ.'K' .OR.
     .               axis_dir.eq.'z' .or. axis_dir.eq.'k') THEN
               line_direction(iaxis) = 'DU'
            ENDIF
         ELSE
            axis_dir = char_init01
         ENDIF
      ENDIF


* ... next check units

      IF (line_direction(iaxis) .EQ. 'NA' .OR. 
     .    line_direction(iaxis) .EQ. 'XX' .OR.
     .    line_direction(iaxis) .EQ. 'YY') THEN

      IF ( units .LT. 0 ) THEN
         line_direction(iaxis) = 'TI'  ! time
      ELSEIF ( units .EQ. 4 ) THEN     ! degrees
        IF  ( TM_HAS_STRING(line_units(iaxis), 'NORTH')
     .   .OR. TM_HAS_STRING(line_units(iaxis), 'SOUTH')
     .   .OR. TM_HAS_STRING(line_units(iaxis), 'degreeN')   ! Check all CF_compliant 
     .   .OR. TM_HAS_STRING(line_units(iaxis), 'degreesN')  ! options
     .   .OR. TM_HAS_STRING(line_units(iaxis), 'degree_N')  ! Case independent
     .   .OR. TM_HAS_STRING(line_units(iaxis), 'degrees_N')
c     .   .OR. TM_HAS_STRING(line_units(iaxis), 'degreesS') ! But this matches all degrees
     .   .OR. TM_HAS_STRING(line_units(iaxis), 'degree_S')
     .   .OR. TM_HAS_STRING(line_units(iaxis), 'degrees_S')
     .   .OR. TM_HAS_STRING(line_units(iaxis), 'LAT') ) THEN 
            line_direction(iaxis) = 'SN'
        ELSEIF ( TM_HAS_STRING(line_units(iaxis), 'EAST')
     .   .OR. TM_HAS_STRING(line_units(iaxis), 'WEST')
     .   .OR. TM_HAS_STRING(line_units(iaxis), 'degreeE')
     .   .OR. TM_HAS_STRING(line_units(iaxis), 'degreesE')
     .   .OR. TM_HAS_STRING(line_units(iaxis), 'degree_E')
     .   .OR. TM_HAS_STRING(line_units(iaxis), 'degrees_E')
     .   .OR. TM_HAS_STRING(line_units(iaxis), 'degreeW')
     .   .OR. TM_HAS_STRING(line_units(iaxis), 'degreesW')
     .   .OR. TM_HAS_STRING(line_units(iaxis), 'degree_W')
     .   .OR. TM_HAS_STRING(line_units(iaxis), 'degrees_W')
     .   .OR. TM_HAS_STRING(line_units(iaxis), 'LON') ) THEN 
            line_direction(iaxis) = 'WE'
        ELSEIF ( line_direction(iaxis) .EQ. 'XX') THEN
                        line_direction(iaxis) = 'WE'
        ELSEIF ( line_direction(iaxis) .EQ. 'YY') THEN
                        line_direction(iaxis) = 'SN'
        ENDIF

      ELSEIF ( units .EQ. pun_millibars
     .   .OR.  units .EQ. pun_decibars ) THEN 
         line_direction(iaxis) = 'UD'
      ELSEIF ( units .EQ. 14
     .   .OR.  units .EQ. 15   ) THEN         ! layer or level
         line_direction(iaxis) = 'UD'
* ... then check stray clues
         ELSEIF ( TM_HAS_STRING(line_units(iaxis), 'LAT') ) THEN
         line_direction(iaxis) = 'SN'
      ELSEIF ( TM_HAS_STRING(line_units(iaxis), 'LON') ) THEN
         line_direction(iaxis) = 'WE'
      ENDIF
      ENDIF

      IF (units .EQ. 4) THEN    ! degrees
         IF (line_direction(iaxis) .EQ. 'NA' ) THEN
            IF ( vupcase(1:1) .EQ. 'Y'
     .      .OR. INDEX(vupcase, 'LAT') .GT. 0 ) THEN
               line_direction(iaxis) = 'SN'
            ELSEIF ( vupcase(1:1) .EQ. 'X'
     .      .OR. INDEX(vupcase, 'LON') .GT. 0 ) THEN
               line_direction(iaxis) = 'WE'
! 8/97: "Y" anywhere in name as a last ditch attempt to id it
            ELSEIF ( INDEX(vupcase, 'Y')   .GT. 0 ) THEN
               line_direction(iaxis) = 'SN'
            ELSE
               line_direction(iaxis) = 'WE'
            ENDIF
         ENDIF
      ENDIF

! See if the long-name contains a string that indicates a z axis
      maxlen = bufflen
      got_it = NC_GET_ATTRIB( dset,ivar,'long_name',
     .           do_warn, vname(:vlen), maxlen, attlen, attoutflag, 
     .           buff, val)
      IF ( got_it .AND. line_direction(iaxis) .EQ. 'NA' ) THEN
         IF (STR_CASE_BLIND_COMPARE('elev',buff(:4)) .EQ. str_eq) 
     .           line_direction(iaxis) = 'DU'
         IF (STR_CASE_BLIND_COMPARE('height',buff(:6)) .EQ. str_eq) 
     .           line_direction(iaxis) = 'DU'
         IF (STR_CASE_BLIND_COMPARE('level',buff(:5)) .EQ. str_eq) 
     .           line_direction(iaxis) = 'DU'
         IF (STR_CASE_BLIND_COMPARE('layer',buff(:5)) .EQ. str_eq) 
     .           line_direction(iaxis) = 'DU'
         IF (STR_CASE_BLIND_COMPARE('depth',buff(:5)) .EQ. str_eq) 
     .           line_direction(iaxis) = 'UD'
      ENDIF

* the attribute "positive=down", etc. may help
      maxlen = bufflen
      got_it = NC_GET_ATTRIB( dset, ivar, 'positive', do_warn, 
     .                        vname(:vlen), maxlen, attlen, attoutflag, 
     .                        buff, val)
      IF ( got_it ) THEN
         IF (STR_CASE_BLIND_COMPARE('west',buff(:4))
     .                                     .EQ.str_eq) THEN
            line_direction(iaxis) = 'WE'
            west_lon = .TRUE.                        ! convert encodings below
         ELSEIF( STR_CASE_BLIND_COMPARE('east',buff(:4))
     .                                     .EQ.str_eq) THEN
            line_direction(iaxis) = 'WE'
         ELSEIF( STR_CASE_BLIND_COMPARE('up',buff(:2))
     .                                     .EQ.str_eq) THEN
            line_direction(iaxis) = 'DU'
         ELSEIF( STR_CASE_BLIND_COMPARE('down',buff(:4))
     .                                     .EQ.str_eq) THEN
            line_direction(iaxis) = 'UD'
         ENDIF
      ENDIF

* standard_names indicating a sigma-axis in Z, and standard names 
* for geophysical axes, longitude, latitude, altitude etc. Note 
* that there seem to be some standard names related to sigma
* coordinates that may not indicate a vertical axis, so simply 
* matching sigma and coordinate is probably not safe.

      maxlen = bufflen
      got_it = NC_GET_ATTRIB( dset, ivar, 'standard_name', do_warn, 
     .                        vname(:vlen), maxlen, attlen, attoutflag, 
     .                        buff, val)
      IF ( got_it ) THEN
         IF (STR_CASE_BLIND_COMPARE(
     .      'atmosphere_hybrid_height_coordinate',buff(:35))
     .                                     .EQ.str_eq) THEN
            line_direction(iaxis) = 'DU'
         ENDIF 
         IF (STR_CASE_BLIND_COMPARE(
     .      'atmosphere_hybrid_sigma_pressure_coordinate',buff(:43))
     .                                     .EQ.str_eq) THEN
            line_direction(iaxis) = 'DU'
         ENDIF
         IF (STR_CASE_BLIND_COMPARE(
     .      'atmosphere_hybrid_height_coordinate',buff(:35))
     .                                     .EQ.str_eq) THEN
            line_direction(iaxis) = 'DU'
         ENDIF
         IF (STR_CASE_BLIND_COMPARE(
     .      'atmosphere_ln_pressure_coordinate',buff(:33))
     .                                     .EQ.str_eq) THEN
            line_direction(iaxis) = 'DU'
         ENDIF
         IF (STR_CASE_BLIND_COMPARE(
     .      'atmosphere_sigma_coordinate',buff(:27))
     .                                     .EQ.str_eq) THEN
            line_direction(iaxis) = 'DU'
         ENDIF
         IF (STR_CASE_BLIND_COMPARE(
     .      'atmosphere_sleve_coordinate',buff(:27))
     .                                     .EQ.str_eq) THEN
            line_direction(iaxis) = 'DU'
         ENDIF
         IF (STR_CASE_BLIND_COMPARE(
     .      'land_ice_sigma_coordinate',buff(:25))
     .                                     .EQ.str_eq) THEN
            line_direction(iaxis) = 'DU'
         ENDIF
         IF (STR_CASE_BLIND_COMPARE(
     .      'ocean_s_coordinate',buff(:19))
     .                                     .EQ.str_eq) THEN
            line_direction(iaxis) = 'UD'
         ENDIF
         IF (STR_CASE_BLIND_COMPARE(
     .      'ocean_sigma_coordinate',buff(:22))
     .                                     .EQ.str_eq) THEN
            line_direction(iaxis) = 'UD'
         ENDIF
         IF (STR_CASE_BLIND_COMPARE(
     .      'ocean_sigma_z_coordinate',buff(:24))
     .                                     .EQ.str_eq) THEN
            line_direction(iaxis) = 'UD'
         ENDIF
         IF (STR_CASE_BLIND_COMPARE(
     .      'ocean_double_sigma_coordinate',buff(:29))
     .                                     .EQ.str_eq) THEN
            line_direction(iaxis) = 'UD'
         ENDIF

      ENDIF  ! end of std names


* longitude axes may be encoded "west" (discontinuous at 180, East negative,
*       West positive) or "east" (standard - 0->360 east from Greenwhich)
      west_lon = west_lon 
     .  .OR.    ( line_direction(iaxis) .EQ. 'WE'
     .      .AND. TM_HAS_STRING(line_units(iaxis),'west') )
     .  .OR.    ( its_epic
     .      .AND. (INDEX(vupcase, 'LON') .GT. 0)
     .      .AND. .NOT.TM_HAS_STRING(line_units(iaxis),'east') )

* was a regular time axis forced on us by the user?
      IF (tregular) THEN
        IF (line_direction(iaxis).EQ."TI" .OR.
     .      line_direction(iaxis).EQ."TT" ) line_regular(iaxis) = .TRUE.
        got_bnds = .FALSE.
      ENDIF


* get the axis coordinates - none, regular, or irregular
      IF ( npts .EQ. 0 ) THEN
* create a dummy one point axis
         CALL TM_NOTE('Axis "'//vname(:vlen)//
     .                '" has no coordinates', tt)
         CALL TM_NOTE('A dummy value of 1 will be used', tt)
         line_start  (iaxis) = 1.0D0
         line_delta  (iaxis) = 1.0D0
         line_subsc1 (iaxis) = unspecified_int4
         line_regular(iaxis) = .TRUE.
         line_dim    (iaxis) = 11
         last_coord          = 1.0D0
         firs_coord          = 1.0D0

* start/n/delta representation for "regular axes"
* (no check is made on validity of the regularity claim)
* read the start and end - compute delta     
      ELSEIF ( line_regular(iaxis) ) THEN
         IF ( epic_time ) THEN
            CALL CD_RD_EP_1( cdfid, ivar,  epic_t2var, line_t0(iaxis),
     .                       line_tunit(iaxis), 1, 
     .                       line_start(iaxis), status )
            IF ( status .NE. merr_ok ) GOTO 1000        ! 3/99 soft error
            CALL CD_RD_EP_1( cdfid, ivar,  epic_t2var, line_t0(iaxis),
     .                       line_tunit(iaxis), npts_cdf, 
     .                       line_delta(iaxis), status )
         ELSEIF ( yyyymmdd_time ) THEN
            CALL CD_RD_YMD_1( cdfid, ivar, line_t0(iaxis),
     .                       line_tunit(iaxis), 1, 
     .                       line_start(iaxis), status )
            IF ( status .NE. merr_ok ) GOTO 1000        ! 3/99 soft error
            CALL CD_RD_YMD_1( cdfid, ivar,  line_t0(iaxis),
     .                       line_tunit(iaxis), npts_cdf, 
     .                       line_delta(iaxis), status )
         ELSE
            CALL CD_RD_R8_1( cdfid, ivar,   1, vartype, vname(:vlen),
     .                       line_start(iaxis), status )
            IF ( status .NE. merr_ok ) GOTO 1000        ! 3/99 soft error
            CALL CD_RD_R8_1( cdfid,ivar,npts_cdf,vartype,vname(:vlen),
     .                       line_delta(iaxis), status )
         ENDIF
         IF ( status .NE. merr_ok ) GOTO 1000   ! 3/99 soft error
	 IF (do_scale) THEN
	    IF (scale_is_dp) THEN
	       line_start(iaxis) = line_start(iaxis)*dscale + doffset
	       line_delta(iaxis) = line_delta(iaxis)*dscale + doffset
	    ELSE
	       line_start(iaxis) = line_start(iaxis)*scale + offset
	       line_delta(iaxis) = line_delta(iaxis)*scale + offset
	    ENDIF
	 ENDIF

         IF ( west_lon ) THEN
            line_start(iaxis) = TM_LONW2LONE( line_start(iaxis) ) 
            line_delta(iaxis) = TM_LONW2LONE( line_delta(iaxis) ) 
         ENDIF
         IF ( npts .GT. 1 ) THEN
* ... allow for reverse ordering of axis coordinates on regular-spaced axes
*     (the variable line_delta actually contains line_end at this moment)
            reversed(iaxis) = line_delta(iaxis) .LT. line_start(iaxis)
            IF ( reversed(iaxis) ) THEN
               tmp8 = line_delta(iaxis)
               line_delta(iaxis) =
     .           (line_start(iaxis)-line_delta(iaxis))/DBLE(npts-1)
               line_start(iaxis) = tmp8
            ELSE
               line_delta(iaxis) =
     .           (line_delta(iaxis)-line_start(iaxis))/DBLE(npts-1)
            ENDIF
         ELSE
            reversed(iaxis) = .FALSE.
            line_delta(iaxis) = 1
         ENDIF
         line_subsc1(iaxis) = unspecified_int4
         last_coord = line_start(iaxis) + DBLE(npts-1)*line_delta(iaxis)
         firs_coord = line_start(iaxis)


* ============= START OF IRREGULAR AXIS PROCESSING ========================
      ELSE 
* adequate storage for the irregular coordinates and box edges ?
* 2*npts + 1 for edges plus coordinate values.

         line_start(iaxis) = unspecified_val8
         line_delta(iaxis) = unspecified_val8
         IF ((next_line_mem_pos+2*npts+1).GT.maxlinestore) GOTO 5300

* read the CDF coordinate values into double precision axis storage
         ipt1 = next_line_mem_pos
         ipte = ipt1 + npts - 1
         IF ( epic_time ) THEN
            CALL CD_RD_EP_ARR( cdfid, ivar, epic_t2var,
     .                         line_t0(iaxis), line_tunit(iaxis),
     .                         npts_cdf, line_mem(ipt1), status )
         ELSEIF ( yyyymmdd_time ) THEN
            CALL CD_RD_YMD_ARR( cdfid, ivar,
     .                         line_t0(iaxis), line_tunit(iaxis),
     .                         npts_cdf, line_mem(ipt1), status )
         ELSE
            CALL CD_RD_R8_ARR( cdfid, ivar, 1, npts_cdf, vartype,
     .                         vname(:vlen), line_mem(ipt1), status )
            IF (days_neg4713 .NE. 0.) THEN
               DO 330 i = ipt1,ipte
                  line_mem(i) = line_mem(i) - days_neg4713
 330           CONTINUE
            ENDIF
         ENDIF
         IF ( status .NE. merr_ok ) GOTO 1000   ! 3/99 soft error
	 
	 IF (do_scale) THEN
	    IF (scale_is_dp) THEN
	       DO 334 i=ipt1,ipte
 334            line_mem(i) = line_mem(i)*dscale + doffset
	    ELSE
	       DO 336 i=ipt1,ipte
 336            line_mem(i) = line_mem(i)*scale + offset
	    ENDIF
	 ENDIF

         IF ( west_lon ) THEN
            DO 340 i=ipt1,ipte
 340        line_mem(i) = TM_LONW2LONE( line_mem(i) ) 
         ENDIF

* if the coordinates are decreasing-ordered then reverse them
         IF (npts .GT. 1) THEN
            first_delta = line_mem(ipt1+1) - line_mem(ipt1)
            reversed(iaxis) = first_delta .LT. 0.0
         ENDIF
         IF ( reversed(iaxis) ) THEN
            DO 345 i=ipt1,ipt1+npts/2-1
               tmp8 = line_mem(i)
               line_mem(i) = line_mem(ipte+ipt1-i)
               line_mem(ipte+ipt1-i) = tmp8
 345        CONTINUE
         ENDIF
            
* Check for non-monotonic ordering of data
         DO 350 i=ipt1+1,ipte
           IF (line_mem(i) .LE. line_mem(i-1) ) THEN
* out of order axis coordinates
              words(1) = TM_LEFINT(i-ipt1+1, istat)
              CALL TM_NOTE(
     .             'Coordinates out of order or missing on axis '//
     .           vname(:vlen)//' at subscript '//words(1), tt)
              CALL TM_NOTE(
     .             'A dummy axis of subscripts will be used', tt)
              GOTO 1000
           ENDIF
  350   CONTINUE

* Check if axis is actually regularly spaced (though not flagged as "even")
* Fix bug 1483; look at each delta rather than trying to compare with first delta.

        IF (its_irregular) GOTO 380  ! If the file says its irreg, treat is at such

* See if double-precision coordinates are regularly-spaced
        IF (vartype .EQ. ncdouble) THEN
           dlast_coord = line_mem(ipte)
           dfirs_coord = line_mem(ipt1)
           dfirst_delta = line_mem(ipt1+1) - line_mem(ipt1)
           depsilon = depsilon_46 * 2.*(ABS(line_mem(ipte)) / first_delta )
           depsilon = depsilon_46 * 2.*(ABS(line_mem(ipt1)) / first_delta )

           DO 355 i=ipt1+2,ipte
              ddelta = line_mem(i) - line_mem(i-1)
              IF (.NOT. TM_DFPEQ_TOL(dfirst_delta, ddelta, depsilon) ) GOTO 380
  355      CONTINUE

* See if single-precision (or integer??) coordinates are regularly-spaced
        ELSE
           last_coord = line_mem(ipte)
           firs_coord = line_mem(ipt1)
           first_delta = line_mem(ipt1+1) - line_mem(ipt1)
           epsilon = epsilon_23 * 2.*(ABS(line_mem(ipte)) / first_delta )
           epsilon = epsilon_23 * 2.*(ABS(line_mem(ipt1)) / first_delta )

           DO 360 i=ipt1+2,ipte
              delta = line_mem(i) - line_mem(i-1)
              IF (.NOT. TM_FPEQ_EPS(epsilon, first_delta, delta) ) GOTO 380
  360      CONTINUE
        ENDIF

* yes, it's actually regular - save it as such
        line_regular(iaxis) = .TRUE.
        line_start  (iaxis) = line_mem(ipt1)
        line_delta  (iaxis) = 
     .              (line_mem(ipte)-line_mem(ipt1))/DBLE(npts-1)
        line_subsc1 (iaxis) = unspecified_int4
        last_coord = line_start(iaxis) + DBLE(npts-1)*line_delta(iaxis)
        firs_coord = line_start(iaxis)

        GOTO 390

* Compute the default locations of boundaries between grid boxes (midpoints)
* Note: this may be modified by a subsequent "EDGES" or "BOUNDS" definition

 380    CALL TM_IRREGAX_MID( ipt1, npts )
      ENDIF

      IF (got_bnds .AND. npts .EQ. 1) THEN
         line_regular(iaxis) = .FALSE.
         ipt1 = next_line_mem_pos
         ipte = next_line_mem_pos
         line_mem(ipt1) = line_start(iaxis)
      ENDIF

* ============= END OF IRREGULAR AXIS PROCESSING ========================

* hack to cope with Gregorian/Julian calendar ambiguities (used by NOAA/CDC)
* Comments from Roland Schweitzer about CDC files:
* These CDCfiles have units of the form: days since 1-1-1 00:00:00 or
* hours since 1-1-1 00:00:00.  Any of these files that encode dates
* after 1590 will look to Ferret as if they are offset by 2 days.  The 
* CDC axes shift around the time axis at various historical breaks as the
* UDUNITS package does. Ferret uses a proleptic Gregorian calendar - same
* for all time without these shifts. (Ferret uses udunits for units 
* definitions only.) Also note there are CDC files which decode to 
* fall entirely within year 0001.  These files represent a climatology,
* and Ferret deals with them correctly.

 390    IF ( (since_T0.GT.2)
     .  .AND.(line_t0(iaxis).EQ.'01-JAN-0001 00:00:00') 
     .  .AND.(cal_id .EQ. GREGORIAN) ) THEN

* ... get time step of L=1
           IF ( line_regular(iaxis) ) THEN
              first = line_start(iaxis)   
           ELSE
              first = line_mem(ipt1)
           ENDIF

* ... compute seconds from 1-jan-0001
           secs2firststep = un_convert(line_unit_code(iaxis)) * first
* compute seconds from 1590
           secs1590 = TM_SECS_FROM_BC (GREGORIAN, 1590,1,1,0,0,0,status)
            IF ( status .NE. merr_ok ) GOTO 5900

* ... correct axis if crosses 1590 threshhold
           IF (secs2firststep .GT. secs1590) THEN
              line_t0(iaxis) = '30-DEC-0000 00:00:00'
              line_shift_origin(iaxis) = .TRUE.
           ENDIF
        ENDIF

*  See if there are box edges or bounds in the file.

****************************************************************************
*                              BOX EDGES                                   *
****************************************************************************

* the attribute "edges=name" may point to a list of box boundaries
      IF ( .NOT.got_edges ) GOTO 500
      IF ( line_regular(iaxis) ) THEN
         CALL TM_NOTE(
     .           'Evenly spaced axis has edges definition: '//
     .           vname(:vlen)//' - ignored', tt )
         GOTO 500
      ENDIF

* Cannot have both "edges" and "bounds" atrtributes
      IF ( got_bnds )THEN
         CALL TM_NOTE(
     .     'Axis has both edges and bounds attributes: '//
     .      vname(:vlen)//' - edges definition ignored', tt )
         GOTO 500
      ENDIF

      CALL TM_CHECK_EDGES_ATTRIB (cdfid, iaxis, vname, vlen, ename, 
     .        edgid, status)

      IF (status .EQ. pcdferr) GO TO 5100
      IF (status .NE. merr_ok) THEN
         CALL TM_IRREGAX_MID( next_line_mem_pos,
     .                          line_dim   (iaxis) )
         GO TO 600
      ENDIF

* read the (npts+1) box edge values into double precision storage

        ipt1 = next_line_mem_pos
        line_subsc1(iaxis) = next_line_mem_pos
        CALL CD_RD_R8_ARR( cdfid, edgid, 1, npts+1, vartype,
     .                     vname(:vlen), line_mem(ipt1+npts), status )
        IF ( status .NE. merr_ok ) GOTO 1000    ! 3/99 soft error

* if the coordinates were reversed then the box bounds must be, too
         IF ( reversed(iaxis) .AND.  npts .GT. 1) THEN
	    iedge = (ipt1+npts)+npts/2-1
	    iedge = ipt1 + 2*npts 
	    irev = iedge
            DO 450 i=ipt1+npts,(ipt1+npts)+npts/2-1
               tmp8 = line_mem(i)
	       line_mem(i) = line_mem(irev)
	       line_mem(irev) = tmp8
	       irev = irev - 1
 450        CONTINUE

         ENDIF

* check to see that each data point is contained inside its box
        IF (.NOT.TM_CHECK_BOX_LOC(iaxis)) THEN
           CALL TM_NOTE('Error in Edges "'//ename(:elen)//
     .         '"or edges do not enclose point on axis '//
     .         vname(:vlen), tt)
           CALL TM_NOTE('Substituting coordinate midpoints', tt)
           CALL TM_IRREGAX_MID( next_line_mem_pos,
     .                          line_dim   (iaxis) )
        ENDIF

****************************************************************************
*                              BOX BOUNDS                                  *
****************************************************************************

* the attribute "bounds=name" may point to a list of box boundaries

500     maxlen = 128
        IF ( .NOT.got_bnds ) GOTO 600

* Check the bounds variable, issue any warnings.
        CALL TM_CHECK_BNDS_ATTRIB (dset, iaxis, vname, vlen, bname,
     .                             bndid, status)
        IF (status .EQ. pcdferr) GO TO 5100
        IF (status .NE. merr_ok) THEN
           IF (npts .EQ. 1) THEN  ! save as a regular axis, no bounds.
              line_regular(iaxis) = .TRUE.
              line_start  (iaxis) = line_mem(ipt1)
              line_delta(iaxis) = 1
              line_subsc1 (iaxis) = unspecified_int4
              last_coord = line_start(iaxis) 
              firs_coord = line_start(iaxis)
           ENDIF
              ! If npts not 1 then we have already set the boundaries
              ! at statement 380  CALL TM_IRREGAX_MID( ipt1, npts )
           GO TO 600
        ENDIF

* read the (npts*2) box edge values into double precision storage 

        line_subsc1(iaxis) = next_line_mem_pos
        ipt1 = next_line_mem_pos
        startpts(1) = 1
        startpts(2) = 1
        numpts(1) = 2
        numpts(2) = npts

        d1s = startpts(1)
        d1e = numpts(1)
        d2s = startpts(2)
        d2e = numpts(2)

        blen = TM_LENSTR1(bname)
        CALL CD_RD_R8_BNDS( cdfid, bndid, startpts, numpts, vartype, 
     .                    bname(:blen), line_mem(ipt1+npts),  
     .                    d1s, d1e, d2s, d2e, status )
        IF ( status .NE. merr_ok ) GOTO 1000    ! 3/99 soft error

* if the coordinates were reversed then the box bounds must be, too 
        IF ( first_delta .LT. 0.0  .AND.  npts .GT. 1) THEN
           DO 550 i=ipt1+npts,(ipt1+npts)+2*npts
 550       line_mem(i) = -1. * line_mem(i)
        ENDIF

* check to see that each data point is contained inside its box,
* and if the bounds are contiguous, warning user if not. Change
* the storage from N*2 bounds to N+1 edges.    

        IF (.NOT. TM_CHECK_BNDS(iaxis, 2*npts, vname) ) THEN  

           CALL TM_NOTE('Error in bounds "'//bname(:blen)//
     .         '" or bounds do not enclose point on axis '//
     .         vname(:vlen), tt)
           CALL TM_NOTE('Substituting coordinate midpoints', tt)
           CALL TM_IRREGAX_MID( next_line_mem_pos,
     .                          line_dim   (iaxis) )
      
        ENDIF

* For axes specified with BOUNDS, check whether coordinates are also 
* centered in the boxes, and whether the size of the boxes are equal.
* If there is one point, use upper and lower bound as given in the file 
* to determine line_delta.

        IF (TM_CHECK_BNDS_CENTERED(iaxis)) THEN
* It's actually regular - save it as such
           IF (.NOT. line_regular(iaxis)) THEN
              line_regular(iaxis) = .TRUE.
              line_start  (iaxis) = line_mem(ipt1)
              IF (npts .GT. 1) THEN
                 line_delta  (iaxis) = 
     .               (line_mem(ipte)-line_mem(ipt1))/DBLE(npts-1)
              ELSE
	         ibx1 = ipt1 + npts
                 line_delta(iaxis) = line_mem(ibx1+1)-line_mem(ibx1)
              ENDIF
              line_subsc1 (iaxis) = unspecified_int4
              last_coord = line_start(iaxis) + 
     .               DBLE(npts-1)*line_delta(iaxis)
              firs_coord = line_start(iaxis)
           ENDIF
        ELSE
           line_regular(iaxis) = .FALSE.
        ENDIF

 600    CONTINUE

* Save the axis name (upper case) and memory storage.  For all irregular axes, 
* we have saved N+1 edges for the axis cells. Increment the pointer.   

        line_name(iaxis) = vupcase
        line_name_orig(iaxis) = vname
        IF ( .NOT.line_regular(iaxis) ) THEN
           line_subsc1(iaxis) = next_line_mem_pos
           next_line_mem_pos =  next_line_storage + 2*npts+1 
        ENDIF

* if it is a longitude axis of 360 degrees or less let it default to MODULO
* unless user has set it explicitly to false.
        
 400    axwwlen = 0.
        axwwlen = TM_WW_AXLEN(iaxis)

        IF ( line_direction(iaxis).EQ.'WE'
     .       .AND. units.EQ.4
     .       .AND. (.NOT.setmodfalse) ) THEN

           delta = axwwlen ! single precision
           IF (TM_FPEQ(delta, 360.0) .OR. delta .LT. 360.0) THEN
              line_modulo(iaxis) = .TRUE.
              IF (line_modulo_len(iaxis).EQ.0.0D0)
     .                    line_modulo_len(iaxis) = 360.D0
           ENDIF
        ENDIF

* if it is a calendar time axis and the first point is in year 0000 or 0001
* then it is implicitly MODULO

        IF (got_torg) THEN
           line_direction(iaxis) = 'TI'
c           secsperyear = cals_yeardays(cal_id)* 24*60*60
           tmp8 = cals_yeardays(cal_id)
           secsperyear = tmp8 * (24.D0*60.D0*60.D0)

           CALL TM_BREAK_DATE(line_t0(iaxis), cal_id, year, month, day,
     .                       hour, minute, second, istat)
           t0_secs = TM_SECS_FROM_BC (cal_id, year, month, day,
     .                               hour, minute, second, status)
           CALL TM_WW_AX_1_N(iaxis, first, tmp8)
 	   secs2start = t0_secs + first*line_tunit(iaxis)
           IF ( (secs2start .LE.  2*secsperyear)
     .	    .AND. (axwwlen*line_tunit(iaxis) .LE. secsperyear) ) THEN
              line_modulo(iaxis) = .TRUE.
              IF (line_modulo_len(iaxis).EQ.0.0D0)
     .          line_modulo_len(iaxis) = secsperyear/line_tunit(iaxis)
           ENDIF
        ENDIF

* sanity check the modulo length
* a whisker too small is assumed a precision error
* a whisker larger is assumed to be deliberate
* ACM 1/08 
* If a whisker too large, this may be a precision error 
* In particular this happens converting single float coordinates to
* double coordinate storage. If coords are single precision do the 
* check in single precision.

        IF (line_modulo(iaxis).AND.line_modulo_len(iaxis).NE.0.D0) THEN
           IF (line_modulo_len(iaxis) .LT. axwwlen) THEN
c             line_modulo_len(iaxis) = 0.0D0
             IF (vartype .EQ. ncdouble) THEN
                ok = TM_DFPEQ(axwwlen, line_modulo_len(iaxis)) 
             ELSE
	        lm_len = line_modulo_len(iaxis)
                delta = axwwlen ! single precision
                ok = TM_FPEQ(delta, lm_len)
             ENDIF
             IF (.NOT.  ok)
     .          CALL TM_NOTE(
     .          'Ignored modulo length exceeding axis length: '
     .          //vname(:vlen), tt )
           ENDIF
        ENDIF

* Save the line direction in the linked-list structure for the coordinate variable

         dir = 0
         dcode = line_direction(iaxis)
         IF (dcode .EQ. 'XX' .OR. dcode .EQ. 'WE') dir = 1
         IF (dcode .EQ. 'YY' .OR. dcode .EQ. 'SN') dir = 2
         IF (dcode .EQ. 'ZZ' .OR. dcode .EQ. 'UD' .OR. dcode .EQ. 'DU') dir = 3
         IF (dcode .EQ. 'TT' .OR. dcode .EQ. 'TI') dir = 4
         CALL CD_SET_ATT_AXDIR (dset, ivar, dir, status)

* successful completion
        status = merr_ok
        RETURN

* 3/99-style "soft" error - invalid axis but CDF file need not be rejected
 1000   CALL TM_DEALLO_DYN_LINE(iaxis)
        iaxis = 0
        status = merr_ok
        RETURN

* error messages
 5000   CALL TM_NOTE( 'netCDF parent axis definition error', tt )
        IF ( soft_err .EQ. 1 ) THEN
           CALL TM_NOTE(
     .    '"true_size" attribute must have only max/min axis coords: '
     .       //vname(:vlen), tt )
        ELSEIF ( soft_err .EQ. 11 ) THEN
           CALL TM_NOTE('Edges definition "'//ename(:elen)//
     .                   '" points to no existing axis', tt )
        ELSEIF ( soft_err .EQ. 12 ) THEN
           CALL TM_NOTE('Edges definition "'//ename(:elen)//
     .                   '" is not 1D', tt )
        ELSEIF ( soft_err .EQ. 13 ) THEN
           CALL TM_NOTE('Edges "'//ename(:elen)//
     .          '" must be 1 pt longer than '//vname(:vlen), tt )
        ENDIF
        IF ( soft_err .LE. 10 ) THEN
           CALL TM_NOTE( 'Axis definition ignored', tt )
        ELSE
           CALL TM_NOTE( 'Edge definitions ignored', tt )
        ENDIF
        GOTO 400

 5100   CALL TM_ERRMSG
     .     ( cdfstat+pcdferr, status, 'CD_GET_1_AXIS', cdfid, ivar,
     .       no_errstring, no_errstring, *5900 )

 5300   CALL TM_ERRMSG (merr_linstorlim, status, 'CD_GET_1_AXIS',
     .                  no_descfile, no_stepfile,
     .                  'MAX='//TM_STRING(DBLE(maxlinestore)),
     .                  no_errstring, *1000)          ! 3/99 soft error


 5400   CALL TM_ERRMSG
     .     ( cdfstat+pcdferr, status, 'CD_GET_1_AXIS', cdfid, ivar,
     .       'Scale and offset attributes not of the same type', 
     .       no_errstring, *5900 )

* error exit
 5900   RETURN
        END
