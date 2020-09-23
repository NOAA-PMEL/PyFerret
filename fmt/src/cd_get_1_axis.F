        SUBROUTINE CD_GET_1_AXIS(dset, cdfid, ivar, iaxis, its_epic,
     .                           reversed, tregular, use_strict, 
     .                           bad_bndsid, has_gdef, status)


*  This software was developed by the Thermal Modeling and Analysis
*  Project(TMAP) of the National Oceanographic and Atmospheric
*  Administration's (NOAA) Pacific Marine Environmental Lab(PMEL),
*  hereafter referred to as NOAA/PMEL/TMAP.
*.
*  Access and use of this software shall impose the following
*  obligations and understandings on the user. The user is granted the
*  right, without any fee or cost, to use, copy, modify, alter, enhance
*  and distribute this software, and any derivative works thereof, and
*  its supporting documentation for any puFrpose whatsoever, provided
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
* V68 *acm* 1/12   Implement micro-adjusting of repeated coordinates (ticket 1910).
*                  To keep the previous behavior the user may specify USE/STRICT.
* V6.74 3/12 *acm* cleanup ifdefs and unnecessary include files
* V6.83 12/12 *acm* ticket 1996: NOTE if reversing an axis
* V685 *acm* 1/13  Fix ticket 2026: dont automatically treat a 1-point longitude
*                  axis as modulo.
* V685 *acm* 8/13  Fix ticket 2088: Report the correct index location in axis when
*                  unrepairable repeated axis coords on axis
* V685 *acm* 8/13  Fix ticket 2090: If there is enough coord storage to read just the coords
*                  but not bounds, still read and check coordiantes for regular spacing.
* V685 *acm* 9/13  Move the NOTE about reversing a decreasing-ordered axis, so that
*                  it shows up for regularly spaced ones.
* V685 *acm* 11/13 REALLY make the check for axis length in single precision when the
*                  coordinate variable came in as a single-precision variable.
* V6.87 4/14 *acm* ticket 2146: if invalid bounds, make the bounds a dependent variale.
* V691  4/15 *acm* Ticket 2170: bub with invalid bounds when point-spacing=true attribute.
* Ticket 2172: If the axis is actually too long, reset the modulo flags
* so it is treated as not modulo.
* V694  5/15 *acm* Match precision for bounds checks (as is done for coords).
* V695+  5/15 *sh* allow "units='none'" as ignorable units string (for Andrew)
* V695+ 10/15 *sh* pass axis length instead of iaxis to TM_CHECK_BNDS_ATTRIB
* V698   4/16 *sh* recognize special E and F non-convertible axis units
* V698  5/16 *acm* for ticket 1432, could write a note when encountering a 
*                  supbspan modolo axis with length almost the modulo length.
*                  (commented out for now.)
* v7+   7/16 *acm* Ticket 2445, regular-spacing attribute and repeated or missing values
*                  in the first 2 coordinates
* V710  4/16 *acm* new TM_UNITS_CAL to account for calendar in units id
* V710  4/16 *acm* Remove the 2-day shift correction. CDC does not seem to 
*                  consistenly applythis logic to current files
* V71 12/16 *acm* Ticket 2158: working with true monthly time axes.
* V7022 1/17 *acm* ticket 2497, Back off the auto-detection of monthly axes.
* 1/13/2017 *acm* Do the repeated-coordinates checking in the new routine
*                 TM_CHECK_COORDS, also called by xeq_define, to consistently
*                 handle almost-equal coordinates, differing by machine precision.
* 1/24/2017 *acm* Ticket 2504: If Ferret is computuing axis bounds, set them such 
*                 that the axis lies within the given modulo length.
* 1/26/2017 *acm* Ticket 1480. change the message if the modulo length is 
*                 less than the axis length. The rest of that ticket is 
*                 addressed in cd_write_axis.
* v720 2/2017 *acm* Ticket 2513. Check that bounds or edges data is in the file when 
*                 attribute is found, before trying to use it.
* v720 3/2017 *acm* Ticket 2246. Call CD_GET_TIME_UNITS routine to get t0 date and time units 
* V72 6/17 *acm* For trac enhancement #767 -- dynamic coordinate storage 
* V73+ 12/17 *acm* Issue #1848: apply set_regular flag once the coordinates have been set up
* V74  ACM  2/2018 As in the DSG branch, pass line_direction(iaxis) to 
*                 cd_get_line_direction as an argument
* v743 9/2018 *acm* Issue 1894. Both point_spacing="even" and bounds are given. The axis
*                  coordinates/bounds are detected as uneven. Neither the coordinates
*                  nor line_start/ line_delta were set. In this case, treat it as irregular.
* v744 11/2011 *acm* Issue 1906: If PROLEPTIC_GREGORIAN is given then 
*                  override mixed-gregorian calendar handling. 
* v7441*acm* 12/18 Issue 1909,1919: ISO 8601 formatted dates and detecting
*                  mixed gregorian/julain calendar axes
* v7445*acm*  4/19 Issue 334: If there was a point_spacing = "even" attr, do more checking.
*                  But, if they've said USE/REGULART then go ahead and treat it as regular.
* v7445*acm*  4/19 Fix the calls to TM_SCALE 
* V7.51 *acm* 5/2019: Issue 1934: error with Epic file where east lons were treated as west
* V7.61 *acm* 7/2020: back to issue 334, test was too stringent. Normalize by the coordinate
*                     data size and use single precision.
*
* argument definitions:
*       cdfid    - netCDF id number for already opened CDF file  (to go away with new attr. hanling)
*       dset     - dataset number for already opened CDF file
*       ivar     - netCDF variable id number
*       iaxis    - returned pointer to the axis created
*       its_epic - flags an EPIC netCDF data set
*       reversed    - logical array to record which axes have reversed coords
*       status   - return status

* include files
        include 'netcdf.inc'
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
        LOGICAL  its_epic, reversed(max_lines:line_ceiling), tregular, 
     .           use_strict, has_gdef
        INTEGER  cdfid, dset, ivar, iaxis, bad_bndsid, status

* local parameter definitions:
      INTEGER     str_eq
      PARAMETER ( str_eq = 0 )

* local variable declarations:
      LOGICAL NC_GET_ATTRIB, CD_GET_ATTVAL,
     .        TM_HAS_STRING, TM_FPEQ_EPS, TM_DFPEQ, CD_GET_ATTVALC, 
     .        CD_GET_ATTVAL_L, TM_CHECK_BNDS, TM_CHECK_BNDS_CENTERED, 
     .        TM_FPEQ, TM_DFPEQ_TOL, NC_GET_ATTRIB_DP, TM_FPEQ_SNGL,
     .        got_it, got_torg, do_warn,
     .        epic_time, west_lon, yyyymmdd_time, 
     .        got_mod, mod_tf, setmodfalse, ok_cal, ok,
     .        got_edges, got_bnds, coordvar, its_irregular, 
     .        do_scale, do_offset, scale_is_dp, reg,
     .        has_repeated, misordered, has_missing, is_double, 
     .        true_month, okmod, edge_reg, set_regular, attr_regular

      INTEGER TM_UNIT_ID, TM_UNITS_CAL, TM_LENSTR1, TM_GET_CALENDAR_ID,      
     .        STR_CASE_BLIND_COMPARE, STR_UPCASE, STR_SAME,
     .        soft_err, 
     .        vartype, vback, npts, vlen, cdfstat, istat,
     .        nvdims, vdims(8), nvatts, i, units, npts_cdf, elen, 
     .        edgid, spacing_atlen, epic_t2var, since_T0,
     .        cal_id, year, month, day, hour, minute, second, slen,
     .        blen, bndid, startpts(2), numpts(2),
     .        d1s, d1e, d2s, d2e, attoutflag, maxlen,
     .        all_outflag, dir, attlen, itop, irev, attype, iatt, ibx1,
     .        i1, i2, i3, i0, npts2

      REAL    rbuff, delta, first_delta, val, val2, lm_len, 
     .        scale, offset, start
      REAL*4  r4delta, r1, r2, rscale
      REAL*8  TM_LONW2LONE, TM_WW_AXLEN, GET_LINE_COORD,
     .        tmp8, axwwlen, dscale, doffset,  micro_adj, madj
      CHARACTER*12 TM_LEFINT
      CHARACTER vname*128, ename*128, vupcase*128, words(8)*24,
     .        spacing_attr*14, bname*128
      CHARACTER*13 TM_STRING
      CHARACTER*1 axis_dir
      CHARACTER*2 dcode

      INTEGER tt  ! nice short name
      EQUIVALENCE (tt,lunit_errors)

      INTEGER bufflen
      PARAMETER (bufflen = 256)
      CHARACTER*256 buff
      REAL*8   TM_SECS_FROM_BC, first, secs2firststep, secs1590,
     .         t0_secs, last_coord, firs_coord, secsperyear, secs2start, 
     .         days_neg4713

* initialize for EPIC
      do_warn = .NOT.its_epic     ! EPIC files can get away with anything
      west_lon = .FALSE.          ! assume longitudes are positive east
      iaxis = 0

* gen; initialization
 
      got_torg = .FALSE.
      days_neg4713 = 0.D0
      bad_bndsid = 0
      cal_id = 1
      misordered = .FALSE.
      has_missing = .FALSE.



* get the vital statistics of the axis 

      CALL CD_GET_VAR_INFO (dset, ivar, vname, vartype, 
     .            nvdims, vdims, nvatts, coordvar, all_outflag, status)

      IF (status .NE. merr_ok) GOTO 5100
      vlen = TM_LENSTR1(vname)
      istat = STR_UPCASE( vupcase, vname )
!      IF ( nvdims .NE. 1 ) CALL TM_ERRMSG
!     .     ( cdfstat+pcdferrmax, status, 'CD_GET_1_AXIS', cdfid, ivar,
!     .     vname(:vlen), 'not a 1D variable', *5900 )


      is_double = vartype .EQ. ncdouble

* get axis length:

      CALL CD_GET_DS_DIMS (dset, vdims(1), buff, npts_cdf, status)
      IF (status .NE. ferr_ok) GOTO 5100

! removed 3/93          IF ( npts_cdf .EQ. 0 ) GOTO 5150
      npts = npts_cdf      ! differ only if "true_size" is used

* grab a temporary dynamic axis slot for it
      CALL TM_ALLO_TMP_LINE(iaxis, status)
      IF (status .NE. merr_ok) GOTO 5900

* Store the data type of the axis

      line_dattype(iaxis) = vartype

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
      set_regular = .FALSE.
      attr_regular = .FALSE.

      IF ( got_it ) THEN
         line_regular(iaxis) = buff(1:1).EQ.'e' .OR. buff(1:1).EQ.'E'
         its_irregular = .NOT.line_regular(iaxis)
	 attr_regular = line_regular(iaxis)
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

* Cannot have both "edges" and "bounds" attributes
* Ticket 2513
* Check that the file has the bounds or edges listed in the attribute.
* Use bounds if they are in the file, else edges.

      IF ( got_bnds .AND. got_edges ) THEN
         bndid = 0
	 edgid = 0
	 CALL CD_GET_VAR_ID (dset, bname, bndid, status)
	 CALL CD_GET_VAR_ID (dset, ename, edgid, status)

* If bounds or edges are in the file, use bounds preferentially, else edges

	 IF (bndid + edgid .GT. 0) THEN
	    IF (bndid .GT. 0) THEN
               CALL TM_NOTE(
     .          'Axis has both edges and bounds attributes: '//
     .           vname(:vlen)//' - edges definition ignored', tt )
               got_bnds = .FALSE.
	    ENDIF
	    IF (edgid .GT. 0 .AND. bndid.EQ.0) THEN
               CALL TM_NOTE(
     .          'Axis has both edges and bounds attributes: '//
     .           vname(:vlen)//'. Bounds not in file. Ignore bounds definition', tt )
               got_bnds = .FALSE.
	    ENDIF
	 ENDIF ! at least one of bounds and edges is in the file
	 
* If attributes but not data for bounds/edges, issue a note	    
	 IF (bndid + edgid .EQ. 0) THEN
            CALL TM_NOTE(
     .           'Axis has both edges and bounds attributes: '//
     .           vname(:vlen)//'. Bounds and edges not in file, '//
     .           'ignore bounds and edges attributes', tt )
            got_bnds = .FALSE.
            got_edges = .FALSE.
	 ENDIF

      ENDIF

* We will be checking bounds or edges. Read the coords too
* in case we are going to mark the axis as irregular.

      IF (got_edges .OR. got_bnds) line_regular(iaxis) = .FALSE.

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
         CALL CD_GET_CALENDAR_NAME(buff, ok_cal)

         IF (ok_cal) THEN
	    line_cal_name(iaxis) = buff
	 ELSE
            slen = TM_LENSTR1(buff)
            CALL TM_NOTE('calendar attribute on axis "'//vupcase(:vlen)//
     .                '" is not recognized: '//buff(1:slen), tt)
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
      IF (STR_SAME (buff(1:4), 'none') .EQ. 0) buff = ' ' ! units = 'none'
      since_T0 = MAX( INDEX(buff,'since'), INDEX(buff,'SINCE') )
      yyyymmdd_time = buff(1:15) .EQ. 'yyyymmddhhmmss'

      IF ( since_T0 .GT. 2 ) THEN
         line_units(iaxis) = buff(:since_T0-1)  ! just the "tunits" part 

         slen = TM_LENSTR1(buff)
         IF ( slen .GT. since_T0 + 5) THEN
	 
	    got_torg = .TRUE.
	    CALL CD_GET_TIME_UNITS (buff, cal_id, line_units(iaxis), 
     .			            line_t0(iaxis), days_neg4713, status)

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

* Named calendar with unit=month or year gets length according to calendar's year length.

      units = TM_UNITS_CAL( units, cal_id, .FALSE. )

      line_unit_code(iaxis) = units
      IF  (   units.EQ.0
     .  .AND. line_units(iaxis) .NE. ' '
     .  .AND. line_units(iaxis) .NE. plag_ax_units_e
     .  .AND. line_units(iaxis) .NE. plag_ax_units_f ) THEN
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

* Use attributes to try to determine the orientation and 
* direction of the axis

      CALL CD_GET_LINE_DIRECTION (dset, ivar, vname, line_units(iaxis),  
     .   do_warn, .TRUE., line_direction(iaxis))

* longitude axes may be encoded "west" (discontinuous at 180, East negative,
* West positive) or "east" (standard - 0->360 east from Greenwhich)

      got_it = NC_GET_ATTRIB( dset, ivar, 'positive', do_warn, 
     .                        vname(:vlen), maxlen, attlen, attoutflag, 
     .                        buff, val)
      IF ( got_it .AND. STR_CASE_BLIND_COMPARE('west',buff(:4))
     .                  .EQ.str_eq) THEN
            line_direction(iaxis) = 'WE'
            west_lon = .TRUE.                        ! convert encodings below
      ENDIF

* If there is a 'positive' attribute, see if it indicates west.


      IF (got_it) THEN
       west_lon = west_lon 
     .  .OR.    ( line_direction(iaxis) .EQ. 'WE'
     .      .AND. TM_HAS_STRING(buff,'west') )
     .  .OR.    ( its_epic
     .      .AND. (INDEX(vupcase, 'LON') .GT. 0)
     .      .AND. .NOT.TM_HAS_STRING(buff,'east') )
      ENDIF

* was a regular time axis forced on us by the user?
      IF ( tregular .AND. line_direction(iaxis)(1:1).EQ."T" ) THEN
        line_regular(iaxis) = .TRUE.
        got_bnds = .FALSE.
	got_edges = .FALSE.
	its_irregular = .FALSE.
	set_regular = .TRUE.
      ENDIF

* get the axis coordinates - none, regular, or irregular

* Allocate line coordinate storage, put the pointer in array linemem(iaxis)
* and lineedg(iaxis) storage.

      CALL GET_LINE_DYNMEM (npts_cdf, iaxis, status) 
      IF (status .NE. merr_ok) GOTO 5900

      CALL PUT_LINE_COORD ( linemem(iaxis)%ptr, 1, 0.0 )

* If there was a point_spacing = "even" attr, do just a bit more checking.
* But, if they've said USE/REGULART then go ahead and treat it as regular.
* (ticket 334)

      IF (npts.GT.1 .AND. line_regular(iaxis) .AND. 
     .    attr_regular.AND.(.NOT.set_regular) ) THEN   

         IF ( epic_time ) THEN
               CALL CD_RD_EP_1( cdfid, ivar, epic_t2var, line_t0(iaxis),
     .                       line_tunit(iaxis), 1, val, status )
               IF ( status .NE. merr_ok ) GOTO 1000
               CALL CD_RD_EP_1( cdfid, ivar, epic_t2var, line_t0(iaxis),
     .                       line_tunit(iaxis), 2, val2, status )
         ELSEIF ( yyyymmdd_time ) THEN
               CALL CD_RD_YMD_1( cdfid, ivar, line_t0(iaxis),
     .                       line_tunit(iaxis), 1, val, status )
               IF ( status .NE. merr_ok ) GOTO 1000
               CALL CD_RD_YMD_1( cdfid, ivar,  line_t0(iaxis),
     .                       line_tunit(iaxis), 2, val2, status )
         ELSE
               CALL CD_RD_R8_1( cdfid, ivar, 1, vartype, vname(:vlen),
     .                       val, status )
               IF ( status .NE. merr_ok ) GOTO 1000
              CALL CD_RD_R8_1( cdfid, ivar, 2, vartype, vname(:vlen),
     .                       val2, status )
         ENDIF

         first_delta = ABS(val2-val)

         IF ( epic_time ) THEN
            CALL CD_RD_EP_1( cdfid, ivar, epic_t2var, line_t0(iaxis),
     .                       line_tunit(iaxis), npts_cdf, val2, status )
         ELSEIF ( yyyymmdd_time ) THEN
            CALL CD_RD_YMD_1( cdfid, ivar,  line_t0(iaxis),
     .                       line_tunit(iaxis), npts_cdf, val2, status )
         ELSE
            CALL CD_RD_R8_1( cdfid,ivar, npts_cdf, vartype, vname(:vlen),
     .                       val2, status )
         ENDIF

*  Do the first-delta and computed delta-coordinates agree? Scale by the coordinate size.
*  val is 1st coord pt, val2 is last pt

	 rscale = (abs(val) + abs(val2))/ 2.
	 r1 = (ABS(val2-val)/DBLE(npts-1) )/ rscale
	 r2 = first_delta/ rscale

	 ok = TM_FPEQ_SNGL(r1, r2) 
         
	 IF (.NOT. ok ) THEN
            CALL TM_NOTE('Axis "'//vname(:vlen)//'" is irregular. Ignoring'//
     .        ' erroneous point_spacing=even attribute. ', tt)
	    line_regular(iaxis) = .FALSE.
	    attr_regular = .FALSE.

	 ENDIF
      ENDIF

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

* Invalid points. Cannot compute line_delta
         IF (ABS(line_start(iaxis)) .GE. ABS(bad_r8)   .OR. 
     .       ABS(line_delta(iaxis)) .GE. ABS(bad_r8) ) THEN 
	    misordered = .TRUE.
	    has_missing = .TRUE.
	    words(1) = '2'
	    GOTO 540
	 ENDIF

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

* Ticket 2445: repeated coordinates on axis marked as regular is an invalid axis
      
	 IF (line_delta(iaxis) .EQ. 0.) THEN
	    misordered = .TRUE.
	    words(1) = '2'
	    GOTO 540
	 ENDIF

         line_subsc1(iaxis) = unspecified_int4
         last_coord = line_start(iaxis) + DBLE(npts-1)*line_delta(iaxis)
         firs_coord = line_start(iaxis)


* ============= START OF IRREGULAR AXIS PROCESSING ========================
      ELSE 


* Read the coords and test for regular spacing. If regular then we're good. 
* If not, then compute the grid cell bounds.

         line_start(iaxis) = unspecified_val8
         line_delta(iaxis) = unspecified_val8


* read the CDF coordinate values into double precision axis storage
* Applying any scaling using TM_SCALE.  Coordinate data does not have missing-flags,
* send the default missing flag which should not match any valid coordinate data.

         IF ( epic_time ) THEN
            CALL CD_RD_EP_ARR( cdfid, ivar, epic_t2var,
     .                         line_t0(iaxis), line_tunit(iaxis),
     .                         npts_cdf, linemem(iaxis)%ptr, status )
         ELSEIF ( yyyymmdd_time ) THEN
            CALL CD_RD_YMD_ARR( cdfid, ivar,
     .                         line_t0(iaxis), line_tunit(iaxis),
     .                         npts_cdf, linemem(iaxis)%ptr, status )
         ELSE
            CALL CD_RD_R8_ARR( cdfid, ivar, 1, npts_cdf, vartype,
     .                         vname(:vlen), linemem(iaxis)%ptr, status )
            IF (days_neg4713 .NE. 0.) THEN
               offset = -1* days_neg4713
               CALL TM_SCALE (1.D0, -1* days_neg4713, bad_r8, 
     .                        linemem(iaxis)%ptr,  npts)
            ENDIF
         ENDIF
         IF ( status .NE. merr_ok ) GOTO 1000   ! 3/99 soft error
	 
	 IF (do_scale) THEN
	    IF (scale_is_dp) THEN
	       CALL TM_SCALE (dscale, doffset, bad_r8, 
     .                         linemem(iaxis)%ptr, npts)
	    ELSE
	       CALL TM_SCALE (scale, offset, bad_r8, 
     .                         linemem(iaxis)%ptr, npts)
	    ENDIF
	 ENDIF

         IF ( west_lon ) THEN
            CALL TM_LONW2E_LINE (linemem(iaxis)%ptr, npts)
         ENDIF

* Check for misordered or repeated data
* If repeated values, apply micro-adjusting, or report as a warning if the 
* user requested /STRICT.

* Check if axis is actually regularly spaced (though not flagged as "even")
* Use the precision of the data  in the file.

	CALL TM_CHECK_LINE ( linemem(iaxis)%ptr, npts, 
     .      reversed(iaxis), vname, vlen, is_double, its_irregular, 
     .      use_strict, line_regular(iaxis), misordered)
	IF (misordered) GOTO 1000

	IF (line_regular(iaxis)) THEN 
* yes, it's actually regular - save it as such
           line_start  (iaxis) = GET_LINE_COORD (linemem(iaxis)%ptr, 1)
           line_delta  (iaxis) = 
     .              ( GET_LINE_COORD (linemem(iaxis)%ptr, npts) - 
     .                line_start(iaxis) )/DBLE(npts-1)
           last_coord = line_start(iaxis) + DBLE(npts-1)*line_delta(iaxis)
           firs_coord = line_start(iaxis)

           GOTO 390
	ENDIF

* Compute the default locations of boundaries between grid boxes (midpoints)
* Note: this may be modified by a subsequent "EDGES" or "BOUNDS" definition
 
        CALL TM_IRREGAX_MID( linemem(iaxis)%ptr, lineedg(iaxis)%ptr, npts )
      ENDIF

* ============= END OF IRREGULAR AXIS PROCESSING ========================


 390  CONTINUE

* Notify user of reversal
* Dont issue the note if west_lon. 

      IF ( reversed(iaxis) .AND. (.NOT.west_lon)) THEN
         CALL TM_NOTE('Axis coordinates are decreasing-ordered. '//
     .         'Reversing ordering for axis '//vname(:vlen), tt)
      ENDIF

* 4/2016 remove this correction. CDC does not seem to consistenly apply
* this logic to current files

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
* 11/2018: If the calendar type is PROLEPTIC_GREGORIAN, then do not apply
*          the shift, even if it the origin is at 01-JAN-0001

      IF (since_T0.GT.2) CALL TM_BREAK_DATE(line_t0(iaxis), cal_id, 
     .       year, month, day, hour, minute, second, istat)
      IF ( (since_T0.GT.2) 
     .  .AND. (istat.EQ.merr_ok) 
     .  .AND. (year.EQ.1) .AND. (month.EQ.1) .AND. (day.EQ.1)
     .  .AND. (hour.EQ.0) .AND. (minute.EQ.0) .AND. (second.EQ.0) 
     .  .AND.(cal_id .EQ. GREGORIAN) ) THEN

* ... get time step of L=1
           IF ( line_regular(iaxis) ) THEN
              first = line_start(iaxis)   
           ELSE
              first = GET_LINE_COORD (lineedg(iaxis)%ptr, 1)
           ENDIF

* ... compute seconds from 1-jan-0001
           secs2firststep = un_convert(line_unit_code(iaxis)) * first
* compute seconds from 1590
           secs1590 = TM_SECS_FROM_BC (GREGORIAN, 1590,1,1,0,0,0,status)
            IF ( status .NE. merr_ok ) GOTO 5900

* ... correct axis if crosses 1590 threshold
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

      CALL TM_CHECK_EDGES_ATTRIB (cdfid, iaxis, vname, vlen, ename, 
     .        edgid, status)

      IF (status .EQ. pcdferr) GO TO 5100

* Invalid edges, issue a note and use coordinate midpoints.
 
       IF (status .NE. merr_ok) THEN
         CALL TM_IRREGAX_MID( linemem(iaxis)%ptr, lineedg(iaxis)%ptr, npts )
         GO TO 600
      ENDIF

* read the (npts+1) box edge values into double precision storage

      CALL CD_RD_R8_ARR( cdfid, edgid, 1, npts+1, vartype,
     .                     vname(:vlen), lineedg(iaxis)%ptr, status )
      IF ( status .NE. merr_ok ) GOTO 1000    ! 3/99 soft error

* Check to see that each data point is contained inside its box.
* See if they're centered after all.

      CALL TM_CHECK_LINE_EDGES ( linemem(iaxis)%ptr, 
     .                   lineedg(iaxis)%ptr, npts, iaxis, reversed(iaxis), 
     .                   vname, vlen, is_double, edge_reg, ok)

      IF (.NOT. ok) THEN
           CALL TM_NOTE(
     .            'Error in Edges or edges do not enclose point on axis '//
     .             vname(:vlen), tt)
           CALL TM_NOTE('Substituting coordinate midpoints', tt)
           CALL TM_IRREGAX_MID( linemem(iaxis)%ptr, lineedg(iaxis)%ptr, npts  )
           bad_bndsid = edgid     
       ENDIF

* The axis had a point_spacing = "even" attribute and also edges.  We have found 
* the axis coordinate and/or edges themselves to be in fact irregular. So don't 
* later on apply the set_regular, which came from the point-spacing attribute.
* If the user said USE/REGULART then go ahead and still treat it as regular

      IF (ok .AND. attr_regular .AND. .NOT.edge_reg ) set_regular = .FALSE.

* It's actually regular - save it as such
      IF (edge_reg .AND. .NOT. line_regular(iaxis)) THEN
         line_regular(iaxis) = .TRUE.
         line_start  (iaxis) = GET_LINE_COORD (lineedg(iaxis)%ptr, 1)
         IF (npts .GT. 1) THEN
	      line_delta  (iaxis) = 
     .        (GET_LINE_COORD (lineedg(iaxis)%ptr, npts) - 
     .         GET_LINE_COORD (lineedg(iaxis)%ptr, 1) )/DBLE(npts-1)
         ELSE
	      line_delta  (iaxis) = 
     .        (GET_LINE_COORD (lineedg(iaxis)%ptr, npts+1) - 
     .         GET_LINE_COORD (lineedg(iaxis)%ptr, 1) )
         ENDIF
         last_coord = line_start(iaxis) + 
     .                DBLE(npts-1)*line_delta(iaxis)
         firs_coord = line_start(iaxis)
      ELSE
	 line_regular(iaxis) = .FALSE.
      ENDIF


****************************************************************************
*                              BOX BOUNDS                                  *
****************************************************************************

* the attribute "bounds=name" may point to a list of box boundaries

500   maxlen = 128
      IF ( .NOT.got_bnds ) GOTO 600

* Check the bounds variable, issue any warnings.
      CALL TM_CHECK_BNDS_ATTRIB (dset, line_dim(iaxis), vname, vlen,
     .                             bname, bndid, status)
      IF (status .EQ. pcdferr) GO TO 5100
      IF (status .NE. merr_ok) THEN
         IF (npts .EQ. 1) THEN  ! save as a regular axis, no bounds.
              line_regular(iaxis) = .TRUE.
              line_delta(iaxis) = 1
              line_subsc1 (iaxis) = unspecified_int4
              last_coord = line_start(iaxis) 
              firs_coord = line_start(iaxis)
         ENDIF
         ! If npts not 1 then we have already set the boundaries
         ! with CALL TM_IRREGAX_MID
         GO TO 600
      ENDIF

* read the (npts*2) box edge values into double precision storage 

      startpts(1) = 1
      startpts(2) = 1
      numpts(1) = 2
      numpts(2) = npts

      d1s = startpts(1)
      d1e = numpts(1)
      d2s = startpts(2)
      d2e = numpts(2)

* read the 2*npts box edge values into double precision storage
* Allocate line-edges coordinate storage, put the pointer in scratch 
* space lineedg(line_0). This will be redone into n+1 edges, so this
* allocation is temporary.

      npts2 = 2*npts
      CALL GET_LINE_DYNMEM (npts2, line_0, status)  
      IF (status .NE. merr_ok) GOTO 5900

      blen = TM_LENSTR1(bname)
      CALL CD_RD_R8_BNDS( cdfid, bndid, startpts, numpts, vartype, 
     .                    bname(:blen), lineedg(line_0)%ptr,  
     .                    d1s, d1e, d2s, d2e, status )
      IF ( status .NE. merr_ok ) GOTO 1000    ! 3/99 soft error

      IF (npts .EQ. 1) CALL PUT_LINE_COORD ( linemem(iaxis)%ptr, 1, line_start(iaxis) )
      CALL TM_CHECK_LINE_BOUNDS (linemem(iaxis)%ptr, 
     .                  lineedg(line_0)%ptr, npts, 
     .                  iaxis, reversed(iaxis), vname, vlen, 
     .                  is_double, line_regular(iaxis), ok)

      IF (.NOT.ok ) THEN
         CALL TM_NOTE('Error in bounds "'//bname(:blen)//
     .         '" or bounds do not enclose point on axis '//
     .         vname(:vlen), tt)
         CALL TM_NOTE('Substituting coordinate midpoints', tt)

         CALL TM_IRREGAX_MID( linemem(iaxis)%ptr, lineedg(iaxis)%ptr, npts )

	 bad_bndsid = bndid
      ENDIF

* The axis had a point_spacing = "even" attribute and also bounds.  We have found 
* the axis coordinate and/or bounds themselves to be in fact irregular. So don't 
* later on apply the set_regular, which came from the point-spacing attribute.

      IF (ok .AND. set_regular .AND. .NOT.line_regular(iaxis) ) set_regular = .FALSE.

      IF (ok .AND. .NOT.line_regular(iaxis)) THEN
	   
* Change from N*2 storage of bounds to N+1 edges.

         CALL TM_CONVERT_BOUNDS_EDGES (
     .                  lineedg(line_0)%ptr, 
     .                  lineedg(iaxis)%ptr, npts)

      ENDIF

* Use bounds to set the line_delta for regular axis.

      IF (ok .AND. line_regular(iaxis)) line_delta(iaxis) = 
     .   GET_LINE_COORD(lineedg(line_0)%ptr, 2) - GET_LINE_COORD(lineedg(line_0)%ptr, 1)

*  deallocate the temporary edge storage used to check bounds.
      CALL FREE_LINE_DYNMEM(line_0)

 600  CONTINUE

* All the checks for regular have been made. If regular, make sure we have set 
* line-start and line_delta, and deallocate the line storage....

      IF (set_regular) THEN
         IF (.NOT.line_regular(iaxis) .AND. npts.GT.1) THEN
	    line_start(iaxis) = GET_LINE_COORD(linemem(iaxis)%ptr, 1)
	    line_delta(iaxis) = GET_LINE_COORD(linemem(iaxis)%ptr, 2) - 
     .                          GET_LINE_COORD(linemem(iaxis)%ptr, 1)
	 ENDIF
	 line_regular(iaxis)= .TRUE.
      ENDIF

      IF (line_regular(iaxis)) CALL FREE_LINE_DYNMEM(iaxis)
 
* Check if axis is a monthly axis for this calendar.  If so store as a 'true-month' 
* axis, stored internally as if regular, but coordinates returned on true axis.
* Do not call this if the file has enhanced header, with parent and child grids. 
* (at least for now.)
* 1/13/2017 ticket 2497, Back off this auto-detection of monthly axes for now.


	true_month = .FALSE.
c	IF (line_direction(iaxis) .EQ. 'TI' .AND. 
c     .      .NOT.line_regular(iaxis) .AND.
c     .      .NOT.has_gdef ) 
c     .       CALL TM_CHECK_MONTHLY_AXIS (linemem(iaxis)%ptr, line_dim(iaxis), cal_id, 
c     .                                   line_t0(iaxis), units, line_units(iaxis),
c     .                                   start, delta, line_tunit(iaxis), true_month)
c
c        IF (true_month) THEN
c           line_regular(iaxis) = .TRUE.
c           line_start  (iaxis) = start !* line_tunit(iaxis)/ un_convert(pun_day)
c           line_delta  (iaxis) = delta
c           line_subsc1 (iaxis) = unspecified_int4
c           last_coord = line_start(iaxis) + DBLE(npts-1)*line_delta(iaxis)
c           firs_coord = start
c           line_unit_code (iaxis) = units
c           line_tunit  (iaxis) = un_convert(units)
c           got_edges = .FALSE.
c           got_bnds  = .FALSE.
c	ENDIF

* Save the axis name (upper case) and original axis name.   

        line_name(iaxis) = vupcase
        line_name_orig(iaxis) = vname

* if it is a longitude axis of 360 degrees or less let it default to MODULO
* unless user has set it explicitly to false.
        
 400    axwwlen = 0.
        axwwlen = TM_WW_AXLEN(iaxis)

        IF ( line_direction(iaxis).EQ.'WE'
     .       .AND. units.EQ.4
     .       .AND. (.NOT.setmodfalse)
     .       .AND. npts.GT.1 ) THEN

           r4delta = axwwlen ! single precision
           delta = r4delta
           IF (TM_FPEQ(delta, 360.0) .OR. delta .LT. 360.0) THEN
              line_modulo(iaxis) = .TRUE.
              IF (line_modulo_len(iaxis).EQ.0.0D0)
     .                    line_modulo_len(iaxis) = 360.D0
           ELSE

* Ticket 2504: Are the coordinates within a 360 deg but the bounds not?

              IF (.NOT.line_regular(iaxis)) THEN
	         IF (.NOT.got_bnds .AND. .NOT.got_edges .AND. 
     .               GET_LINE_COORD (lineedg(iaxis)%ptr, npts) - 
     .               GET_LINE_COORD (lineedg(iaxis)%ptr,1) .LE. 360.D0 ) THEN
	            CALL TM_ADJUST_BOUNDS(iaxis, npts, 360.0, axwwlen, okmod)
	            IF (okmod) THEN
		       line_modulo(iaxis) = .TRUE.
		       line_modulo_len(iaxis) = 360.D0
		    ENDIF
	         ENDIF
	      ENDIF
           ENDIF
        ENDIF

* if it is a calendar time axis and the first point is in year 0000 or 0001
* then it is implicitly MODULO

        IF (got_torg) THEN
           IF (line_direction(iaxis)(1:1).EQ.'T') line_direction(iaxis) = 'TI'
           IF (line_direction(iaxis)(1:1).EQ.'F') line_direction(iaxis) = 'FI'
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

* Ticket 2504: Are the coordinates within a year but the bounds not?
           ELSEIF (.NOT.line_regular(iaxis) .AND. .NOT.got_bnds .AND. .NOT.got_edges) THEN  
	      delta = tmp8 - first
              IF ( (secs2start .LE.  2*secsperyear)
     .	           .AND. (delta*line_tunit(iaxis) .LE. secsperyear) ) THEN
	         CALL TM_ADJUST_BOUNDS(iaxis, npts, 
     .                 secsperyear/line_tunit(iaxis), axwwlen, okmod)

	         IF (okmod) THEN
		    line_modulo(iaxis) = .TRUE.
		    line_modulo_len(iaxis) = axwwlen
	         ENDIF
              ENDIF
           ENDIF
        ENDIF

* sanity check the modulo length

* Ticket 2504: If the coordinates lie within the modulo length but the auto-
* generated midpoint bounds make the axis too long, reset them so the axis 
* is the prescribed length. (may have been done for special cases above)

	IF (.NOT.line_regular(iaxis) .AND. line_modulo(iaxis) .AND. .NOT.got_bnds .AND. .NOT.got_edges) THEN
	   CALL TM_ADJUST_BOUNDS(iaxis, npts,  
     .                 line_modulo_len(iaxis), axwwlen, okmod)
	   IF (.NOT.okmod ) THEN
	      line_modulo(iaxis) = .FALSE.
	      line_modulo_len(iaxis) = 0.
	   ENDIF
	ENDIF

* a whisker too small is assumed a precision error
* a whisker larger is assumed to be deliberate
* ACM 1/08 
* If a whisker too large, this may be a precision error 
* In particular this happens converting single float coordinates to
* double coordinate storage. If coords are single precision do the 
* check in single precision.

* Ticket 2172: If the axis is actually too long, reset the modulo flags
* so it is treated as not modulo.
* Ticket 1480, change the message; the modulo length is invalid if 
* it's shorter than the axis.

        IF (line_modulo(iaxis).AND.line_modulo_len(iaxis).NE.0.D0) THEN
           IF (line_modulo_len(iaxis) .LT. axwwlen) THEN
             IF (is_double) THEN
                ok = TM_DFPEQ(axwwlen, line_modulo_len(iaxis)) 
             ELSE
	        lm_len = line_modulo_len(iaxis)
                delta = axwwlen ! single precision
		IF (vartype.NE.ncdouble) delta = r4delta
                ok = TM_FPEQ(delta, lm_len)
             ENDIF
             IF (.NOT.  ok) THEN
	        line_modulo_len(iaxis) = 0
		line_modulo(iaxis) = .FALSE.
                CALL TM_NOTE(
     .          'Ignored modulo length less than axis length: '
     .          //vname(:vlen), tt )
             ENDIF
           ENDIF
        ENDIF

* Write a note when subspan modulo axis length within a grid cell of modulo length
c	CALL TM_WARN_SUBSPAN (iaxis)

* Save the line direction in the linked-list structure for the coordinate variable

         dir = 0
         dcode = line_direction(iaxis)
         IF (dcode .EQ. 'XX' .OR. dcode .EQ. 'WE') dir = 1
         IF (dcode .EQ. 'YY' .OR. dcode .EQ. 'SN') dir = 2
         IF (dcode .EQ. 'ZZ' .OR. dcode .EQ. 'UD' .OR. dcode .EQ. 'DU') dir = 3
         IF (dcode .EQ. 'TT' .OR. dcode .EQ. 'TI') dir = 4
         IF (dcode .EQ. 'EE') dir = 5
         IF (dcode .EQ. 'FF' .OR. dcode .EQ. 'FI') dir = 6
         CALL CD_SET_ATT_AXDIR (dset, ivar, dir, status)

* successful completion
        status = merr_ok
        RETURN

 540	IF (misordered) THEN
	    IF (has_missing) THEN
	       CALL TM_NOTE(
     .           'Coordinates missing on axis '//
     .           vname(:vlen)//' at subscript '//words(1), tt)
	    ELSE
	       CALL TM_NOTE(
     .           'unrepairable repeated axis coords on axis '//
     .           vname(:vlen)//' at subscript '//words(1), tt)
	    ENDIF
	    CALL TM_NOTE(
     .           'A dummy axis of subscripts will be used', tt)

	 ENDIF

* 3/99-style "soft" error - invalid axis but CDF file need not be rejected
 1000   CONTINUE

	CALL TM_DEALLO_DYN_LINE(iaxis)
	CALL FREE_LINE_DYNMEM (iaxis)

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

 5400   CALL TM_ERRMSG
     .     ( cdfstat+pcdferr, status, 'CD_GET_1_AXIS', cdfid, ivar,
     .       'Scale and offset attributes not of the same type', 
     .       no_errstring, *5900 )

* error exit
 5900   RETURN
        END
