        SUBROUTINE CD_GET_1_AXIS(cdfid, ivar, iaxis, its_epic,
     .                           reversed, tregular, status)

*
*
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
* Load the common block XGRID with a single axis definition from a
* netCDF file

* Programmer Steve Hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program

* revision 0.0 - 2/92
*   8/19/92: search for "LAT" in axis units was setup wrong
*  10/20/92: a more systematic approach to identifying the axis orientation
*  11/30/92:  changed Z axis orientation "UP" to "DU"
* 12/92: use TM_FPEQ to test regularity of axes
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

* argument definitions:
*       cdfid    - netCDF id number for already opened CDF file
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

* argument declarations
        LOGICAL  its_epic, reversed(max_lines:line_ceiling), tregular
        INTEGER  cdfid, ivar, iaxis, status

* local parameter definitions:
      INTEGER     str_eq
      PARAMETER ( str_eq = 0 )

* local variable declarations:
      LOGICAL CD_GET_ATTRIB, CD_GET_ATTVAL, TM_CHECK_BOX_LOC,
     .        TM_HAS_STRING, TM_FPEQ, got_it, got_torg, do_warn,
     .        epic_time, west_lon, yyyymmdd_time
      INTEGER TM_UNIT_ID, TM_LENSTR1,
     .        STR_CASE_BLIND_COMPARE, STR_UPCASE, NCVID, soft_err,
     .        ipt1, ipte, vartype, vback, npts, vlen, cdfstat, istat,
     .        nvdims, vdims(8), nvatts, i, units, npts_cdf, elen, 
     .        edgid, spacing_atlen, epic_t2var, since_T0,
     .        TM_GET_CALENDAR_ID, cal_id, slen, len_cal
      REAL    rbuff, delta, first_delta
      REAL*8  TM_LONW2LONE, tmp
      CHARACTER*12 TM_LEFINT
* 4/99 *kob* increase vname and vupcase from 24 to 64 chars 
      CHARACTER vname*64, ename*30, vupcase*64, words(8)*24,
     .        spacing_attr*14
      CHARACTER*13 TM_STRING
      CHARACTER*132 buff

      INTEGER tt  ! nice short name
      EQUIVALENCE (tt,lunit_errors)

      REAL*8   TM_SECS_FROM_BC, first, secs2firststep, secs1590,
     .          t0_secs, last_secs, last_coord, firs_coord, yr_len
      INTEGER   year, month, day, hour, minute, second

* parameter declarations
        REAL*8          psecs_per_day
        PARAMETER     ( psecs_per_day = 60.*60.*24. )

* initialize for EPIC
      do_warn = .NOT.its_epic     ! EPIC files can get away with anything
      west_lon = .FALSE.          ! assume longitudes are positive east
      iaxis = 0

* gen; initialization
      got_torg = .FALSE.

* get the vital statistics of the axis
      CALL NCVINQ(cdfid, ivar, vname, vartype, nvdims, vdims,
     .            nvatts,cdfstat)
      IF (cdfstat.NE.NCNOERR) GOTO 5100
      vlen = TM_LENSTR1(vname)
      istat = STR_UPCASE( vupcase, vname )
!      IF ( nvdims .NE. 1 ) CALL TM_ERRMSG
!     .     ( cdfstat+pcdferrmax, status, 'CD_GET_1_AXIS', cdfid, ivar,
!     .     vname(:vlen), 'not a 1D variable', *5900 )

* get axis length:
      CALL NCDINQ( cdfid, vdims, buff, npts_cdf, cdfstat )
      IF (cdfstat.NE.NCNOERR) GOTO 5100
! removed 3/93          IF ( npts_cdf .EQ. 0 ) GOTO 5150
      npts = npts_cdf      ! differ only if "true_size" is used

* grab a temporary dynamic axis slot for it
      CALL TM_ALLO_TMP_LINE(iaxis, status)
      IF (status .NE. merr_ok) GOTO 5900

* optional point spacing parameter: even, uneven, disordered
      IF ( its_epic ) THEN
         spacing_attr = 'type'          ! type = "EVEN" or ="E"
         spacing_atlen = 4
      ELSE
         spacing_attr = 'point_spacing' ! point_spacing = "even"
         spacing_atlen = 13
      ENDIF
      got_it = CD_GET_ATTRIB(cdfid,ivar,spacing_attr(1:spacing_atlen),
     .                       do_warn,vname(:vlen),buff,132 )
      IF ( got_it ) THEN
         line_regular(iaxis) = buff(1:1).EQ.'e' .OR. buff(1:1).EQ.'E'
      ELSE
         line_regular(iaxis) = .FALSE. ! default
      ENDIF

* optional "true_size" parameter for 2 point representations of even axes
      IF ( line_regular(iaxis) ) THEN
         got_it = CD_GET_ATTVAL(cdfid,ivar,'true_size',
     .                          do_warn,vname(:vlen),rbuff,1,vback)
         IF ( got_it ) THEN
            IF ( npts_cdf .NE. 2 ) THEN
               soft_err = 1
               GOTO 5000
            ENDIF
            npts = rbuff
         ENDIF
      ENDIF
      line_dim( iaxis ) = npts
      IF ( npts .LE. 2 ) line_regular(iaxis) = .TRUE.

* *ACM get the calendar name.

      line_cal_name(iaxis) = 'GREGORIAN'   ! default
      got_it = CD_GET_ATTRIB( cdfid, ivar, 'calendar', do_warn,
     .                        vname(:vlen), buff, 48 )
      IF (got_it) then
         istat = STR_UPCASE (buff, buff)
         line_cal_name(iaxis) = buff
         IF (line_cal_name(iaxis) .EQ. '360') 
     .        line_cal_name(iaxis) = '360_DAY' 
         IF (line_cal_name(iaxis) .EQ. 'STANDARD') 
     .        line_cal_name(iaxis) = 'GREGORIAN' 
         IF (line_cal_name(iaxis)(1:6) .EQ. 'COMMON') 
     .        line_cal_name(iaxis) = 'NOLEAP' 
      ENDIF
      cal_id = TM_GET_CALENDAR_ID (line_cal_name(iaxis))

* optional units for axis (may contain "tunits since T0_date" 2/95)
      line_units(iaxis) = ' '   ! default
      got_it = CD_GET_ATTRIB( cdfid, ivar, 'units', do_warn,
     .                        vname(:vlen), buff, 48 )
      since_T0 = MAX( INDEX(buff,'since'), INDEX(buff,'SINCE') )
      yyyymmdd_time = buff .EQ. 'yyyymmddhhmmss'

      IF ( since_T0 .GT. 2 ) THEN
         line_units(iaxis) = buff(:since_T0-1)  ! just the "tunits" part 
         IF ( TM_LENSTR1(buff) .GT. since_T0 + 5 ) THEN
            CALL TM_DATE_REFORMAT(buff(since_T0+6:),pdate_numslashnum,
     .                            cal_id, line_t0(iaxis), 
     .                            pdate_vax, status )
            IF ( status .NE. merr_ok ) GOTO 1000        ! 3/99 soft error 

*  Append calendar name to units string.
c           IF (cal_id .GT. gregorian .AND. cal_id .le. max_calendars) THEN
c                slen = TM_LENSTR1 (line_units(iaxis))
c                len_cal = TM_LENSTR1 (line_cal_name(iaxis))
c                line_units(iaxis) = line_units(iaxis)(:slen)//
c     .                 '('//line_cal_name(iaxis)(:len_cal)//')'
c           ENDIF

            got_torg = .TRUE.
         ELSE
            since_t0 = 0        ! flag that no T0 was present
         ENDIF
      ELSEIF ( yyyymmdd_time ) THEN
         line_units(iaxis) = 'days'
*  Append calendar name to units string.
c        IF (cal_id .GT. gregorian .AND. cal_id .le. max_calendars) THEN
c           slen = TM_LENSTR1 (line_units(iaxis))
c           len_cal = TM_LENSTR1 (line_cal_name(iaxis))
c           line_units(iaxis) = line_units(iaxis)(:slen)//
c     .                 '('//line_cal_name(iaxis)(:len_cal)//')'
c         ENDIF
         line_t0(iaxis) = ' ' ! to be determined when coords are read
         got_torg = .TRUE.
      ELSE
         line_units(iaxis) = buff
         got_torg = .FALSE.
      ENDIF

* EPIC time axes are encoded as 2 integers in variables TIME and TIME2
      IF ( its_epic .AND.  vupcase .EQ. 'TIME' ) THEN
         epic_t2var = NCVID( cdfid, 'time2', cdfstat )
         epic_time  = cdfstat .EQ. NCNOERR
      ELSE
         epic_time  = .FALSE.
      ENDIF
      IF ( epic_time ) line_units(iaxis) = 'Days'     ! arbitrary

* ... decode the units
      units = TM_UNIT_ID( line_units(iaxis) )
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

* optional "modulo" flag
      line_modulo(iaxis) = CD_GET_ATTRIB( cdfid, ivar, 'modulo',
     .                             do_warn, vname(:vlen), buff, 1 )

* optional time_origin and time_unit values for time axis
* (time origin could have been set by the units "... since time_origin")
      IF (.NOT.got_torg) got_torg = CD_GET_ATTRIB( cdfid, ivar,
     .          'time_origin', do_warn, vname(:vlen), line_t0(iaxis), 20 )
      IF ( .NOT.got_torg ) THEN
         IF ( epic_time ) THEN
            line_t0(iaxis) = '15-JAN-1901' ! arbitrary
         ELSEIF (since_t0 .LT. 2) THEN
            line_t0(iaxis) = char_init20
         ENDIF
      ENDIF

* a user can impose non-standard time units by setting the time_unit attribute
* (not fully tested, I suspect - *sh* 4/93)
      got_it = CD_GET_ATTVAL(cdfid,ivar,'time_unit',
     .           do_warn,vname(:vlen),line_tunit(iaxis),1,vback)
*!      got_it = CD_GET_ATTVAL(cdfid,ivar,'month_lengths',do_warn,

* if a time_origin was given then some units are required (4/93)
* default to "seconds" if no units were specified
      IF ( got_torg .AND. line_tunit(iaxis).EQ.real4_init ) THEN
           line_units(iaxis) = 'seconds'
           line_unit_code(iaxis) = TM_UNIT_ID( line_units(iaxis) )
           line_tunit(iaxis) = un_convert(line_unit_code(iaxis))
      ENDIF

* try to determine the orientation and direction of the axis
      line_direction(iaxis) = 'NA'
* ... first check units
      IF ( units .LT. 0 ) THEN
         line_direction(iaxis) = 'TI'  ! time
      ELSEIF ( units .EQ. 4 ) THEN     ! degrees
        IF  ( TM_HAS_STRING(line_units(iaxis), 'NORTH')
     .   .OR. TM_HAS_STRING(line_units(iaxis), 'SOUTH')
     .   .OR. TM_HAS_STRING(line_units(iaxis), 'LAT') ) THEN 
            line_direction(iaxis) = 'SN'
         ELSEIF ( TM_HAS_STRING(line_units(iaxis), 'EAST')
     .   .OR. TM_HAS_STRING(line_units(iaxis), 'WEST')
     .   .OR. TM_HAS_STRING(line_units(iaxis), 'LON') ) THEN 
            line_direction(iaxis) = 'WE'
         ELSEIF ( vupcase(1:1) .EQ. 'Y'
     .   .OR. INDEX(vupcase, 'LAT') .GT. 0 ) THEN
            line_direction(iaxis) = 'SN'
         ELSEIF ( vupcase(1:1) .EQ. 'X'
     .   .OR. INDEX(vupcase, 'LON') .GT. 0 ) THEN
            line_direction(iaxis) = 'WE'
! 8/97: "Y" anywhere in name as a last ditch attempt to id it
         ELSEIF ( INDEX(vupcase, 'Y')   .GT. 0 ) THEN
            line_direction(iaxis) = 'SN'
         ELSE
            line_direction(iaxis) = 'WE'
         ENDIF
      ELSEIF ( units .EQ. pun_millibars
     .   .OR.  units .EQ. pun_decibars ) THEN 
         line_direction(iaxis) = 'UD'
      ELSEIF ( units .EQ. 14
     .   .OR.  units .EQ. 15   ) THEN         ! layer or level
         line_direction(iaxis) = 'UD'
* ... then check axis name for clues
      ELSEIF ( INDEX(vupcase, 'TIME') .GT. 0
     .   .OR.  INDEX(vupcase, 'DATE') .GT. 0
     .   .OR.  vupcase .EQ. 'T'               ) THEN
         line_direction(iaxis) = 'TI'  ! time
      ELSEIF ( INDEX(vupcase, 'DEPTH') .GT. 0 ) THEN
         line_direction(iaxis) = 'UD'
      ELSEIF ( INDEX(vupcase, 'ELEV') .GT. 0
     .   .OR.  INDEX(vupcase, 'HEIGHT') .GT. 0
     .   .OR.  INDEX(vupcase, 'Z'   ) .GT. 0  ) THEN
         line_direction(iaxis) = 'DU'
* ... then check stray clues
      ELSEIF ( TM_HAS_STRING(line_units(iaxis), 'LAT') ) THEN
         line_direction(iaxis) = 'SN'
      ELSEIF ( TM_HAS_STRING(line_units(iaxis), 'LON') ) THEN
         line_direction(iaxis) = 'WE'
      ENDIF
* Note: there are some unfortunate ambiguities about how the TMAP library
*     uses line_direction:  it refers both to orientation and to the formatting
*     that should be used for the axis.  Thus an axis name of X* is sufficient
*     to pin down orientation but not to pin down formatting.  At a future
*     time it may be desirable to separate these two concepts into separate
*     variables and to make this routine (and TM_READ_GRIDFILE) sensitive to
*     axis names X*, Y*, and Z* (among others?).

* the attribute "positive=down", etc. may help
      got_it = CD_GET_ATTRIB( cdfid, ivar, 'positive',
     .                        do_warn, vname(:vlen), buff, 6 )
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

* longitude axes may be encoded "west" (discontinuous at 180, East negative,
*       West positive) or "east" (standard - 0->360 east from Greenwhich)
      west_lon = west_lon 
     .  .OR.    ( line_direction(iaxis) .EQ. 'WE'
     .      .AND. TM_HAS_STRING(line_units(iaxis),'west') )
     .  .OR.    ( its_epic
     .      .AND. vupcase .EQ. 'LON'
     .      .AND. .NOT.TM_HAS_STRING(line_units(iaxis),'east') )


* was a regular time axis forced on us by the user?
      IF (tregular) THEN
        IF (line_direction(iaxis).EQ."TI") line_regular(iaxis) = .TRUE.
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
         IF ( west_lon ) THEN
            line_start(iaxis) = TM_LONW2LONE( line_start(iaxis) ) 
            line_delta(iaxis) = TM_LONW2LONE( line_delta(iaxis) ) 
         ENDIF
         IF ( npts .GT. 1 ) THEN
* ... allow for reverse ordering of axis coordinates on regular-spaced axes
*     (the variable line_delta actually contains line_end at this moment)
            reversed(iaxis) = line_delta(iaxis) .LT. line_start(iaxis)
            IF ( reversed(iaxis) ) THEN
               tmp = line_delta(iaxis)
               line_delta(iaxis) =
     .           (line_start(iaxis)-line_delta(iaxis))/(npts-1)
               line_start(iaxis) = tmp
            ELSE
               line_delta(iaxis) =
     .           (line_delta(iaxis)-line_start(iaxis))/(npts-1)
            ENDIF
         ELSE
            reversed(iaxis) = .FALSE.
            line_delta(iaxis) = 1
         ENDIF
         line_subsc1(iaxis) = unspecified_int4
         last_coord = line_start(iaxis) + (npts-1)*line_delta(iaxis)
         firs_coord = line_start(iaxis)


* ============= START OF IRREGULAR AXIS PROCESSING ========================
      ELSE
* adequate storage for the irregular coordinates and box edges ?
         line_start(iaxis) = unspecified_val8
         line_delta(iaxis) = unspecified_val8
         IF ((next_line_mem_pos+2*npts).GT.maxlinestore) GOTO 5300

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
         ENDIF
         IF ( status .NE. merr_ok ) GOTO 1000   ! 3/99 soft error
         IF ( west_lon ) THEN
            DO 340 i=ipt1,ipte
 340        line_mem(i) = TM_LONW2LONE( line_mem(i) ) 
         ENDIF

* if the coordinates are decreasing-ordered then reverse them
         first_delta = line_mem(ipt1+1) - line_mem(ipt1)
         reversed(iaxis) = first_delta .LT. 0.0
         IF ( reversed(iaxis) ) THEN
            DO 345 i=ipt1,ipt1+npts/2-1
               tmp = line_mem(i)
               line_mem(i) = line_mem(ipte+ipt1-i)
               line_mem(ipte+ipt1-i) = tmp
 345        CONTINUE
         ENDIF
            
* Check for non-monotonic ordering of data
         DO 350 i=ipt1+1,ipte
           IF (line_mem(i) .LE. line_mem(i-1) ) THEN
* out of order axis coordinates
              words(1) = TM_LEFINT(i-ipt1+1, istat)
              CALL TM_NOTE(
     .             'Coordinates out of order or missing on axis '
     .              //vname(:vlen)//' at subscript '//words(1), tt)
              CALL TM_NOTE(
     .             'A dummy axis of subscripts will be used', tt)
              GOTO 1000
           ENDIF
  350   CONTINUE
        last_coord = line_mem(ipte)
        firs_coord = line_mem(ipt1)

* Check if axis is actually regularly spaced (though not flagged as "even")
        delta = line_mem(ipt1+1) - line_mem(ipt1)  ! sing. prec., only
        DO 360 i=ipt1+2,ipte
           IF (.NOT. TM_FPEQ( delta,
     .                   SNGL(line_mem(i)-line_mem(i-1))) ) GOTO 380
  360   CONTINUE
* yes, it's actually regular - save it as such
        line_regular(iaxis) = .TRUE.
        line_start  (iaxis) = line_mem(ipt1)
        line_delta  (iaxis) = 
     .              (line_mem(ipte)-line_mem(ipt1))/(npts-1)
        line_subsc1 (iaxis) = unspecified_int4
        last_coord = line_start(iaxis) + (npts-1)*line_delta(iaxis)
        firs_coord = line_start(iaxis)
        GOTO 390

* compute the default locations of boundaries between grid boxes (midpoints)
* Note: this may be modified by a subsequent "EDGES" definition
 380    CALL TM_IRREGAX_MID( ipt1, npts )

      ENDIF
* ============= END OF IRREGULAR AXIS PROCESSING ========================

* hack to cope with Gregorian/Julian calendar ambiguities (used by NOAA/CDC)
 390    IF ( (since_T0.GT.2)
     .  .AND.(line_t0(iaxis).EQ.'01-JAN-0001 00:00:00') 
     .  .AND.(cal_id .EQ. gregorian) ) THEN

* ... get time step of L=1
           IF ( line_regular(iaxis) ) THEN
              first = line_start(iaxis)   
           ELSE
              first = line_mem(ipt1)
           ENDIF

* ... compute seconds from 1-jan-0001
           secs2firststep = un_convert(line_unit_code(iaxis)) * first
* compute seconds from 1590
           secs1590 = TM_SECS_FROM_BC (gregorian, 1590,1,1,0,0,0,status)
            IF ( status .NE. merr_ok ) GOTO 5900

* ... correct axis if crosses 1590 threshhold
           IF (secs2firststep .GT. secs1590) line_t0(iaxis)
     .                                          = '30-DEC-0000 00:00:00'
        ENDIF

* Save the axis name (upper case) and memory storage     
      line_name(iaxis) = vupcase
      IF ( .NOT.line_regular(iaxis) ) THEN
             line_subsc1(iaxis) = next_line_mem_pos
             next_line_mem_pos =  next_line_mem_pos + 2*npts + 1
      ENDIF

* the attribute "edges=name" may point to a list of box boundaries
      got_it = CD_GET_ATTRIB( cdfid, ivar, 'edges',
     .                       do_warn, vname(:vlen), ename, 30 )
      IF ( .NOT.got_it ) GOTO 500
      IF ( line_regular(iaxis) ) THEN
         CALL TM_NOTE(
     .           'Evenly spaced axis has edges definition: '//
     .           vname(:vlen)//' - ignored', tt )
         GOTO 500
      ENDIF

****************************************************************************
*                              BOX EDGES                                   *
****************************************************************************

* does the variable pointed to really exist ?
        elen = TM_LENSTR1(ename)
        edgid = NCVID( cdfid, ename(:elen), cdfstat )
        IF (cdfstat.NE.NCNOERR) THEN
           soft_err = 11
           GOTO 5000
        ENDIF

* get the vital statistics of the edges variable
        CALL NCVINQ(cdfid, edgid, ename(:elen), vartype, nvdims,
     .               vdims, nvatts,cdfstat)
        IF ( nvdims .NE. 1 ) THEN
           soft_err = 12
           GOTO 5000
        ENDIF

* get axis length:
        CALL NCDINQ( cdfid, vdims, buff, npts, cdfstat )
        IF (cdfstat.NE.NCNOERR) GOTO 5100   ! should never happen

* check for error conditions
        IF ( npts .NE. line_dim(iaxis)+1 ) THEN
           soft_err = 13
           GOTO 5000
        ENDIF

* read the (npts+1) box edge values into double precision storage
* (Note: "npts" is now +1 larger than in the previous section)
        ipt1 = line_subsc1(iaxis)
        CALL CD_RD_R8_ARR( cdfid, edgid, 1, npts, vartype, vname(:vlen),
     .                     line_mem(ipt1+npts-1), status )
        IF ( status .NE. merr_ok ) GOTO 1000    ! 3/99 soft error

* if the coordinates were reversed then the box bounds must be, too
         IF ( first_delta .LT. 0.0 ) THEN
            DO 450 i=ipt1+npts-1,(ipt1+npts-1)+npts
 450        line_mem(i) = -1. * line_mem(i)
         ENDIF

* check to see that each data point is contained inside it's box
        IF (.NOT.TM_CHECK_BOX_LOC(iaxis)) THEN
           CALL TM_NOTE('Edges "'//ename(:elen)//
     .         '" do not enclose point on axis '//line_name(iaxis), tt)
           CALL TM_NOTE('Substituting coordinate midpoints', tt)
           CALL TM_IRREGAX_MID( line_subsc1(iaxis),
     .                          line_dim   (iaxis) )
        ENDIF

* if it is a longitude axis of 360 degrees let it default to MODULO
 500    IF (line_direction(iaxis).EQ.'WE' .AND. units.EQ.4) THEN
           npts = line_dim(iaxis)
           IF ( line_regular(iaxis) ) THEN
              delta = npts * line_delta(iaxis)
           ELSE
              ipt1  = line_subsc1(iaxis)
              delta = line_mem(ipt1+2*npts) - line_mem(ipt1+npts)
           ENDIF
           IF ( TM_FPEQ(delta,360.0) ) line_modulo(iaxis) = .TRUE.
        ENDIF

* if it is a calendar time axis it might be implicitly MODULO
* must have T0 in year 0000 or 0001, have a year length roughly correct,
* and have a final tstep in year 0000 or 0001

        IF (got_torg .AND. .NOT.line_modulo(iaxis)) THEN
           CALL TM_BREAK_DATE(line_t0(iaxis), cal_id, year, month, day,
     .                            hour, minute, second, istat)
           IF (istat.EQ.merr_ok .AND. year.LE.1) THEN
              t0_secs = TM_SECS_FROM_BC (cal_id, year, month, day,
     .                                   hour, minute, second, status)
              last_secs = t0_secs + last_coord*line_tunit(iaxis)
              yr_len = last_secs - 
     .                 t0_secs + firs_coord*line_tunit(iaxis)
              IF ( yr_len.GE.359.*psecs_per_day
     .       .AND. yr_len.LE.367.*psecs_per_day) 
     .             line_modulo(iaxis)
     .                   = last_secs .LE. 2.*366.*psecs_per_day
           ENDIF
        ENDIF

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
        GOTO 500

 5100   CALL TM_ERRMSG
     .     ( cdfstat+pcdferr, status, 'CD_GET_1_AXIS', cdfid, ivar,
     .       no_errstring, no_errstring, *5900 )

 5300   CALL TM_ERRMSG (merr_linstorlim, status, 'CD_GET_1_AXIS',
     .                  no_descfile, no_stepfile,
     .                  'MAX='//TM_STRING(DBLE(maxlinestore)),
     .                  no_errstring, *1000)                    ! 3/99 soft error

* error exit
 5900   RETURN
        END
