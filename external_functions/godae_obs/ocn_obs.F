* support code for Ferret EF taken from "program ocn_obs"
* *sh* 12/00

* original documentation follows
* for the Ferret EF application the code has been reduced to two subroutines
*   INTEGER FUNCTION OPEN_OCN_OBS(filename,errmsg)
*   SUBROUTINE READ_OCN_OBS(result, nvars, nobs, errmsg)

*** START OF ORIGINAL DOCUMENTATION ***
c
c.............................START PROLOGUE............................
c
c SCCS IDENTIFICATION:  %W% %G%
c
c MODULE NAME:  ocn_obs
c
c DESCRIPTION:  simple driver for processing FNMOC ocean obs files
c                  altim - TOPEX and ERS altimeter SSHA
c                  mcsst - AVHRR SST retrievals
c                profile - subsurface measures of in situ T and S
c                   ship - in situ surface measures of T
c                   ssmi - DMSP sea ice concentration retrievals
c
c               documentation of the variables for each ocean obs
c               file is contained within the respective "rd" routines:
c                  rd_altim, rd_mcssst, rd_prof, rd_ship, rd_ssmi
c
c               the first record in all of the FNMOC ocean obs files
c               contains the number the observations in that file
c               and the maximum number of observation levels (can be
c               greater than 1 for the {dtg}.profile files).  as in
c               this simple driver routine, these variables can be
c               read and passed as automatic array dimensions to a
c               subroutine.
c
c               the code assumes that the ocean obs data files are
c               in a directory structure given by
c                  $data_dir/altim
c                           /mcsst
c                           /profile
c                           /ship
c                           /ssmi
c               where $data_dir is an environmental variable
c               defined by "OCEAN_OBS_DIR" that describes the
c               directory path to the top of ocean obs tree
c
c               all of the code required is concatentated in
c               the ocn_obs.f file.  a compilation can be
c               done using the following (for SGI)
c                  f90 -r4 -i4 -o ocn_obs obs_obs.f
c
c               the GETARG and GETENV routines may be different
c               on SUN SPARC workstations
c
c               to see what command line arguments are required,
c               execute ocn_obs with no command line arguments
c
c               temperature units are deg C, salinity units are
c               PSU, sea surface height units are meters, and
c               sea ice concentration units are per cent.
c
c               unless otherwise specified, missing values are
c               set to -999 in all of the ocean obs data files
c
c....................MAINTENANCE SECTION................................
c
c MODULES CALLED:
c        Name                    Description
c   --------------     -------------------------------------
c   error_exit         standard error processing
c   GETARG             retrieve command line argument
c   GETENV             retrieve environmental variable
c   rd_altim           read satellite altimeter SSHA observations
c   rd_mcsst           read AVHRR satellite SST retrievals
c   rd_prof            read profile temperatures and salinities
c   rd_ship            read surface temperature observations
c   rd_ssmi            read ssmi sea ice retrievals
c
c..............................END PROLOGUE.............................
c

*** END OF ORIGINAL DOCUMENTATION ***


      SUBROUTINE READ_OCN_OBS(file_dtg, result, nvars, nobs, errmsg)

      INTEGER nvars, nobs
      REAL*4 result(nobs, nvars)
      character file_dtg * 10
      CHARACTER*(*) errmsg

      integer    UNIT
      parameter (UNIT = 60)

      call rd_ship (UNIT, file_dtg, result, nvars, nobs)
      close (UNIT)
      errmsg = ' '
      RETURN
      END
  
*
**********************
*

      INTEGER FUNCTION OPEN_OCN_OBS(file_dtg,err_msg)


      implicit  none
c
c     ..set local data file fortran unit number
c
      integer    UNIT
      parameter (UNIT = 60)
c
      character*(*) err_msg
      character file_dtg * 10

      character arg * 80
      character data_dir * 256
      logical   exist
      character file_name * 256
      integer   len
      integer   n_arg
      integer*4 n_lvl
      integer*4 n_obs
      character obs_typ * 5

        INTEGER TM_LENSTR
c
c...............................executable..............................
c

c
c     ..retrieve file root directory
c
        obs_typ = 'ship'

c
c     ..retrieve file root directory
c
      call GETENV ('OCN_OBS_DIR', data_dir)
      len = TM_LENSTR (data_dir)
      if (len .eq. 0) then
         data_dir = '.'
         len = TM_LENSTR (data_dir)
      endif
c
c-----------------------------------------------------------------------
c
c     ..profile observations
c
#ifdef 0
      if (obs_typ(1:4) .eq. 'prof') then
         open (45, file='report.profile', status='unknown',
     *             form='formatted')
         write (45, '(''   ****** Reading PROFILE Data ******'')')
         write (45, '(''   file date time group: '', a)') file_dtg
         write (45, '(''    data directory path: '', a)')
     *          data_dir(1:len)
         file_name = data_dir(1:len) // '/profile/' //
     *               file_dtg // '.profile'
         len = TM_LENSTR (file_name)
         inquire (file=file_name(1:len), exist=exist)
         if (exist) then
            open (UNIT, file=file_name(1:len), status='old',
     *                  form='unformatted')
            read (UNIT) n_obs, n_lvl
            write (45, '(''        number profiles: '', i10)') n_obs
            write (45, '(''      max number levels: '', i10)') n_lvl
            if (n_obs .gt. 0) then
               call rd_prof (UNIT, n_obs, n_lvl)
            endif
            close (UNIT)
         else
            write (err_msg, '(''file "'', a, ''" does not exist'')')
     *             file_name(1:len)
            call error_exit ('OCN_OBS', err_msg)
         endif
# endif
c
c-----------------------------------------------------------------------
c
c     ..surface ship/buoy observations
c
!      else if (obs_typ(1:4) .eq. 'ship') then
      if (obs_typ(1:4) .eq. 'ship') then
         open (45, file='report.ship', status='unknown',
     *             form='formatted')
         write (45, '(''   ****** Reading SHIP Data ******'')')
         write (45, '(''   file date time group: '', a)') file_dtg
         write (45, '(''    data directory path: '', a)')
     *          data_dir(1:len)
         file_name = data_dir(1:len) // '/ship/' //
     *               file_dtg // '.ship'
         len = TM_LENSTR (file_name)
         inquire (file=file_name(1:len), exist=exist)
         if (exist) then
            open (UNIT, file=file_name(1:len), status='old',
     *                  form='unformatted')
            read (UNIT) n_obs
            write (45, '(''        number ship obs: '', i10)') n_obs

c
c   successfully opened the file
c
            err_msg = ' '
            OPEN_OCN_OBS = n_obs
            RETURN
         else
            err_msg = 'file "'//file_name(:len)//'" does not exist'
            OPEN_OCN_OBS = 0
         endif
c
c-----------------------------------------------------------------------
c
c     ..mcsst observations
c
#ifdef 0
      else if (obs_typ(1:5) .eq. 'mcsst') then
         open (45, file='report.mcsst', status='unknown',
     *             form='formatted')
         write (45, '(''   ****** Reading MCSST Data ******'')')
         write (45, '(''   file date time group: '', a)') file_dtg
         write (45, '(''    data directory path: '', a)')
     *          data_dir(1:len)
         file_name = data_dir(1:len) // '/mcsst/' //
     *               file_dtg // '.mcsst'
         len = TM_LENSTR (file_name)
         inquire (file=file_name(1:len), exist=exist)
         if (exist) then
            open (UNIT, file=file_name(1:len), status='old',
     *                  form='unformatted')
            read (UNIT) n_obs
            write (45, '(''       number mcsst obs: '', i10)') n_obs
            if (n_obs .gt. 0) then
               call rd_mcsst (UNIT, n_obs)
            endif
            close (UNIT)
         else
            write (err_msg, '(''file "'', a, ''" does not exist'')')
     *             file_name(1:len)
            call error_exit ('OCN_OBS', err_msg)
         endif
# endif
c
c-----------------------------------------------------------------------
c
c     ..altimeter observations
c
#ifdef 0
      else if (obs_typ(1:5) .eq. 'altim') then
         open (45, file='report.altim', status='unknown',
     *             form='formatted')
         write (45, '(''   ****** Reading ALTIM Data ******'')')
         write (45, '(''   file date time group: '', a)') file_dtg
         write (45, '(''    data directory path: '', a)')
     *          data_dir(1:len)
         file_name = data_dir(1:len) // '/altim/' //
     *               file_dtg // '.altim'
         len = TM_LENSTR (file_name)
         inquire (file=file_name(1:len), exist=exist)
         if (exist) then
            open (UNIT, file=file_name(1:len), status='old',
     *                  form='unformatted')
            read (UNIT) n_obs
            write (45, '(''       number altim obs: '', i10)') n_obs
            if (n_obs .gt. 0) then
               call rd_altim (UNIT, n_obs)
            endif
            close (UNIT)
         else
            write (err_msg, '(''file "'', a, ''" does not exist'')')
     *             file_name(1:len)
            call error_exit ('OCN_OBS', err_msg)
         endif
# endif
c
c-----------------------------------------------------------------------
c
c     ..SSM/I sea ice observations
c
#ifdef 0
      else if (obs_typ(1:4) .eq. 'ssmi') then
         open (45, file='report.ssmi', status='unknown',
     *             form='formatted')
         write (45, '(''   ****** Reading SSMI Data ******'')')
         write (45, '(''   file date time group: '', a)') file_dtg
         write (45, '(''    data directory path: '', a)')
     *          data_dir(1:len)
         file_name = data_dir(1:len) // '/ssmi/' //
     *               file_dtg // '.ssmi'
         len = TM_LENSTR (file_name)
         inquire (file=file_name(1:len), exist=exist)
         if (exist) then
            open (UNIT, file=file_name(1:len), status='old',
     *                  form='unformatted')
            read (UNIT) n_obs
            write (45, '(''        number ssmi obs: '', i10)') n_obs
            if (n_obs .gt. 0) then
               call rd_ssmi (UNIT, n_obs)
            endif
            close (UNIT)
         else
            write (err_msg, '(''file "'', a, ''" does not exist'')')
     *             file_name(1:len)
            call error_exit ('OCN_OBS', err_msg)
         endif
# endif
c
c-----------------------------------------------------------------------
c
c     ..unknown obs data type 
c
      else
        err_msg = 'unknown obs TYPE'
        OPEN_OCN_OBS = 0
      endif
c
      stop
      end


      subroutine rd_altim (UNIT, n_obs)
c
c.............................START PROLOGUE............................
c
c SCCS IDENTIFICATION:  %W% %G%
c
c MODULE NAME:  rd_altim
c
c DESCRIPTION:  reads the ALTIM ocean obs files and produces a report
c
c PARAMETERS:
c      Name          Type        Usage            Description
c   ----------     ---------     ------    ---------------------------
c   n_obs           integer      input     number altim obs
c   unit            integer      input     FORTRAN unit number
c
c ALTIMETER VARIABLES:
c     ame       Type                     Description
c   --------  --------    ----------------------------------------------
c   ob_age     real       age of the observation in hours since
c                         January 1, 1992.  provides a continuous
c                         time variable
c   ob_clim    real       SSHA climatological estimate at obs
c                         location and sampling time
c   ob_cycle   integer    satellite cycle number
c   ob_dtg     character  SSHA obs date time group in the form
c                         year, month, day, hour, minute,
c                         second (YYYYMMDDHHMMSS)
c   ob_glbl    real       SSHA global analysis estimate at obs
c                         location and receipt time at FNMOC
c   ob_lat     real       SSHA obs latitude (south negative)
c   ob_lon     real       SSHA obs longitude (west negative)
c   ob_qc      real       SSHA obs probability of a gross error
c                         (assumes normal pdf of SSHA errors)
c   ob_regn    real       SSHA regional analysis estimate at obs
c                         location and receipt time at FNMOC
c   ob_sat     integer    satellite ID (TOPEX, ERS, GFO) - see
c                         ocn_types.h for codes
c   ob_sgma    real       climatological variability of SSHA at
c                         obs location and time of year
c   ob_smpl    integer    sequential sample number along a
c                         satellite track
c   ob_ssh     real       SSHA observation (meters) in terms of
c                         deviation from a long term TOPEX mean
c   ob_track   integer    satellite track number for cycle
c
c..............................END PROLOGUE.............................
c
      implicit none
c
      include 'ocn_types.h'
c
c     ..local array dimensions
c
      integer*4 n_obs
c
      integer   i, k
      real*4    ob_age (n_obs)
      real*4    ob_clim (n_obs)
      integer*4 ob_cycle (n_obs)
      character ob_dtg (n_obs) * 14
      real*4    ob_glbl (n_obs)
      real*4    ob_lat (n_obs)
      real*4    ob_lon (n_obs)
      real*4    ob_qc (n_obs)
      real*4    ob_regn (n_obs)
      integer*4 ob_smpl (n_obs)
      integer*4 ob_sat (n_obs)
      real*4    ob_sgma (n_obs)
      real*4    ob_ssh (n_obs)
      integer*4 ob_track (n_obs)
      integer   UNIT
c
c...............................executable..............................
c
c     ..read altimeter variables
c
      read (UNIT) (ob_age(i), i = 1, n_obs)
      read (UNIT) (ob_clim(i), i = 1, n_obs)
      read (UNIT) (ob_cycle(i), i = 1, n_obs)
      read (UNIT) (ob_glbl(i), i = 1, n_obs)
      read (UNIT) (ob_lat(i), i = 1, n_obs)
      read (UNIT) (ob_lon(i), i = 1, n_obs)
      read (UNIT) (ob_qc(i), i = 1, n_obs)
      read (UNIT) (ob_regn(i), i = 1, n_obs)
      read (UNIT) (ob_smpl(i), i = 1, n_obs)
      read (UNIT) (ob_sat(i), i = 1, n_obs)
      read (UNIT) (ob_sgma(i), i = 1, n_obs)
      read (UNIT) (ob_ssh(i), i = 1, n_obs)
      read (UNIT) (ob_track(i), i = 1, n_obs)
      read (UNIT) (ob_dtg(i), i = 1, n_obs)
c
c     ..produce altimeter report
c
      k = 100
      write (45, '(''  reporting skip factor: '', i10)') k
      write (45, '(11x,''dtg'', 5x,''lat'', 5x,''lon'', 5x,''sat'',
     *             5x,''ssh'', 4x,''clim'', 4x,''glbl'', 4x,''regn'',
     *             6x,''qc'',  3x,''cycle'', 4x,''trak'', 4x,''smpl'',
     *             4x,''sgma'')')
      do i = 1, n_obs, k
         write (45, '(a,2f8.2,i8,f8.4,3f8.3,f8.4,3i8,f8.4,2x,a)')
     *          ob_dtg(i), ob_lat(i), ob_lon(i), ob_sat(i),
     *          ob_ssh(i), ob_clim(i), ob_glbl(i), ob_regn(i),
     *          ob_qc(i), ob_cycle(i), ob_track(i), ob_smpl(i),
     *          ob_sgma(i), data_lbl(ob_sat(i))
      enddo
c
      return
      end


      subroutine rd_mcsst (UNIT, n_obs)
c
c.............................START PROLOGUE............................
c
c SCCS IDENTIFICATION:  %W% %G%
c
c MODULE NAME:  rd_mcsst
c
c DESCRIPTION:  reads the MCSST ocean obs files and produces a report
c
c PARAMETERS:
c      Name          Type        Usage            Description
c   ----------     ---------     ------    ---------------------------
c   n_obs           integer      input     number mcsst obs
c   unit            integer      input     FORTRAN unit number
c
c MCSST VARIABLES:
c      Name      Type                      Description
c   ----------  -------     --------------------------------------------
c   ob_age      real        age of the observation in hours since
c                           January 1, 1992.  provides a continuous
c                           time variable
c   ob_aod      real        NESDIS aeorosol optical depth at SST 
c                           obs and sampling time (to within +/- 
c                           24 hrs).  missing AODs are set to -1.
c   ob_clim     real        NCEP SST climatological estimate at obs
c                           location and sampling time
c   ob_csgm     real        NCEP SST climatology variability estimate 
c                           at obs location and receipt time at FNMOC
c                           (not implemented yet)
c   ob_dtg      character   SST obs date time group in the form year,
c                           month, day, hour, minute (YYYYMMDDHHMM)
c   ob_glbl     real        SST global analysis estimate at obs
c                           location and receipt time at FNMOC
c   ob_gsgm     real        global SST analysis variability estimate at
c                           obs location and receipt time at FNMOC
c                           (not implemented yet)
c   ob_lat      real        SST obs latitude (south negative)
c   ob_lon      real        SST obs longitude (west negative)
c   ob_qc       real        SST obs probability of a gross error
c                           (assumes normal pdf of SST errors)
c   ob_regn                 SST regional analysis estimate at obs
c                           location and receipt time at FNMOC
c   ob_rsgm     real        regional SST analysis variability estimate
c                           at obs location and receipt time at
c                           FNMOC (not implemented yet)
c   ob_sst      real        SST observation
c   ob_type     integer     SST obseration data type; NOAA14 or NOAA15
c                           day, night, relaxed day retrievals (see 
c                           ocn_types.h for codes)
c   ob_wm       integer     SST water mass indicator from Bayesian
c                           classification scheme.  only valid in
c                           western North Pacific, western North
c                           Atlantic or GIN-Seas regions.  used to
c                           prevent averaging of SST observations
c                           from different water mass during the
c                           computation of SST super-obs; helps
c                           maintain SST fronts and eddies in the
c                           analysis of western boundary currents
c
c..............................END PROLOGUE.............................
c
      implicit none
c
      include 'ocn_types.h'
c
c     ..local array dimensions
c
      integer*4 n_obs
c
      integer   i, k
      real*4    ob_age (n_obs)
      real*4    ob_aod (n_obs)
      real*4    ob_clim (n_obs)
      real*4    ob_csgm (n_obs)
      character ob_dtg (n_obs) * 12
      real*4    ob_glbl (n_obs)
      real*4    ob_gsgm (n_obs)
      real*4    ob_lat (n_obs)
      real*4    ob_lon (n_obs)
      real*4    ob_qc (n_obs)
      real*4    ob_regn (n_obs)
      real*4    ob_rsgm (n_obs)
      real*4    ob_sst (n_obs)
      integer*4 ob_type (n_obs)
      integer*4 ob_wm (n_obs)
      integer   UNIT
c
c...............................executable..............................
c
c     ..read mcsst variables
c
      read (UNIT) (ob_wm(i), i = 1, n_obs)
      read (UNIT) (ob_glbl(i), i = 1, n_obs)
      read (UNIT) (ob_lat(i), i = 1, n_obs)
      read (UNIT) (ob_lon(i), i = 1, n_obs)
      read (UNIT) (ob_age(i), i = 1, n_obs)
      read (UNIT) (ob_clim(i), i = 1, n_obs)
      read (UNIT) (ob_qc(i), i = 1, n_obs)
      read (UNIT) (ob_type(i), i = 1, n_obs)
      read (UNIT) (ob_regn(i), i = 1, n_obs)
      read (UNIT) (ob_sst(i), i = 1, n_obs)
      read (UNIT) (ob_aod(i), i = 1, n_obs)
      read (UNIT) (ob_dtg(i), i = 1, n_obs)
c     read (UNIT) (ob_csgm(i), i = 1, n_obs)
c     read (UNIT) (ob_gsgm(i), i = 1, n_obs)
c     read (UNIT) (ob_rsgm(i), i = 1, n_obs)
      do i = 1, n_obs
         ob_csgm(i) = -999.
         ob_gsgm(i) = -999.
         ob_rsgm(i) = -999.
      enddo
c
c     ..produce mcsst report
c
      k = 100
      write (45, '(''  reporting skip factor: '', i10)') k
      write (45, '(9x,''dtg'', 5x,''lat'', 5x,''lon'', 4x,''type'',
     *             5x,''sst'', 4x,''clim'', 4x,''glbl'', 4x,''regn'',
     *             6x,''qc'',  5x,''aod'', 4x,''csgm'', 4x,''gsgm'',
     *             4x,''rsgm'', 2x,''wm'')')
      do i = 1, n_obs, k
         write (45, '(a,2f8.2,i8,4f8.2,2f8.4,3f8.2,i4,2x,a)')
     *          ob_dtg(i), ob_lat(i), ob_lon(i), ob_type(i),
     *          ob_sst(i), ob_clim(i), ob_glbl(i), ob_regn(i),
     *          ob_qc(i), ob_aod(i), ob_csgm(i), ob_gsgm(i),
     *          ob_rsgm(i), ob_wm(i), data_lbl(ob_type(i))
      enddo
c
      return
      end


      subroutine rd_prof (UNIT, n_obs, n_lvl)
c
c.............................START PROLOGUE............................
c
c SCCS IDENTIFICATION:  %W% %G%
c
c MODULE NAME:  rd_prof
c
c DESCRIPTION:  reads the PROFILE ocean obs files and produces a report
c
c PARAMETERS:
c      Name          Type        Usage            Description
c   ----------     ---------     ------    ---------------------------
c   n_obs           integer      input     number profile obs 
c   n_lvl           integer      input     number profile levels
c   unit            integer      input     FORTRAN unit number
c
c PROFILE VARIABLES:
c     ame       Type                     Description
c   --------  --------    ----------------------------------------------
c   ob_btm     real       bottom depth in meters from DBDB5 data base
c                         at profile lat,lon
c   ob_clim    real       MODAS temperature climatology estimate at
c                         profile location, levels and sampling time
c   ob_dtg     character  profile observation sampling date time group
c                         in the form  year, month, day, hour, minute,
c                         second (YYYYMMDDHHMMSS)
c   ob_glbl    real       global analysis estimate of profile
c                         temperatures at profile obs location,
c                         levels, and sampling time
c   ob_lat     real       profile observation latitude (south negative)
c   ob_lon     real       profile observation longitude (west negative)
c   ob_ls      integer    number of observed profile salinity levels
c                         (a zero indicates temperature-only profile)
c   ob_lt      integer    number of observed profile temperature levels
c   ob_lvl     real       observed (and extended) profile levels
c   ob_modas   real       modas synthetic profile estimate at profile
c                         location, levels, and sampling time.  the
c                         predictor variables used in the generation
c                         of the modas synthetic profile are the
c                         ob_sst (SST) and ob_ssh (SSHA) variables.
c   ob_nd      integer    total number of levels for profile observation
c                         includes observed levels (ob_lt) plus levels
c                         to vertically extend the profile to the bottom
c   ob_rcpt    character  profile observation receipt time at FNMOC in
c                         the form year, month, day, hour, minute
c                         (YYYYMMDDHHMM); the difference between
c                         ob_rcpt and ob_dtg gives the timeliness 
c                         of the observation at FNMOC
c   ob_regn    real       regional analysis estimate of profile
c                         temperatures at profile obs location,
c                         levels, and sampling time
c   ob_sal     real       observed (and extended) profile salinities,
c                         if salinity has not been observed it has been
c                         estimated from climatological T/S regressions
c   ob_scr     character  profile obs security classification code; "U"
c                         for unclassified
c   ob_sign    character  profile observation call sign
c   ob_sprb    real       salinity profile level-by-level probability
c                         of a gross error
c   ob_sqc     real       salinity profile overall probability of gross
c                         error (integrates level-by-level errors taking
c                         into account layer thcknesses)
c   ob_ssh     real       SSHA of profile dynamic height from long-term
c                         hydrographic mean.  dynamic height has been
c                         calculated relative to 2000 m or the bottom
c                         whichever is shallower.  profile SSHA is
c                         directly comparable to satellite altimeter
c                         SSHA.  vertically extended levels may have
c                         been used in the dynamic height computation,
c                         so the profile SSHA values must be used with
c                         care for shallow observed profiles (ob_lt)
c   ob_sst     real       SST estimate (in order of high resoloution
c                         regional analysis if available, global
c                         analysis if available, profile SST if
c                         observed shallow enough or SST climatology
c                         (MODAS or GDEM)) valid at profile observation
c                         location and sampling time
c   ob_sstd    real       climatolgical estimates of variability of
c                         salinity at profile location, levels and
c                         sampling time (one standard deviation)
c   ob_tmp     real       observed (and extended) profile temperatures
c   ob_tqc     real       temperature profile overall probability of
c                         gross error (integrates level-by-level errors
c                         taking into account layer thcknesses)
c   ob_tprb    real       temperature profile level-by-level probability
c                         of a gross error
c   ob_tstd    real       climatolgical estimates of variability of
c                         temperature at profile location, levels and
c                         sampling time (one standard deviation)
c   ob_typ     integer    profile data type (see ocean_types.h for
c                         codes)
c 
c..............................END PROLOGUE.............................
c
      implicit none         
c
      include 'ocn_types.h'
c
c     ..local array dimensions
c
      integer*4 n_obs
      integer*4 n_lvl
c
      integer   i, j
      real*4    ob_btm (n_obs)
      real*4    ob_clim (n_lvl, n_obs)
      character ob_dtg (n_obs) * 12
      real*4    ob_glbl (n_lvl, n_obs)
      real*4    ob_lat (n_obs)
      real*4    ob_lon (n_obs)
      real*4    ob_lvl (n_lvl, n_obs)
      integer*4 ob_ls (n_obs)
      integer*4 ob_lt (n_obs)
      real*4    ob_modas (n_lvl, n_obs)
      integer*4 ob_nd (n_obs)
      real*4    ob_sprb (n_lvl, n_obs)
      real*4    ob_sqc (n_obs)
      real*4    ob_tprb (n_lvl, n_obs)
      real*4    ob_tqc (n_obs)
      character ob_rcpt (n_obs) * 12
      real*4    ob_regn (n_lvl, n_obs)
      character ob_scr (n_obs) * 1
      character ob_sign (n_obs) * 7
      real*4    ob_sal (n_lvl, n_obs)
      real*4    ob_ssh (n_obs)
      real*4    ob_sst (n_obs)
      real*4    ob_sstd (n_lvl, n_obs)
      real*4    ob_tmp (n_lvl, n_obs)
      real*4    ob_tstd (n_lvl, n_obs)
      integer*4 ob_typ (n_obs)
      integer   UNIT
c
c...............................executable..............................
c
c     ..read profile variables
c
      read (UNIT) (ob_lt(i),  i = 1, n_obs)
      read (UNIT) (ob_ssh(i), i = 1, n_obs)
      read (UNIT) (ob_btm(i), i = 1, n_obs)
      read (UNIT) (ob_lat(i), i = 1, n_obs)
      read (UNIT) (ob_lon(i), i = 1, n_obs)
      read (UNIT) (ob_nd(i),  i = 1, n_obs)
      read (UNIT) (ob_sqc(i), i = 1, n_obs)
      read (UNIT) (ob_tqc(i), i = 1, n_obs)
      read (UNIT) (ob_typ(i), i = 1, n_obs)
      do i = 1, n_obs
         read (UNIT) (ob_lvl(j,i), j = 1, ob_nd(i))
         read (UNIT) (ob_modas(j,i), j = 1, ob_nd(i))
         read (UNIT) (ob_clim(j,i), j = 1, ob_nd(i))
         read (UNIT) (ob_glbl(j,i), j = 1, ob_nd(i))
         read (UNIT) (ob_regn(j,i), j = 1, ob_nd(i))
         read (UNIT) (ob_sprb(j,i), j = 1, ob_nd(i))
         read (UNIT) (ob_tprb(j,i), j = 1, ob_nd(i))
         read (UNIT) (ob_sal(j,i), j = 1, ob_nd(i))
         read (UNIT) (ob_tmp(j,i), j = 1, ob_nd(i))
      enddo
      read (UNIT) (ob_dtg(i), i = 1, n_obs)
      read (UNIT) (ob_rcpt(i), i = 1, n_obs)
      read (UNIT) (ob_scr(i), i = 1, n_obs)
      read (UNIT) (ob_sign(i), i = 1, n_obs)
      read (UNIT) (ob_ls(i), i = 1, n_obs)
      read (UNIT) (ob_sst(i), i = 1, n_obs)
      do i = 1, n_obs
         read (UNIT) (ob_sstd(j,i), j = 1, ob_nd(i))
         read (UNIT) (ob_tstd(j,i), j = 1, ob_nd(i))
      enddo
c
c     ..produce profile report
c
      do i = 1, n_obs
         write (45, '(110(''-''))')
         write (45, '(''profile call sign           : "'', a, ''"'')')
     *          ob_sign(i)
         write (45, '(''profile latitude            : '', f12.2)')
     *          ob_lat(i)
         write (45, '(''profile longitude           : '', f12.2)')
     *          ob_lon(i)
         write (45, '(''profile observed DTG        : "'', a, ''"'')')
     *          ob_dtg(i)
         write (45, '(''profile received DTG        : "'', a, ''"'')')
     *          ob_rcpt(i)
         write (45, '(''DBDB5 bottom depth          : '', f12.1)')
     *          ob_btm(i)
         write (45, '(''profile data type code      : '', i12)')
     *          ob_typ(i)
         write (45, '(''profile data type           : "'', a, ''"'')')
     *          data_lbl(ob_typ(i))
         write (45, '(''total number levels         : '', i12)')
     *          ob_nd(i)
         write (45, '(''observed temperature levels : '', i12)')
     *          ob_lt(i)
         write (45, '(''observed salinity levels    : '', i12)')
     *          ob_ls(i)
         write (45, '(''temperature gross error     : '', f12.4)')
     *          ob_tqc(i)
         write (45, '(''salinity gross error        : '', f12.4)')
     *          ob_sqc(i)
         write (45, '(''sea surface height anomaly  : '', f12.4)')
     *          ob_ssh(i)
         write (45, '(''sea surface temperature     : '', f12.2)')
     *          ob_sst(i)
         write (45, '(''security classification     : '', 9x,
     *          ''"'', a, ''"'')') ob_scr(i)
         write (45, '(5x,''depth'', 6x,''temp'', 6x,''salt'',
     *               3x,''tmp_std'', 3x,''sal_std'',
     *               2x,''tmp_prob'', 2x,''sal_prob'',
     *               6x,''clim'', 5x,''modas'',
     *               6x,''glbl'', 6x,''regn'')')
         do j = 1, ob_nd(i)
            write (45, '(f10.1, 4f10.2, 2f10.4, 4f10.2)')
     *             ob_lvl(j,i), ob_tmp(j,i), ob_sal(j,i),
     *             ob_tstd(j,i), ob_sstd(j,i), ob_tprb(j,i),
     *             ob_sprb(j,i), ob_clim(j,i), ob_modas(j,i),
     *             ob_glbl(j,i), ob_regn(j,i)
            if (ob_nd(i) .gt. ob_lt(i)) then
               if (j .eq. ob_lt(i)) then
                  write (45, '(10x, 100(''-''))') 
               endif
            endif
         enddo
      enddo
c
      return
      end


      subroutine rd_ship (UNIT, file_dtg, result, nvars, n_obs)
c
c.............................START PROLOGUE............................
c
c SCCS IDENTIFICATION:  %W% %G%
c
c MODULE NAME:  rd_ship
c
c DESCRIPTION:  reads the SHIP ocean obs files and produces a report
c
c PARAMETERS:
c      Name          Type        Usage            Description
c   ----------     ---------     ------    ---------------------------
c   n_obs           integer      input     number ship obs
c   unit            integer      input     FORTRAN unit number
c
c MCSST VARIABLES:
c      Name      Type                     Description
c   ----------  -------     -------------------------------------------
c   ob_age      real        age of the observation in hours since
c                           January 1, 1992.  provides a continuous
c                           time variable
c   ob_clim     real        NCEP SST climatological estimate at obs
c                           location and sampling time
c   ob_csgm     real        NCEP SST climatology variability estimate
c                           at obs location and receipt time at FNMOC
c                           (not implemented yet)
c   ob_dtg      character   SST obs date time group in the form year,
c                           month, day, hour, minute (YYYYMMDDHHMM)
c   ob_glbl     real        SST global analysis estimate at obs
c                           location and receipt time at FNMOC
c   ob_gsgm     real        global SST analysis variability estimate
c                           at obs location and receipt time at FNMOC
c                           (not implemented yet)
c   ob_lat      real        SST obs latitude (south negative)
c   ob_lon      real        SST obs longitude (west negative)
c   ob_qc       real        SST obs probability of a gross error
c                           (assumes normal pdf of SST errors)
c   ob_rcpt     character   SST observation receipt time at FNMOC in
c                           the form year, month, day, hour, minute
c                           (YYYYMMDDHHMM); the difference between
c                           ob_rcpt and ob_dtg gives the timeliness 
c                           of the observation and the validity of
c                           ob_glbl and ob_regn background estimates 
c   ob_regn                 SST regional analysis estimate at obs
c                           location and receipt time at FNMOC
c   ob_rsgm     real        regional SST analysis variability estimate
c                           at obs location and receipt time at FNMOC
c                           (not implemented yet)
c   ob_scr      character   SST obs security classification code; "U"
c                           for unclassified
c   ob_sign     character   SST observation call sign
c   ob_sst      real        SST observation
c   ob_type     integer     SST obseration data type; ship (ERI, bucket,
c                           hull contact), buoy (fixed, drifting), CMAN
c                           (see ocn_types.h for codes)
c   ob_wm       integer     SST water mass indicator from Bayesian
c                           classification scheme.  only valid in
c                           western North Pacific, western North
c                           Atlantic or GIN-Seas regions.  used to
c                           prevent averaging of SST observations
c                           from different water mass during the
c                           computation of SST super-obs; helps
c                           maintain SST fronts and eddies in the
c                           analysis of western boundary currents
c
c..............................END PROLOGUE.............................
c
      implicit none
c
      include 'ocn_types.h'
c
c     ..local array dimensions
c
      integer*4 n_obs
      integer*4 nvars
      character file_dtg * 10
      REAL*4 result(n_obs, nvars)
c
      integer   i
      real*4    ob_age (n_obs)
      real*4    ob_clim (n_obs)
      real*4    ob_csgm (n_obs)
      character ob_dtg (n_obs) * 12
      real*4    ob_glbl (n_obs)
      real*4    ob_gsgm (n_obs)
      real*4    ob_lat (n_obs)
      real*4    ob_lon (n_obs)
      real*4    ob_qc (n_obs)
      character ob_rcpt (n_obs) * 12
      real*4    ob_regn (n_obs)
      real*4    ob_rsgm (n_obs)
      character ob_scr (n_obs) * 1
      character ob_sign (n_obs) * 6
      real*4    ob_sst (n_obs)
      integer*4 ob_type (n_obs)
      integer*4 ob_wm (n_obs)
      integer   UNIT

      integer iyr ,imo ,idy ,ihr ,imn,
     .        iyr0,imo0,idy0,ihr0,imn0
      REAL*8          TM_SECS_FROM_BC

      integer
     .  POB_WM,
     .  POB_GLBL,
     .  POB_LAT,
     .  POB_LON,
     .  POB_AGE,
     .  POB_CLIM,
     .  POB_QC,
     .  POB_REGN,
     .  POB_SST,
     .  POB_TYPE,
     .  POB_DTG,
     .  POB_RCPT,
     .  POB_SCR,
     .  POB_SIGN

      PARAMETER (
     .  POB_WM = 2,
     .  POB_GLBL = 3,
     .  POB_LAT = 4,
     .  POB_LON = 5,
     .  POB_AGE = 6,
     .  POB_CLIM = 7,
     .  POB_QC = 8,
     .  POB_REGN = 9,
     .  POB_SST = 10,
     .  POB_TYPE = 11,
     .  POB_DTG = 12,
     .  POB_RCPT = 13,
     .  POB_SCR = 14,
     .  POB_SIGN = 15 )

c
c...............................executable..............................
c
c     ..read ship variables
c
      read (UNIT) (ob_wm(i), i = 1, n_obs)
      read (UNIT) (ob_glbl(i), i = 1, n_obs)
      read (UNIT) (ob_lat(i), i = 1, n_obs)
      read (UNIT) (ob_lon(i), i = 1, n_obs)
      read (UNIT) (ob_age(i), i = 1, n_obs)
      read (UNIT) (ob_clim(i), i = 1, n_obs)
      read (UNIT) (ob_qc(i), i = 1, n_obs)
      read (UNIT) (ob_regn(i), i = 1, n_obs)
      read (UNIT) (ob_sst(i), i = 1, n_obs)
      read (UNIT) (ob_type(i), i = 1, n_obs)
      read (UNIT) (ob_dtg(i), i = 1, n_obs)
      read (UNIT) (ob_rcpt(i), i = 1, n_obs)
      read (UNIT) (ob_scr(i), i = 1, n_obs)
      read (UNIT) (ob_sign(i), i = 1, n_obs)
c     read (UNIT) (ob_csgm(i), i = 1, n_obs)
c     read (UNIT) (ob_gsgm(i), i = 1, n_obs)
c     read (UNIT) (ob_rsgm(i), i = 1, n_obs)
      do i = 1, n_obs
         ob_csgm(i) = -999.
         ob_gsgm(i) = -999.
         ob_rsgm(i) = -999.
      enddo

* RETURN THE RESULTS TO FERRET
      do  i = 1, n_obs
        result(i,1) = -999.
        result(i,POB_WM)   = ob_wm(i)
        result(i,POB_GLBL) = ob_glbl(i)
        result(i,POB_LAT)  = ob_lat(i)
        result(i,POB_LON)  = ob_lon(i)
        result(i,POB_AGE)  = ob_age(i)
        result(i,POB_CLIM) = ob_clim(i)
        result(i,POB_QC)   = ob_qc(i)
        result(i,POB_REGN) = ob_regn(i)
        result(i,POB_SST)  = ob_sst(i)
        result(i,POB_TYPE) = ob_type(i)
* compute lag from observation to receipt of observation
        READ ( ob_rcpt(i),'(I4,4I2)') iyr ,imo ,idy ,ihr ,imn
        READ ( ob_dtg (i),'(I4,4I2)') iyr0,imo0,idy0,ihr0,imn0
        result(i,POB_DTG)  =
     .          (TM_SECS_FROM_BC(iyr,imo,idy,ihr,imn,0)
     .        -  TM_SECS_FROM_BC(iyr0,imo0,idy0,ihr0,imn0,0)) / 3600.

        result(i,POB_RCPT) = -999.   ! not needed
        result(i,POB_SCR)  = -999.   ! ob_scr(i)
        result(i,POB_SIGN) = -999.   ! ob_sign(i) -- useful?
      enddo

* return the number of obs in result(1,1)
      result(1,1) = n_obs

* return the numerical-encoded filename in result(2,1)
      READ (file_dtg,'(I4,4I2)') iyr,imo,idy,ihr,imn
      result(2,1) = (TM_SECS_FROM_BC(iyr,imo,idy,ihr,imn,0)
     .            -  TM_SECS_FROM_BC(1992,1,1,0,0,0)) / 3600.

c
c     ..produce ship report
c
      write (45, '(9x,''dtg'', 3x,''sign '', 5x,''lat'', 5x,''lon'',
     *             4x,''type'', 5x,''sst'', 4x,''clim'', 4x,''glbl'',
     *             4x,''regn'', 6x,''qc'', 4x,''csgm'', 4x,''gsgm'',
     *             4x,''rsgm'', 2x,''wm'', 2x,''sc'')')
      do i = 1, n_obs
         write (45, '(a,2x,a,2f8.2,i8,4f8.2,f8.4,3f8.2,i4,3x,a,2x,a)')
     *          ob_dtg(i), ob_sign(i), ob_lat(i), ob_lon(i),
     *          ob_type(i), ob_sst(i), ob_clim(i), ob_glbl(i),
     *          ob_regn(i), ob_qc(i), ob_csgm(i), ob_gsgm(i),
     *          ob_rsgm(i), ob_wm(i), ob_scr(i), data_lbl(ob_type(i))
      enddo
c
      return
      end


      subroutine rd_ssmi (UNIT, n_obs)
c
c.............................START PROLOGUE............................
c
c SCCS IDENTIFICATION:  %W% %G%
c
c MODULE NAME:  rd_ssmi
c
c DESCRIPTION:  reads the SSMI sea ice ocean obs files and produces a
c               report whether you want one or not
c
c PARAMETERS:
c      Name          Type        Usage            Description
c   ----------     ---------     ------    ---------------------------
c   n_obs           integer      input     number ssmi obs
c   unit            integer      input     FORTRAN unit number
c
c SSMI VARIABLES:
c     ame       Type                     Description
c   --------  --------    ----------------------------------------------
c   ob_age     real       age of the observation in hours since
c                         January 1, 1992.  provides a continuous
c                         time variable
c   ob_clim    real       ECMWF sea ice climatological estimate at
c                         SSM/I sea ice location and sampling time
c   ob_dtg     character  SSM/I sea ice retrieval date time group in
c                         the form  year, month, day, hour, minute,
c                         second (YYYYMMDDHHMMSS)
c   ob_glbl    real       SSM/I sea ice global analysis estimate at
c                         obs location and receipt time at FNMOC
c   ob_ice     real       SSM/I sea ice concentration (per cent) 
c   ob_lat     real       SSM/I sea ice latitude (south negative)
c   ob_lon     real       SSM/I sea ice longitude (west negative)
c   ob_mean    real       SSM/I 30 day mean sea ice concentration at
c                         obs location and samping time
c   ob_qc      real       SSM/I sea ice probability of a gross error
c                         (assumes normal pdf of sea ice retrieval
c                         errors)
c   ob_regn    real       SSM/I sea ice regional analysis estimate at
c                         obs location and receipt time at FNMOC
c   ob_sat     integer    satellite ID (DMSP F11, F13, F14, F15);
c                         see ocn_types.h for codes
c
c..............................END PROLOGUE.............................
c
      implicit none
c
      include 'ocn_types.h'
c
c     ..local array dimensions
c
      integer*4 n_obs
c
      integer   i, k
      real*4    ob_age (n_obs)
      real*4    ob_clim (n_obs)
      character ob_dtg (n_obs) * 12
      real*4    ob_glbl (n_obs)
      real*4    ob_ice (n_obs)
      real*4    ob_lat (n_obs)
      real*4    ob_lon (n_obs)
      real*4    ob_qc (n_obs)
      real*4    ob_regn (n_obs)
      integer*4 ob_sat (n_obs)
      real*4    ob_mean (n_obs)
      integer   UNIT
c
c...............................executable..............................
c
c     ..read ssmi variables
c
      read (UNIT) (ob_glbl(i), i = 1, n_obs)
      read (UNIT) (ob_ice(i), i = 1, n_obs)
      read (UNIT) (ob_lat(i), i = 1, n_obs)
      read (UNIT) (ob_lon(i), i = 1, n_obs)
      read (UNIT) (ob_qc(i), i = 1, n_obs)
      read (UNIT) (ob_age(i), i = 1, n_obs)
      read (UNIT) (ob_regn(i), i = 1, n_obs)
      read (UNIT) (ob_sat(i), i = 1, n_obs)
      read (UNIT) (ob_clim(i), i = 1, n_obs)
      read (UNIT) (ob_mean(i), i = 1, n_obs)
      read (UNIT) (ob_dtg(i), i = 1, n_obs)
c
c     ..produce ssmi report
c
      k = 1000
      write (45, '(''  reporting skip factor: '', i10)') k
      write (45, '(9x,''dtg'', 5x,''lat'', 5x,''lon'', 5x,''sat'',
     *             5x,''ice'', 4x,''clim'', 4x,''glbl'', 4x,''regn'',
     *             4x,''mean'', 6x,''qc'')')
      do i = 1, n_obs, k
         write (45, '(a,2f8.2,i8,5f8.2,f8.4,2x,a)')
     *          ob_dtg(i), ob_lat(i), ob_lon(i), ob_sat(i),
     *          ob_ice(i), ob_clim(i), ob_glbl(i), ob_regn(i),
     *          ob_mean(i), ob_qc(i), data_lbl(ob_sat(i))
      enddo
c
      return
      end


      subroutine error_exit (routine, message)
c
c.............................START PROLOGUE............................
c
c SCCS IDENTIFICATION:  %W% %G%
c
c MODULE NAME:  error_exit
c
c DESCRIPTION:  prints a fatal error message and terminates the program.
c
c PARAMETERS:
c    Name          Type       Usage            Description
c   -------     ----------    ------    ---------------------------
c   routine     char * (*)    input     name of routine
c   message     char * (*)    input     user supplied error message
c
c..............................END PROLOGUE.............................
c
      implicit  none
c
      integer   ln
      character message * (*)
      character routine * (*)

        INTEGER TM_LENSTR1

c
c..............................executable...............................
c
c     ..determine message string length
c
      ln = TM_LENSTR1 (message)
c
      write (*, '(//, ''*** FATAL ERROR ('', a, '') ***'')') routine
      write (*, '(/, a)') message(1:ln)
      write (*, '(/, ''*** PROGRAM TERMINATED ***'', /)')
c
c     ..exit with non zero completion code
c
      stop 15
      end
