	SUBROUTINE TM_INIT_DSET(dset_name,perm,dset_num,tregular,status)

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
* Initialize a TMAP data set:  GT, TS, or CDF
* For GT and TS: opens dataset DESCRIPTOR/SET file,reads the FORMAT_RECORD and
* BACKGROUND_RECORD, reads in any GRID files given, then determines correct
* reading routine to load common blocks XDOtSET_INFO and XSTEP_FILES with vital
* system and background information
* For netCDF file allocates a data set slot and proceded to CD_ routines

* Programmer Mark Verschell (from programs by Steve Hankin)
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX/VMS
*
* revision 0.00 - 06/23/88 - From GT_INIT_DSET version 1.20
* revision 0.01 - 11/21/88 - variable not_used removed (wasn't used)
* revision 0.02 - 12/12/88 - removed arguement dset_num from TM_READ_GRIDFILE
* revision 0.03 - 02/15/89 - changed GT_CLOSE_SET to TM_CLOSE_SET, eliminated
* revision 0.04 - 02/17/89 - Added initialization of D_TIME_MODULO
* revision 0.05 - 04/17/89 - Added message in case file not descriptor file
* revision 0.06 - 01/10/91 - added ifdef unix for default file *sh*
* Unix/RISC port *sh* 3/91 - found bug: CLOSE called on error when lunit wasn't
*                          - opened.  Added logical "opened" and "got_lun"
*                          - use TM_INQ_PATH for (nearly) OS-independent access
* SUN port   - 1/20/92 kob --- moved all NAMELIST definitions to immediately
*                             after declarations.  SUN Fortran didn't like them
*                             embedded in executable code
* netCDF support *sh* 2/92 - added new calling argument dset_type
*                          - added TM_CLEAN_FILENAME
* 12/93 *sh* - bug fix: check for filename ending ".cdf" BEFORE determining
*			    which path list to use
*  4/94 *sh* - ... and if no match try the other choice (.des v.s..cdf) 
*  5/94 *sh* - recognize URL-style names as netCDF names (DODS networking)
*
* 6/2/94 *kob* - IBM Port
*		  Need to initialize d_time_modulo with .FALSE. instead of 0
*		  Need ifdef READONLY_ALLOWED around readonly in open statemnt
* *sh* 3/2/95 - informative messages issued through TM_NOTE (for GUI)
* *sh* 11/96 - recognize .nc as a NetCDF file and ignore "(no cache)"
* Linux port -kob - 3/97 - Modify include for tmap_dset.parm to be a
*			   preprocessor include for F90 - needed because
*			   tmap_dset.parm contains an ifdef
* *kob* 7/97 - modification to eliminate unnecessary error messages when
*              searching for files w/out extensions (ie ".cdf")
* *kob* 4/99 - increase size of TM_INQ_PATH and des_name
*              to accomodate lenghty DODS dataset names
* *sh*  6/99 - added "perm" (permutation) argument
* V510 *sh* 3/00 - added tregular argumant
*	    4/00 - converted to dynamic grids -- bump use counts here
* v530 *kob* 1/00 - minor mods for remote mc aggregations
* v530 *acm* 1/01 - initialize ds_cal_name  calendar name
* v540 *acm* 10/01 increase length of ds_name
* v540 *acm* 10/01  remove VMS include statements
* v541 *acm*  4/02  New descriptor d_caltype for nonstandard calendar types.
*                   **also changes to descript.def**
*      *acm*  5/02  When extension not specified, try .nc as well as .des, .cdf

* arguments:
*	dset_name	- filename of DESCRIPTOR file (input)
*	dset_num	- number assigned to this data set (output)
*			  ( 1 <= dset_num <= maxdsets - see COMMON XDSET_INFO)
*	dset_type	- coaching for data set type (GT,TS, or CDF)
*                       - blank to determine based on filename extension
*	status		- return status
*

* Argument definitions
*	CHARACTER*(*)	dset_name, dset_type  removed 4/92 kob
	LOGICAL		tregular
	INTEGER		perm(4), dset_num, status
	CHARACTER*(*)	dset_name

* Parameter and common files
	include 'tmap_errors.parm'		! error codes
#include "tmap_dset.parm"
	include 'tmap_dims.parm'		! data set dimensions
#include "gt_lib.parm"
	include 'xio.parm'
	include 'xdset_info.cmn_text'	! data set common
	external xdset_info_data
        include 'xsys_names.cmn_text'
        external xsys_names_data
        include 'xdiag_ctrl.cmn_text'
        external xdiag_ctrl_data

* Define variables from Set File namelist records
	include 'descript.def'

* Parameter for number of known types and number of format_revs
	INTEGER*4	num_known, num_revs
	PARAMETER	(num_known = 5)
	PARAMETER	(num_revs  = 1)

* Internal declarations
	CHARACTER*4	known_type(num_known)
	CHARACTER*4	format_rev(num_revs,num_known)
	CHARACTER*13	TM_STRING
	CHARACTER	def_ext*4, def_cdf*4, def_nc*3, ext*4, 
     .                  cdf_path*12, path*12
        CHARACTER*256   TM_CLEAN_FILENAME
	CHARACTER*42	reading_nml
	CHARACTER*256   TM_INQ_PATH, des_name
	INTEGER*4	set_cnt, lunit, cnt, aux_cnt, tcnt, dlen, dlen2,
     .			rcnt, parm_cnt, grd_cnt, equal_pos, istat
	INTEGER*4	TM_LENSTR, TM_LENSTR1
	LOGICAL		TM_HAS_STRING, match,
     .			opened, got_lun, dods_url, nocache,
     .                  explicit_ext                 ! 7/97
	DATA		known_type/'  GT','  TS','  MC','  MG','  MT'/
	DATA		format_rev/'  1C',
     .				   '  1A',
     .				   '    ',
     .				   '    ',
     .				   '    '/
	DATA		def_ext  /'.des'/, def_cdf/'.cdf'/
        DATA            def_nc/'.nc'/

        DATA            cdf_path/'FER_DATA'/

* namelist declarations
* record 1 - descriptor format record
	NAMELIST /FORMAT_RECORD/d_type,d_format,d_source_class,
     .				d_source,d_subsource,d_prog_rev
* record 2 - file background record
	NAMELIST /BACKGROUND_RECORD/d_expnum,d_modnum,d_title,
     .				   d_mod_title,d_forcing_set,d_t0time,
     .				   d_time_unit,d_time_modulo,
     .				   d_ntegrate_tstep,d_ntuple,d_time_run,
     .				   d_who_ranit,d_time_made,d_who_madeit,
     .				   d_aux_modnum,d_add_parm,
     .				   d_grid_filename,d_caltype

* initialize
        got_lun = .FALSE.    ! controls error clean-up
        opened = .FALSE.     ! controls error clean-up
	explicit_ext = .FALSE.  !controls missing file error reporting 7/97
* is this a network data request based on a DODS URL?
	dods_url = INDEX(dset_name,'://') .GT. 0
	IF ( dods_url ) THEN
	   nocache = TM_HAS_STRING( dset_name, 'no cache' )
	ELSE
	   nocache = .FALSE.
	ENDIF
	IF ( nocache ) THEN
	   dlen = INDEX(dset_name,'(') - 1
	   IF (dlen .LE. 0 ) GOTO 5300
	   dlen = TM_LENSTR1( dset_name(:dlen) )	  ! length up to "(no cache)"
	ELSE
	   dlen = TM_LENSTR1( dset_name )
	ENDIF

* search for TMAP-formatted data set or NetCDF ?
* (ds_dset_type may have been set externally to 'CDF' already)
* 5/94: also recognize (e.g.) dods://dcz.gso.uri.edu/usr/data/gsc.nc
* 11/96: also recognize ".nc" (and insist that "cdf" be at the name end
* 7/97: Set explict_ext if there is an extension, and also check for
*       a .des extension.  
	IF ( dods_url ) ds_dset_type = 'CDF'
	IF ( dlen .GT. 2 ) THEN
	   IF (TM_HAS_STRING(dset_name(dlen-2:dlen), '.nc')) THEN
     		ds_dset_type = 'CDF'
	        explicit_ext = .TRUE.
	   ENDIF
	ENDIF
	IF ( dlen .GT. 3 ) THEN
	   IF (TM_HAS_STRING(dset_name(dlen-3:dlen), '.cdf')) THEN
     		ds_dset_type = 'CDF'
	        explicit_ext = .TRUE.
	   ENDIF
	   IF (TM_HAS_STRING(dset_name(dlen-3:dlen), '.des')) THEN
	        explicit_ext = .TRUE.
	   ENDIF
	ENDIF
        IF ( ds_dset_type .EQ. 'CDF' ) THEN
           ext  = def_cdf       ! default netCDF file extension
           path = cdf_path
        ELSE
           ext  = def_ext       ! default TMAP file extension
           path = tmap_sets    ! FER_DESCR
        ENDIF
* if there is an explicit extension, we don't want to pass the default
* extension on, because the error messages are misleading *kob*
	IF ( explicit_ext ) ext = ' '    ! 7/97

* Find full file specification using INQUIRE with dset_name
* OS-independent path search
	IF ( dods_url ) THEN
	   des_name = dset_name
	   status = merr_OK
	ELSE
* 7/97 *kob* pass explicit_ext into tm_inq_path to indicate whether
*            or not an error message should be output in tm_path_list
           des_name = TM_INQ_PATH(dset_name,path,ext,explicit_ext,
     .                    status)
	ENDIF
	IF ( status .NE. merr_ok .AND. explicit_ext ) GOTO 9950
	IF ( status .NE. merr_ok) THEN
* interchange and try the other (.des/.cdf) data set type
	   IF ( ext .EQ. def_ext ) THEN
	      ext  = def_cdf	! was .des
	      path = cdf_path
	      ds_dset_type = 'CDF'
	   ELSE
	      ext  = def_ext	! was .cdf
	      path = tmap_sets
	      ds_dset_type = ' '
	   ENDIF
	   des_name = TM_INQ_PATH(dset_name,path,ext,.FALSE.,status)

* 4/02 *acm* try .nc extension
	   IF ( status .NE. merr_ok ) THEN
	      ext  = def_nc
	      path = cdf_path
	      ds_dset_type = 'CDF'
           ENDIF
	   des_name = TM_INQ_PATH(dset_name,path,ext,.FALSE.,status)


	   IF ( status .NE. merr_ok ) THEN
	      ext  = def_nc
	      path = tmap_sets
	      ds_dset_type = 'CDF'
           ENDIF
	   des_name = TM_INQ_PATH(dset_name,path,ext,.FALSE.,status)



* 7/97 *kob* check to see which error message should be output - don't
*            want to output same error message twice.
	   IF ( status .NE. merr_ok ) THEN

	      IF (status .EQ. merr_syntax .OR. status .EQ. merr_erreq) THEN
		 GOTO 9950
	      ELSE 
* 7/97 *kob* if file contains a path, only that path was searched, not all 
*            default paths
		 IF ( INDEX(dset_name,PathSeparator) .GT. 0 ) THEN
		    GOTO 5500
		 ELSE
		    GOTO 5400
		 ENDIF
	      ENDIF
	   ENDIF
	ENDIF
* check to see if this descriptor has already been initialized
	DO 100 set_cnt = 1, maxdsets
	   IF (nocache) THEN
* ... imperfect logic: "abcd(no cache)" will match "abc"
	      dlen2 = TM_LENSTR1(ds_des_name(set_cnt))
	      match = dlen2 .EQ. dlen
     .	        .AND. ds_des_name(set_cnt) .EQ. des_name(:dlen)
	   ELSE
	      match = ds_des_name(set_cnt) .EQ. des_name
	   ENDIF
	   IF ( match ) THEN
	      dset_num = set_cnt
	      status = merr_no_action
	      GOTO 9999
	    ENDIF
  100	CONTINUE

* Assign data set number, exiting if maximum number of data sets are open
	DO 110 set_cnt = 1,maxdsets
	  IF (ds_name(set_cnt) .EQ. char_init40) THEN
	    dset_num = set_cnt
	    ds_des_name(dset_num) = des_name
            ds_name(dset_num) = TM_CLEAN_FILENAME( dset_name )
	    GOTO 120
	  ENDIF
  110	CONTINUE
* Too many data sets are open
	CALL TM_ERRMSG ( merr_dsetlim, status, 'TM_INIT_DSET',
     .			 no_descfile, no_stepfile,
     .			 'MAX='//TM_STRING(DBLE(maxdsets)),
     .			 no_errstring, *9950)

* netCDF file ?
 120    IF ( ds_dset_type .EQ. 'CDF' ) THEN
           ds_dset_type = ' '     ! reset the coaching shared by outside world
           CALL CD_INIT_DSET( dset_num, perm, tregular, status )
           IF ( status .NE. merr_ok ) GOTO 9900
           RETURN
        ENDIF

* Get free VMS logical unit # for temporary open
 	CALL LIB_GET_LUN (lunit)
        got_lun = .TRUE.

* Open descriptor file for reading
	OPEN   (UNIT		= lunit,
     .		FILE		= des_name,
#ifdef READONLY_ALLOWED
     .		READONLY,
#endif
     .		STATUS		= 'OLD',
     .		ERR 		= 5000)
        opened = .TRUE.
* record 1 - descriptor format record


	reading_nml = 'FORMAT_RECORD; or file not descriptor file'
	READ  (lunit, NML=FORMAT_RECORD, END=5100, ERR=5200)

* check to see if we know how to handle this data descriptor
	DO 130 tcnt = 1,num_known
	  IF (d_type .EQ. known_type(tcnt)) GOTO 140
  130	CONTINUE
	CALL TM_ERRMSG (merr_unktype, status, 'TM_INIT_DSET',
     .			dset_num, no_stepfile,
     .			'Type declared '//d_type,
     .			no_errstring, *9900)

  140	DO 150 rcnt = 1,num_revs
	DO 155 tcnt = 1,num_known
	  IF (d_format .EQ. format_rev(rcnt,tcnt)) GOTO 160
  155	CONTINUE
  150	CONTINUE
* ... des_name used as a buffer ...
	CALL TM_NOTE( 'BEWARE: Unrecognized descriptor format '//
     .		   d_format, tooplu ) 
!	WRITE (tooplu,1000) d_format, format_rev(1,tcnt)
! 1000	FORMAT(' Unrecognized descriptor format ',A4,'.'/
!     .	       ' Should be at least ',A4,'.'/
!     .	       ' Routines may not work correctly')

  160	ds_type(dset_num)         = d_type
	ds_format(dset_num)       = d_format
	ds_source_class(dset_num) = d_source_class
	ds_source(dset_num)       = d_source
	ds_subsource(dset_num)    = d_subsource
	ds_prog_rev(dset_num)     = d_prog_rev



* Initialize some BACKROUND variables
	d_mod_title = ' '
	d_ntuple    = 1
* 6/94 initialize d_time_modulo with .FALSE instead of 0 for IBM *kob*
	d_time_modulo = .FALSE.
	DO 170 cnt = 1,num_aux
  170	d_aux_modnum(cnt)    = ' '
	DO 180 cnt = 1,num_parm
  180	d_add_parm(cnt)      = ' '
	DO 190 cnt = 1,num_gridfiles
  190	d_grid_filename(cnt) = ' '


        d_caltype       = 'GREGORIAN'

	reading_nml = 'BACKGROUND_RECORD'
	READ  (lunit, NML=BACKGROUND_RECORD, END=5100, ERR=5200)

* Check for unsupported features
	IF (d_ntuple .NE. 1.) CALL TM_ERRMSG
     .		(merr_notsupport, status, 'TM_INIT_DSET',
     .		 dset_num, no_stepfile,
     .		 'D_NTUPLE in descriptor file must = 1',
     .		 no_errstring, *9900)

	ds_expnum(dset_num)         = d_expnum
	ds_modnum(dset_num)         = d_modnum
	ds_title(dset_num)          = d_title
	ds_mod_title(dset_num)      = d_mod_title
	ds_forcing_set(dset_num)    = d_forcing_set
	ds_t0time(dset_num)         = d_t0time
	ds_time_unit(dset_num)      = d_time_unit
	ds_time_modulo(dset_num)    = d_time_modulo
	ds_ntegrate_tstep(dset_num) = d_ntegrate_tstep
	ds_ntuple(dset_num)         = d_ntuple
	ds_time_run(dset_num)       = d_time_run
        ds_cal_name(dset_num)       = d_caltype

	DO 200 aux_cnt = 1,num_aux
	  ds_aux_modnum(aux_cnt,dset_num)    = d_aux_modnum(aux_cnt)
  200	CONTINUE

* Break add_parm into parm_name and parm_text
	DO 210 parm_cnt = 1, num_parm
	  equal_pos = INDEX(d_add_parm(parm_cnt),'=')
	  IF (equal_pos .EQ. 0) THEN
	    IF (TM_LENSTR(d_add_parm(parm_cnt)) .GT. 0) THEN
	      ds_parm_name(parm_cnt,dset_num) = d_add_parm(parm_cnt)
	      ds_parm_text(parm_cnt,dset_num) = ' '
	    ELSE
	      ds_parm_name(parm_cnt,dset_num) = ' '
	      ds_parm_text(parm_cnt,dset_num) = ' '
	    ENDIF
	  ELSE
	    ds_parm_name(parm_cnt,dset_num) = 
     .			d_add_parm(parm_cnt)(1:equal_pos-1)
	    ds_parm_text(parm_cnt,dset_num) =
     .			d_add_parm(parm_cnt)(equal_pos+1:)
	  ENDIF
  210	CONTINUE

* Read in any gridfiles associated with data set
	DO 220 grd_cnt = 1,num_gridfiles
	  IF (d_grid_filename(grd_cnt) .NE. ' ') THEN
	    CALL TM_READ_GRIDFILE(d_grid_filename(grd_cnt),status)
	    IF ( status .EQ. merr_no_action ) GOTO 220
	    IF ( status .NE. merr_ok) GOTO 9900
	  ENDIF
  220	CONTINUE

* Call specific _INIT_DSET's now
	IF (d_type .EQ. '  GT') THEN
	  CALL GT_INIT_DSET(dset_num, lunit, status)
	  IF (status .NE. merr_ok) GOTO 9900
	  CALL TM_MAKE_4D_GRIDS(dset_num, status)
* ... bump the use counts on grids and axes used by variables in this data set
	  CALL TM_DSET_USE_GRIDS( dset_num )
	ELSE IF (d_type .EQ. '  TS') THEN
	  CALL TS_INIT_DSET(dset_num, lunit, status)
	  IF (status .NE. merr_ok) GOTO 9900
	  CALL TM_MAKE_4D_GRIDS(dset_num, status)
* ... bump the use counts on grids and axes used by variables in this data set
	  CALL TM_DSET_USE_GRIDS( dset_num )
!*kob*
	ELSE IF (d_type .EQ. '  MC') THEN
	  CALL MC_INIT_DSET(dset_num, lunit, perm, tregular, status)
	  IF (status .NE. merr_ok) GOTO 9900
	  CALL TM_MAKE_4D_GRIDS(dset_num,status)
* 1/00 - need to do this for remote mc aggregations
	  CALL TM_DSET_USE_GRIDS( dset_num )
	ENDIF
!	IF (d_type .EQ. '  MG') CALL MG_INIT_DSET(dset_num, lunit, status)
!	IF (d_type .EQ. '  MT') CALL MT_INIT_DSET(dset_num, lunit, status)
	IF (status .NE. merr_ok) GOTO 9900

* success
	GOTO 9990

* errors
 5000	CALL TM_ERRMSG (merr_erreq, status, 'TM_INIT_DSET',
     .			dset_num, no_stepfile,
     .			no_errstring, no_errstring, *9900)

 5100	CALL TM_ERRMSG (merr_nmlerr, status, 'TM_INIT_DSET',
     .			dset_num, no_stepfile,
     .			'Didn''t find: '//reading_nml,
     .			no_errstring, *9900)

 5200	CALL TM_ERRMSG (merr_nmlerr, status, 'TM_INIT_DSET',
     .			dset_num, no_stepfile,
     .			'Error in: '//reading_nml,
     .			no_errstring, *9900)

 5300	CALL TM_ERRMSG (merr_syntax, status, 'TM_INIT_DSET',
     .			dset_num, no_stepfile,
     .			dset_name,
     .			no_errstring, *9900)

 5400	CALL TM_ERRMSG (merr_notmounted, status, 'TM_INIT_DSET',
     .			unspecified_int4, no_stepfile,
     .			'Searched $FER_DESCR and $FER_DATA for '//dset_name//
     .                  '('//def_ext//','//def_cdf//',.nc)',
     .			no_errstring, *9950)

 5500	CALL TM_ERRMSG (merr_notmounted, status, 'TM_INIT_DSET',
     .			unspecified_int4, no_stepfile,
     .			dset_name//'('//def_ext//','//def_cdf//',.nc)',
     .			no_errstring, *9950)


* get out from error
 9900	CALL TM_CLOSE_SET ( dset_num, istat )
 9950	dset_num = unspecified_int4

* get out
 9990	IF ( opened ) CLOSE (UNIT = lunit)
	IF ( got_lun) CALL LIB_FREE_LUN (lunit)
 9999	RETURN
	END
