	SUBROUTINE CD_OPEN_DSET ( dset, cdfid, remote_dods_id, status)

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
* Open a netCDF file and stores preliminaries in TMAP XDSET_INFO common

* Programmer Steve Hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program

* revision 0.0 - 2/92
* 2/93 - ds_aux_modnum replaced by cd_what_isit (equivalenced (but unused))
* 3/93 - initialize ds_type for PMEL EPIC CDF file
* 11/94 - ignore the global "history" attribute.  Use "title_mod" instead.
* *sh* 3/2/95 - informative messages issued through TM_NOTE (for GUI)
*
* *kob* 7/95 - added logic for multi-file netcdf datasets
* *kob* 3/21/96 - add in changes originally made 7/21/95 by -sh- to 
*		 initialize ds_basic_axes as per TMAP model output
* *kob* 6/4/96	- prevent title from being overwritten for mc datasets
* *kob* 6/18/96 - bug fix in mc loop - loop w/ maxstepfiles rather than
*		  maxdsets.  Also reorganize loop for efficiency
*		- clarify error message at 5300
* 10/96 *kob* - Linux port 
*		- had to move variable assignments off of
*		  declaration line.
*		- hsd to declare hex constants as implicit
*		  integers in DATA statements.  Added a 
*		  FORTRAN_90 ifdef to accomplish this.
* *sh* 11/12/96 - changes to accomodate caching of DODS data 
* *kob* 4/99 - increase size of TM_INQ_PATH, cdf_file and raw_file 
*              to accomodate lenghty DODS dataset names
* 4/03 *acm* - Use STR_SAME to compare strings

* argument definitions:
*       dset    - pointer to TMAP data set
*       cdfid   - returned pointer to open CDF file (or cache)
*       remote_dods_id   - returned pointer to open remote URL (or zero)
*	status	- return status

* argument declarations:
	INTEGER	      dset, cdfid, remote_dods_id, status

* include files
        include 'netcdf.decl'
#include "netcdf.inc"
        include 'tmap_errors.parm'
	include 'tmap_dims.parm'
#include "gt_lib.parm"
#include "tmap_dset.parm"
	include 'xstep_files.cmn_text'
        external xstep_files_data
	include 'xdset_info.cmn_text'
	external xdset_info_data
        include 'xio.cmn_text'
        external xio_data
        include 'xdset_info.cd_equiv'
! kob 7/95
	include 'xsys_names.cmn_text'	            


* local variable declarations:
* *kob* explicit definitions not allowed on decl line in linux...
*       moved it to parameter line 10/96
      LOGICAL CD_GET_ATTRIB, CD_ISIT_EPIC, TM_HAS_STRING, got_it,
     .        do_warn, file_exists
      PARAMETER (do_warn = .TRUE. )
      INTEGER TM_HASH_NAME, NCOPN, TM_LENSTR1, STR_SAME,
     .        ndims, nvars, ngatts, recdim, cdfstat, i, istep
      INTEGER flen, iin, iout, ilast, dlen
      CHARACTER    TM_STRING*13, ch*1
* *kob* 4/99
* *kob* 11/00 - increase size of below to accomidate DODS files in mc sets
      CHARACTER*512 TM_INQ_PATH, cdf_file, raw_file

#ifdef FORTRAN_90
	INTEGER HEX40
	DATA HEX40/Z'40'/
#endif

****** DODS *********
* If this is a DODS remote file - see if it is cached (11/96)
	remote_dods_id = -1	! the default: "NO"
	IF ( ds_des_name(dset)(1:7) .EQ. 'http://') THEN

* ... have they specifically indicated "(no cache)"
	   IF ( TM_HAS_STRING(ds_des_name(dset),'(no cach') ) THEN
* ... do not use cache!
* ... crude parse: remove everything beyond last "("
	      dlen = TM_LENSTR1(ds_des_name(dset))
	      DO 5 iin = dlen, 1, -1
	         IF (ds_des_name(dset)(iin:iin) .NE. '(' ) THEN
	            ds_des_name(dset)(iin:iin) = ' '
	         ELSE
	            ds_des_name(dset)(iin:iin) = ' '
	            GOTO 50
	         ENDIF
 5	      CONTINUE
	   ENDIF

* ... create the full file name path for the cache file
	   dlen = TM_LENSTR1(ds_des_name(dset))
	   raw_file = ' '
	   iout = 1				
	   DO 10 iin = 8, dlen			! just past the "http://"
	      ch = ds_des_name(dset)(iin:iin)	
	      IF ( ch .EQ. PathSeparator ) THEN
	         raw_file(iout:iout) = '_'
	      ELSE
	         raw_file(iout:iout) = ch
	      ENDIF
	      iout = iout + 1
 10	   CONTINUE

* ... does it exist?
	   cdf_file = TM_INQ_PATH( raw_file, 'FER_DODS_CACHE_DIR',
     .		' ', .FALSE., status )
	   IF ( status .EQ. merr_notmounted ) GOTO 50
	   IF ( status .NE. merr_ok ) GOTO 5900		! syntax err

* ... open the remote URL in order to initialize the DODS client libraries
* This time penalty is unavoidable (though HTTP caching may eliminate it)
* Using the cache file will remove the need to read remote coordinates
	   CALL TM_NOTE( 'Using cache '//cdf_file, tooplu)
	   remote_dods_id = NCOPN(ds_des_name(dset), NCNOWRIT, cdfstat)
	   IF ( cdfstat .NE. NCNOERR ) GOTO 5150
	   ds_des_name(dset) = cdf_file	! replace remote with cache!
	ENDIF
****** END DODS *********

* open the netCDF file and get its vital statistics
 50      CALL NCPOPT(0)  ! no error processing by netCDF
* check for mutli-file netcdf datasets then loop through the list
* of stepfiles to find the first one that is on line    *kob* 7/95
* * kob*  bug fix in loop logic 9/18/96
        IF (STR_SAME(ds_type(dset), '  MC') .EQ. 0) THEN
 		DO 100 i=1,maxstepfiles
		     IF (sf_setnum(i) .NE. dset) GOTO 100
		     cdf_file = TM_INQ_PATH(sf_name(i),tmap_tsteps,
     .                                      ' ',.FALSE., status)
		     IF (status .EQ. merr_ok)  THEN
	   	         cdfid = NCOPN(cdf_file, NCNOWRIT, cdfstat)
			 mc_init_stepfile = i
			 IF ( cdfstat .EQ. NCNOERR) GOTO 110
		     ENDIF
 100	        CONTINUE
* error exit now.....no file on line found after looping through
* all stepfiles
		GOTO 5300		
	else
	   	cdfid = NCOPN(ds_des_name(dset), NCNOWRIT, cdfstat)
	endif
	        IF ( cdfstat .NE. NCNOERR ) GOTO 5100
 110    CALL NCINQ( cdfid, ndims, nvars, ngatts, recdim, cdfstat )
        IF ( cdfstat .NE. NCNOERR ) GOTO 5100

* Default contents as per TMAP "Background record"
* if this is a MC data set, we don't want to reset these values
        IF (STR_SAME(ds_type(dset), '  MC') .NE. 0) THEN
		ds_title(dset)		= ' '
		ds_mod_title(dset)	= ' '
		ds_forcing_set(dset)	= ' '
		ds_t0time(dset)		= ' '
		ds_ntegrate_tstep(dset)	= 1
		ds_time_run(dset)	= ' '
		DO 120 i = 1,num_aux
  120		cd_what_isit(i,dset)= ' '
	ENDIF

* time axis
	ds_time_axis(dset) = unspecified_int4

* Load message record
	ds_alert_on_open(dset) 	= .FALSE.
	ds_alert_on_output(dset)= .FALSE.
	ds_message(dset)	= ' '

* initialize incidental variables
        ds_basic_axes(1,dset) = mpsxt
        ds_basic_axes(2,dset) = mpsxu
        ds_basic_axes(3,dset) = mpsyt
        ds_basic_axes(4,dset) = mpsyu
        ds_basic_axes(5,dset) = mpszt
        ds_basic_axes(6,dset) = mpszw

* save the "record" axis
        ds_time_axis(dset) = recdim

* does it have a (readable) title ?
* *kob* 6/96 - don't do this if we have a mc dataset - title comes
*	       from the descriptor rather than the cdf file
        IF (STR_SAME(ds_type(dset), '  MC') .NE. 0) THEN
        	got_it = CD_GET_ATTRIB( cdfid, NCGLOBAL, 'title',
     .                     .NOT.do_warn, ' ', ds_title(dset), 80 )
	ENDIF
* title modifier ?
        got_it = CD_GET_ATTRIB( cdfid, NCGLOBAL, 'title_mod',
     .                  .NOT.do_warn, ' ', ds_mod_title(dset), 80 )

* does it have a message to display upon opening ?
        ds_alert_on_output(dset) = .FALSE.
        ds_alert_on_open(dset) = 
     .       CD_GET_ATTRIB( cdfid, NCGLOBAL, 'message',
     .                  .NOT.do_warn, ' ', ds_message(dset), 80 )
* holler the message
        IF ( ds_alert_on_open(dset) ) THEN
	   CALL TM_NOTE('regarding '//
     .		ds_des_name(dset)(:TM_LENSTR1(ds_des_name(dset)))
     .		//' ...', tooplu)
	   CALL TM_NOTE(ds_message(dset), tooplu)
	ENDIF

* special TMAP parameters ?
* ... read global attributes beginning with "TMAP"

* Default contents as per TMAP "Format record" (is it EPIC? 3/93)
* -kob- only set ds_type to cdf if it isn't already set to MC 7/95

        IF ( CD_ISIT_EPIC(dset, cdfid, status) ) THEN
           ds_type(dset) = 'ECDF'
        ELSEIF ( ds_type(dset) .NE. '  MC' ) THEN
           ds_type(dset) = 'CDF'
        ENDIF
	ds_format(dset)	= '  1A'

* "unique" 2 letter code for this data set
        ds_prog_rev(dset) = ' '
        i = TM_HASH_NAME( ds_name(dset), 26*26-1 )
#ifdef FORTRAN_90
        ds_prog_rev(dset)(1:1) = CHAR( IOR(HEX40,i/26 + 1  ) )
        ds_prog_rev(dset)(2:2) = CHAR( IOR(HEX40,MOD(i,26) + 1 ) )
#else
        ds_prog_rev(dset)(1:1) = CHAR( IOR('40'X,i/26 + 1  ) )
        ds_prog_rev(dset)(2:2) = CHAR( IOR('40'X,MOD(i,26) + 1 ) )
#endif
* Allocate a "step file" slot to save CDF id and filename
* Find the next storage position for stepfiles in common
* only if the file is not a multi-file netcdf dataset
        IF (STR_SAME(ds_type(dset), '  MC') .NE. 0) THEN
c commented out diagnostic write statements
c	  write(6,*) 'sf_setnum......',sf_setnum(1)
c	  write(6,*) 'set_not_open...',set_not_open
          DO 190 istep = 1,maxstepfiles
            IF (sf_setnum(istep) .EQ. set_not_open) GOTO 200
 190      CONTINUE
* No room left for new stepfiles
          GOTO 5200

 200      sf_lunit ( istep ) = cdfid
          sf_setnum( istep ) = dset
          sf_name  ( istep ) = ds_des_name(dset) 
          cd_stepfile( dset ) = istep    ! step file containing desired data
          ds_open  ( dset )   = istep    ! which step file is currently open
	ENDIF

* successful completion
	status = merr_ok
	RETURN

* error messages
 5100   CALL TM_ERRMSG
     .     ( cdfstat+pcdferr, status, 'CD_OPEN_DSET', cdfid,
     .        no_varid, 'is this a CDF file ?', no_errstring, *5900 )

 5150   CALL TM_ERRMSG
     .     ( cdfstat+pcdferr, status, 'CD_OPEN_DSET', cdfid,
     .        no_varid, 'Unable to access remote URL', no_errstring,
     .		*5900 )

 5200   CALL TM_ERRMSG ( merr_filim, status, 'CD_OPEN_DSET',
     .                     dset, no_stepfile,
     .                     'MAX='//TM_STRING(DBLE(maxstepfiles)),
     .                     no_errstring, *5900)

*kob* 9/18
 5300  CALL TM_ERRMSG (merr_notmounted,status, 'CD_OPEN_DSET',dset,
     .                     no_stepfile,
     .                    'No files on-line from data set',
     .			   no_errstring,
     .                     *5900)
* erase any record of tampering with this data set
 5900   CALL TM_CLOSE_SET( dset, i )

* error exit
	RETURN
	END
