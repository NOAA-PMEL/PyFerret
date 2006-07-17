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
* 9/04 *acm* - Fix bug 982 If same variable name in multiple descriptor files, 
*              mc_init_stepfile needs to know which file its from.
* V581: *acm*/3/05 - increase length of strings for dataset title to 1024 
* V600 *acm* 6/05 - Attribute control. just open the dataset here; getting info
*                   is done in cd_load_dset_attrs. See sequence of calls in 
*                   cd_init_dset.
* V600 *acm* 7/06 - To return correct netCDF error messages, call NF_OPEN and
*                   NF_STRERROR instead of NCOPN. Requires DODS library v3.5.2 or
*                   higher.

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

#ifdef usingDODSf2cUnderscore
	INTEGER NF_OPEN_
	CHARACTER*80 NF_STRERROR_
#endif 

* local variable declarations:
* *kob* explicit definitions not allowed on decl line in linux...
*       moved it to parameter line 10/96

      LOGICAL TM_HAS_STRING, got_it,
     .        do_warn, file_exists
      PARAMETER (do_warn = .TRUE. )
      REAL val
      INTEGER TM_HASH_NAME, TM_LENSTR1, STR_SAME,
     .        ndims, nvars, ngatts, recdim, cdfstat, i, istep
      INTEGER flen, iin, iout, ilast, dlen
      CHARACTER ch*1
* *kob* 4/99
* *kob* 11/00 - increase size of below to accomidate DODS files in mc sets
      CHARACTER*512 TM_INQ_PATH, cdf_file, raw_file
      CHARACTER*80 TM_FMT, buff  ! for nf_ error message
      CHARACTER*10 errcode

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
cc	   remote_dods_id = NCOPN(ds_des_name(dset), NCNOWRIT, cdfstat)

#ifdef usingDODSf2cUnderscore 
           cdfstat = NF_OPEN_(ds_des_name(dset), NF_NOWRITE, cdfid) 
#else
           cdfstat = NF_OPEN(ds_des_name(dset), NF_NOWRITE, cdfid) 
#endif 

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
cc	   	         cdfid = NCOPN(cdf_file, NCNOWRIT, cdfstat)

#ifdef usingDODSf2cUnderscore 
			 cdfstat = NF_OPEN_(cdf_file, NF_NOWRITE, cdfid)
#else
			 cdfstat = NF_OPEN(cdf_file, NF_NOWRITE, cdfid) 
#endif 

			 mc_init_stepfile(dset) = i
			 IF ( cdfstat .EQ. NCNOERR) GOTO 110
		     ENDIF
 100	        CONTINUE

#endif 

* error exit now.....no file on line found after looping through
* all stepfiles
		GOTO 5300
                
	ELSE
cc	   	cdfid = NCOPN(ds_des_name(dset), NCNOWRIT, cdfstat)
                
#ifdef usingDODSf2cUnderscore 
	   	cdfstat = NF_OPEN_(ds_des_name(dset), NF_NOWRITE, cdfid) 
#else
	   	cdfstat = NF_OPEN(ds_des_name(dset), NF_NOWRITE, cdfid) 
#endif 

	ENDIF
	        IF ( cdfstat .NE. NCNOERR ) GOTO 5100
 110   CONTINUE

       status = merr_ok

       RETURN

* error messages
 5100   CONTINUE
#ifdef usingDODSf2cUnderscore 
        buff =  NF_STRERROR_(cdfstat)
#else
        buff =  NF_STRERROR(cdfstat)
#endif  
        dlen = TM_LENSTR1(buff)
        val = cdfstat
        errcode = TM_FMT(val, 0, 10, len)
        buff(dlen+2:)  = '(OPeNDAP/netCDF Error code '//
     .      errcode(1:TM_LENSTR1(errcode))//')'
        dlen = TM_LENSTR1(buff)


        CALL TM_ERRMSG
     .     ( merr_nc_open, status, 'CD_OPEN_DSET', dset,
     .        no_stepfile, buff, no_errstring, *5900 )

 5150   CALL TM_ERRMSG
     .     ( merr_nc_open, status, 'CD_OPEN_DSET', dset,
     .        no_varid, 'Unable to access remote URL', no_errstring,
     .		*5900 )
*kob* 9/18
 5300   CONTINUE

#ifdef usingDODSf2cUnderscore 
        buff =  NF_STRERROR_(cdfstat)
#else
        buff =  NF_STRERROR(cdfstat)
#endif
        dlen = TM_LENSTR1(buff)
        val = cdfstat
        errcode = TM_FMT(val, 0, 10, len)
        buff(dlen+2:)  = '(OPeNDAP/netCDF Error code '//
     .      errcode(1:TM_LENSTR1(errcode))//')'
        dlen = TM_LENSTR1(buff)

        CALL TM_ERRMSG (merr_notmounted,status, 'CD_OPEN_DSET',dset,
     .    no_stepfile,
     .    'No files on-line from data set. '//buff(1:dlen),
     .    no_errstring,
     .    *5900)
* erase any record of tampering with this data set
 5900   CALL TM_CLOSE_SET( dset, i )

* error exit
	RETURN
	END
