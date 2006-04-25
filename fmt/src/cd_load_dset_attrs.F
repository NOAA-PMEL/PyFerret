	SUBROUTINE CD_LOAD_DSET_ATTRS ( dset, cdfid, status)

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
* get attribute information from linked lists, and store in TMAP XDSET_INFO common
* (based on cd_open_dset)

* Programmer Ansley Manke
* NOAA/PMEL, Seattle, WA - TMAP
* V600 6/05 *acm* Attribute control.


* argument definitions:
*       dset    - pointer to TMAP data set
*	status	- return status

* argument declarations:
	INTEGER	      dset, status

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
      INTEGER TM_HASH_NAME, TM_LENSTR1, STR_SAME,
     .        cdfid, ndims, nvars, ngatts, recdim, cdfstat, 
     .        i, istep, varid, iatt, attlen, attoutflag, maxlen
      CHARACTER    TM_STRING*13, name*128
      REAL val
      LOGICAL CD_ISIT_EPIC, NC_GET_ATTRIB, got_it, do_warn
      PARAMETER (do_warn = .TRUE. )

#ifdef FORTRAN_90
	INTEGER HEX40
	DATA HEX40/Z'40'/
#endif

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

        CALL NCF_INQ_DS (dset, ndims, nvars, ngatts, recdim )
        ds_time_axis(dset) = recdim

* does it have a (readable) title ?

* get global attribute from linked list structure
* loop through global attributes looking for title, title_mod

* get global attribute pseudo-variable id

        IF (ngatts .GT. 0) THEN
           CALL CD_GET_VAR_ID (dset, '.', varid, status)

           DO 100 iatt = 1, ngatts

              name = ' '
              CALL CD_GET_VAR_ATT_NAME( dset, varid, iatt, name, status)

* *kob* 6/96 - don't do this if we have a mc dataset - title comes
*	       from the descriptor rather than the cdf file

              IF (STR_SAME(ds_type(dset), '  MC') .NE. 0) THEN
                 maxlen = LEN(ds_title(dset))
                 IF (name .EQ. 'title' .OR. name .EQ. 'TITLE')
     .              got_it = NC_GET_ATTRIB ( dset, varid, name, do_warn, 
     .                 '.', maxlen, attlen, attoutflag, ds_title(dset), 
     .                 val)
              ENDIF

* title modifier ?

              maxlen = LEN(ds_mod_title(dset))
              IF (name .EQ. 'title_mod' .OR. name .EQ. 'TITLE_MOD')
     .           got_it = NC_GET_ATTRIB ( dset, varid, name, do_warn, 
     .              '.', maxlen, attlen, attoutflag, 
     .              ds_mod_title(dset), val)
     
* does it have a message to display upon opening ?

* does it have a message to display upon opening ?
              ds_alert_on_output(dset) = .FALSE.
              maxlen = LEN(ds_message(dset))
              IF (name .EQ. 'message' .OR. name .EQ. 'MESSAGE')
     .           ds_alert_on_open(dset) = NC_GET_ATTRIB ( dset, varid, 
     .                    name, do_warn, '.', maxlen, attlen, 
     .                    attoutflag, ds_message(dset), val)
  
* holler the message
              IF ( ds_alert_on_open(dset) ) THEN
	         CALL TM_NOTE('regarding '//
     .	   	   ds_des_name(dset)(:TM_LENSTR1(ds_des_name(dset)))
     .		   //' ...', tooplu)
	         CALL TM_NOTE(ds_message(dset), tooplu)
              ENDIF
 100       CONTINUE

        ENDIF  ! ngatts gt 0

* special TMAP parameters ?
* ... read global attributes beginning with "TMAP"

* Default contents as per TMAP "Format record" (is it EPIC? 3/93)
* -kob- only set ds_type to cdf if it isn't already set to MC 7/95

        IF ( CD_ISIT_EPIC(dset, status) ) THEN
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
