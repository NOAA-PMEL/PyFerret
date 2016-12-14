	SUBROUTINE CD_QUIK_NCOPEN (dset, filename, vname,
     .	    cdfid, vid, bounds_id, units_str, cal_str, n, status)
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
* open a netcdf file; get the netCDF variable id and 1d array length
* of the given var name
* if that variable has a 'bounds' attribute, get its vid, too

* programmer - Steve Hankin
* NOAA/PMEL, Seattle, WA - Thermal Modeling and Analysis Program

* V695+ 10/15 *sh*

*
* arguments:
*	dset		- (input) Ferret dataset number
*	filename	- (input) name of file to be opened
*       vname		- (input) name of variable to look for (case-sensitive)
*	cdfid		- (output) netCDF id of the open dataset
*	vid		- (output) netCDF id of the requested variable
*	bounds_id	- (output) netCDF id of the bounds variable
*	units_str	- (output) string value of the units attribute
*	cal_str		- (output) string value of the calendar attribute
*	n		- (output) length of the named variable
*	status		- (output) TMAP status code

        include 'netcdf.inc'
        include 'tmap_errors.parm'
	include 'tmap_dims.parm'
#include "gt_lib.parm"
	include 'xdset_info.cmn_text'
	include 'xstep_files.cmn_text'

* Argument variable declarations
	CHARACTER*(*) filename, vname, units_str, cal_str
	INTEGER   dset, cdfid, vid, bounds_id, n, status

* Local variable declarations
	LOGICAL	CD_GET_ATTRIB, coordvar, got_it
	INTEGER	TM_LENSTR1, 
     .		vlen, vartyp, nvdim, vdims(8), nvatts, all_outflag,
     .		maxlen, attlen, tmap_status, cdfstat, blen, 
     .		dlen, elen, nb
	CHARACTER*2048   TM_INQ_PATH
	CHARACTER	varname*128, buff*128, errmsg*80, dummy*20
	REAL vals  ! potential numerical attributes


* local parameter definitions:
	LOGICAL	    do_warn
	PARAMETER ( do_warn = .TRUE. )

*initialize
	vlen = TM_LENSTR1(vname)
	maxlen = 128

* locate the file
	ds_des_name(dset) =
     .		TM_INQ_PATH(filename, 'FER_DATA', ' ',.FALSE., status)
        IF (status .NE. merr_ok)  THEN
	   tmap_status = merr_notmounted
	   GOTO 5050
	ENDIF
	dlen = TM_LENSTR1(ds_des_name(dset))

* open the file
#ifdef usingDODSf2cUnderscore 
	cdfstat = NF_OPEN_(ds_des_name(dset), NF_NOWRITE, cdfid) 
#else
	cdfstat = NF_OPEN(ds_des_name(dset), NF_NOWRITE, cdfid) 
#endif
	IF ( cdfstat .NE. NF_NOERR ) THEN
	   tmap_status = merr_nc_open
	   errmsg = 'unable to open file'
	   GOTO 5100
	ENDIF

* get the netCDF id of the named time axis variable
#ifdef usingDODSf2cUnderscore
        cdfstat = NF_INQ_VARID_(cdfid, vname(:vlen), vid)
#else
        cdfstat = NF_INQ_VARID(cdfid, vname(:vlen), vid)
#endif
	IF (cdfstat.NE.NF_NOERR) THEN
	   tmap_status = merr_unkvar
	   errmsg = 'time axis variable is missing: '//vname
	   GOTO 5100
	ENDIF

* inquire after the netCDF metadata for this variable
#ifdef usingDODSf2cUnderscore
        cdfstat = NF_INQ_VAR_(cdfid, vid, varname, vartyp, nvdim, 
     .              vdims, nvatts)
#else
        cdfstat = NF_INQ_VAR(cdfid, vid, varname, vartyp, nvdim, 
     .              vdims, nvatts)
#endif
	IF (cdfstat.NE.NF_NOERR) THEN
	   tmap_status = merr_badfileatt
	   errmsg = 'unable to read attributes of '//vname
	   GOTO 5100
	ENDIF

* sanity check the 1d axis variable
	IF (nvdim .NE. 1) THEN
	   tmap_status = merr_badtimedef
	   errmsg = 'time axis variable not 1D: '//vname
	   GOTO 5100
	ENDIF

* get the axis length
#ifdef usingDODSf2cUnderscore
	cdfstat = NF_INQ_DIM_( cdfid, vdims(1), varname, n )
#else
	cdfstat = NF_INQ_DIM( cdfid, vdims(1), varname, n )
#endif
	IF ( cdfstat .NE. NF_NOERR ) THEN
	   tmap_status = merr_badtimedef
	   errmsg = 'unable to read time ax dim length '//vname
	   GOTO 5100
	ENDIF

* does this 1d var have a bounds attribute?
	blen = LEN(buff)  ! character array length
	got_it = CD_GET_ATTRIB( cdfid, vid, 'bounds',
     .                     do_warn, vname(:vlen), buff, attlen, blen )
	IF (.NOT.got_it) THEN
	   bounds_id = unspecified_int4
	ELSE
!	   blen = TM_LENSTR1(buff)
* ... check the bounds variable, issue any warnings.
cc This call goes and gets info out of the list structure for member 1.
cc          CALL TM_CHECK_BNDS_ATTRIB (dset, n, vname, vlen, buff(:attlen),
cc     .                             bounds_id, tmap_status)


#ifdef usingDODSf2cUnderscore
           cdfstat = NF_INQ_VARID_(cdfid, buff(:attlen), bounds_id)
           cdfstat = NF_INQ_VAR_(cdfid, bounds_id, buff, vartyp, nvdim, 
     .              vdims, nvatts)
           cdfstat = NF_INQ_DIM_( cdfid, vdims(2), dummy, nb )
#else
           cdfstat = NF_INQ_VARID(cdfid, buff(:attlen), bounds_id)
           cdfstat = NF_INQ_VAR(cdfid, bounds_id, buff, vartyp, nvdim, 
     .              vdims, nvatts)
           cdfstat = NF_INQ_DIM( cdfid, vdims(2), dummy, nb )
#endif

	   IF ( cdfstat .NE. NF_NOERR .OR.nb.NE.n) THEN
	      tmap_status = merr_badtimedef
	      errmsg = 'errors with time axis bounds '//buff(:attlen)
	      bounds_id = unspecified_int4    ! invalidate the bounds variable
	      GOTO 5100
	   ENDIF

	ENDIF

* get the units attribute if it exists
	blen = LEN(units_str)  ! character array length
	got_it = CD_GET_ATTRIB( cdfid, vid, 'units',
     .                  do_warn, vname(:vlen), units_str, attlen, blen )
	IF (.NOT.got_it) units_str = ' '

* get the calendar attribute if it exists
	blen = LEN(cal_str)  ! character array length
	got_it = CD_GET_ATTRIB( cdfid, vid, 'calendar',
     .                  do_warn, vname(:vlen), cal_str, attlen, blen )
	IF (.NOT.got_it) cal_str = ' '

* successful completion
 1000	status = merr_ok
 5000	RETURN

* error exits
 5050	CALL TM_ERRMSG
     .     ( tmap_status, status, 'CD_QUIK_NCOPEN',
     .       no_descfile, no_stepfile,
     .       filename, no_errstring,  *5000 )
 5100	elen = TM_LENSTR1(errmsg)
	CALL TM_ERRMSG
     .     ( tmap_status, status, 'CD_QUIK_NCOPEN',
     .       no_descfile, no_stepfile,
     .	    'in file '//ds_des_name(dset)(:dlen)//' '//errmsg,
     .	     no_errstring,  *5000 )
 5200   CALL TM_ERRMSG
     .     ( tmap_status+pcdferr, status, 'CD_QUIK_NCOPEN', cdfid, vid,
     .       no_errstring, no_errstring, *5000 )

	END
