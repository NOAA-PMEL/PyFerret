	SUBROUTINE CD_DSG_READ ( dset, ivar, var, vtype, status )
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
*
* Read a netCDF3 1D DSG contiguous ragged array variable 

* V720 *sh* 4/17
* V74+ 4/18 *acm* Getting DSG grids
* V74+ 4/18 *acm* Allow for the NCEI-style datasets which are one feature
*                 per file but they put the obs variables on 2D grids, 
*                 temp(feature, nobs)
* V74+ 7/18 *acm* More on reading NCEI-type grids

* include files
        include 'netcdf.inc'
	include 'tmap_errors.parm'
	include 'tmap_dims.parm'		! data set dimension
	include 'tmap_dtypes.parm'
#include "gt_lib.parm"
	include 'xdset_info.cmn_text'	! data set info common
	external xdset_info_data
	include 'xstep_files.cmn_text'	! step file common
	external xstep_files_data
        include 'xdset_info.cd_equiv'


* arguments:
*	dset        - internal data set # (from GT_INIT_DSET)
*	ivar        - position in common of variable information
*	var         - array to receive DSG values (output)
*	vtype       - data type of the variable
*	status      - error status upon completion (output)
*
	INTEGER		dset, ivar, vtype, status
	REAL     	var(*)

* local definitions
	LOGICAL one_by_nobs, nobs_by_one
	INTEGER sf_num, cdfid, varid, dimlen, dtype

	INTEGER ndim, iflag,
     .          start(nferdims), count(nferdims), 
     .          stride(nferdims), imap(nferdims),
     .          nvdims, vdims(nferdims), ax1len, ax2len
	CHARACTER*128 buff

* initialize
	sf_num = cd_stepfile(dset)
        cdfid  = sf_lunit(sf_num)
        varid  = cd_varid(ivar)
	dimlen = dsg_obs_dimlen(dset)
	dtype = 0    ! arg to CD_RD_R8_ARR, but not used there


* See if its a file like the NCEI ones with 1 feature and dimensions 
* for the obs variables of (obs,traj) or (traj,obs)

	one_by_nobs = .FALSE.
	nobs_by_one = .FALSE.
	CALL CD_GET_VAR_DIMS_INFO (dset, varid, buff, 
     .        nvdims, vdims, status)
	IF (nvdims.EQ.2 .AND. vtype.NE.ptype_string) THEN
	   status = NF_INQ_DIMLEN(cdfid, vdims(1), ax1len)
	   status = NF_INQ_DIMLEN(cdfid, vdims(2), ax2len)
	   IF (ax1len.EQ.1 .AND. ax2len.EQ.dimlen) one_by_nobs = .TRUE. 
	   IF (ax1len.EQ.dimlen .AND. ax2len.EQ.1) nobs_by_one = .TRUE. 
	   IF (.NOT.one_by_nobs .AND. .NOT.nobs_by_one) GOTO 5001
	ENDIF

* read the 1D variable from the file

	IF (vtype .NE. ptype_string) THEN

	   IF (nvdims.EQ.2 .AND. vtype.NE.ptype_string) THEN
	      start(1) = 1
	      start(2) = 1
	      count(1) = 1
	      count(2) = 1
	      IF (nobs_by_one) count(1) = dimlen
	      IF (one_by_nobs) count(2) = dimlen
	      CALL CD_RD_R8_BNDS( cdfid, varid, start, count, dtype,
     .                          "reading "//ds_var_code(ivar)
     .                          //" for DSG",
     .                           var, 1, 1, 1, dimlen, status )
	   ELSE
              CALL CD_RD_R8_ARR( cdfid, varid, 1, dimlen, dtype,
     .                     "reading DSG variable"//ds_var_code(ivar),
     .                      var, status )
	   ENDIF

* String data types. 

	ELSE

	   IF (one_by_nobs) THEN  ! untested so far...
	      start(1) = 1
	      start(2) = 1
	      start(3) = 1
	      count(1) = 1
	      count(2) = dimlen
	      count(3) = 1
	   ELSe
	      ndim = 1
	      iflag = 0
	      start(1) = 1
	      count(1) = dimlen
	      stride(1) = 1
	      imap(1) = 1
	   ENDIF
	   
	   CALL CD_READ_SUB(cdfid, varid, ndim, start, count,
     .		   stride, imap, var, iflag, iflag, status) 
           IF (status .EQ. NF_NOERR ) status = merr_ok
	ENDIF

! DSGTBD: scale and offset need to be applied to the variable
!    Can the logic from CD_READ be packaged into a subroutine and reused?


 5000	RETURN
* Error, should never get here - reading data on too many dimensions
 5001    CALL TM_ERRMSG
     .     ( pcdferr, status, 'CD_DSG_READ', cdfid,
     .       varid, 'Failed attempting READ', no_errstring, *5000 )


	END
