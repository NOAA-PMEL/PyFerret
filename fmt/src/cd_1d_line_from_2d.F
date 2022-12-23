	SUBROUTINE CD_1D_LINE_FROM_2D (dset, cdfid, varid, vname,
     .			axcalt, axlagf, axlagt, status)

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

* this routine gets the 2D FMRC time array into memory and calls
* CD_MAKE_FMRC_AXES to synthesize the T calendar time and F lead time
* axes that are needed to support FMRC visualization and analysis
* it also modifies the time axis (native_tax) of the dataset -- which TDS
* provides only as a undocumented dimension, making a lag time axis (axlagt)

* V695+ *acm* and *sh*
* V698  *sh* 3/16 -- modify the native taxis (axlagt) to become a lag time
*                 -- look for and read time_bounds 2d array if it exists
* V72 6/17 *acm* For trac enhancement #767 -- dynamic coordinate storage 
* v751 *acm*  6/19 Allow for longer strings as attribute values (line_name is length 64)
C 12/2022 *acm* Code cleanup: Declare uppermost as an array to avoid rank-mismatch errors
*               when compiling under gfortran v10.

* include files
        include 'netcdf.inc'
	include 'tmap_errors.parm'
#include "gt_lib.parm"
	include 'tmap_dims.parm'
#include "tmap_dset.parm"

	include 'xio.cmn_text'
	include 'xtm_grid.cmn_text'
	include 'xunits.cmn_text'
	external xgt_grid_data


* calling argument declarations
	INTEGER dset, cdfid, varid, axcalt, axlagf, axlagt, status
	CHARACTER*(*) vname

* internal variable declarations
	LOGICAL NC_GET_ATTRIB_STRING, coordvar, has_edges, do_warn, got_it
	INTEGER TM_LENSTR1, cdfstat, i, outflag, nvatts, vartype,
     .		start(nferdims),count(nferdims), zero(nferdims),
     .		npts, nT, nF, npts_cdf, vdims(8), nvdim,
     .		vlen, blen, bndid, attid, attype, attlen, attoutflag
	REAL*8  TM_WORLD_AX, avg_lag, uppermost(1)
	CHARACTER  TM_STRING*13, buff*256
	CHARACTER bname*64, attname*64

	INTEGER max_slen
	PARAMETER (max_slen = 64)

* initialize
	do_warn = .FALSE.
	vlen = TM_LENSTR1(vname)


* start out by getting an overview of the 2D time variable
	CALL CD_GET_VAR_INFO (dset, varid, vname, vartype, 
     .            nvdim, vdims, nvatts, coordvar, outflag, status)

* Read the 2d time coordinate array from the netCDF dataset
* Save it at the start of available line storage memory

	npts = 1
	DO i = 1, nvdim
	   CALL CD_GET_DS_DIMS (dset, vdims(i), buff, npts_cdf, status)
	   start(i) = 1
	   count(i) = npts_cdf
	   zero(i) = 0
	   npts = npts* npts_cdf
	ENDDO
	nT = count(1)   ! number of time steps per forecast
	nF = count(2)   ! number of forecasts in collection

* grab a temporary dynamic axis slot for it. Store it in the axis axcalt 
* as that axis will be set up with the coordinates

	CALL TM_ALLO_TMP_LINE(axcalt, status)
	IF (status .NE. merr_ok) GOTO 5000

	CALL GET_LINE_DYNMEM (npts+1, axcalt, status)
	IF (status .NE. merr_ok) GOTO 5000 
	CALL CD_READ_SUB(cdfid, varid, nvdim, start, count, zero, zero, 
     .			 linemem(axcalt)%ptr, 0, 0, cdfstat)
	IF ( cdfstat .NE. NF_NOERR ) GOTO 5000

************* start bounds reading
	has_edges = .FALSE.   ! assume failure
* if a 2d time_bounds array exists, read the bounds from it
* ... first look for a "bounds" attribute on the 2d time array
	CALL CD_GET_VAR_ATT_ID (dset, varid, 'bounds', attid, status)
	IF (status.EQ.atom_not_found) GOTO 1000

* ... get the name of the variable pointed to by the bounds array
        CALL CD_GET_VAR_ATT_INFO (dset, varid, attid, 
     .                attname, attype, attlen, attoutflag, status )
	IF (status.EQ.atom_not_found) GOTO 999
	IF (attype .NE. NCCHAR) GOTO 999
	got_it = NC_GET_ATTRIB_STRING (dset, varid, attname, 
     .                                do_warn, vname(:vlen), max_slen, 
     .                                attlen, attoutflag, bname)
	IF (.NOT.got_it) GOTO 999

        CALL TM_CHECK_FMRC_BNDS_ATTRIB (dset, nT, nF, vname, vlen, 
     .                                  bname, bndid, status)
        IF (status .NE. merr_ok) GOTO 999
	blen = TM_LENSTR1(bname)

* ... set up limits for reading the lower bounds
	nvdim = 3
	start(1) = 1  ! lower bounds
	count(1) = 1
	zero (1) = 0
	start(2) = 1
	count(2) = nT
	zero (2) = 0
	start(3) = 1
	count(3) = nF
	zero (3) = 0

* ... read the lower bounds into line memory above the 2d time coordinates
	CALL CD_READ_SUB(cdfid, bndid, nvdim, start, count, zero, zero, 
     .			 lineedg(axcalt)%ptr, 0, 0, cdfstat)
	IF ( cdfstat .NE. NF_NOERR ) GOTO 999

* ... alter limits for reading the single upper bound of the top point
	start(1) = 2  ! upper bound
	count(1) = 1
	zero (1) = 0
	start(2) = nT
	count(2) = 1
	zero (2) = 0
	start(3) = nF
	count(3) = 1
	zero (3) = 0

* ... read the upper-upper bound into line memory above the 2d lower bounds
	CALL CD_READ_SUB(cdfid, bndid, nvdim, start, count, zero, zero, 
     .			 uppermost, 0, 0, cdfstat)
	IF ( cdfstat .NE. NF_NOERR ) GOTO 999
	CALL PUT_LINE_COORD ( lineedg(axcalt)%ptr, 2*npts+1, uppermost )

* ... success!
	has_edges = .TRUE.
	GOTO 1000

* ... report corruption in bounds linkage
 999	CALL TM_NOTE('2D bounds attribute corruption', lunit_errors)
************* end bounds reading

* make the axes needed by the FMRC dataset
 1000	CALL CD_MAKE_FMRC_AXES (dset, has_edges, varid, vname, vartype,
     .			    npts, nT, nF, axcalt, axlagf, status)
	IF (status .NE. merr_ok) GOTO 5000

* 3/16 - correct the lag-time axis of the FMRC (created by Unidata TDS)
*    from a simplistic 1, ..Nt axis to a true delta (lag) time
	line_unit_code(axlagt) = line_unit_code(axcalt)
	line_units(axlagt) = un_name(line_unit_code(axlagt))
	line_direction(axlagt) = 'TT'
	line_dattype  (axlagt) = nf_double
	line_dim_only (axlagt) = .FALSE.  ! no longer netCDF dim w/o coord var

	avg_lag = ( TM_WORLD_AX(nT, axcalt, box_middle )
     .	          - TM_WORLD_AX(1,  axcalt, box_middle ) ) / (nT-1)
	line_start(axlagt)     = avg_lag/2.  ! box 1 starts at 0.0
	line_delta(axlagt)     = avg_lag

* successful completion
        status = merr_ok
	RETURN

* error exit
 5000	RETURN

	END
