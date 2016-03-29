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

* include files
        include 'netcdf.inc'
	include 'tmap_errors.parm'
#include "gt_lib.parm"
	include 'tmap_dims.parm'
#include "tmap_dset.parm"

	include 'xtm_grid.cmn_text'
	include 'xunits.cmn_text'
	external xgt_grid_data


* calling argument declarations
	INTEGER dset, cdfid, varid, axcalt, axlagf, axlagt, status
	CHARACTER*(*) vname

* internal variable declarations
	LOGICAL coordvar, has_edges
	INTEGER cdfstat, i, outflag, nvatts, vartype,
     .		start(nferdims),count(nferdims), zero(nferdims),
     .		npts, nT, nF, npts_cdf, vdims(8), nvdim
	REAL*8  TM_WORLD_AX, avg_lag
	CHARACTER  TM_STRING*13, buff*256


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

	IF ((next_line_mem_pos+ npts+1).GT.maxlinestore) GOTO 5300
	CALL CD_READ_SUB(cdfid, varid, nvdim, start, count, zero, zero, 
     .			 line_mem(next_line_mem_pos), 0, 0, cdfstat)
	IF ( cdfstat .NE. NF_NOERR ) GOTO 5000

* make the axes needed by the FMRC dataset
	has_edges = .FALSE.
	CALL CD_MAKE_FMRC_AXES (dset, has_edges, varid, vname, vartype,
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
 5300	CALL TM_ERRMSG (merr_linstorlim, status, 'CD_1D_LINE_FROM_2D',
     .                  no_descfile, no_stepfile,
     .                  'MAX='//TM_STRING(DBLE(maxlinestore)),
     .                  no_errstring, *5000)

	END
