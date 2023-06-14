	SUBROUTINE CD_DODS_CONTROL ( dset, cdfid, remote_dods_id, cdfstat )

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
* Remote DODS file open has just been simulated using a local cache
* Now pass control over to the remote file to create the illusion that
* it was legitimately opened

* Programmer Steve Hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program

* revision 0.0 - 11/96
* Linux Port - 6/97 - reorder if def logic when setting cd_data_type to 
*		      BYTE or INTEGER*1
*		    - use a cpp include for tmap_dset.parm
* V65  *acm* 1/10  Change all netcdf2-style calls and parameter names to 
*                  netcdf3-style calls, which are also consistent with netcdf4.
* V65  *acm* 1/10  Add #ifdef usingDODSf2cUnderscore, for 32-bit linux build
*                  linked with OPeNDAP library.
* V6.74 3/12 *acm* cleanup ifdefs and unnecessary include files

* argument definitions:
*    dset - data set number
*    cdfid -  NetCDF open id for cache file
*    remote_dods_id - NetCDF open id for remote DODS file
*    cdfstat - return cdfstat

* argument declarations:
	INTEGER	dset, cdfid, remote_dods_id, cdfstat

* include files
        include 'netcdf.inc'
#include "gt_lib.parm"
	include 'tmap_dims.parm'
#	include "tmap_dset.parm"
	include 'tmap_errors.parm'
	include 'xdset_info.cmn_text'
	external xdset_info_data
        include 'xdset_info.cd_equiv'
	include 'xtm_grid.cmn_text'
	include 'xstep_files.cmn_text'

! Declare netcdf function names. 
! The ones without trailing underscore are declared in netcdf.inc
#include "netcdf_declareUnderscore.inc"

* reorder loop logic because linux cpp didn't like it *kob* 10/96

* local variable declarations:
	LOGICAL CD_GET_ATTRIB, CD_GET_ATTVAL,
     .	got_it, do_warn
	INTEGER tmvar, grid, idim, line, istep, attlen
	REAL	val

* local data initialization
	DATA do_warn/ .TRUE. /

* loop through the variables in this data set
	DO 100 tmvar = 1, maxvars
	   IF ( ds_var_setnum(tmvar) .NE. dset ) GOTO 100

* get the correct NetCDF data type from attributes in the cache file
	   got_it = CD_GET_ATTVAL( cdfid, cd_varid(tmvar),
     .			'netcdf_data_type', do_warn, ' ', val, 1, cdfstat )
	   IF ( .NOT. got_it ) GOTO 5100
	   cd_data_type(tmvar) = val

* get the correct NetCDF variable id from attributes in the cache file
	   got_it = CD_GET_ATTVAL( cdfid, cd_varid(tmvar),
     .			'netcdf_var_id', do_warn, ' ', val, 1, cdfstat )
	   IF ( .NOT. got_it ) GOTO 5100
	   cd_varid(tmvar) = val

* restore the full span of the grids to the variables
	   grid = ds_grid_number(tmvar)
	   DO 50 idim = 1, 4
	      line = grid_line(idim,grid)
	      IF (line .NE. mpsnorm) ds_grid_end(idim,tmvar)
     .				     = line_dim( line )
 50	   CONTINUE

 100	CONTINUE

* get the full URL back again (cache name was substituted in ds_des_name)
	got_it = CD_GET_ATTRIB( cdfid, NF_GLOBAL, 'DODS_URL',
     .                     do_warn, ' ', ds_des_name(dset), attlen, 128 )
	IF ( .NOT. got_it ) GOTO 5100

* close the cache file and substitute the remote file in its place

	cdfstat = NF_CLOSE ( cdfid )

* insert the remote file info in place of the cache info
	istep = cd_stepfile(dset)
	sf_lunit(istep) = remote_dods_id
	cdfid = remote_dods_id
	sf_name(istep) = ds_des_name(dset)

* successful completion
      cdfstat = merr_ok
      RETURN

* error exit
 5100	CALL TM_ERRMSG( merr_dsetnf, cdfstat, 'CD_DODS_CONTROL',
     .                   dset, no_stepfile,
     .                   'DODS cache file lacks necessary attributes',
     .                   no_errstring, *5900)
 5900	CALL CD_ABORT_GRIDS

	cdfstat = NF_CLOSE ( remote_dods_id )
 5990	RETURN
	END
