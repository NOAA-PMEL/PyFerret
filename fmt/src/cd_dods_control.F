	SUBROUTINE CD_DODS_CONTROL ( dset, cdfid, remote_dods_id, status )

* Remote DODS file open has just been simulated using a local cache
* Now pass control over to the remote file to create the illusion that
* it was legitimately opened

* Programmer Steve Hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program

* revision 0.0 - 11/96
* Linux Port - 6/97 - reorder if def logic when setting cd_data_type to 
*		      BYTE or INTEGER*1
*		    - use a cpp include for tmap_dset.parm
* argument definitions:
*    dset - data set number
*    cdfid -  NetCDF open id for cache file
*    remote_dods_id - NetCDF open id for remote DODS file
*    status - return status

* argument declarations:
	INTEGER	dset, cdfid, remote_dods_id, status

* include files
#ifdef unix
	include 'tmap_format/netcdf.decl'
	include 'tmap_format/netcdf.inc'
	include 'tmap_format/gt_lib.parm'
	include 'tmap_format/tmap_dims.parm'
#	include "tmap_format/tmap_dset.parm"
	include 'tmap_format/tmap_errors.parm'
	include 'tmap_format/xdset_info.cmn_text'
	external xdset_info_data
        include 'tmap_format/xdset_info.cd_equiv'
	include 'tmap_format/xtm_grid.cmn_text'
	include 'tmap_format/xstep_files.cmn_text'
#else
	INCLUDE 'TMAP_FORMAT:NETCDF.DECL'
	INCLUDE 'TMAP_FORMAT:NETCDF.INC'
	INCLUDE 'TMAP_FORMAT:GT_LIB.PARM'
	INCLUDE 'TMAP_FORMAT:TMAP_DIMS.PARM'
	INCLUDE 'TMAP_FORMAT:TMAP_DSET.PARM'
	INCLUDE 'TMAP_FORMAT:TMAP_ERRORS.PARM'
	INCLUDE 'TMAP_FORMAT:XDSET_INFO.CMN'
        INCLUDE 'TMAP_FORMAT:XDSET_INFO.CD_EQUIV'
	INCLUDE 'TMAP_FORMAT:XTM_GRID.CMN'
	INCLUDE 'TMAP_FORMAT/XSTEP_FILES.CMN_TEXT'
#endif

* special equivalence (not in tmap_format/xdset_info.cd_equiv because "BYTE"
* isn't supported on DECstation)
* reorder loop logic because linux cpp didn't like it *kob* 10/96

#ifdef unix
#  ifdef sun
      BYTE cd_data_type(maxvars)
#   else
      INTEGER*1 cd_data_type(maxvars)
#   endif
#else
      BYTE cd_data_type(maxvars)
#endif
	EQUIVALENCE ( ds_precision, cd_data_type )

* local variable declarations:
	LOGICAL CD_GET_ATTRIB, CD_GET_ATTVAL,
     .	got_it, do_warn
	INTEGER tmvar, grid, idim, line, istep
	REAL	val

* local data initialization
	DATA do_warn/ .TRUE. /

* loop through the variables in this data set
	DO 100 tmvar = 1, maxvars
	   IF ( ds_var_setnum(tmvar) .NE. dset ) GOTO 100

* get the correct NetCDF data type from attributes in the cache file
	   got_it = CD_GET_ATTVAL( cdfid, cd_varid(tmvar),
     .			'netcdf_data_type', do_warn, ' ', val, 1, status )
	   IF ( .NOT. got_it ) GOTO 5100
	   cd_data_type(tmvar) = val

* get the correct NetCDF variable id from attributes in the cache file
	   got_it = CD_GET_ATTVAL( cdfid, cd_varid(tmvar),
     .			'netcdf_var_id', do_warn, ' ', val, 1, status )
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
	got_it = CD_GET_ATTRIB( cdfid, NCGLOBAL, 'DODS_URL',
     .                     do_warn, ' ', ds_des_name(dset), 128 )
	IF ( .NOT. got_it ) GOTO 5100

* close the cache file and substitute the remote file in its place
	CALL NCCLOS( cdfid, status )

* insert the remote file info in place of the cache info
	istep = cd_stepfile(dset)
	sf_lunit(istep) = remote_dods_id
	cdfid = remote_dods_id
	sf_name(istep) = ds_des_name(dset)

* successful completion
      status = merr_ok
      RETURN

* error exit
 5100	CALL TM_ERRMSG( merr_dsetnf, status, 'CD_DODS_CONTROL',
     .                   dset, no_stepfile,
     .                   'DODS cache file lacks necessary attributes',
     .                   no_errstring, *5900)
 5900	CALL CD_ABORT_GRIDS
	CALL NCCLOS( remote_dods_id, status )
 5990	RETURN
	END
