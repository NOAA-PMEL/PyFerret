	SUBROUTINE CD_INIT_DSET ( dset, status )

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

* Programmer Steve Hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program

* revision 0.0 - 2/92
* 2/95 - added support for reverse-ordered coordinate axes
* *sh* 11/12/96 - changes to accomodate caching of DODS data 
* Linux Port - 6/97 *kob*  - use a cpp include for tmap_dset.parm
* argument definitions:
*    dset   - pointer to TMAP data set
*    status - return status
* 11/97 *kob* - added a close on the cdfid for the netcdf file for
*               mc datasets only.  
* 4/98 *kob* - need to explicitely declare istat
* 5/98 *kob* - make sure that the sf_lunit is reset to file_not_open
*              when cdf file is closed - for mc datasets only

* argument declarations:
	INTEGER	dset, status

* include files
#ifdef unix
        include 'tmap_format/xbuild_grids.cmn'
	include 'tmap_format/gt_lib.parm'
	include 'tmap_format/tmap_dims.parm'
#	include "tmap_format/tmap_dset.parm"
	include 'tmap_format/tmap_errors.parm'
* add below *kob* 5/98
	include 'tmap_format/xstep_files.cmn_text'
	include 'tmap_format/xdset_info.cmn_text'
	external xdset_info_data
        include 'tmap_format/xdset_info.cd_equiv'
#else
        include 'tmap_format/xbuild_grids.cmn'
	INCLUDE 'TMAP_FORMAT:GT_LIB.PARM'
	INCLUDE 'TMAP_FORMAT:TMAP_DIMS.PARM'
	INCLUDE 'TMAP_FORMAT:TMAP_DSET.PARM'
	INCLUDE 'TMAP_FORMAT:TMAP_ERRORS.PARM'
	INCLUDE 'TMAP_FORMAT:XDSET_INFO.CMN'
        INCLUDE 'TMAP_FORMAT:XDSET_INFO.CD_EQUIV'
#endif

* local variable declarations:
      INTEGER cdfid, remote_dods_id,istat
      CHARACTER*16 temp_axnams(4,max_temp_grid)
      LOGICAL reversed(0:max_lines)

* open the netCDF data file and save global information
      CALL CD_OPEN_DSET ( dset, cdfid, remote_dods_id, status )
      IF ( status .NE. merr_ok ) GOTO 5990

* get "TMAP" special parameters
      CALL CD_GET_TMAP_PARMS ( dset, cdfid, status )
      IF ( status .NE. merr_ok ) GOTO 5980

* read through the netCDF variables to determine which are TMAP axes
* and to read variable attributes
      CALL CD_SCAN_VARS( dset, cdfid, status )
      IF ( status .NE. merr_ok ) GOTO 5900

* read through the netCDF variables to load the TMAP axes
      CALL CD_GET_AXES ( dset, cdfid, reversed, status )
      IF ( status .NE. merr_ok ) GOTO 5900

* build grids for variables w/out parent grids and associate with variables
      CALL CD_GET_GENERIC_GRIDS (dset,cdfid,temp_axnams,reversed,
     .					status)
      IF ( status .NE. merr_ok ) GOTO 5900

* build grids for variables with parent grids
      CALL CD_GET_PARENT_GRIDS (cdfid,temp_axnams,status)
      IF ( status .NE. merr_ok ) GOTO 5900

* associate the parent grids with data set variables      
      CALL CD_ASSOC_GRIDS (dset, cdfid, reversed, temp_axnams, status)
      IF ( status .NE. merr_ok ) GOTO 5900

* replace name linkages between axes and grids with pointers.  Delete dups
      CALL CD_CLEAN_GRIDS ( dset, cdfid, temp_axnams, status )
      IF ( status .NE. merr_ok ) GOTO 5900

* close up holes left by duplicate definitions
      CALL TM_GARB_COL_LINES ( firstax, lastax )
      CALL TM_GARB_COL_GRIDS ( firstgrid, lastgrid )

* summarize the special axes in the data set
      CALL CD_GET_TIME_AXIS( dset, t_dim, status )
      IF ( status .NE. merr_ok ) GOTO 5900

* if using a DODS cache then create the illusion of remote access
      IF (remote_dods_id .NE. -1) THEN
         CALL CD_DODS_CONTROL( dset, cdfid, remote_dods_id, status )
         IF ( status .NE. merr_ok ) GOTO 5900
      ENDIF

! for multiple files ... maybe
! *kob* 11/97 - close cdf files which are a part of a multi-file netcdf set
!* close the netCDF file
! 5/98 *kob* - reset sf_lunit as well
      IF (ds_type(dset) .EQ. '  MC') THEN
        CALL NCCLOS( cdfid, istat )
	sf_lunit(dset) = file_not_open
      ENDIF
* successful completion
      status = merr_ok
      RETURN

* error exit
 5900 CALL CD_ABORT_GRIDS
 5980 IF (remote_dods_id .NE. 0) CALL NCCLOS( remote_dods_id, status )
 5990 RETURN
      END
