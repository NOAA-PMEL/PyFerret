	SUBROUTINE CD_GET_AXES ( dset, cdfid, reversed, status )

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
* Loads the common block XGRID with all of the axis definitions from a
* netCDF file.
* An "axis" is a 1D CDF variable where its dimension name is also its
* variable name.
* The boundaries between points on an axis can be separately defined using
* another variable with length n+1.  The attribute "edges = variable" is used
* to point to this list of edge locations.

* Programmer Steve Hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program

* revision 0.0 - 2/92
* 3/4/93 - pass to CD_GET_1_AXIS a flag for EPIC
* 2/95 - added support for reverse-ordered coordinate axes
* 10/96 *kob* - Linux port - had to move variable assignments off of
*		declaration line.

* argument definitions:
*       dset        - TMAP data set pointer
*	cdfid	    - netCDF id number for already opened CDF file
*	reversed    - logical array to record which axes have reversed coords
*	status	    - return status

* include files
#ifdef unix
        include 'netcdf.decl'
#include "netcdf.inc"
        include 'xbuild_grids.cmn'
	include 'tmap_errors.parm'
	include 'tmap_dims.parm'
#include "tmap_dset.parm"
#include "gt_lib.parm"
	include 'xdset_info.cmn_text'
	external xdset_info_data
	include 'xtm_grid.cmn_text'
	external xgt_grid_data
#else
xx        include 'netcdf.decl'  ! untested
#include "netcdf.inc"
        INCLUDE 'TMAP_FORMAT:XBUILD_GRIDS.CMN'
	INCLUDE 'TMAP_FORMAT:TMAP_DIMS.PARM'
	INCLUDE 'TMAP_FORMAT:TMAP_DSET.PARM'
	INCLUDE 'TMAP_FORMAT:GT_LIB.PARM'
	INCLUDE 'TMAP_FORMAT:XDSET_INFO.CMN'
	INCLUDE 'TMAP_FORMAT:XTM_GRID.CMN'
#endif

* argument declarations
	INTEGER	dset, cdfid, status
	LOGICAL reversed(0:max_lines)

* local parameter definitions:
      INTEGER     str_eq
      PARAMETER ( str_eq = 0 )

* local variable declarations:
* *kob* explicit definitions not allowed on decl line in linux...
*       moved it to parameter line 10/96
      LOGICAL CD_GET_ATTRIB,
     .        got_it, do_warn
      PARAMETER (do_warn = .TRUE. )
      INTEGER TM_LENSTR1, TM_FIND_GRID_SLOT, TM_FIND_LINE_SLOT, NCDID,
     .        ndims, iaxis, cdfstat, nvars, ngatts, recdim,
     .        ivar, vartyp, nvdim, vdims(8), nvatts, i, vlen
      CHARACTER*24 vname*24, name*16

* initialize: save the state of grid memory prior to any new definitions
* ... save original start of free line memory
	orig_line_mem = next_line_mem_pos

* ... flag all axes as non-reversed as the default
	DO 10 i = 0, max_lines
 10	reversed(i) = .FALSE.

* ... how many grid spaces used before we start ?
	status = TM_FIND_GRID_SLOT ( firstgrid )
        lastgrid = firstgrid - 1
	IF (status .NE. merr_ok) GOTO 5900

* ... how many line spaces used before we start ?
	status = TM_FIND_LINE_SLOT ( firstax )
        lastax = firstax - 1
	IF (status .NE. merr_ok) GOTO 5900

* get the CDF file vital statistics
        CALL NCINQ( cdfid, ndims, nvars, ngatts, recdim, cdfstat )
        IF ( cdfstat .NE. NCNOERR ) GOTO 5100

* * * * loop through each variable in the CDF file * * *
* if its variable name matches its dimension name create an axis from it
	DO 100 ivar = 1,nvars

* get the vital statistics for this variable
           CALL NCVINQ(cdfid, ivar, vname, vartyp, nvdim, vdims,
     .                 nvatts,cdfstat)
           IF ( cdfstat .NE. NCNOERR ) GOTO 5100
           vlen = TM_LENSTR1(vname)

* if for some reason it is desirable for this axis name not to match its dim
* name then the "parent_axis" attribute can flag it is a parent axis. 
           got_it = CD_GET_ATTRIB( cdfid, ivar, 'parent_axis',
     .                            do_warn, vname(:vlen), name, 16 )
           IF ( got_it ) GOTO 50

* CDF "coordinate variable" ? (i.e. is it also a dimension name?)
           i = NCDID( cdfid, vname(:vlen), cdfstat )
           IF (cdfstat .NE. NCNOERR ) GOTO 100

* or is it really an edge definition ?
           got_it = CD_GET_ATTRIB( cdfid, ivar, 'edges',
     .                            do_warn, vname(:vlen), name, 16 ) !linenamlen
           IF ( got_it .AND. name.EQ.' ' ) GOTO 100

* or is it really a child axis (avail for. generic CDF apps but not needed) ?
           got_it = CD_GET_ATTRIB( cdfid, ivar, 'child_axis',
     .                            do_warn, vname(:vlen), name, 16 ) !linenamlen
           IF ( got_it ) GOTO 100

* define the center points of an axis
 50        CALL CD_GET_1_AXIS(cdfid, ivar, iaxis,
     .                        ds_type(dset).EQ.'ECDF', reversed, status)
           IF ( status .NE. merr_ok ) GOTO 5900

 100    CONTINUE

* successful completion
 	status = merr_ok
        RETURN

* error messages
 5100   CALL TM_ERRMSG
     .     ( cdfstat+pcdferr, status, 'CD_GET_AXES', cdfid,
     .        no_varid, no_errstring, no_errstring, *5900 )

* error exit
 5900   RETURN
	END
