	SUBROUTINE CD_GET_GENERIC_GRIDS (dset,cdfid,temp_axnams,reversed,
     .					 status)

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
* Loads the common block XGRID with implicitly given grid definitions from a
* netCDF file - i.e. grid definitions for variables that lack a 
* "parent_grid = name" attribute
* also initialize grid limits (lo hi bounds on each axis) for these variables

* Programmer Steve Hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program

* revision 0.0 - 2/92
* 2/95 - added support for reverse-ordered coordinate axes
* 7/96 - increase "name" from 16 to 24 characters for consistence
*	(still dimension names cannot be 24 chars --> line names in COMMON)
* 10/96 *kob* - Linux port - had to move variable assignments off of
*		declaration line.

* argument definitions:
*       dset        - TMAP data set pointer
*	cdfid	    - netCDF id number for already opened CDF file
*       temp_axnams - 4 axis names per grid: temp until grids are assembled
*       reversed    - logical array to record which axes have reversed coords
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
	include 'xio.cmn_text'
	external xio_data
	include 'xtm_grid.cmn_text'
	external xgt_grid_data
        include 'xdset_info.cd_equiv'
#else
xx        include 'netcdf.decl'  ! untested
#include "netcdf.inc"
        INCLUDE 'TMAP_FORMAT:XBUILD_GRIDS.CMN'
	INCLUDE 'TMAP_FORMAT:TMAP_DIMS.PARM'
	INCLUDE 'TMAP_FORMAT:TMAP_DSET.PARM'
	INCLUDE 'TMAP_FORMAT:GT_LIB.PARM'
	INCLUDE 'TMAP_FORMAT:XDSET_INFO.CMN'
	INCLUDE 'TMAP_FORMAT:XIO.CMN'
	INCLUDE 'TMAP_FORMAT:XTM_GRID.CMN'
        INCLUDE 'TMAP_FORMAT:XDSET_INFO.CD_EQUIV'
#endif

* argument declarations
	LOGICAL reversed(0:max_lines)
	INTEGER	 dset, cdfid, status
        CHARACTER*16 temp_axnams(4,max_temp_grid)

* local parameter definitions:
      INTEGER     str_eq
      PARAMETER ( str_eq = 0 )

* local variable declarations:
* *kob* explicit definitions not allowed on decl line in linux...
*       moved them to parameter line 10/96
      LOGICAL CD_GET_ATTRIB,
     .        got_it, do_warn, centers
      PARAMETER (do_warn = .TRUE. )
      PARAMETER (centers = .TRUE. )
      INTEGER TM_LENSTR1, TM_FIND_GRID_SLOT, STR_CASE_BLIND_COMPARE,
     .        NCVID, igrid, iaxis, grid_cnt, cdfstat, istat, idim,
     .        ivar, vartyp, nvdims, vdims(8), nvatts, vlen,
     .        axid, tmvar, tmpgrid, npts, dim, tmplines(4)
      CHARACTER name*24, vname*24	! 7/96: name was 16 chars

      INTEGER tt  ! nice short name
      EQUIVALENCE (tt,lunit_errors)

* * * * loop through each variable in this TMAP data set * * *
* if it doesn't have a grid pointer create a suitable one and point to it
        DO 500 tmvar = 1,maxvars
           IF ( ds_var_setnum(tmvar) .NE. dset ) GOTO 500
           ivar = cd_varid(tmvar)

* does it point explicitly to a parent grid ?
          got_it = CD_GET_ATTRIB( cdfid, ivar, 'parent_grid',
     .                       .NOT.do_warn, ' ', name, 16 )  !linenamlen
          IF ( got_it ) GOTO 500  ! yes - already has a pointer

* get the vital statistics for this variable
           CALL NCVINQ(cdfid, ivar, vname, vartyp, nvdims, vdims,
     .                 nvatts,cdfstat)
           IF (cdfstat.NE.NCNOERR) CALL TM_ERRMSG
     .     ( cdfstat+pcdferr,status,'CD_GET_GENERIC_GRIDS',cdfid,ivar,
     .       no_errstring, no_errstring, *5900 )
           vlen = TM_LENSTR1(vname)

* use the next grid building slot to build a grid for this variable
           tmpgrid = lastgrid - firstgrid + 2

* initialize the axis names in case there are fewer than 4 from CDF variable
           DO 10 idim = 1, 4
              tmplines(idim) = unspecified_int4
              temp_axnams(idim,tmpgrid)='NORMAL'
 10        CONTINUE

* find the axis names from the CDF file
           DO 200 idim = 1, nvdims

* ... get name of dimension
              CALL NCDINQ( cdfid, vdims(idim), name, npts, cdfstat )
              IF (cdfstat.NE.NCNOERR) CALL TM_ERRMSG
     .     ( cdfstat+pcdferr,status,'CD_GET_GENERIC_GRIDS',cdfid,ivar,
     .       vname(:vlen), 'cant get dimensions', *5900 )
 
* ... locate the axis by that name (processed in CD_GET_AXES)
              DO 100 iaxis = firstax, lastax
                 istat = STR_CASE_BLIND_COMPARE(name, line_name(iaxis))
                 IF ( istat .EQ. str_eq ) GOTO 150
 100          CONTINUE

* ... axis doesn't exist - perhaps it is a dimension with no variable defn
              axid = NCVID( cdfid, name, cdfstat ) 
              IF (cdfstat.NE.NCNOERR) THEN
                 CALL TM_MAKE_BASIC_AXIS( name, 1.0D0, 1.0D0, npts,
     .                                    iaxis, status )
                 IF ( status .NE. merr_ok ) GOTO 5900
              ELSE
                 CALL TM_NOTE('Internal TMAP/CDF error: mystery axis: '
     .                //name, tt )
                 status = 0
                 GOTO 5900
              ENDIF

* save a pointer to the axis just found
 150          IF (idim.LE.4) tmplines(idim) = iaxis

 200       CONTINUE

* reorder the axes into the most likely order if not explicitly given
           IF ( ds_ordering(1,tmvar) .EQ. unspecified_int4 ) THEN
              CALL TM_AXIS_ORDER(tmplines, ds_ordering(1,tmvar),
     .                           nvdims, vname(:vlen))
           ENDIF

* save the axis names in the proper order
           DO 240 idim = 1, 4
              ds_grid_start(idim,tmvar) = 1
              ds_grid_end  (idim,tmvar) = 1
 240       CONTINUE
           DO 250 idim = 1, 4
              iaxis = tmplines(idim)
              dim = ds_ordering(idim,tmvar)
              IF ( dim .NE. 0 ) THEN
                 temp_axnams(dim,tmpgrid) = line_name(iaxis)
                 ds_grid_end  (dim,tmvar) = line_dim(iaxis)
* ... flag reverse-ordered coordinate axis of variable (2/95)
	         IF ( reversed(iaxis) ) ds_ordering(idim,tmvar) = -1*dim
              ENDIF
 250       CONTINUE

* is the temporary grid we just created unique ?
           DO 310 igrid = firstgrid, lastgrid
              grid_cnt = igrid - firstgrid + 1
              DO 300 idim = 1, 4
                 IF ( temp_axnams(idim,tmpgrid )
     .          .NE.  temp_axnams(idim,grid_cnt) ) GOTO 310
 300          CONTINUE
* ... not unique - exact match with one that already exists
              GOTO 350
 310       CONTINUE

* allocate a new grid
* find next location to store info; if too many grids - give error
	  status = TM_FIND_GRID_SLOT ( igrid )
	  IF (status .NE. merr_ok) GOTO 5900

* make up a unique grid name "GXXnn" where XX is hashed from data set name
          grid_name(igrid) = 'G'//cd_dset_code(dset)(:2)
          CALL TM_NEW_GRID_NAME( grid_name(igrid), name )
          grid_name(igrid) = name
          lastgrid = igrid

* basic defaults
          grid_rotation(igrid) = 0.0
          DO 340 idim = 1, 4
 340      grid_out_prod(idim,igrid) = .TRUE.

* save a pointer to this grid for the TMAP variable
 350      ds_grid_number(tmvar) = igrid

 500   CONTINUE

* successful completion
 	status = merr_ok
        RETURN

* error exit
 5900   RETURN
	END
