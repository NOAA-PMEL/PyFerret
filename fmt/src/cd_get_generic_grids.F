	SUBROUTINE CD_GET_GENERIC_GRIDS (dset,cdfid,temp_axnams,reversed,
     .					 perm, status)

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
* V500: 3/99 *sh* - if an axis doesn't exist matching the dimension name, then
*	create a 1,2,3...n axis without error message
*       4/99 *kob* - increase size of temp_axnams, vname and name to 64
* *sh*  6/99 - added "perm" (permutation) argument
* V510: *sh* 1/00 - when a "BASIC" axis is created cuz the file has no
*			coords that axis needs to bump "lastax" to get
*			checked for dup names later
*	     3/00 - convert to dynamic grids and axes

* argument definitions:
*       dset        - TMAP data set pointer
*	cdfid	    - netCDF id number for already opened CDF file
*       temp_axnams - 4 axis names per grid: temp until grids are assembled
*       reversed    - logical array to record which axes have reversed coords
*       perm        - user-specified permutation (e.g. "YXZT"=2,1,3,4)
*	status	    - return status

* include files
#ifdef unix
        include 'netcdf.decl'
#include "netcdf.inc"
	include 'tmap_errors.parm'
	include 'tmap_dims.parm'
#include "tmap_dset.parm"
#include "gt_lib.parm"
        include 'xbuild_grids.cmn'
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
	LOGICAL  reversed(max_lines:line_ceiling)
	INTEGER	 dset, cdfid, perm(4), status
	CHARACTER*64 temp_axnams(4,max_temp_grid)

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
     .        ivar, vartyp, nvdims, vdims(8), nvatts, vlen, idim2,
     .        axid, tmvar, tmpgrid, npts, dim, tmplines(4), ndim,
     .	      grid_dims(4), trans(4), indices(4), i
      CHARACTER name*64, vname*64	! 4/99:both were  16 chars

      INTEGER tt  ! nice short name
      EQUIVALENCE (tt,lunit_errors)

* initialize temporary grid name pointer
	num_tmp_grids = 0
	DO 5 i = max_grids, grid_ceiling
 5	   tmp_nam_ptr(i) = 0

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
           tmpgrid = num_tmp_grids + 1

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
	      iaxis = 0
 100	      CALL TM_NEXT_TMP_LINE(iaxis, *120)  ! 120 if no more
	         istat = STR_CASE_BLIND_COMPARE(name, line_name(iaxis))
	         IF ( istat .EQ. str_eq ) GOTO 150  ! got a match
	      GOTO 100

* ... axis doesn't exist - perhaps it is a dimension with no variable defn
* *sh* 3/99: or perhaps a 2D variable with name matching the dimension name
*  either way -- not an error condition if we need to synthesize an axis ...
 120	      CALL TM_MAKE_BASIC_AXIS( name, 1.0D0, 1.0D0, npts,
     .                                 iaxis, status )
              IF ( status .NE. merr_ok ) GOTO 5900

* save a pointer to the axis just found
 150          IF (idim.LE.4) tmplines(idim) = iaxis

 200       CONTINUE

* reorder the axes into the most likely order if not explicitly given
           IF ( ds_ordering(1,tmvar) .EQ. unspecified_int4 ) THEN
              CALL TM_AXIS_ORDER(tmplines, perm, ds_ordering(1,tmvar),
     .                           nvdims, vname(:vlen))
           ENDIF

* ... make a sorted list of the relevant axes in grid_dims
*     and a list of the index ordering of the axes in indices
	ndim = 0
	DO 210 idim = 1, 4
	   trans(idim) =  ABS(ds_ordering(idim,tmvar))
	   grid_dims(idim) = trans(idim)
	   indices(idim) = idim
	   IF (trans(idim) .NE. 0 ) THEN
	      ndim = ndim + 1
	   ENDIF
 210	CONTINUE
	DO 220 idim = 1, ndim
	   DO 215 idim2 = idim+1, ndim
	      IF (grid_dims(idim) .GT. grid_dims(idim2)) THEN
	         dim = grid_dims(idim)
	         grid_dims(idim ) = grid_dims(idim2)
	         grid_dims(idim2) = dim
	      ENDIF
 215	   CONTINUE
 220	CONTINUE
	DO 230 idim = 1, ndim
	   i = 1
	   DO 225 idim2 = 1, ndim
	      IF (trans(idim2) .LT. trans(idim)) i = i + 1
 225	   CONTINUE
	   indices(idim) = i
 230	CONTINUE

* save the axis names in the proper order
           DO 240 idim = 1, 4
              ds_grid_start(idim,tmvar) = 1
              ds_grid_end  (idim,tmvar) = 1
 240       CONTINUE
           DO 250 idim = 1, ndim
              dim = trans(idim)
              iaxis = tmplines(indices(idim))
	      idim2 = grid_dims(idim)
              temp_axnams(idim2,tmpgrid) = line_name(iaxis)
              ds_grid_end(idim2,tmvar)   = line_dim(iaxis)
* ... flag reverse-ordered coordinate axis of variable (2/95)
	      IF ( reversed(iaxis) ) ds_ordering(idim,tmvar)
     .			      = -1 * ds_ordering(idim,tmvar)
 250       CONTINUE

* is the temporary grid we just created unique ?
	   igrid = 0
 300	   CALL TM_NEXT_TMP_GRID( igrid, *320)
	      grid_cnt = tmp_nam_ptr(igrid)
              DO 310 idim = 1, 4
                 IF ( temp_axnams(idim,tmpgrid )
     .          .NE.  temp_axnams(idim,grid_cnt) ) GOTO 300
 310          CONTINUE
* ... not unique - exact match with one that already exists
              GOTO 350
 320	   CONTINUE

* allocate a new grid
* find next location to store info; if too many grids - give error
	  CALL TM_ALLO_TMP_GRID(igrid, status)
	  IF (status .NE. merr_ok) GOTO 5900

* make up a unique grid name "GXXnn" where XX is hashed from data set name
          grid_name(igrid) = 'G'//cd_dset_code(dset)(:2)
          CALL TM_NEW_GRID_NAME( grid_name(igrid), name )
          grid_name(igrid) = name

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
 5900	RETURN

	END
