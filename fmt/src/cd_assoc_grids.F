	SUBROUTINE CD_ASSOC_GRIDS (dset, cdfid, reversed, 
     .				   temp_axnams, status)

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
* within a netCDF file
* Checks that all variables in the data set have associated grids.

* Programmer Steve Hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program

* revision 0.0 - 2/92
* 2/95 - added support for reverse-ordered coordinate axes
* Linux port -kob - 3/97 - Modify include for tmap_dset.parm to be a
*			   preprocessor include for F90 - needed because
*			   tmap_dset.parm contains an ifdef

* argument definitions:
*       dset        - TMAP data set pointer
*	cdfid	    - netCDF id number for already opened CDF file
*       reversed    - logical array to record which axes have reversed coords
*       temp_axnams - temporary storage of axis names to construct grids
*	status	    - return status

* include files
#ifdef unix
        include 'tmap_format/netcdf.decl'
        include 'tmap_format/netcdf.inc'
        include 'tmap_format/xbuild_grids.cmn'
	include 'tmap_format/tmap_errors.parm'
	include 'tmap_format/tmap_dims.parm'
#include "tmap_format/tmap_dset.parm"
	include 'tmap_format/gt_lib.parm'
	include 'tmap_format/xdset_info.cmn_text'
	external xdset_info_data
	include 'tmap_format/xtm_grid.cmn_text'
	external xgt_grid_data
        include 'tmap_format/xdset_info.cd_equiv'
#else
        INCLUDE 'TMAP_FORMAT:NETCDF.DECL'  ! untested
        INCLUDE 'TMAP_FORMAT:NETCDF.INC'
        INCLUDE 'TMAP_FORMAT:XBUILD_GRIDS.CMN'
	INCLUDE 'TMAP_FORMAT:TMAP_DIMS.PARM'
	INCLUDE 'TMAP_FORMAT:TMAP_DSET.PARM'
	INCLUDE 'TMAP_FORMAT:GT_LIB.PARM'
	INCLUDE 'TMAP_FORMAT:XDSET_INFO.CMN'
	INCLUDE 'TMAP_FORMAT:XTM_GRID.CMN'
        INCLUDE 'TMAP_FORMAT:XDSET_INFO.CD_EQUIV'
#endif

* argument declarations
	INTEGER	 dset, cdfid, status
        LOGICAL reversed(0:max_lines)
	CHARACTER*16 temp_axnams(4,max_temp_grid)

* local parameter definitions:
      INTEGER     str_eq
      PARAMETER ( str_eq = 0 )

* local variable declarations:
* move do_warn assignment to paramater for linux *kob*
      LOGICAL CD_GET_ATTRIB,
     .        got_it, do_warn
      PARAMETER (do_warn = .TRUE.)
      INTEGER STR_CASE_BLIND_COMPARE,
     .        igrid, iaxis, istat, tmvar, ivar, ndim, idim
      CHARACTER*16 gname
* variables for final (unnecessary?) check
      CHARACTER TM_STRING*13, vname*16, buff*64
      INTEGER TM_LENSTR1,
     .        vartyp, nvdim, vdims(6), nvatts, dim, npts, vlen
* *kob* explicit definitions not allowed on decl line in linux...moved it to parameter line
      CHARACTER*1     dimnam(4)
      DATA dimnam/'I', 'J', 'K', 'L'/

* loop through each variable in this TMAP data set
* if it has a "parent_grid" then associate it with that grid
* if not then make sure it has a grid associated

        DO 1000 tmvar = 1,maxvars
           IF ( ds_var_setnum(tmvar) .NE. dset ) GOTO 1000
           ivar = cd_varid(tmvar)

* does it point explicitly to a parent grid ?
          got_it = CD_GET_ATTRIB( cdfid, ivar, 'parent_grid',
     .                         .NOT.do_warn, ' ', gname, 16 )   !gridnamlen
          IF ( .NOT.got_it ) THEN
             IF ( ds_grid_number(tmvar) .LT. firstgrid
     .       .OR. ds_grid_number(tmvar) .GT. lastgrid ) GOTO 5100
             GOTO 1000
          ENDIF

* locate the grid it points to
          DO 100 igrid = firstgrid, lastgrid
             istat = STR_CASE_BLIND_COMPARE(gname, grid_name(igrid))
             IF ( istat .EQ. str_eq ) THEN
                ds_grid_number(tmvar) = igrid
                GOTO 110
             ENDIF
 100      CONTINUE
          GOTO 5100  ! no parent grid defined

* set ds_ordering if not explicitly given in file
* e.g. map X NORMAL Z T to 1 3 4 0
 110      IF ( ds_ordering(1,tmvar) .EQ. unspecified_int4 ) THEN
             DO 190 idim = 1, 4
 190         ds_ordering(idim,tmvar) = 0
             ndim = 0
             DO 200 idim = 1, 4
                IF ( grid_line(idim,igrid) .EQ. mpsnorm ) GOTO 200
                ndim = ndim + 1
                ds_ordering(ndim,tmvar) = idim
 200         CONTINUE
          ENDIF

* count the number of axes in the grid
          ndim = 0
          DO 220 idim = 1, 4
             IF ( grid_line(idim,igrid).NE.mpsnorm ) ndim = ndim + 1
 220      CONTINUE
          
* set ds_grid_start if not explicitly given in file
          IF ( ds_grid_start(1,tmvar) .EQ. unspecified_int4 ) THEN
             DO 300 idim = 1, 4
 300         ds_grid_start(idim,tmvar) = 1
          ENDIF

* set ds_grid_end if not explicitly given in file
          IF ( ds_grid_start(1,tmvar) .EQ. unspecified_int4 ) THEN
             DO 400 idim = 1, 4
                iaxis = grid_line(idim,igrid)
                IF ( iaxis .EQ. mpsnorm ) THEN
                   ds_grid_end(idim,tmvar) = 1
                ELSE
                   ds_grid_end(idim,tmvar) = line_dim(iaxis)
                ENDIF
 400         CONTINUE
          ENDIF

* check that the parent lo/hi limits are consistent with the variable slab size
          CALL NCVINQ(cdfid, ivar, vname, vartyp, nvdim, vdims,
     .                 nvatts,istat)
          vlen = TM_LENSTR1(vname)
          IF ( ndim .NE. nvdim ) GOTO 5300 
          DO 500 idim = 1, 4
             dim = ds_ordering(idim,tmvar)
             IF ( dim .EQ. 0 ) GOTO 500
!             IF ( dim .EQ. unspecified_int4 ) dim = idim
             CALL NCDINQ( cdfid, vdims(idim), buff, npts, istat )
             IF ( npts .NE.
     .          ds_grid_end(dim,tmvar)-ds_grid_start(dim,tmvar)+1 )
     .               GOTO 5200
 500      CONTINUE

* flag (negative) the axes that have reverse-ordered coordinates
	  DO 610 idim = 1, 4
	     dim = ds_ordering(idim,tmvar)
             IF ( dim .EQ. 0 ) GOTO 610
* At this stage the axis pointers have not yet been inserted into the
* parent grid definitions (we still have to check for duplicate def'ns, etc.)
* So we need to check by name (among the axes defined in this data set)
* to find this axis and see if this axis is reversed.
	     IF ( grid_line(dim,igrid) .EQ. unspecified_int4 ) THEN
	       vname = temp_axnams(dim, igrid - firstgrid + 1)
	       DO 600 iaxis = firstax, lastax
                 IF ( vname .EQ. line_name(iaxis) ) THEN
	           IF (reversed(iaxis)) ds_ordering(idim,tmvar) = -1*dim
                   GOTO 610
                 ENDIF
 600           CONTINUE
	    ENDIF
 610	  CONTINUE

 1000   CONTINUE

* successful completion
 	status = merr_ok
        RETURN

* error messages
 5100   CALL TM_ERRMSG(merr_unkgrid, status, 'CD_ASSOC_GRIDS',
     .       dset, no_stepfile,
     .       'CDF grid name: '//gname, no_errstring, *5900)

 5200   CALL TM_ERRMSG(merr_badsubscr, status, 'CD_ASSOC_GRIDS',
     .       dset, no_stepfile, 'CDF variable "'//vname(:vlen)//
     .       '" on '//dimnam(dim)//'='//
     .       TM_STRING(DBLE(ds_grid_start(dim,tmvar)))//'-'//
     .       TM_STRING(DBLE(ds_grid_end  (dim,tmvar))),
     .       no_errstring, *5900)

 5300   CALL TM_ERRMSG(merr_gridpredef, status, 'CD_ASSOC_GRIDS',
     .       dset, no_stepfile,
     .       '# of dims in variable '//vname(:vlen)//
     .       ' does not match grid '//gname, no_errstring, *5900)

* error exit
 5900   RETURN
	END
