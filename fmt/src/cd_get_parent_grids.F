	SUBROUTINE CD_GET_PARENT_GRIDS (cdfid,temp_axnams,status)

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
* Loads the common block XGRID with explicitly given grid definitions from a
* netCDF file - i.e. grid definitions for variables that have a
* "parent_grid = name" attribute

* Programmer Steve Hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program

* revision 0.0 - 2/92 - adds a new data set format to TMAP libraries
*              - based on TM_READ_GRIDFILE
* 10/96 *kob* - Linux port - had to move variable assignments off of
*		declaration line.
* V500: 4/99 *kob* - increase size of temp_axnams to 64

* argument definitions:
*	cdfid	       - netCDF id number for already opened CDF file
*       temp_axnams    - 4 axis names per grid: temp until grids are assembled
*	status	       - return status

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
	include 'xunits.cmn_text'
	external xunits_data
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
	INCLUDE 'TMAP_FORMAT:XUNITS.CMN'
#endif

* argument declarations
	INTEGER	 cdfid, status
* 4/99 *kob*
        CHARACTER*64 temp_axnams(4,max_temp_grid)

* local parameter definitions:
      INTEGER     str_eq
      PARAMETER ( str_eq = 0 )

* local variable declarations:
* *kob* explicit definitions not allowed on decl line in linux...
*       moved it to parameter line 10/96
      LOGICAL CD_GET_ATTRIB, CD_GET_ATTVAL,
     .        got_it, do_warn, centers
      PARAMETER (do_warn = .TRUE. ) 
      PARAMETER (centers = .TRUE. )
      INTEGER TM_LENSTR1, TM_FIND_GRID_SLOT,
     .        STR_CASE_BLIND_COMPARE, STR_UPCASE, 
     .        ndims, nvars, ngatts, recdim, igrid, vback,
     .        ivar, vartyp, nvdims, vdims(8), nvatts, vlen,
     .        grid_cnt, cdfstat, istat, idim, npts
      CHARACTER*24 vname, words(8)
      CHARACTER*132 TM_CMPRSS, buff, cbuff

      INTEGER tt  ! nice short name
      EQUIVALENCE (tt,lunit_errors)

* get the vital statistics for this netCDF file
        CALL NCINQ( cdfid, ndims, nvars, ngatts, recdim, cdfstat )
        IF ( cdfstat .NE. NCNOERR ) CALL TM_ERRMSG
     .     ( cdfstat+pcdferr, status, 'CD_GET_PARENT_GRIDS', cdfid,
     .        no_varid, no_errstring, no_errstring, *5900 )

!* is it a FERRET standard file ?
!        got_it = CD_GET_ATTRIB( cdfid, NCGLOBAL, 'DATA_TYPE',
!     .                        .NOT.do_warn, ' ', data_type, 24 )
!        IF (.NOT.got_it .OR. data_type.NE.'FERRET') RETURN 

* * * * loop through all variables in the net CDF file * * *
        DO 500 ivar = 1,nvars
      
* get the vital statistics about this variable
           CALL NCVINQ(cdfid, ivar, vname, vartyp, nvdims, vdims,
     .                 nvatts,cdfstat)
           IF (cdfstat.NE.NCNOERR) CALL TM_ERRMSG
     .     ( cdfstat+pcdferr,status,'CD_GET_PARENT_GRIDS',cdfid,ivar,
     .       no_errstring, no_errstring, *5900 )
           vlen = TM_LENSTR1(vname)

* is it a definition for a parent grid ? (1D based on 'grid_definition' )
          IF ( nvdims .NE. 1 ) GOTO 500
          CALL NCDINQ( cdfid, vdims, buff, npts, cdfstat )
          istat = STR_CASE_BLIND_COMPARE('grid_definition', buff(:15) )
          IF ( istat .NE. str_eq ) GOTO 500

* find next location to store info; if too many grids - give error
	  status = TM_FIND_GRID_SLOT ( igrid )
	  IF (status .NE. merr_ok) GOTO 5900

* get the mandatory list of axis names for this grid
          got_it = CD_GET_ATTRIB( cdfid, ivar, 'axes',
     .                            do_warn, vname(:vlen), buff, 132 )
          IF ( .NOT.got_it ) THEN
             CALL TM_NOTE(
     .            '"axes" attribute missing from grid definition', tt)
             CALL TM_NOTE('grid defintion ignored: '//vname(:vlen), tt)
             GOTO 500
          ENDIF

* store grid name (upper case)
          grid_cnt = igrid - firstgrid + 1
          lastgrid = igrid
          istat = STR_UPCASE( grid_name(igrid), vname )

* parse the list and store the axis names in the temporary grid definition
	  cbuff = TM_CMPRSS(buff)
	  DO 180 idim =1,4
 180      words(idim) = ' '
	  CALL TM_BREAK_STR(cbuff, words, 4)
          DO 185 idim = 1, 4
 185      istat = STR_UPCASE( temp_axnams(idim,grid_cnt), words(idim) )

* flag the axes that are normal in the grid so grid associations can be made
          DO 186 idim = 1, 4
             IF ( temp_axnams(idim,grid_cnt) .EQ. 'NORMAL' ) THEN
                grid_line(idim, igrid) = mpsnorm
             ELSE
                grid_line(idim, igrid) = unspecified_int4
             ENDIF
 186      CONTINUE

* get optional rotation parameter (not a string parameter)
          got_it = CD_GET_ATTVAL( cdfid, ivar, 'rotation', do_warn,
     .                      vname(:vlen), grid_rotation(igrid),1,vback )
          IF ( got_it .AND. grid_rotation(igrid).NE.0.0) THEN
             CALL TM_NOTE ('rotated grids not supported - using 0.0: '
     .                      //vname(:vlen), tt)
             grid_rotation(igrid) = 0.0
          ELSE
             grid_rotation(igrid) = 0.0    ! default
          ENDIF

* get optional inner/outer product list
          got_it = CD_GET_ATTRIB( cdfid, ivar, 'axis_assn',
     .                        do_warn, vname(:vlen), buff, 132 )
          IF ( got_it ) THEN
* ... parse the inner/outer list and store it in the grid definition
	     cbuff = TM_CMPRSS(buff)
	     DO 190 idim =1,4
 190         words(idim) = ' '
	     CALL TM_BREAK_STR(cbuff, words, 4)
             DO 200 idim = 1, 4
                istat = STR_CASE_BLIND_COMPARE('OUTER',words(idim))
                IF ( istat .EQ. str_eq ) THEN
                   grid_out_prod(idim,igrid) = .TRUE.
                ELSE
                   CALL TM_NOTE ('inner product grids not supported: '
     .                         //vname(:vlen), tt )
                   grid_out_prod(idim,igrid) = .TRUE.
                ENDIF
 200         CONTINUE
          ELSE
             DO 210 idim = 1, 4
 210         grid_out_prod(idim,igrid) = .TRUE.    ! default
          ENDIF

 500    CONTINUE

* successful completion
 	status = merr_ok
        RETURN

* error exit
 5900   RETURN
	END
