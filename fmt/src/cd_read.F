	SUBROUTINE CD_READ ( dset, ivar,
     .                       a_lo_x, a_lo_y, a_lo_z, a_lo_t,
     .                       a_hi_x, a_hi_y, a_hi_z, a_hi_t,
     .			     use_lo, use_hi,
     .			     var, sf_num, status )
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
* Reads a 4D block of data (lower D being a degenerate case with dimensions
* being 1 point thick) from a netCDF file
*
* replaced "elif" syntax with
*       else
*          if
*  for SGI port  - kob 4/8/92
*
* written 2/92 by Steve Hankin for PMEL/TMAP

* bug fix 10/92 - modulo reads require use_lo/hi .NE. dimensioned lo/hi
*         11/92 - allow 24 character variable names
* 5/94 - allow reading of DOUBLE variables using routine cd_rd_r8_as_r4.c
* 8/94 - modulo error message (5400) prints wrong axis
* 2/95 - small change to allow negative ds_ordering to represent reversed axis
* 10/96 *kob* - Linux port - had to move variable assignments off of
*		declaration line.

* arguments:
*	dset        - internal data set # (from GT_INIT_DSET)
*	ivar        - position in common of variable information
*       actual_dims - "a_" full dimensions of data_grid array passed
*	use_lo      - low indices of model data block desired
*	use_hi      - high indices of model data block desired
*	var         - array containing model data (output)
*	sf_num      - stepfile number (returned from GT_FIND_TSTEP (output)
*	status      - error status upon completion (output)
*
	INTEGER		dset, ivar, sf_num, status, ndim,
     .                  a_lo_x, a_lo_y, a_lo_z, a_lo_t,
     .                  a_hi_x, a_hi_y, a_hi_z, a_hi_t,
     .			use_lo(4), use_hi(4)
	REAL		var(a_lo_x:a_hi_x,
     .                      a_lo_y:a_hi_y,
     .                      a_lo_z:a_hi_z,
     .                      a_lo_t:a_hi_t)

* include files
#ifdef unix
        include 'tmap_format/netcdf.decl'
        include 'tmap_format/netcdf.inc'
	include 'tmap_format/tmap_errors.parm'
	include 'tmap_format/tmap_dims.parm'		! data set dimension
	include 'tmap_format/gt_lib.parm'		! axis setup params
	include 'tmap_format/xdset_info.cmn_text'	! data set info common
	external xdset_info_data
	include 'tmap_format/xstep_files.cmn_text'	! step file common
	external xstep_files_data
        include 'tmap_format/xdset_info.cd_equiv'
#ifdef sun
        include 'tmap_format/xmachine_byte.cmn'
#else
        include 'tmap_format/xmachine_int1.cmn'
#endif
#else
        INCLUDE 'TMAP_FORMAT:NETCDF.DECL'
        INCLUDE 'TMAP_FORMAT:NETCDF.INC'
	INCLUDE 'TMAP_FORMAT:TMAP_DIMS.PARM'		! Data set dimension
	INCLUDE 'TMAP_FORMAT:TMAP_ERRORS.PARM'		! err code PARM defs
	INCLUDE 'TMAP_FORMAT:GT_LIB.PARM'		! Axis setup params
	INCLUDE 'TMAP_FORMAT:XDSET_INFO.CMN'		! Data set info common
	INCLUDE 'TMAP_FORMAT:XSTEP_FILES.CMN'		! Step file common
	INCLUDE 'TMAP_FORMAT:XMACHINE_BYTE.CMN'
        INCLUDE 'TMAP_FORMAT:XDSET_INFO.CD_EQUIV'
#endif

* local definitions
* *kob* explicit definitions not allowed on decl line in linux...
*       moved it to parameter line 10/96
        INTEGER      TM_LENSTR1,
     .               cdfid, varid, idim, lasdim, cdfstat, dim, idim2,
     .               start(4), size(4), totsize, dim_lo(4), dim_hi(4)
	CHARACTER*1	dimnam(4)
	DATA  		dimnam/'I','J','K','L'/
	CHARACTER*13	TM_STRING

* special equivalence (cannot include in tmap_format/xdset_info.cd_equiv 
* because "BYTE" isn't supported on DECstation)
* 10/96 reorder the if def because linux cpp didn't like it

#ifdef unix
#ifdef sun
      BYTE cd_data_type(maxvars)
#else
      INTEGER*1 cd_data_type(maxvars)
#endif
#else
      BYTE cd_data_type(maxvars)
#endif
      EQUIVALENCE ( ds_precision, cd_data_type )

* initialize
        cdfid = sf_lunit(sf_num)
        varid = cd_varid(ivar)
        dim_lo(1) = a_lo_x
        dim_lo(2) = a_lo_y
        dim_lo(3) = a_lo_z
        dim_lo(4) = a_lo_t
        dim_hi(1) = a_hi_x
        dim_hi(2) = a_hi_y
        dim_hi(3) = a_hi_z
        dim_hi(4) = a_hi_t

* check for error conditions
        DO 10 idim = 1, 4
	   IF ( use_lo(idim) .LT. ds_grid_start(idim,ivar) .OR.
     .	        use_hi(idim) .GT. ds_grid_end(idim,ivar) ) GOTO 5200
	   IF ( use_lo(idim) .GT. use_hi(idim)
     .    .OR.  use_lo(idim) .LT. dim_lo(idim)
     .    .OR.  use_hi(idim) .GT. dim_hi(idim) ) GOTO 5300
 10     CONTINUE
        lasdim = 0
        DO 20 idim = 1, 4
           IF ( ds_ordering(idim,ivar) .EQ. 0 ) GOTO 20
           IF ( ABS(ds_ordering(idim,ivar)) .LE. lasdim )
     .          write (6,*) 'permuted CDF order not supported'
           lasdim = ABS(ds_ordering(idim,ivar))
 20     CONTINUE

* check for an attempt to read into a sub-portion of a grid
* (fully supported in GT and TS libs but only partially supported here)
* it is supported if the sub-region is contiguous - i.e. the sub-limits
* are along the slowest axis of non-unity size
        DO 60 idim = 1, 4
           IF ( dim_lo(idim) .NE. use_lo(idim)
     .     .OR. dim_hi(idim) .NE. use_hi(idim) ) THEN
              DO 50 idim2 = idim+1, 4
 50           IF ( dim_lo(idim2) .NE. dim_hi(idim2) ) GOTO 5400
           ENDIF
 60     CONTINUE

* determine size from the low/hi limits requested
* and compute lower corner relative to the hyper slab in the  netCDF file
* skip unused dimensions resulting from NORMAL axes
        totsize = 1
	ndim = 0
        DO 100 idim = 1, 4
           dim = ABS(ds_ordering(idim,ivar))
           IF ( dim .EQ. 0 ) GOTO 500
	   ndim = ndim + 1
           size(idim) = use_hi(dim) - use_lo(dim) + 1
           start(idim) = use_lo(dim) - ds_grid_start(dim,ivar) + 1
           totsize = totsize*size(idim)
 100    CONTINUE

* read the data
 500	IF ( cd_data_type(ivar) .EQ. NCDOUBLE ) THEN
* read double precision variables (requires special routine) 5/94
	   CALL CD_RD_R8_AS_R4(cdfid, varid, start, size, ndim,
     .             var(use_lo(1),
     .                 use_lo(2),
     .                 use_lo(3),
     .                 use_lo(4)) , cdfstat)
	   IF ( cdfstat .NE. NCNOERR ) GOTO 5500
	ELSE
 	   CALL NCVGT(cdfid, varid, start, size,
     .             var(use_lo(1),
     .                 use_lo(2),
     .                 use_lo(3),
     .                 use_lo(4)) , cdfstat)
           IF (cdfstat .NE. NCNOERR ) GOTO 5100

* convert data type if necessary
           IF ( cd_data_type(ivar) .EQ. NCFLOAT ) THEN
              CONTINUE
           ELSEIF ( cd_data_type(ivar) .EQ. NCLONG ) THEN
              CALL TM_I4_TO_R4( var, var, totsize)
           ELSEIF ( cd_data_type(ivar) .EQ. NCSHORT ) THEN
              CALL TM_I2_TO_R4( var, var, totsize)
           ELSE   ! NCBYTE or NCCHAR
              CALL TM_I1_TO_R4( var, var, totsize)
           ENDIF
	ENDIF

* if data has mean or range removed rescale it
        IF ( cd_scaled( ivar ) ) CALL TM_SCALE(
     .       cd_scale_factor(ivar),cd_add_offset(ivar),var,totsize)

* successful completion
	status = merr_ok
	RETURN

* error messages
 5100      CALL TM_ERRMSG
     .     ( cdfstat+pcdferr, status, 'CD_READ', cdfid,
     .       varid, 'Failed attempting READ', no_errstring, *5900 )

 5200      CALL TM_ERRMSG (merr_outofgrid, status, 'CD_READ',
     .     dset, no_stepfile, 'variable '//
     .     ds_var_code(ivar)(:TM_LENSTR1(ds_var_code(ivar)))//
     .     ' requested '//
     .	   dimnam(idim)//'='//TM_STRING(DBLE(use_lo(idim)))//'-'//
     .	   TM_STRING(DBLE(use_hi(idim))), no_errstring, *5900)

 5300      CALL TM_ERRMSG(merr_badsubscr, status, 'CD_READ',
     .     dset, no_stepfile, 'for requested '//dimnam(idim)//
     .     ' subscripts', no_errstring, *5900)

 5400      CALL TM_ERRMSG(merr_notsupport, status, 'CD_READ',
     .     dset, no_stepfile,
     .     'modulo or random access on netCDF mid-dimension: '//
     .     dimnam(idim), ' ', *5900)

 5500	IF ( status .EQ. -1 ) THEN
	   CALL TM_ERRMSG( cdfstat+pcdferr, status, 'CD_RD_R8_AS_R4',
     .	             cdfid, varid,
     .		    'unable to allocate memory for doub. prec. read',
     .		     no_errstring, *5900 )
	ELSE
	   CALL TM_ERRMSG( cdfstat+pcdferr, status, 'CD_RD_R8_AS_R4',
     .	             cdfid, varid, no_errstring, no_errstring, *5900 )
	ENDIF

 5900	RETURN
	END
