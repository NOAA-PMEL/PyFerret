	SUBROUTINE CD_READ_REMOTE ( dset, rvar_varid,
     .		a_lo_x, a_lo_y, a_lo_z, a_lo_t, a_lo_e, a_lo_f,
     .		a_hi_x, a_hi_y, a_hi_z, a_hi_t, a_hi_e, a_hi_f,
     .		use_lo, use_hi, stride, grid_start, 
     .		grid_end, grid_order, var, sf_num, status )

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
*
* Reads a N-D block of data (3D, 2D being degenerate cases w/ 1 dimension being
* 1 time series) from TMAP data sets with LET/REMOTE definition.
*
* written 2/2013 by Ansley Manke PMEL/TMAP, from TM_CHOOSE_READ and CD_READ
* V751 *acm* 6/19 Remove unused declarations
* 11/2022 *acm* Code cleanup: Modernize DO-loop syntax. 

* include files
        include 'netcdf.inc'
        include 'tmap_dims.parm'                ! data set dimension
        include 'tmap_errors.parm'              ! err code parm defs
#include "gt_lib.parm"
        include 'xtm_grid.cmn_text'             ! grid info
        external xgt_grid_data
        include 'xdset_info.cmn_text'   ! data set info common
        external xdset_info_data
        include 'xdset_info.cd_equiv'
	include 'xstep_files.cmn_text'	! step file common
	external xstep_files_data

* arguments:
*	dset        - internal data set # (from GT_INIT_DSET)
*       rvar_varid  - varid in ftds dataset to read
*       actual_dims - "a_" full dimensions of data_grid array passed
*	use_lo      - low indices of model data block desired
*	use_hi      - high indices of model data block desired
*	stride	    - every nth point to be read from disk file
*       gridlines   - lists the line ids that are part of the remote-variable grid
*	var         - array containing model data (output)
*	sf_num      - stepfile number (returned from GT_FIND_TSTEP (output)
*	status      - error status upon completion (output)
*
	INTEGER		dset, rvar_varid, sf_num, status, ndim,
     .		a_lo_x, a_lo_y, a_lo_z, a_lo_t, a_lo_e, a_lo_f,
     .		a_hi_x, a_hi_y, a_hi_z, a_hi_t, a_hi_e, a_hi_f,
     .		use_lo(*), use_hi(*), stride(*), 
     .		grid_start(*), grid_end(*), grid_order(*)
	REAL     	var(a_lo_x:a_hi_x,
     .                      a_lo_y:a_hi_y,
     .                      a_lo_z:a_hi_z,
     .                      a_lo_t:a_hi_t,
     .                      a_lo_e:a_hi_e,
     .                      a_lo_f:a_hi_f),
     .			baddat


* local definitions
* *kob* explicit definitions not allowed on decl line in linux...
*       moved it to parameter line 10/96
	LOGICAL      do_warn, already_scaled, got_it, tf2, 
     .               coordvar
        INTEGER      TM_LENSTR1,
     .               cdfid, varid, idim, lasdim, cdfstat, dim, idim2,
     .               start(nferdims), filsize(nferdims), totsize, 
     .               dim_lo(nferdims),dim_hi(nferdims), imap(nferdims), 
     .		     trans(nferdims), memsize(nferdims), count(nferdims),
     .		     grid_dims(nferdims), fildim, memdim, filstride(nferdims),
     .               maxlen, attlen, attoutflag, vtype, nvdims, nvatts, 
     .               outflag, dlen, err_status, i, permuted, strided
        REAL vdims(10), val
	CHARACTER*1	dimnam(nferdims)
	CHARACTER*13	TM_STRING
        CHARACTER*128   buff, vname
	DATA  		dimnam/'I','J','K','L','M','N'/
	CHARACTER*1024 ebuff, upstr  ! for nf_ error message

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
        cdfid = sf_lunit(dset)
        varid = rvar_varid
        dim_lo(1) = a_lo_x
        dim_lo(2) = a_lo_y
        dim_lo(3) = a_lo_z
        dim_lo(4) = a_lo_t
        dim_lo(5) = a_lo_e
        dim_lo(6) = a_lo_f
        dim_hi(1) = a_hi_x
        dim_hi(2) = a_hi_y
        dim_hi(3) = a_hi_z
        dim_hi(4) = a_hi_t
        dim_hi(5) = a_hi_e
        dim_hi(6) = a_hi_f
	permuted  = 0
	strided  = 0
        do_warn = .TRUE.

* check for error conditions
	DO 10 idim = 1, nferdims
	      IF ( use_lo(idim) .LT. grid_start(idim) .OR.
     .	        use_hi(idim) .GT. grid_end(idim) ) GOTO 5200
	      IF ( use_lo(idim) .GT. use_hi(idim)
     .       .OR.  use_lo(idim) .LT. dim_lo(idim)
     .       .OR.  use_hi(idim) .GT. dim_hi(idim) ) GOTO 5300
 10     CONTINUE
* summarize the axis permutation specification in ds_order

        lasdim = 0
	ndim = 0
        DO 20 idim = 1, nferdims
           IF ( grid_order(idim) .EQ. 0 ) GOTO 20
	   ndim = ndim + 1
           IF ( ABS(grid_order(idim)) .LE. lasdim )
     .          permuted = 1
           lasdim = ABS(grid_order(idim))
 20     CONTINUE

* ... compress the list of relevant axes into grid_dims
	DO 30 idim = 1, nferdims
	   trans(idim) = grid_order(idim)
	   grid_dims(idim) = trans(idim)
 30	CONTINUE

	IF ( trans(1) .EQ. 0 ) THEN
* ... trap for scalar (no dimension) netCDF variables
	   trans(1) = 1
	   grid_dims(1) = 1
	ENDIF
	DO 34 idim = 1, ndim
	   DO 32 idim2 = idim+1, ndim
	      IF (grid_dims(idim) .GT. grid_dims(idim2)) THEN
	         dim = grid_dims(idim)
	         grid_dims(idim ) = grid_dims(idim2)
	         grid_dims(idim2) = dim
	      ENDIF
 32	   CONTINUE
 34	CONTINUE

* check for an attempt to read into a sub-portion of a grid
* (fully supported in GT and TS libs but only partially supported here)
* it is supported if the sub-region is contiguous - i.e. the sub-limits
* are along the slowest axis of non-unity size
        DO 60 idim = 1, nferdims
           IF ( dim_lo(idim) .NE. use_lo(idim)
     .     .OR. dim_hi(idim) .NE. use_hi(idim) ) THEN
              DO 50 idim2 = idim+1, nferdims
                 IF ( dim_lo(idim2) .NE. dim_hi(idim2) ) GOTO 5400
 50	      CONTINUE
           ENDIF
 60     CONTINUE

* determine size from the low/hi limits requested
* and compute lower corner relative to the hyper slab in the  netCDF file
* skip unused dimensions resulting from NORMAL axes
        DO 100 idim = 1, ndim
           fildim = trans(idim)
	   memdim = grid_dims(idim)
           filsize(fildim) = use_hi(memdim) - use_lo(memdim) + 1
	   count(fildim)   = (filsize(fildim)-1)/stride(memdim) + 1
           start(fildim)   = use_lo(memdim) 
	   filstride(fildim) = stride(memdim)
           IF (stride(memdim) .GT. 1) strided = 1
 100    CONTINUE

*  compute the memory array size (for cases where strides are used)
*  note: transposed compared to filsize
* *kob* 7/00 - totsize calculation now takes striding into account
	totsize = 1
	DO 200 idim = 1, nferdims
	   memsize(idim) =  (use_hi(idim)-use_lo(idim))/stride(idim) + 1
	   totsize = totsize * memsize(idim)
 200	CONTINUE

* compute the map vector (for strides, reversals, and permutations)
	DO 210 idim = 1, nferdims
 	   imap(idim) = 0
 210	CONTINUE
* for netCDF V3 the imap array is set up in number of array elements, not bytes
	imap(trans(1)) = 1

	DO 220 idim = 2,ndim
	   imap(trans(idim)) = imap(trans(idim-1))
     .			     * (1 + (memsize(grid_dims(idim-1))-1))
 220	CONTINUE
* ... compress the gaps out of the map -- e.g. XYT ==> has a zero in imap(3)
*     and other arrays
	idim2 = 0
	DO 225 idim = 1, nferdims
	   IF (imap(idim) .EQ. 0) GOTO 225
	   idim2 = idim2 + 1
	   imap(idim2)      = imap(idim)
	   filsize(idim2)   = filsize(idim)
	   start(idim2)     = start(idim)
	   count(idim2)     = count(idim)
	   filstride(idim2) = filstride(idim)
 225	CONTINUE

* read the data

        already_scaled = .FALSE.
        status = merr_ok

        CALL CD_READ_SUB(cdfid, varid, ndim, start, count,
     .		filstride, imap,
     .		var(use_lo(1),
     .              use_lo(2),
     .              use_lo(3),
     .              use_lo(4),
     .              use_lo(5),
     .              use_lo(6)), permuted, strided, cdfstat)
        IF ( cdfstat .NE. NF_NOERR ) GOTO 5500

* Data from F-TDS data reads should not be scaled. If it were,
* handle that here...


* successful completion

	status = merr_ok
	RETURN

* error messages
 5100	CALL TM_ERRMSG
     .     ( cdfstat+pcdferr, status, 'CD_READ', cdfid,
     .       varid, 'Failed attempting READ', no_errstring, *5900 )

 5200   CONTINUE 
	CALL CD_GET_VAR_INFO (dset, varid, vname, vtype, nvdims,
     .          vdims, nvatts, coordvar, outflag, status)
	CALL TM_ERRMSG (merr_outofgrid, status, 'CD_READ_REMOTE',
     .     dset, no_stepfile, 'variable '//
     .     vname(:TM_LENSTR1(vname))//
     .     ' requested '//
     .	   dimnam(idim)//'='//TM_STRING(DBLE(use_lo(idim)))//'-'//
     .	   TM_STRING(DBLE(use_hi(idim))), no_errstring, *5900)

 5300	CALL TM_ERRMSG(merr_badsubscr, status, 'CD_READ',
     .     dset, no_stepfile, 'for requested '//dimnam(idim)//
     .     ' subscripts', no_errstring, *5900)

 5400	CALL TM_ERRMSG(merr_notsupport, status, 'CD_READ',
     .     dset, no_stepfile,
     .     'modulo or random access on netCDF mid-dimension: '//
     .     dimnam(idim), ' ', *5900)

 5500	IF ( status .EQ. -1 ) THEN
 
	   CALL TM_ERRMSG( cdfstat+pcdferr, status, 'CD_READ',
     .	             cdfid, varid,
     .		    'unable to allocate memory for doub. prec. read',
     .		     no_errstring, *5900 )
	ELSE
	
           IF (cdfstat .LT. NF_NOERR) THEN
	      err_status = merr_nc_open 
	      CALL CD_TRANSLATE_ERROR (cdfstat, ebuff)
           ELSE IF (cdfstat .EQ. nf_interrupt) THEN
	      err_status = merr_read_interrupt
      
	      CALL CD_GET_VAR_INFO (dset, varid, vname, vtype, nvdims,
     .          vdims, nvatts, coordvar, outflag, status)

	      ebuff = 'Reading variable '//vname(:TM_LENSTR1(vname))//
     .          ', interrupted from command line'
           ENDIF

           CALL TM_ERRMSG
     .     ( err_status, status, 'CD_REMOTE_READ', dset,
     .        no_stepfile, ebuff, no_errstring, *5900 )

	ENDIF

 5900	RETURN
	END
