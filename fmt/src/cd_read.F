	SUBROUTINE CD_READ ( dset, ivar,
     .		a_lo_x, a_lo_y, a_lo_z, a_lo_t, a_lo_e, a_lo_f,
     .		a_hi_x, a_hi_y, a_hi_z, a_hi_t, a_hi_e, a_hi_f,
     .		use_lo, use_hi, stride,
     .		var, sf_num, status )
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
* 5/99 *sh* (V5.0) - implement strides, reversals, and permutations 
* 3/00 *kob* (v5.1) - use netcdf v3 calls to read double precision data
*		rather than having to call cd_rd_r8_as_r4 
* V510 6/00 *sh* - remove obsolete code left behind by 3/00 fix
*		 - if variable is scalar fill in trans and grid_dims
* V521 7/00 *kob* - modify calculation of totsize to take any striding into
*                   account.
* v531 5/01 *kob* - fix erroneous error report referencing cd_rd_r8_as_r4
* v552 4/03 *acm* - Changes for SET VAR/SCALEFAC/OFFSET: apply new scale
*                   and offset.
* V600 11/05 *acm*  Fix bug 491 properly -- send missing value to tm_scale
*                   so that missing data is NOT scaled by scale and offset attrs.
* V600  6/06 *acm*  needed call to CD_GET_VAR_INFO to define vname for NC_GET_ATTRIB
* V65  *acm* 1/10  Change all netcdf2-style calls and parameter names to 
*                  netcdf3-style calls, which are also consistent with netcdf4.
* V65  *acm* 1/10  Add #ifdef usingDODSf2cUnderscore, for 32-bit linux build
*                  linked with OPeNDAP library.
* V67  *acm* 1/11  Choose between routinesi from the NetCDF API: varm if permutations,
*                  vars if strides but no perm, var if no strides or perm.
* V68  *acm* 1/12  ifdef double_p for double-precision ferret.
* V674 2/12 *acm* 6D Ferret: use nferdims rather than 4 for dimension indexing
* v73+ *acm* 12/2017 Issue # 1846, writing user-variables with scale_factor, add_offset
*                    changes to tm_scale_nobad call.
* V74  *acm*  2/2018 Issue # 1861, handling missing- and _FillValue with scale and offset
*             4/2018 for the 1861 fix, need to deal with different missing_value and _FillValue
* V751 *acm*  5/2019 Issue # 1935.  If there is a _FillValue but no missing_value don't treat
*                    things as if missing_value = 0 ! 
* V751 *acm*  8/2019 Issue # 1948. Correct the logic for when there is scaled variable that has
*                    a missing_value but no _FillValue.
* 11/2022 *acm* Code cleanup: Modernize DO-loop syntax. 

* include files
        include 'netcdf.inc'
	include 'tmap_errors.parm'
	include 'tmap_dims.parm'		! data set dimension
#include "gt_lib.parm"
	include 'xdset_info.cmn_text'	! data set info common
	external xdset_info_data
	include 'xstep_files.cmn_text'	! step file common
	external xstep_files_data
        include 'xdset_info.cd_equiv'
        include 'xmachine_int1.cmn'


* arguments:
*	dset        - internal data set # (from GT_INIT_DSET)
*	ivar        - position in common of variable information
*       actual_dims - "a_" full dimensions of data_grid array passed
*	use_lo      - low indices of model data block desired
*	use_hi      - high indices of model data block desired
*	stride	    - every nth point to be read from disk file
*	var         - array containing model data (output)
*	sf_num      - stepfile number (returned from GT_FIND_TSTEP (output)
*	status      - error status upon completion (output)
*
	INTEGER		dset, ivar, sf_num, status, ndim,
     .		a_lo_x, a_lo_y, a_lo_z, a_lo_t, a_lo_e, a_lo_f,
     .		a_hi_x, a_hi_y, a_hi_z, a_hi_t, a_hi_e, a_hi_f,
     .			use_lo(nferdims), use_hi(nferdims), stride(nferdims)
	REAL     	var(a_lo_x:a_hi_x,
     .                      a_lo_y:a_hi_y,
     .                      a_lo_z:a_hi_z,
     .                      a_lo_t:a_hi_t,
     .                      a_lo_e:a_hi_e,
     .                      a_lo_f:a_hi_f)


* local definitions
* *kob* explicit definitions not allowed on decl line in linux...
*       moved it to parameter line 10/96
	LOGICAL      NC_GET_ATTRIB, 
     .               do_warn, already_scaled, got_it, tf1, tf2, 
     .               coordvar
        INTEGER      TM_LENSTR1,
     .               cdfid, varid, idim, lasdim, cdfstat, dim, idim2,
     .               start(nferdims), filsize(nferdims), totsize, 
     .               dim_lo(nferdims),dim_hi(nferdims), imap(nferdims), 
     .		     trans(nferdims), memsize(nferdims), count(nferdims),
     .		     grid_dims(nferdims), fildim, memdim, filstride(nferdims),
     .               maxlen, attlen, attoutflag, vtype, nvdims, nvatts, 
     .               outflag, dlen, err_status, i, permuted, strided
        REAL vdims(10), val, fill_in, bad_in
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
        cdfid = sf_lunit(sf_num)
        varid = cd_varid(ivar)
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
	   IF ( use_lo(idim) .LT. ds_grid_start(idim,ivar) .OR.
     .	        use_hi(idim) .GT. ds_grid_end(idim,ivar) ) GOTO 5200
	   IF ( use_lo(idim) .GT. use_hi(idim)
     .    .OR.  use_lo(idim) .LT. dim_lo(idim)
     .    .OR.  use_hi(idim) .GT. dim_hi(idim) ) GOTO 5300
 10     CONTINUE

* summarize the axis permutation specification in ds_order
        lasdim = 0
	ndim = 0
        DO 20 idim = 1, nferdims
           IF ( ds_ordering(idim,ivar) .EQ. 0 ) GOTO 20
	   ndim = ndim + 1
           IF ( ABS(ds_ordering(idim,ivar)) .LE. lasdim )
     .          permuted = 1
           lasdim = ABS(ds_ordering(idim,ivar))
 20     CONTINUE
* ... sort the list of relevant axes into grid_dims
	DO 30 idim = 1, nferdims
	   trans(idim) =  ABS(ds_ordering(idim,ivar))
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
 50           CONTINUE
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
           start(fildim)   = use_lo(memdim) - 
     .                         ds_grid_start(memdim,ivar) + 1
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
* v5.1 *kob* - Major revision.  Use nc v3 call to read data and convert values
*              to real/float.  Should work for all datatypes.
* v5.11 *kob* - add ifdef to check whether or not to append an underscore
*               to the v3 routine name.  This necessary because DODS netcdf
*               fortran jackets were configured assuming f2c, which appends
*               an additional underscore to routine names already containing
*               and underscore.  The real solution is to configure dods 
*               correctly, but for now, this kludge is applied
* v5.33 *sh* - actual reading is buried in a C subroutine
* V600  *acm*- Fix bug 419 (incompletely fixed before) send missing data
*              flag to TM_SCALE so that bad data is not scaled.
* V600  *acm* -Hack to let us read double precision data to be used to 
*              create an axis, by specifying an offset and reading the
*              data as double precision, applying the offset and THEN
*              converting to single precision.
*      ... when compiling as double precision, the data is read as double
*      ... and it will be scaled below.  So we just skip this hack.

        already_scaled = .FALSE.
        status = merr_ok
#ifndef double_p 

        IF (ds_var_scaleit( ivar ) ) THEN   ! reads data as double precision, if available.
           CALL CD_READ_SCALE(cdfid, varid, ndim, 
     .             ds_var_off(ivar), ds_var_scale(ivar), 
     .             ds_bad_flag(ivar),
     .             start, count,
     .		   filstride, imap,
     .             var(use_lo(1),
     .                 use_lo(2),
     .                 use_lo(3),
     .                 use_lo(4),
     .                 use_lo(5),
     .                 use_lo(6)), 
     .             permuted, strided, already_scaled, cdfstat, status)
           IF ( cdfstat .NE. NF_NOERR ) GOTO 5500           

           IF (status .NE. merr_ok) THEN   ! try to read as single precision floats
           
	      CALL CD_READ_SUB(cdfid, varid, ndim, start, count,
     .		   filstride, imap,
     .             var(use_lo(1),
     .                 use_lo(2),
     .                 use_lo(3),
     .                 use_lo(4),
     .                 use_lo(5),
     .                 use_lo(6)), permuted, strided, cdfstat)
	      IF ( cdfstat .NE. NF_NOERR ) GOTO 5500
           ENDIF

        ELSE
#endif
	   CALL CD_READ_SUB(cdfid, varid, ndim, start, count,
     .		   filstride, imap,
     .             var(use_lo(1),
     .                 use_lo(2),
     .                 use_lo(3),
     .                 use_lo(4),
     .                 use_lo(5),
     .                 use_lo(6)), permuted, strided, cdfstat)
	   IF ( cdfstat .NE. NF_NOERR ) GOTO 5500
#ifndef double_p
        ENDIF
#endif

* if data has mean or range removed rescale it.
* Do not scale missing data.

* Check for the scale factors again here (done on SET DATA in cd_get_1_dep_var,
* but the user may have reset the attributes by now.) If this is a file in a 
* descriptor file, and if scaling values exist for the variable, then do not reset 
* scale factors again here.
* Ticket 1980: initialize the scale attributes in case they don't exist.

        IF (.NOT.is_mc .OR. .NOT.cd_scaled(ivar)) THEN
           maxlen = 2
	   cd_scale_factor(ivar) = 1.
           CALL CD_GET_VAR_INFO (dset, varid, vname, vtype, nvdims,
     .         vdims, nvatts, coordvar, outflag, status)
           got_it = NC_GET_ATTRIB( dset, varid, 'scale_factor',
     .                     do_warn, vname, maxlen, attlen, 
     .                     attoutflag, buff, cd_scale_factor(ivar))
           maxlen = 2
	   cd_add_offset(ivar) = 0.
           tf2    = NC_GET_ATTRIB( dset, varid, 'add_offset',
     .                     do_warn, vname, maxlen, attlen, 
     .                     attoutflag, buff, cd_add_offset(ivar))

           cd_scaled(ivar) = got_it .OR. tf2
        ENDIF

* Scaled data - read, and switch Ferret's default missing flag for the file missing flag.
* If the file variable has a missing-flag use it, else the _FillValue

        IF ( cd_scaled( ivar ) ) THEN
           tf1 = NC_GET_ATTRIB( dset, varid, '_FillValue',
     .                     do_warn, vname, maxlen, attlen, 
     .                     attoutflag, buff, fill_in)
           tf2 = NC_GET_ATTRIB( dset, varid, 'missing_value',
     .                     do_warn, vname, maxlen, attlen, 
     .                     attoutflag, buff, bad_in)

* Consolidate to one bad-value - needed for the fix to 1861...

	   IF (tf1 .AND. tf2) CALL TM_SWITCH_BAD ( fill_in, bad_in, var, totsize )
	   IF (tf1 .AND. .NOT.tf2) bad_in = fill_in

	   CALL TM_SCALE_SWITCH_BAD ( cd_scale_factor(ivar), cd_add_offset(ivar), 
     .                     bad_in, ds_bad_flag(ivar), var, totsize )
        ENDIF

* if user has asked for new offset and/or scale, apply those, but
* not to any missing data.

        IF ( ds_var_scaleit(ivar) .AND. (.NOT. already_scaled) )
     .    CALL TM_SCALE_NOBAD(
     .       ds_var_scale(ivar), ds_var_off(ivar), var, totsize,
     .       ds_bad_flag(ivar), vtype )

* If its an Epic file, the missing-flag is not in the file, and
* precision conversion means they won't match. Fill the missing data
* with the right double precision values.
        IF (ds_type(dset) .EQ. 'ECDF') THEN 
           CALL TM_BAD_EPIC(var, totsize, ds_bad_flag(ivar) )
        ENDIF

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
     .     ( err_status, status, 'CD_READ', dset,
     .        no_stepfile, ebuff, no_errstring, *5900 )

	ENDIF

 5900	RETURN
	END
