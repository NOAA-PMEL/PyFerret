	SUBROUTINE CD_WRITE_DSG_DATA ( cdfid, vname, grid, lo, hi, dat, 
     .                           mode_upcase_output, fer_bad, status )

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
* Write a hyperslab of data into the current output netCDF file
* pared down simplified version of CD_WRITE_VAR for writing a dsg 1-d variable

* Programmer Ansley Manke
* NOAA/PMEL, Seattle, WA - Science Data Integration Group
*  2/5/2019

* argument definitions:
*     cdfid   - netCDF id number for already opened CDF file
*     vname   - name of variable
*     grid    - defining grid for variable
*     lo      - low indices of variable hyperslab
*     hi      - high indices of variable hyperslab
*     dat     - data values to write
*     mode_upcase_output
*     fer_bad - bad flag, for handling packed  data
*     status  - return status

* include files
        include 'netcdf.inc'
	include 'tmap_errors.parm'
	include 'tmap_dims.parm'
#include "tmap_dset.parm"
        include 'cdf_tmap.parm'
        include 'ferret.parm'
        include 'xvariables.cmn'
	include 'xdset_info.cmn_text'
	external xdset_info_data
        include 'xunits.cmn_text'
	include 'xtm_grid.cmn_text'
	external xgt_grid_data

* argument declarations
      LOGICAL mode_upcase_output
      INTEGER cdfid, grid, lo(*), hi(*), status
      CHARACTER*(*) vname
      REAL dat(*), fer_bad

* local variable declarations:

      INTEGER TM_LENSTR1, STR_SAME,
     .        start(nferdims), count(nferdims), idim, iaxis, cdfstat, vlen,
     .        slab_min(nferdims), slab_max(nferdims), tmrecax,
     .	      dims, varid, wrkspc, i, j, k, l, m, n,
     .	      vartyp, nvdim, vdims(8), nvatts, npts

      LOGICAL CD_GET_BOUNDS, TM_DFPEQ, CD_GET_ATTVAL, 
     .        got_it, scaled, do_warn, user_var
      CHARACTER*128 vbuff     
      REAL   bad, r_scale, r_offset
      REAL*8 scalefac, addoff
      REAL*8 r100_8(200)
      REAL*8 TM_WORLD
    
      INTEGER elen, numvals, nv 
      CHARACTER name*128

      REAL*4 r100_4(200)

* For error message
      CHARACTER*9 typnam(6)
      DATA typnam/'NC_BYTE', 'NC_CHAR', 'NC_SHORT', 'NC_INT', 
     .            'NC_FLOAT', 'NC_DOUBLE'/

* initialize
      name = vname
      vlen = TM_LENSTR1( name )

* get the variable id

      cdfstat = NF_INQ_VARID(cdfid, vname(:vlen), varid)
      IF ( cdfstat .NE. NF_NOERR ) GOTO 5100

* get the vital statistics for this variable

      cdfstat = NF_INQ_VAR (cdfid, varid, vbuff, vartyp, nvdim,
     .             vdims, nvatts)

* Get values for scalefac, addoff and bad-flag set in cdf_list for this variable

      scalefac = 1.
      addoff = 0.
      numvals = 1
      do_warn = .FALSE.
      got_it = CD_GET_ATTVAL ( cdfid, varid, "scale_factor",
     .                        do_warn, vname,
     .                        scalefac, numvals, nv )
      got_it = CD_GET_ATTVAL ( cdfid, varid, "add_offset",
     .                        do_warn, vname,
     .                        addoff, numvals, nv )
      got_it = CD_GET_ATTVAL ( cdfid, varid, "missing_value",
     .                        do_warn, vname,
     .                        bad, numvals, nv )
      IF (.NOT.got_it) got_it = 
     .         CD_GET_ATTVAL ( cdfid, varid, "_FillValue",
     .                        do_warn, vname,
     .                        bad, numvals, nv )

* enter data mode so data can be read and written
* (CD_GET_GENERIC_BOUNDS needs to read coordinate values)
      CALL CD_SET_MODE(cdfid, pcd_mode_data, status)
      IF ( status .NE. merr_ok ) RETURN

* get the hyperslab limits of this variable as stored in the file
      got_it = CD_GET_BOUNDS( cdfid, varid, slab_min, slab_max,
     .                        tmrecax, status )
      IF ( status .NE. merr_ok ) GOTO 5900
    
      IF (.NOT.got_it) CALL CD_GET_GENERIC_BOUNDS
     . ( cdfid, varid, grid, lo, slab_min, slab_max, tmrecax, status )

      IF ( status .NE. merr_ok ) GOTO 5900

* determine the start and count as netCDF will want them

      dims = 0
      DO idim = 1, nferdims
         IF ( lo(idim).NE.unspecified_int4) THEN
            dims = dims + 1
            start(dims) = lo(idim) - slab_min(idim) + 1
            count(dims) = hi(idim) - lo(idim) + 1
         ENDIF
      ENDDO

*xxxx   Never scale the missing flag: instead we read data, insert the default missing-flag
*xxxx   for internal use in computing etc, and then on output restore the
*xxxx   packing, and the incoming missing-data flag.

      scaled = .FALSE.
      r_offset = 0.
      r_scale = 1.

      IF ( .NOT. TM_DFPEQ(scalefac, 1.D0) .OR. 
     .     .NOT. TM_DFPEQ(addoff, 0.D0) ) THEN

         IF (.NOT. TM_DFPEQ(scalefac, 0.D0) ) THEN
	    r_scale = 1. /scalefac
            r_offset = -1.D0* addoff* r_scale
         ENDIF

         npts = 1
         DO 250  idim = 1, nferdims
            IF ( count(idim) .GT. 0 ) npts = npts* count(idim)
 250     CONTINUE

	 IF (user_var) THEN
	    CALL TM_SCALE_NOBAD( r_scale, r_offset, dat, npts, bad, vartyp )
	 ELSE

c file var: rescale, skipping missing and restoring the _FillValue from the input file.
            CALL TM_SCALE_SWITCH_BAD ( r_scale, r_offset, fer_bad, bad, dat, npts )
	 ENDIF

         scaled = .TRUE.
      ENDIF

* write the slab

      CALL CD_WRITE_VAR_SUB( cdfid, varid, vartyp, dims, start, count, 
     .	vdims(1), dat, cdfstat )

* If rescaled above, restore it to previous values so it can be reused 

      IF ( scaled ) THEN
         r_scale = scalefac
         r_offset = addoff
	 
	 IF (user_var) THEN
	    CALL TM_SCALE_NOBAD( r_scale, r_offset, dat, npts, bad, vartyp )
	 ELSE

c unpack again, skipping missing and going back to Ferret internal bad-flag.
            CALL TM_SCALE_SWITCH_BAD ( r_scale, r_offset, bad, fer_bad, dat, npts )

	 ENDIF
      ENDIF

      IF ( cdfstat .EQ. -60) GOTO 5260
      IF ( cdfstat .NE. NF_NOERR ) GOTO 5300

* successful completion
 1000 status = merr_ok
      RETURN

* error messages
 5100 CALL TM_ERRMSG
     . ( merr_unkvar, status, 'CD_WRITE_DSG_DATA', unspecified_int4,
     .   unspecified_int4,
     .   'variable was never created in CDF file',
     .   vname(:vlen), *5900 )

 5260 vbuff = vname 
      CALL TM_ERRMSG (cdfstat+pcdferr, status, 'CD_WRITE_DSG_DATA',
     .     no_descfile, no_stepfile, 
     .    'data in variable '//vbuff(:vlen)//
     .    ' not representable in output type '//typnam(vartyp),
     .     no_errstring, *5900 )

 5300 CALL TM_ERRMSG
     . ( cdfstat+pcdferr, status, 'CD_WRITE_DSG_DATA', unspecified_int4,
     . no_varid, 'failure writing to CDF output file',
     . no_errstring, *5900 )

* error exit
 5900   RETURN
	END

