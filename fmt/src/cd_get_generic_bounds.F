	SUBROUTINE CD_GET_GENERIC_BOUNDS
     .       ( cdfid, ivar, grid, tmlo, cdflo, cdfhi, tmrecdim, status )

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
* Get the bounds of the hyperslab of the given netCDF variable
* from its (generic) dimension limits

* Programmer Steve Hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program

* revision 0.0 - 3/92
*  1/95 *sh*: synchronize the start coordinates of the axes in the CDF file
*		with the coordinates from the grid being written
*  11/96 *kob* - Linux Port - had to use F90 intrinsic function TRANSFER
*			    - to properly pass cvdim variable (not array)
*		   	      to the NCVINQ routine.
* V500 4/99 *kob*: increase size of vname and axnam from 24 to 64
*	- for wacky DODS variables
* V533 *sh* 6/01 - add support for string variables (NCCHAR)


* argument definitions:
*	cdfid   - id of currently open CDF file
*       ivar    - CDF variable ID of the desired variable
*       grid    - TMAP grid number
*       tmlo    - lo subscript limits from TMAP grid
*       cdflo   - array of (4) lower bounds subscripts (returned)
*       cdfhi   - array of (4) upper bounds subscripts
*       tmrecdim- returned axis (1-4) which is CDF record dimension (or 0)

* argument declarations:
	INTEGER	 cdfid, ivar, grid, tmlo(4), cdflo(4), cdfhi(4),
     .		tmrecdim, status

* include files
        include 'netcdf.decl'
#include "netcdf.inc"
	include 'tmap_errors.parm'
	include 'tmap_dims.parm'
#include "gt_lib.parm"
	include 'cd_lib.parm'
        include 'xio.cmn_text'
        external xio_data
        include 'xtm_grid.cmn_text'
        external xgt_grid_data

* local variable declarations:
      LOGICAL CD_USE_RECDIM, TM_FPEQ, no_file_coords
      INTEGER TM_LENSTR1, TM_SUBSC, NCVID, 
     .        idim, vartyp, nvdim,
     .        ndims, cdfstat, vdims(6), nvatts,
     .	      cvartyp, ncvdims, cvdim, npts, alen, axid, offset
* 4/99 *kob* 
      CHARACTER vname*64, axnam*64
      REAL*8 TM_WORLD, file_val, mem_val

* determine if this variable uses the record dimension
        IF ( CD_USE_RECDIM( cdfid, ivar, status ) ) THEN
            tmrecdim = t_dim
         ELSE
            tmrecdim = 0
         ENDIF

* get the vital statistics for this variable
        CALL NCVINQ(cdfid, ivar, vname, vartyp, nvdim, vdims,
     .              nvatts,cdfstat)

* if it is a character variable, then the first dimension of it is the strlen
        IF (vartyp .EQ. NCCHAR) THEN
	   DO idim = 1, 5
	      vdims(idim) = vdims(idim+1)
	   ENDDO
	   nvdim = nvdim - 1
	ENDIF

* get the lo/hi bounds
        ndims = 0
        DO 100 idim = 1, 4
           IF ( grid_line(idim,grid) .NE. mpsnorm
     .    .AND. tmlo(idim) .NE. unspecified_int4  ) THEN
              ndims = ndims + 1
!              cdfhi(idim) = CD_AXLEN( cdfid, vdims(ndims), status )
* 1/95: get info about the coordinate variable (previously w/ CD_AXLEN)
* and then synchronize the start subscript in the output file with the axis
* position on the grid being written (this permits writing into files that
* lack the slab_min/max_limits machinery)

* ... get the size and name of the dimension
              CALL NCDINQ( cdfid, vdims(ndims), axnam, npts, cdfstat )
              alen = TM_LENSTR1( axnam )

* ... time axis as a record axis will be synchronized in another routine
	      IF ( idim .EQ. tmrecdim ) THEN
	          cdflo(idim) = 1
	          cdfhi(idim) = npts
	          GOTO 100
	      ENDIF

* ... find the coordinate variable by this name
              axid = NCVID( cdfid, axnam(:alen), cdfstat )
              no_file_coords = cdfstat .NE. NCNOERR

* ... determine the first coordinate value
              IF ( no_file_coords ) THEN
	         file_val = 1.0D0	! like abstract axis
	      ELSE
* ... get the vital statistics of the coordinate variable
                 CALL NCVINQ(cdfid, axid, axnam, cvartyp, ncvdims,
#ifdef FORTRAN_90
     .			TRANSFER(cvdim,cvdim,1), nvatts, cdfstat)
#else
     .                  cvdim, nvatts, cdfstat)
#endif
* ... read the first coordinate
                 CALL CD_RD_R8_1( cdfid, axid, 1, cvartyp,
     .                      'axis: '//axnam(:alen), file_val, status )
	         IF ( status .NE. merr_ok ) GOTO 5000
	      ENDIF

* ... locate the first file coordinate on the TMAP grid axis
	      offset = TM_SUBSC ( file_val, grid, idim, 1 )

* ... check that the starting alignment is "exact"
	      mem_val = TM_WORLD ( offset, grid, idim, box_middle )
	      IF (.NOT.TM_FPEQ(SNGL(file_val), SNGL(mem_val))) GOTO 5100

* .. here are the synchronized subscripts
              cdflo(idim) = offset
              cdfhi(idim) = offset + npts - 1

* ... safety check - make sure the upper axis limit is also exact
*     (no check is made for points in between)
	      IF ( no_file_coords ) THEN
	         file_val = npts
	      ELSE
                 CALL CD_RD_R8_1( cdfid, axid, npts, cvartyp,
     .                      'axis: '//axnam(:alen), file_val, status )
	         IF ( status .NE. merr_ok ) GOTO 5000
	      ENDIF
	      offset = TM_SUBSC ( file_val, grid, idim,  cdfhi(idim) )
	      mem_val = TM_WORLD ( offset, grid, idim, box_middle )
	      IF (.NOT.TM_FPEQ(SNGL(file_val), SNGL(mem_val))) GOTO 5100

           ELSE
              cdflo(idim) = 1
              cdfhi(idim) = 1
           ENDIF
 100    CONTINUE

* successful completion
	status = merr_ok
        RETURN

* error exits
 5000	RETURN
 5100  CALL TM_ERRMSG
     .      ( merr_badlinedef, status, 'CD_GET_GENERIC_BOUNDS',
     .	      no_descfile, no_stepfile,
     .       'file coords dont match variable coords on axis '
     .       //axnam(:alen),
     .       no_errstring , *5000 )

	END

