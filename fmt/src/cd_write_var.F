	SUBROUTINE CD_WRITE_VAR ( cdfid, vname, grid, lo, hi, dat, 
     .                            status )

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

* Programmer Steve Hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program

* revision 0.0 - 3/92
*  performance: 10/92 consolidated record axis data writes under 1 ncendf call
*                     and use CD_SET_MODE to change between data & define modes
*  12/92: coordinate variables -> double precision
*   1/95: use cd_get_generic_bounds to synchronize output in files without
*	  the slab_(min/max)_limits machinery 
* 10/96 *kob* - Linux port - had to move variable assignments off of
*		declaration line.
*	      - had to use F90 Intrinsic function TRANSFER to properly
*		pass single values to NCVPT as arrays of 1 element
* 6/97	*kob* - fix introduced bug in call to NCVPT
* V533 *sh* 6/01 - add support for string variables (NCCHAR)
* V552 *acm* 5/03   increase vbuff to 128 chars 

* argument definitions:
*     cdfid   - netCDF id number for already opened CDF file
*     vname   - name of variable
*     grid    - defining grid for variable
*     lo      - low indices of variable hyperslab
*     hi      - high indices of variable hyperslab
*     dat     - data values to write
*     status  - return status

* include files
        include 'netcdf.decl'
#include "netcdf.inc"
	include 'tmap_errors.parm'
	include 'tmap_dims.parm'
#include "tmap_dset.parm"
        include 'cdf_tmap.parm'
#include "gt_lib.parm"
	include 'xdset_info.cmn_text'
	external xdset_info_data
	include 'xtm_grid.cmn_text'
	external xgt_grid_data

* argument declarations
      INTEGER cdfid, grid, lo(4), hi(4), status
      CHARACTER*(*) vname
      REAL dat(*)

* local variable declarations:
* *kob* explicit definitions not allowed on decl line in linux...
*       moved it to parameter line 10/96
      INTEGER TM_LENSTR1, NCVID,
     .        start(4), count(4), idim, iaxis, cdfstat, vlen,
     .        slab_min(4), slab_max(4), tmrecax, cdrecax, dims, varid,
     .        hilim, i100, i, recaxvar, lorec, hirec, rectyp,
     .	      vartyp, nvdim, vdims(6), nvatts
      LOGICAL CD_GET_BOUNDS, TM_ABSTRACT_AXIS, got_it
      CHARACTER*13 TM_STRING
      CHARACTER*1  dimnam(4)
      DATA  	   dimnam/'I','J','K','L'/
      CHARACTER*128 vbuff     ! needed because of fussy f772.1 DEC compiler
      REAL*8 r100_8(100)
      REAL*8 TM_WORLD

      REAL*4 r100_4(100)
      EQUIVALENCE (r100_8, r100_4)

* initialize
      vlen = TM_LENSTR1( vname )

* get the variable id
      varid = NCVID( cdfid, vname(:vlen), cdfstat )
      IF ( cdfstat .NE. NCNOERR ) GOTO 5100

* get the vital statistics for this variable
      CALL NCVINQ(cdfid, varid, vbuff, vartyp, nvdim, vdims,
     .              nvatts,cdfstat)

* enter data mode so data can be read and written
* (CD_GET_GENERIC_BOUNDS needs to read coordinate values)
      CALL CD_SET_MODE(cdfid, pcd_mode_data, status)
      IF ( status .NE. merr_ok ) RETURN

* get the hyperslab limits of this variable as stored in the file
      got_it = CD_GET_BOUNDS( cdfid, varid, slab_min, slab_max,
     .                        tmrecax, status )
      IF ( status .NE. merr_ok ) GOTO 5900
      IF ( .NOT. got_it ) CALL CD_GET_GENERIC_BOUNDS
     .   ( cdfid, varid, grid, lo, slab_min, slab_max, tmrecax, status )
      IF ( status .NE. merr_ok ) GOTO 5900

* special check for abstract axes (kludge!) - length, only
* if i=5:15 overwrites i=1:10 it will be detected but
* if i=6:15 overwrite i=1:10 it still will not be detected resulting in an
* unhelpful error message, below
         DO 10 idim = 1, 4
            IF ( idim .EQ. tmrecax ) GOTO 10
            iaxis = grid_line(idim, grid)
            IF ( TM_ABSTRACT_AXIS(iaxis)
     .     .AND. lo(idim).NE.unspecified_int4) THEN
               IF ( hi(idim)-lo(idim)
     .         .GT. slab_max(idim)-slab_min(idim) ) GOTO 5250
               slab_min(idim) = lo(idim)
               slab_max(idim) = hi(idim)
            ENDIF
 10      CONTINUE
* ... possible unnecessary record axis check
         IF ( lo(t_dim).EQ.unspecified_int4 ) tmrecax = 0

* check for valid limits (must be consistent with first time write)
      DO 100 idim = 1, 4
         IF ( idim .EQ. tmrecax ) GOTO 100
         IF ( lo(idim) .EQ. unspecified_int4 ) THEN
            IF ( slab_min(idim) .NE. 1 ) GOTO 5200
         ELSE
            IF ( lo(idim) .LT. slab_min(idim)
     .      .OR. hi(idim) .GT. slab_max(idim) ) GOTO 5200
         ENDIF
 100  CONTINUE

* determine the start and count as netCDF will want them
* and determine which axis (cdrecax = 1-4) of THIS (possibly less than 4D)
* variable is the record axis.
* (the complex logic for this is to allow future record axes not to be T)
      cdrecax = tmrecax
      dims = 0
      DO 200 idim = 1, 4
         iaxis = grid_line(idim, grid)
         IF ( iaxis.EQ.mpsnorm .OR. lo(idim).EQ.unspecified_int4) THEN
            IF ( dims .LE. tmrecax ) cdrecax = cdrecax - 1   ! logic ok ??
         ELSE
            dims = dims + 1
            start(dims) = lo(idim) - slab_min(idim) + 1
            count(dims) = hi(idim) - lo(idim) + 1
         ENDIF
 200  CONTINUE

* synchronize with data already in the file along the record dimension
* then write the coordinate values to the file

* is record axis involved in this variable (e.g. time-independent) ?
      IF ( tmrecax .NE. 0 ) THEN
         lorec = lo(tmrecax)
         hirec = hi(tmrecax)
      ELSE
         lorec = unspecified_int4
      ENDIF
      IF ( lorec .NE. unspecified_int4 ) THEN
         CALL CD_REC_SYNCH( cdfid, lorec, hirec, tmrecax,
     .               grid, start(cdrecax), recaxvar, rectyp, status )
         IF ( status .NE. merr_ok ) GOTO 5900

* write the record axis coords (buffer 100 at a time) (12/92 REAL*8 option)
* (code lifted from CD_REC_SYNCH 10/92)
         DO 310 i100 = lorec, hirec, 100
            hilim = MIN(i100+99, hirec)
            DO 300 i = i100, hilim
               r100_8(i-i100+1) = TM_WORLD(i,grid,tmrecax,box_middle)
               IF ( rectyp .EQ. NCFLOAT ) r100_4(i-i100+1)
     .                                  = r100_8(i-i100+1)
 300        CONTINUE

#ifdef FORTRAN_90
            CALL NCVPT(cdfid, recaxvar, 
     .                 TRANSFER(i100-lorec+start(cdrecax),start,1),
     .                 TRANSFER(hilim-i100+1,count,1), 
     .		       TRANSFER(r100_8,r100_4), 
     . 		        cdfstat)
#else
            CALL NCVPT(cdfid, recaxvar, i100-lorec+start(cdrecax),
     .                 hilim-i100+1, r100_8, cdfstat)
#endif
            IF ( cdfstat .NE. NCNOERR ) GOTO 5300
 310     CONTINUE
      ENDIF

* write the slab
      CALL CD_WRITE_VAR_SUB( cdfid, varid, vartyp, dims, start, count, 
     .	vdims(1), dat, cdfstat )
      IF ( cdfstat .NE. NCNOERR ) GOTO 5300
!      CALL NCVPT( cdfid, varid, start,count, dat, cdfstat )
!      IF ( cdfstat .NE. NCNOERR ) GOTO 5300

* successful completion
 1000 status = merr_ok
      RETURN

* error messages
 5100 CALL TM_ERRMSG
     . ( merr_unkvar, status, 'CD_WRITE_VAR', unspecified_int4,
     .   unspecified_int4,
     .   'variable was never created in CDF file',
     .   vname(:vlen), *5900 )

 5200 vbuff = vname   ! needed because of fussy DEC f77 v2.1 compiler
      CALL TM_ERRMSG (merr_outofgrid, status, 'CD_WRITE_VAR',
     .     no_descfile, no_stepfile,
     .     vbuff(:vlen)//'['//dimnam(idim)//'='//
     .     TM_STRING(DBLE(lo(idim)))//':'//
     .     TM_STRING(DBLE(hi(idim)))//']',
     .     vbuff(:vlen)//'['//dimnam(idim)//'='//
     .     TM_STRING(DBLE(slab_min(idim)))//':'//
     .     TM_STRING(DBLE(slab_max(idim)))//']',
     .     *5900)

 5250 vbuff = vname 
      CALL TM_ERRMSG (merr_outofgrid, status, 'CD_WRITE_VAR',
     .     no_descfile, no_stepfile, 
     .    'inconsistent '//dimnam(idim)//' axis length for '//
     .     vbuff(:vlen), no_errstring, *5900 )

 5300 CALL TM_ERRMSG
     . ( cdfstat+pcdferr, status, 'CD_WRITE_VAR', unspecified_int4,
     . no_varid, 'failure writing to CDF output file',
     . no_errstring, *5900 )

* error exit
 5900   RETURN
	END
