	LOGICAL FUNCTION CD_GET_BOUNDS ( cdfid, ivar, lo, hi, tmrecdim,
     .                                   status )

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
* Get the bounds of the hyperslab of the given netCDF variable relative
* to its parent grid in the TMAP Common

* Programmer Steve Hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program

* revision 0.0 - 3/92
*
* 15apr94 *kob* Need to initialzie cd_get_bounds in case of error 
*		exit
*
* 2jun94 *kob* cd_get_bounds needs to be initialazed with boolean .FALSE.
* 		instead of 0.
* 10/96 *kob* - Linux port - had to move variable assignments off of
*		declaration line.
* V500 4/99 *kob*: increase size of vname and buff from 24 to 64
* V533 *sh* 6/01 - add support for string variables (NCCHAR)
* V500 4/99 *kob*: increase size of vname and buff from 24 to 64
* V552 *acm* 5/03   increase vname and buff to 128 chars 


* argument definitions:
*	cdfid   - id of currently open CDF file
*       ivar    - CDF variable ID of the desired variable
*       lo      - array of (4) lower bounds subscripts
*       hi      - array of (4) upper bounds subscripts
*       tmrecdim- returned axis (1-4) which is CDF record dimension (or 0)
*       status  - returned TMAP status

* argument declarations:
	INTEGER	 cdfid, ivar, lo(4), hi(4), tmrecdim, status

* include files
        include 'netcdf.decl'
#include "netcdf.inc"
	include 'tmap_errors.parm'
	include 'cd_lib.parm'
        include 'xio.cmn_text'
        external xio_data

* local variable declarations:
* *kob* explicit definitions not allowed on decl line in linux...
*       moved it to parameter line 10/96
      LOGICAL CD_GET_ATTVAL, got_lo, got_hi,
     .        do_warn
      PARAMETER (do_warn = .TRUE. )
      INTEGER TM_LENSTR1,
     .        ndims, nvars, cdfstat, vbacklo, vbackhi, idim,
     .        vartyp, nvdim, vdims(6), nvatts, vlen,
     .        ngatts, npts, recdim
      REAL    r4_lo(4), r4_hi(4)
      CHARACTER vname*128, buff*128

      INTEGER tt  ! nice short name
      EQUIVALENCE (tt,lunit_errors)

* initialize
        tmrecdim = 0   ! may be changed, below
* ibm compiler didn't like initializing cd_get_bounds to 0, so *kob* 
* changed it to .false.  2jun94
	CD_GET_BOUNDS = .FALSE.     !in case of error exit
* get the vital statistics for the CDF file
        CALL NCINQ( cdfid, ndims, nvars, ngatts, recdim, cdfstat )

* if it is a character variable, then the first dimension of it is the strlen
        IF (vartyp .EQ. NCCHAR) THEN
	   DO idim = 1, 5
	      vdims(idim) = vdims(idim+1)
	   ENDDO
	   nvdim = nvdim - 1
	ENDIF

* get the vital statistics for this variable
        CALL NCVINQ(cdfid, ivar, vname, vartyp, nvdim, vdims,
     .              nvatts,cdfstat)
        vlen = TM_LENSTR1(vname)


* get the lower bounds
        got_lo = CD_GET_ATTVAL( cdfid, ivar, 'slab_min_index',
     .           do_warn,vname(:vlen),r4_lo,4,vbacklo )
        IF ( got_lo ) THEN
          DO 100 idim = 1, 4
 100      lo(idim) = r4_lo(idim) !fp->int
        ENDIF

* get the upper bounds
        got_hi = CD_GET_ATTVAL( cdfid, ivar, 'slab_max_index',
     .           do_warn,vname(:vlen),r4_hi, 4, vbackhi )
        IF ( got_hi ) THEN
          DO 200 idim = 1, 4
 200      hi(idim) = r4_hi(idim) !fp->int
        ENDIF

* check for consistency
        IF ( got_lo .NEQV. got_hi ) GOTO 5200
        IF ( got_lo .AND. (vbacklo.NE.4.OR.vbackhi.NE.4) ) GOTO 5300
        IF ( .NOT.got_lo ) GOTO 1000
        IF ( recdim .EQ. -1 ) GOTO 1000     ! no record axis in file ?

* the record dimension of the variable may have a flag, only
* if so substitute the full size of the record dimension
        DO 300 idim = 1, 4
           IF ( hi(idim) .EQ. pcdf_recdim ) THEN
              IF ( tmrecdim .NE. 0 ) GOTO 5100
              tmrecdim = idim
           ENDIF
 300    CONTINUE
        IF ( tmrecdim .NE. 0 ) THEN
           CALL NCDINQ( cdfid, recdim, buff, npts, cdfstat )
           lo(tmrecdim) = 1
           hi(tmrecdim) = npts
        ENDIF

* check for reasonable slab limits
        DO 400 idim = 1, 4
           IF ( idim .EQ. tmrecdim ) GOTO 400
           IF ( lo(idim).LE.0 .OR. hi(idim).LE.0
     .     .OR. lo(idim) .GT. hi(idim) )           GOTO 5400
 400    CONTINUE

* successful completion
 1000   CD_GET_BOUNDS = got_lo
        status = merr_ok
        RETURN

* error exits
 5100   CALL TM_ERRMSG(pcdferr+pcdferrmax,
     .         status, 'CD_GET_BOUNDS', cdfid, ivar, 
     .         'More than one record dimension in "slab_min_index": '
     .         //vname(:vlen), no_errstring, *5000 )

 5200    CALL TM_ERRMSG(pcdferr+pcdferrmax,
     .         status, 'CD_GET_BOUNDS', cdfid, ivar,
     .         'Must give both min and max "slab_m**_index" values: '
     .         //vname(:vlen),  no_errstring, *5000 )

 5300    CALL TM_ERRMSG(pcdferr+pcdferrmax,
     .         status, 'CD_GET_BOUNDS', cdfid, ivar,
     .         'Must be exactly 4 "slab_m**_index" values: '
     .         //vname(:vlen),  no_errstring, *5000 )

 5400    CALL TM_ERRMSG(pcdferr+pcdferrmax,
     .         status, 'CD_GET_BOUNDS', cdfid, ivar,
     .         'Non-positive or disordered "slab_m**_index" values: '
     .         //vname(:vlen),  no_errstring, *5000 )

 5000   RETURN
	END

