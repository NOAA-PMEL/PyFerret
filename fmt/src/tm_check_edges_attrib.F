        SUBROUTINE TM_CHECK_EDGES_ATTRIB (cdfid, iaxis, vname, vlen, 
     .                                    ename, edgid, status)

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
* V570 5/2004 *acm*
* Check the edges attribute, issue any warnings
* V600 *acm* 10/05 should not have npts as an argument; npts from cd_get_1_axis
*                  was being reset  as npts bounds or edges, which is npts+1
* V65  *acm* 1/10  Change all netcdf2-style calls and parameter names to 
*                  netcdf3-style calls, which are also consistent with netcdf4.
* V65  *acm* 1/10  Add #ifdef usingDODSf2cUnderscore, for 32-bit linux build
*                  linked with OPeNDAP library.

        INCLUDE 'tmap_dims.parm'
        INCLUDE 'xtm_grid.cmn_text'
        INCLUDE 'xio.cmn_text'
        INCLUDE 'tmap_errors.parm'
#include "netcdf.inc"

! Declare netcdf function names. 
! The ones without trailing underscore are declared in netcdf.inc
#include "netcdf_declareUnderscore.inc"

* argument declarations
      INTEGER cdfid, iaxis, vlen, edgid, status
      CHARACTER*(*) ename, vname

* Local variable declarations
      INTEGER TM_LENSTR1
      INTEGER elen, cdfstat, soft_err, vartype, nvdims,
     .          vdims(8), nvatts, npts
      CHARACTER*132 buff, vbuff

* does the variable pointed to really exist ?

        elen = TM_LENSTR1(ename)
#ifdef usingDODSf2cUnderscore
	cdfstat = NF_INQ_VARID_(cdfid, ename(:elen), edgid)
#else
	cdfstat = NF_INQ_VARID(cdfid, ename(:elen), edgid)
#endif

        IF (cdfstat.NE.NF_NOERR) THEN
           soft_err = 11
           GOTO 5000
        ENDIF

* get the vital statistics of the edges variable
#ifdef usingDODSf2cUnderscore
        cdfstat = NF_INQ_VAR_(cdfid, edgid, ename(:elen), vartype,
     .               nvdims, vdims, nvatts)
#else
        cdfstat = NF_INQ_VAR (cdfid, edgid, ename(:elen), vartype,
     .               nvdims, vdims, nvatts)
#endif
        IF ( nvdims .NE. 1 ) THEN
           soft_err = 12
           GOTO 5000
        ENDIF

* get axis length:
#ifdef usingDODSf2cUnderscore
	cdfstat = NF_INQ_DIM_(cdfid, vdims, buff, npts)
#else
	cdfstat = NF_INQ_DIM (cdfid, vdims, buff, npts)
#endif

        IF (cdfstat.NE.NF_NOERR) GOTO 5100   ! should never happen

* check for error conditions
        IF ( npts .NE. line_dim(iaxis)+1 ) THEN
           soft_err = 13
           GOTO 5000
        ENDIF

        status = merr_ok
        RETURN

* error messages

 5000   CALL TM_NOTE( 'netCDF parent axis definition error', lunit_errors )
        buff = ename
        vbuff = vname
        IF ( soft_err .EQ. 1 ) THEN
           CALL TM_NOTE(
     .    '"true_size" attribute must have only max/min axis coords: '
     .       //vbuff(:vlen), lunit_errors )
        ELSEIF ( soft_err .EQ. 11 ) THEN
           CALL TM_NOTE('Edges definition "'//buff(:elen)//
     .                   '" points to no existing axis', lunit_errors )
        ELSEIF ( soft_err .EQ. 12 ) THEN
           CALL TM_NOTE('Edges definition "'//buff(:elen)//
     .                   '" is not 1D', lunit_errors )
        ELSEIF ( soft_err .EQ. 13 ) THEN
           CALL TM_NOTE('Edges "'//buff(:elen)//
     .          '" must be 1 pt longer than '//vbuff(:vlen), lunit_errors )
        ENDIF
        IF ( soft_err .LE. 10 ) THEN
           CALL TM_NOTE( 'Axis definition ignored', lunit_errors )
        ELSE
           CALL TM_NOTE( 'Edge definitions ignored', lunit_errors )
        ENDIF
        status = merr_ok + 1
        RETURN

 5100   status = pcdferr
        RETURN

        END
