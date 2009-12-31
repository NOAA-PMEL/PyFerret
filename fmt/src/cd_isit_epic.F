	LOGICAL FUNCTION CD_ISIT_EPIC ( dset, status )

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
* determine if this netCDF file is an EPIC file
* the following approach is taken to determine if this is an EPIC file:
* 1) locate one of the axes, "time" or "lat"
* 2) check for the attribute "epic_code"

* the common variable cd_what_isit is modified only if it is EPIC

* Programmer Steve Hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program

* revision 0.0 - 2/93
* V600 *ACM* attribute control. 
*             Replace NCVID with CD_GET_VAR_ID

* argument definitions:
*	dset	- dataset pointer for already opened CDF file

* argument declarations
      INTEGER dset, status

      include 'netcdf.decl'
#include "netcdf.inc"
      include 'tmap_errors.parm'
      include 'tmap_dims.parm'
      include 'xdset_info.cmn_text'
      external xdset_info_data
      include 'xdset_info.cd_equiv'

* internal variable declarations:
      INTEGER i, axid, attype, attlen, attoutflag, attid
      INTEGER axnamlen(2)
      CHARACTER*4 axname(2)

* internal data statements
      DATA axname(1)/'time'/, axnamlen(1)/4/, 
     .     axname(2)/'lat' /, axnamlen(2)/3/

* initialize
      status = merr_ok

* locate an axis that's in the file
      DO 100 i = 1, 2
         CALL CD_GET_VAR_ID ( dset, axname(i)(1:axnamlen(i)), 
     .         axid, status )
         IF ( axid .GT. 0 ) GOTO 200
 100  CONTINUE

* axis not found - it must not be EPIC
      CD_ISIT_EPIC = .FALSE.
      RETURN

* check for "epic_code" attribute
 200  CALL CD_GET_VAR_ATT_ID (dset, axid, 'epic_code', attid, status)

      CD_ISIT_EPIC =  status .EQ. merr_ok

* successful completion (always)
      RETURN
      END
