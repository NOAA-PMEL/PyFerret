	CHARACTER*(*) FUNCTION CD_CHILDAX_NAME( axname, lo, hi, tlen )

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

* Programmer Steve Hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program

* revision 0.0 - 3/92
* 1/95 - allow negative subscripts.  Substitute "N" for "-"
* v4.91 - *kob* - 12/97 - added checks for axis name length

* argument definitions:
*     axis    - original axis name
*     lo      - lo subscript
*     hi      - hi subscript
*     slen    - returned length of name

#ifdef unix
        include 'tmap_format/netcdf.decl'
        include 'tmap_format/netcdf.inc'
	include 'tmap_format/tmap_errors.parm'
	include 'tmap_format/tmap_dims.parm'
#include "tmap_format/tmap_dset.parm"
	include 'tmap_format/gt_lib.parm'
	include 'tmap_format/xtm_grid.cmn_text'
	external xgt_grid_data
#else
xx        include 'xxtmap_format/netcdf.decl'  ! untested
xx        include 'xxtmap_format/netcdf.inc'
	INCLUDE 'TMAP_FORMAT:TMAP_DIMS.PARM'
	INCLUDE 'TMAP_FORMAT:TMAP_DSET.PARM'
	INCLUDE 'TMAP_FORMAT:GT_LIB.PARM'
	INCLUDE 'TMAP_FORMAT:XTM_GRID.CMN'
#endif

* argument declarations
      INTEGER lo, hi, tlen
      CHARACTER*(*) axname

* local variable declarations:
      INTEGER TM_LENSTR1, slen
      CHARACTER TM_LEFINT*8, int_buff*8
* 12/97
      INTEGER maxlen

      maxlen = LEN(CD_CHILDAX_NAME)

* generate "axis_nameLO_HI"
* append the low subscript limit
      tlen = TM_LENSTR1(axname)
      IF (tlen .GT. maxlen) tlen = maxlen    ! hack to prevent crash
      int_buff = TM_LEFINT(lo,slen)
      IF (int_buff(1:1) .EQ. '-') int_buff(1:1) = 'N'
      CD_CHILDAX_NAME = axname(:tlen)//int_buff
      tlen = tlen+slen
      IF (tlen .GT. maxlen) tlen = maxlen    ! hack to prevent crash


* append the high subscript limit
      int_buff = TM_LEFINT(hi,slen)
      IF (int_buff(1:1) .EQ. '-') int_buff(1:1) = 'N'
      CD_CHILDAX_NAME = CD_CHILDAX_NAME(:tlen)//'_'//int_buff
      tlen = tlen+slen+1
      IF (tlen .GT. maxlen) tlen = maxlen    ! hack to prevent crash
      RETURN
      END
