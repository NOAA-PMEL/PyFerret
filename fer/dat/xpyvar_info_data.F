      BLOCK DATA XPYVAR_INFO_DATA

*
*  This software was developed by the Thermal Modeling and Analysis
*  Project(TMAP) of the National Oceanographic and Atmospheric
*  Administrations (NOAA) Pacific Marine Environmental Lab(PMEL),
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
* initialize variables given in COMMON/XPYVAR_INFO
*

      IMPLICIT NONE

      INCLUDE 'xpyvar_info.cmn'

      INTEGER     max4pyvars
      PARAMETER ( max4pyvars = 4 * maxpyvars )

* tmap_dset.parm contains C-preprocessor flags
#include "tmap_dset.parm"

      DATA pyvar_code_head    / 0 /
      DATA pyvar_ndarray_obj  / maxpyvars * 0 /
      DATA pyvar_code         / maxpyvars * char_init128 /
      DATA pyvar_title        / maxpyvars * char_init128 /
      DATA pyvar_units        / maxpyvars * char_init64 /
      DATA pyvar_missing_flag / maxpyvars * real4_init /
      DATA pyvar_dset_number  / maxpyvars * int4_init /
      DATA pyvar_grid_number  / maxpyvars * int4_init /
      DATA pyvar_grid_start   / max4pyvars * int4_init /
      DATA pyvar_grid_end     / max4pyvars * int4_init /

      END

