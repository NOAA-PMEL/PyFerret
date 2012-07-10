*
*  ef_get_axis_modulo_len.F
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
*  ARE DISCLAIMED. IN NO EVENT SHALL NOAA/PMEL/TMAP BE LIABLE FOR ANY
*  SPECIAL, INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER
*  RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF
*  CONTRACT, NEGLIGENCE OR OTHER TORTUOUS ACTION, ARISING OUT OF OR IN
*  CONNECTION WITH THE ACCESS, USE OR PERFORMANCE OF THIS SOFTWARE. 
*
*
* Returns the modulo length (double precision) of the modulo axis
* indicated, or zero if the axis is not a modulo axis.
*

      SUBROUTINE EF_GET_AXIS_MODULO_LEN(id, iarg, iaxis, modlen)

      IMPLICIT NONE
      INCLUDE 'tmap_dims.parm'
      INCLUDE 'ferret.parm'
      INCLUDE 'xcontext.cmn'
      INCLUDE 'EF_Util.parm'
      INCLUDE 'xtm_grid.cmn_text'

*  Input arguments
      INTEGER id, iarg, iaxis

*  Output arguments
      REAL*8 modlen

*  Local vairables
      INTEGER cx_list(EF_MAX_ARGS)
      INTEGER grid, axis

c* The context list was cached by Ferret and must be retrieved
c* to be accessible to the external function.

      CALL EF_GET_CX_LIST(cx_list)
      grid = cx_grid(cx_list(iarg))
      axis = grid_line(iaxis, grid) 

      IF ( (axis .EQ. munknown) .OR. (axis .EQ. mnormal) ) THEN
         modlen = 0.0D0
      ELSE IF ( .NOT. line_modulo(axis) ) THEN
         modlen = 0.0D0
      ELSE
         modlen = line_modulo_len(axis)
      ENDIF

      RETURN 
      END
