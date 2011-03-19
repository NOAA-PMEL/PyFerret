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
* Retrieves information about an axis of a grid argument to an
* external function.  Designed to be an C-callable interface
* (assuming Hollerith strings in C are just character arrays
* with array length appended to the argument list), so strings
* are null-terminated and 0/1 are assigned for logicals.
*

      SUBROUTINE EF_GET_SINGLE_AXIS_INFO(id, iarg, iaxis, axis_name,
     .         axis_unit, backwards_axis, modulo_axis, regular_axis)

      IMPLICIT NONE
      INTEGER id, iarg, iaxis
      CHARACTER*(*) axis_name, axis_unit
      INTEGER backwards_axis, modulo_axis, regular_axis

      CHARACTER*(64) axnames(4), axunits(4)
      LOGICAL axbackwards(4), axmodulos(4), axregulars(4)
      INTEGER TM_LENSTR, namelen

*     Get the information for all the axes (avoid duplication of code)
      CALL EF_GET_AXIS_INFO(id, iarg, axnames, axunits,
     .                      axbackwards, axmodulos, axregulars)

*     Assign the values for the axis requested
      namelen = TM_LENSTR(axnames(iaxis))
      IF ( namelen .GT. 0 ) THEN
          axis_name(1:namelen) = axnames(iaxis)
      ENDIF
      axis_name(namelen+1:namelen+1) = CHAR(0)

      namelen = TM_LENSTR(axunits(iaxis))
      IF ( namelen .GT. 0 ) THEN
          axis_unit(1:namelen) = axunits(iaxis)
      ENDIF
      axis_unit(namelen+1:namelen+1) = CHAR(0)

      IF ( axbackwards(iaxis) ) THEN
          backwards_axis = 1
      ELSE
          backwards_axis = 0
      ENDIF

      IF ( axmodulos(iaxis) ) THEN
          modulo_axis = 1
      ELSE
          modulo_axis = 0
      ENDIF

      IF ( axregulars(iaxis) ) THEN
          regular_axis = 1
      ELSE
          regular_axis = 0
      ENDIF

      RETURN
      END

