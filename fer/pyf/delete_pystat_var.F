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
* Delete a pystat variable.
* NOTE: this does NOT purge ferret-memory copies of this variable.
*       CALL PURGE_PYSTAT_VAR prior to calling this function
*       to purge ferret-memory copies of only this variable.
*
* Input:
*     ivar - the (Fortran) index of the pystat variable to delete.  The
*            reference count to the python memory associated with this 
*            data is decremented.
*
* Output:
*     errmsg - error message if unsuccessful; blank if and only if successful
*     lenerr - length of the actual error message; zero if and only if successful
*

      SUBROUTINE DELETE_PYSTAT_VAR(ivar, errmsg, lenerr)
      IMPLICIT NONE

      INCLUDE 'tmap_dims.parm'
      INCLUDE 'xpyvar_info.cmn'

*     Passed arguments
      CHARACTER*(*) errmsg
      INTEGER       ivar, lenerr

      CHARACTER*2 undef_name
      PARAMETER ( undef_name = '%%' )

*     Function declarations
      INTEGER TM_LENSTR

      IF ( (ivar .LT. 1) .OR. (ivar .GT. maxpyvars) ) THEN
          WRITE(errmsg,*) 'Invalid pystat variable number: ', ivar
          lenerr = TM_LENSTR(errmsg)
          RETURN
      ENDIF
      IF ( pyvar_ndarray_obj(ivar) .EQ. 0 ) THEN
          WRITE(errmsg,*) 'No pystat variable at position ', ivar
          lenerr = TM_LENSTR(errmsg)
          RETURN
      ENDIF

*     Decrement the reference count to the ndarray object
      CALL DECREF_PYOBJ(pyvar_ndarray_obj(ivar))
      pyvar_ndarray_obj(ivar) = 0

*     Clear the name from the array and hash table
      CALL STRING_ARRAY_MODIFY(pyvar_code_head, ivar,
     .                         undef_name, LEN(undef_name))

*     Decrement the reference count to the associated grid and its lines
      CALL DEALLO_DYN_GRID(pyvar_grid_number(ivar))

*     Success
      errmsg = ' '
      lenerr = 0
      RETURN

      END

