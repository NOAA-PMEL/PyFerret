c*
c* ef_get_axis_dates.F
c*
c* Ansley Manke
c* Dec 9 1998
c*
c*
*
*  This software was developed by the Thermal Modeling and Analysis
*  Project(TMAP) of the National Oceanographic and Atmospheric
*  Administration''s (NOAA) Pacific Marine Environmental Lab(PMEL),
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
c*
c* This returns the date buffer associated with the 
c* time axis of an argument.
c* datebuf is the date string. slen is the maximum length of the date strings
c*
*  V665 *acm*/11/09 Fix ticket 1766, as in #762, if the axis is modulo
*                   the year 0000 or 0001 returns as blank.  If not modulo
*		    then we want to return the year correctly.
* v695 *acm* 2/15  Get axis dates from an F or T axis
* v75  *acm* 3/18  Changes to allow for timesteps in fractional seconds

      SUBROUTINE EF_GET_AXIS_DATES(id, iarg, tax, dim, numtimes, precision, slen, datebuf )

      IMPLICIT NONE
      INCLUDE 'tmap_dims.parm'
      INCLUDE 'ferret.parm'
      INCLUDE 'xcontext.cmn'
      INCLUDE 'EF_Util.parm'
      INCLUDE 'xtm_grid.cmn_text'

      INTEGER id, iarg, dim, numtimes, precision, slen
      REAL*8 tax(*)
      CHARACTER*(*) datebuf(*)

      INTEGER TM_LENSTR1, l, grid, seconds
      REAL*8  tstep_r8, test(5)
      INTEGER cx_list(EF_MAX_ARGS)

c* The context list was cached by Ferret and must be retrieved
c* to be accessible to the external function.

      CALL EF_GET_CX_LIST(cx_list)
      grid = CX_GRID(cx_list(iarg))

      slen = 0
      IF (precision .EQ. 0) precision = 6
      DO 100 L = 1, numtimes
         tstep_r8 = tax(L)
         CALL EF_GET_DATE_TSTEP ( grid, dim, tstep_r8, precision, datebuf(l) )
	 IF ( TM_LENSTR1(datebuf(l)) .GT. slen ) slen = TM_LENSTR1(datebuf(l))
100   CONTINUE


      RETURN 
      END
