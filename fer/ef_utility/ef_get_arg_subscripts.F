c*
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
c*
c* Initialize the context limits the EF will use.  
c*
* 4D subroutine version which calls the 6D subroutine version and
* checks the E and F dimensions are unused
*

      SUBROUTINE EF_GET_ARG_SUBSCRIPTS(id, arg_lo_ss, arg_hi_ss, 
     .                                 arg_incr)

      IMPLICIT NONE
      INCLUDE 'tmap_dims.parm'
      INCLUDE 'EF_Util.parm'

* argument declarations - 4D
      INTEGER id,
     .        arg_lo_ss(4,EF_MAX_ARGS+1), 
     .        arg_hi_ss(4,EF_MAX_ARGS+1),
     .        arg_incr (4,EF_MAX_ARGS+1)

* local variables
      INTEGER ndx, adx,
     .        arg_lo_ss_6d(nferdims,EF_MAX_ARGS+1), 
     .        arg_hi_ss_6d(nferdims,EF_MAX_ARGS+1),
     .        arg_incr_6d (nferdims,EF_MAX_ARGS+1)
      CHARACTER*128 buf

* call the 6D version
      CALL EF_GET_ARG_SUBSCRIPTS_6D(id, arg_lo_ss_6d, arg_hi_ss_6d,
     .                              arg_incr_6d)

* check E and F are unused
      DO 110 adx = 1, EF_MAX_ARGS
         DO 100 ndx = 5, nferdims
            IF ( arg_lo_ss_6d(ndx, adx) .NE.
     .           arg_hi_ss_6d(ndx, adx) ) THEN
               WRITE (buf,50) ndx, adx
 50            FORMAT ('Dimension ', I1, ' of argument ', I2,
     .            ' is used; use EF_GET_ARG_SUBSCRIPTS_6D instead')
               CALL EF_BAIL_OUT(id, buf)
               STOP 'EF_BAIL_OUT returned in EF_GET_ARG_SUBSCRIPTS'
            ENDIF
 100     CONTINUE
 110  CONTINUE

* copy to the 4D arrays
      DO 210 adx = 1, EF_MAX_ARGS
         DO 200 ndx = 1, 4
            arg_lo_ss(ndx, adx) = arg_lo_ss_6d(ndx, adx)
            arg_hi_ss(ndx, adx) = arg_hi_ss_6d(ndx, adx)
            arg_incr (ndx, adx) = arg_incr_6d (ndx, adx)
 200     CONTINUE
 210  CONTINUE

      RETURN 
      END

