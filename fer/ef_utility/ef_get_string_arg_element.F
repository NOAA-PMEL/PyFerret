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
c* This returns an element of the string array which was passed as an argument
c* to the function and for convenience, its length.
*
* 4D subroutine version which calls the 6D subroutine version and
* checks the E and F dimensions are unused
*


      SUBROUTINE EF_GET_STRING_ARG_ELEMENT(id, iarg, arg, i, j, k, l,
     .                                     slen, text)

      IMPLICIT NONE
      INCLUDE 'tmap_dims.parm'
      INCLUDE 'EF_Util.parm'

* argument declarations - 4D
      INTEGER id, iarg, slen, i, j, k, l
      CHARACTER*(*) text
      REAL arg(*)

* local variables
      INTEGER ndx, m, n,
     .        arg_lo_ss(nferdims,EF_MAX_ARGS),
     .        arg_hi_ss(nferdims,EF_MAX_ARGS),
     .        arg_incr (nferdims,EF_MAX_ARGS)
      CHARACTER*128 buff

* get the dimensions so m and n can be checked and assigned
      CALL EF_GET_ARG_SUBSCRIPTS_6D(id, arg_lo_ss, arg_hi_ss, arg_incr)
 
* make sure the E and F dimensions are not used
      DO 100 ndx = 5, nferdims
         IF ( arg_lo_ss(ndx, iarg) .NE.
     .        arg_hi_ss(ndx, iarg) ) THEN
            WRITE (buff,50) ndx, iarg
 50         FORMAT ('Dimension ', I1, ' of argument ', I2,
     .         ' is used; use EF_GET_STRING_ARG_ELEMENT_6D instead')
            CALL EF_BAIL_OUT(id, buff)
            STOP 'EF_BAIL_OUT returned in EF_GET_STRING_ARG_ELEMENT'
         ENDIF
 100  CONTINUE

* call the 6D routine with the appropriate m and n
      m = arg_lo_ss(5, iarg)
      n = arg_lo_ss(6, iarg)
      CALL EF_GET_STRING_ARG_ELEMENT_6D(id, iarg, arg,
     .                   i, j, k, l, m, n, slen, text)

      RETURN
      END
