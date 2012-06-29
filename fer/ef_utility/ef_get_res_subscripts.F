c*
c* ef_get_res_subscripts.F
c*
c* Jonathan Callahan
c* Apr 6th 1998
c*
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
c* Initialize the result context limits the EF will use.  
c*
* 4D subroutine version which calls the 6D subroutine version and
* checks the E and F dimensions are unused
*

      SUBROUTINE EF_GET_RES_SUBSCRIPTS(id, res_lo_ss,
     .                                 res_hi_ss, res_incr)

      IMPLICIT NONE
      INCLUDE 'tmap_dims.parm'

* argument declarations - 4D
      INTEGER id, res_lo_ss(4), res_hi_ss(4), res_incr(4)

* local variables
      INTEGER res_lo_ss_6d(nferdims),
     .        res_hi_ss_6d(nferdims),
     .        res_incr_6d (nferdims)
      INTEGER ndx
      CHARACTER*128 buff

* call the 6D version
      CALL EF_GET_RES_SUBSCRIPTS_6D(id, res_lo_ss_6d, res_hi_ss_6d,
     .                              res_incr_6d)

* check E and F are unused
      DO 100 ndx = 5, nferdims
         IF ( res_lo_ss_6d(ndx) .NE. res_hi_ss_6d(ndx) ) THEN
            WRITE (buff,50) ndx
 50         FORMAT ('Dimension ', I1,
     .         ' is used; use EF_GET_RES_SUBSCRIPTS_6D instead')
            CALL EF_BAIL_OUT(id, buff)
            STOP 'EF_BAIL_OUT returned in EF_GET_RES_SUBSCRIPTS'
         ENDIF
 100  CONTINUE

* copy to the 4D arrays
      DO 200 ndx = 1, 4
         res_lo_ss(ndx) = res_lo_ss_6d(ndx)
         res_hi_ss(ndx) = res_hi_ss_6d(ndx)
         res_incr(ndx)  = res_incr_6d(ndx)
 200  CONTINUE

      RETURN 
      END

