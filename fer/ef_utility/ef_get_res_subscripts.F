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

      SUBROUTINE EF_GET_RES_SUBSCRIPTS(id, res_lo_ss,
     .     res_hi_ss, res_incr)

      INCLUDE 'ferret.parm'
      INCLUDE 'xvariables.cmn'
      INCLUDE 'EF_Util.parm'

      INTEGER id, res_lo_ss(4), res_hi_ss(4), res_incr(4)
      INTEGER idim

      INTEGER mres

c* The memory resident id for the rsult, "mres", was cached by 
c* Ferret and must be retrieved to be accessible to the external function.
      CALL EF_GET_MRES(mres)

      DO 100 idim=1,4

         res_lo_ss(idim) = mr_lo_ss(mres, idim)
         res_hi_ss(idim) = mr_hi_ss(mres, idim)
         IF (res_lo_ss(idim) .EQ. res_hi_ss(idim)) 
     .        THEN
            res_incr(idim) = 0
         ELSE
            res_incr(idim) = 1
         ENDIF

 100  CONTINUE

      RETURN 
      END



*
* Return the array memory limits in the common blocks for those
* times when they cannot be retrieved directly.
*
      SUBROUTINE EF_GET_RES_MEM_SUBSCRIPTS(id, res_memlo, res_memhi)

      INCLUDE 'EF_Util.parm'
      INCLUDE 'EF_mem_subsc.cmn'

* argument declarations
      INTEGER id, res_memlo(4), res_memhi(4)

      res_memlo(1) = memreslox
      res_memlo(2) = memresloy
      res_memlo(3) = memresloz
      res_memlo(4) = memreslot

      res_memhi(1) = memreshix
      res_memhi(2) = memreshiy
      res_memhi(3) = memreshiz
      res_memhi(4) = memreshit

      RETURN
      END
