c*
c* ef_get_coordinates_sub.F
c*
c* Jonathan Callahan
c* Mar 10th 1998
c*
c*

c*
c* This fills in the coordinates of an axis in a grid.  
c*
c* JC_TODO: What about REAL*8 values being passed back to C
c*

      SUBROUTINE EF_GET_COORDINATES_SUB(cx_list, iarg, idim,
     .     ss_lo_lim, ss_hi_lim, values )

      INCLUDE 'ferret_cmn/ferret.parm'
      INCLUDE 'ferret_cmn/xvariables.cmn'
      INCLUDE 'ferret_cmn/xcontext.cmn'
      INCLUDE 'ferret_cmn/EF_Util.cmn'

      INTEGER cx_list(EF_MAX_ARGS), iarg
      INTEGER ss_lo_lim, ss_hi_lim
      INTEGER idim, grid, i

      REAL    values( ss_lo_lim:ss_hi_lim )
      REAL*8  TM_WORLD


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
      grid = cx_grid( cx_list(iarg) )

      DO 100 i = ss_lo_lim, ss_hi_lim
         values( i ) = TM_WORLD( i,grid,idim,box_middle )
 100  continue

      RETURN 
      END
