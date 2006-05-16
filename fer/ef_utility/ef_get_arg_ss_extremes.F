c*
c* ef_get_arg_ss_extremes.F
c*
c* Jonathan Callahan
c* Mar 27th 1998
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
c* Return the minimum and maximum subscripts for a grid.  
c*

      SUBROUTINE EF_GET_ARG_SS_EXTREMES(id, num_args, ss_min, ss_max)

      INCLUDE 'ferret.parm'
      INCLUDE 'xcontext.cmn'
      INCLUDE 'EF_Util.parm'

      INTEGER id, num_args, ss_min(4,EF_MAX_ARGS), 
     .     ss_max(4,EF_MAX_ARGS)

      INTEGER grid, idim, iarg
      INTEGER cx_list(EF_MAX_ARGS)

c* The context list was cached by Ferret and must be retrieved
c* to be accessible to the external function.
      CALL EF_GET_CX_LIST(cx_list)

      DO 200 iarg=1,num_args

         grid = cx_grid( cx_list(iarg) )

         DO 100 idim=1,4

	   CALL GRID_SUBSCRIPT_EXTREMES(ss_min(idim,iarg), 
     .           ss_max(idim,iarg), grid, idim)

 100     CONTINUE
 200  CONTINUE

      RETURN 
      END
