c*
c* ef_get_arg_info.F
c*
c* Jonathan Callahan
c* Sep 28th 1998
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
c* This returns the name, title and dataset associated 
c* with an argument.
c*
* V552 *acm* 4/03- up VAR_CODE decl from 64 to 128 chars

      SUBROUTINE ef_get_arg_info(id, iarg, name, title, units )

      INCLUDE 'ferret.parm'
      INCLUDE 'xcontext.cmn'
      INCLUDE 'EF_Util.parm'

      INTEGER id, iarg
      INTEGER cat, var

      CHARACTER*(*) name, title, units
      CHARACTER var_code*128, var_title*128, var_units*32

      INTEGER cx_list(EF_MAX_ARGS)

c* The context list was cached by Ferret and must be retrieved
c* to be accessible to the external function.

      CALL ef_get_cx_list(cx_list)

      cat = cx_category(cx_list(iarg))
      var = cx_variable(cx_list(iarg)) 

      name  = var_code(cat,var)
      title = var_title(cx_list(iarg))
      units = var_units(cx_list(iarg))


      RETURN 
      END
