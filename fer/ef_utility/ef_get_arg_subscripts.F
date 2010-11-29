c*
c* ef_get_arg_subscripts.F
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
c* Jonathan Callahan
c* Feb 19th 1998
c*
c* V510 12/99 *sh* - cope with unspecified limits (default=full axis range)
c*               and limits embedded inside of user variable definitions
c*               as in LET a = sst[L=5:9]; LIST FFT(a) (info from uvar_*)
c*      3/00 *sh* - bug fix: above logic applies only to CUSTOM axes 


c*
c* Initialize the context limits the EF will use.  
c*

      SUBROUTINE EF_GET_ARG_SUBSCRIPTS(id, arg_lo_ss, arg_hi_ss, 
     .     arg_incr)

      INCLUDE 'ferret.parm'
      INCLUDE 'xcontext.cmn'
      INCLUDE 'xvariables.cmn'
      INCLUDE 'EF_Util.parm'

* argument declarations
      INTEGER id, arg_lo_ss(4,EF_MAX_ARGS+1), 
     .     arg_hi_ss(4,EF_MAX_ARGS+1), arg_incr(4,EF_MAX_ARGS+1)


* internal variable declarations
      LOGICAL embedded_limits
      INTEGER idim, iarg, num_reqd_args, cat, var
      INTEGER cx_list(EF_MAX_ARGS), ax_src(4)
      INTEGER efcn_get_num_reqd_args

      num_reqd_args = efcn_get_num_reqd_args(id)
      CALL efcn_get_axis_will_be(id, ax_src)

c* The context list was cached by Ferret and must be retrieved
c* to be accessible to the external function.
      CALL EF_GET_CX_LIST(cx_list)

      DO 200 idim=1,4

         DO 100 iarg=1, num_reqd_args

            IF ( cx_given(idim,cx_list(iarg)) ) THEN
* limits are explicit in the top-level expression
              arg_lo_ss(idim,iarg) = cx_lo_ss(cx_list(iarg), idim)
              arg_hi_ss(idim,iarg) = cx_hi_ss(cx_list(iarg), idim)
            ELSE
* are there explicit limits embedded within user-var definitions?
              cat = cx_category( cx_list(iarg) )
              var = cx_variable( cx_list(iarg) )
              IF ( cat .EQ. cat_user_var ) THEN
                 embedded_limits =
     .                 uvar_given(idim,var) .GE. uvlim_gvn_gnl
     .           .AND. uvar_given(idim,var) .LE. uvlim_gvn_xact
              ELSE
                 embedded_limits = .FALSE.
              ENDIF
              IF ( embedded_limits ) THEN
* ... get embedded explicit limits from the user-variable definition
                arg_lo_ss(idim,iarg) = uvar_lo_ss(var, idim)
                arg_hi_ss(idim,iarg) = uvar_hi_ss(var, idim)
              ELSE
                IF (ax_src(idim) .NE. CUSTOM) THEN
* ... result axis inherits limits from SET REGION and command qualifiers, etc.
* NOTE: ARG LIMITS WILL BE WRONG IF USED DURING GRID-CREATION
*       (which should never happen ...)
                  arg_lo_ss(idim,iarg) = cx_lo_ss(cx_list(iarg), idim)
                  arg_hi_ss(idim,iarg) = cx_hi_ss(cx_list(iarg), idim)
                ELSE
* ... result axis is imposed by the EF and this arg has no explicit limits
*     so infer the limits from the nature of the variable
                  IF (  cat .EQ. cat_file_var ) THEN
*    ... get limits of file variable (possibly less than full-axis)
                    CALL VAR_SS_LIMS( idim, cx_list(iarg),
     .                                arg_lo_ss(idim,iarg),
     .                                arg_hi_ss(idim,iarg) )
                  ELSE
*    ... assume the full span of the defining axis as the default range
                    CALL GRID_SUBSCRIPT_EXTREMES_NO_MOD(
     .                  arg_lo_ss(idim,iarg),
     .                  arg_hi_ss(idim,iarg),
     .                  cx_grid(cx_list(iarg)), idim)
                  ENDIF
                ENDIF
              ENDIF
            ENDIF

* fill in the delta to indicate if a range is available
            IF (arg_lo_ss(idim,iarg) .EQ. arg_hi_ss(idim,iarg)) 
     .           THEN
               arg_incr(idim, iarg) = 0
            ELSE
               arg_incr(idim, iarg) = 1
            ENDIF

 100     CONTINUE

 200  CONTINUE

      RETURN 
      END


*
* Return the array memory limits in the common blocks for those
* times when they cannot be retrieved directly.
*
      SUBROUTINE EF_GET_ARG_MEM_SUBSCRIPTS(id, arg_memlo, arg_memhi)

      INCLUDE 'EF_Util.parm'
      INCLUDE 'EF_mem_subsc.cmn'

* argument declarations
      INTEGER id, arg_memlo(4,EF_MAX_ARGS),
     .            arg_memhi(4,EF_MAX_ARGS)

* this assumed EF_MAX_ARGS .EQ. 9
      arg_memlo(1,1) = mem1lox
      arg_memlo(1,2) = mem2lox
      arg_memlo(1,3) = mem3lox
      arg_memlo(1,4) = mem4lox
      arg_memlo(1,5) = mem5lox
      arg_memlo(1,6) = mem6lox
      arg_memlo(1,7) = mem7lox
      arg_memlo(1,8) = mem8lox
      arg_memlo(1,9) = mem9lox

      arg_memlo(2,1) = mem1loy
      arg_memlo(2,2) = mem2loy
      arg_memlo(2,3) = mem3loy
      arg_memlo(2,4) = mem4loy
      arg_memlo(2,5) = mem5loy
      arg_memlo(2,6) = mem6loy
      arg_memlo(2,7) = mem7loy
      arg_memlo(2,8) = mem8loy
      arg_memlo(2,9) = mem9loy

      arg_memlo(3,1) = mem1loz
      arg_memlo(3,2) = mem2loz
      arg_memlo(3,3) = mem3loz
      arg_memlo(3,4) = mem4loz
      arg_memlo(3,5) = mem5loz
      arg_memlo(3,6) = mem6loz
      arg_memlo(3,7) = mem7loz
      arg_memlo(3,8) = mem8loz
      arg_memlo(3,9) = mem9loz

      arg_memlo(4,1) = mem1lot
      arg_memlo(4,2) = mem2lot
      arg_memlo(4,3) = mem3lot
      arg_memlo(4,4) = mem4lot
      arg_memlo(4,5) = mem5lot
      arg_memlo(4,6) = mem6lot
      arg_memlo(4,7) = mem7lot
      arg_memlo(4,8) = mem8lot
      arg_memlo(4,9) = mem9lot

      arg_memhi(1,1) = mem1hix
      arg_memhi(1,2) = mem2hix
      arg_memhi(1,3) = mem3hix
      arg_memhi(1,4) = mem4hix
      arg_memhi(1,5) = mem5hix
      arg_memhi(1,6) = mem6hix
      arg_memhi(1,7) = mem7hix
      arg_memhi(1,8) = mem8hix
      arg_memhi(1,9) = mem9hix

      arg_memhi(2,1) = mem1hiy
      arg_memhi(2,2) = mem2hiy
      arg_memhi(2,3) = mem3hiy
      arg_memhi(2,4) = mem4hiy
      arg_memhi(2,5) = mem5hiy
      arg_memhi(2,6) = mem6hiy
      arg_memhi(2,7) = mem7hiy
      arg_memhi(2,8) = mem8hiy
      arg_memhi(2,9) = mem9hiy

      arg_memhi(3,1) = mem1hiz
      arg_memhi(3,2) = mem2hiz
      arg_memhi(3,3) = mem3hiz
      arg_memhi(3,4) = mem4hiz
      arg_memhi(3,5) = mem5hiz
      arg_memhi(3,6) = mem6hiz
      arg_memhi(3,7) = mem7hiz
      arg_memhi(3,8) = mem8hiz
      arg_memhi(3,9) = mem9hiz

      arg_memhi(4,1) = mem1hit
      arg_memhi(4,2) = mem2hit
      arg_memhi(4,3) = mem3hit
      arg_memhi(4,4) = mem4hit
      arg_memhi(4,5) = mem5hit
      arg_memhi(4,6) = mem6hit
      arg_memhi(4,7) = mem7hit
      arg_memhi(4,8) = mem8hit
      arg_memhi(4,9) = mem9hit

      RETURN
      END
