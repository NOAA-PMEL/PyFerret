      SUBROUTINE ISIT_CONST_ARR_VAR ( cx, const_arr_var, status )

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
      
* See if the expression is a constant-array variable.  Only checks the
* simple definition of the variable, not a full-on Ferret parsing of the
* variable and handling as a stack.

* 9/2014 for ticket 981

        INCLUDE	'ferret.parm'
	INCLUDE 'errmsg.parm'
	include 'tmap_dims.parm'
	include	'interp_stack.parm'
	include	'xvariables.cmn'
	include	'xcontext.cmn'
	include	'xprog_state.cmn'

* calling argument declarations
      LOGICAL const_arr_var
      INTEGER cx, status

* local variable declarations

      INTEGER cat, var, lo_isp, item, itype
      CHARACTER*128 buff
   
      const_arr_var = .FALSE.

      cat = cx_category( cx )
      var = cx_variable( cx )

      IF (cat .EQ. cat_const_var) const_arr_var = .TRUE.

      IF (cat .EQ. cat_user_var) THEN 

        DO item = 1, uvar_num_items( var )
	   itype = uvar_item_type ( item, var )
	   IF (itype .EQ. alg_const_var) const_arr_var = .TRUE.
	ENDDO

      ENDIF

      RETURN
      END
