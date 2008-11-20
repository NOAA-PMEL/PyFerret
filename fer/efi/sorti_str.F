*
* sorti_str.F
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
* Ansley Manke
* Nov 2008
* V6.2 *acm* SORTI for strings
*
* This function sorts data on x axis in increasing order.
* Returns index of sorted values.  
*
*  NOTE:
*  IT IS GENERALLY ADVISABLE TO INCLUDE EXPLICIT LIMITS WHEN WORKING WITH
*  FUNCTIONS THAT REPLACE AXES. FOR EXAMPLE, THE CONSIDER THE FUNCTION
*  sorti_str(v). THE EXPRESSION
*  	LIST/L=6:10 sortl_str(v)
*  IS NOT EQUIVALENT TO
*  	LIST sortl_str(v[L=6:10])
*  THE FORMER WILL LIST THE 6TH THROUGH 10TH SORTED INDICES FROM THE ENTIRE
*  L RANGE OF VARIABLE V. THE LATTER WILL LIST ALL OF THE INDICES THAT
*  RESULT FROM SORTING v[L=6:10].
*  
*
* In this subroutine we provide information about
* the function.  The user configurable information 
* consists of the following:
*
* descr              Text description of the function
*
* num_args           Required number of arguments
*
* axis_inheritance   Type of axis for the result
*                       ( CUSTOM, IMPLIED_BY_ARGS, NORMAL, ABSTRACT )
*                       CUSTOM          - user defined axis
*                       IMPLIED_BY_ARGS - same axis as the incoming argument
*                       NORMAL          - the result is normal to this axis
*                       ABSTRACT        - an axis which only has index values
*
* piecemeal_ok       For memory optimization:
*                       axes where calculation may be performed piecemeal
*                       ( YES, NO )
* 
*
* For each argument we provide the following information:
*
* name               Text name for an argument
*
* unit               Text units for an argument
*
* desc               Text description of an argument
*
* axis_influence     Are this argument's axes the same as the result grid?
*                       ( YES, NO )
*
* axis_extend       How much does Ferret need to extend arg limits relative to result 
*


      SUBROUTINE sorti_str_init(id)

      INCLUDE 'EF_Util.cmn'

      INTEGER id, arg

***********************************************************************
*                                           USER CONFIGURABLE PORTION |
*                                                                     |
*                                                                     V
      CHARACTER*100 fcn_desc
      WRITE (fcn_desc, 10)
   10 FORMAT ('Returns indices of string data, sorted on the I axis ',
     .        'in increasing order, null strings at the end')
      CALL ef_set_desc(id, fcn_desc)

      CALL ef_set_num_args(id, 1)
      CALL ef_set_has_vari_args(id, NO)
      CALL ef_set_axis_inheritance(id, ABSTRACT, IMPLIED_BY_ARGS,
     .     IMPLIED_BY_ARGS, IMPLIED_BY_ARGS)

      CALL ef_set_piecemeal_ok(id, NO, NO, NO, NO)

      CALL ef_set_num_work_arrays(id, 2)

      arg = 1
      CALL ef_set_arg_name(id, arg, 'STR')
      CALL ef_set_arg_desc(id, arg, 'String variable to sort in I')
      CALL ef_set_axis_influence(id, arg, NO, YES, YES, YES)
      CALL ef_set_arg_type (id, arg, STRING_ARG)

*                                                                     ^
*                                                                     |
*                                           USER CONFIGURABLE PORTION |
***********************************************************************

      RETURN 
      END


*
* In this subroutine we provide information about the lo and hi
* limits associated with each abstract or custom axis.   The user 
* configurable information consists of the following:
*
* lo_ss               lo subscript for an axis
*
* hi_ss               hi subscript for an axis
*

      SUBROUTINE sorti_str_result_limits(id)

      INCLUDE 'EF_Util.cmn'

      INTEGER id

* **********************************************************************
*                                           USER CONFIGURABLE PORTION |
*                                                                     |
*                                                                     V

      INTEGER my_lo_l, my_hi_l

      INTEGER arg
      INTEGER arg_lo_ss(4,EF_MAX_ARGS), arg_hi_ss(4,EF_MAX_ARGS),
     .     arg_incr(4,EF_MAX_ARGS)

*
*     Use utility functions to get context information about the arguments.
*

      CALL ef_get_arg_subscripts(id, arg_lo_ss, arg_hi_ss, arg_incr)

      my_lo_l = 1
      my_hi_l = 1

      arg = 1

      my_hi_l = arg_hi_ss(X_AXIS,arg) - arg_lo_ss(X_AXIS,arg) + 1


      CALL ef_set_axis_limits(id, X_AXIS, my_lo_l, my_hi_l)
*                                                                     ^
*                                                                     |
*                                           USER CONFIGURABLE PORTION |
* **********************************************************************

      RETURN 
      END


*
* In this subroutine we request an amount of storage to be supplied
* by Ferret and passed as an additional argument.
*
      SUBROUTINE sorti_str_work_size(id)

      INCLUDE 'EF_Util.cmn'
      INCLUDE 'EF_mem_subsc.cmn'

      INTEGER id

* **********************************************************************
*                                            USER CONFIGURABLE PORTION |
*                                                                      |
* 
* Set the work arrays,  X/Y/Z/T dimensions
*
* ef_set_work_array_dims(id,array #,xlo,ylo,zlo,tlo,xhi,yhi,zhi,thi)
*
      INTEGER mxdat
      INTEGER arg_lo_ss(4,1:EF_MAX_ARGS), arg_hi_ss(4,1:EF_MAX_ARGS),
     .     arg_incr(4,1:EF_MAX_ARGS)

      CALL ef_get_arg_subscripts(id, arg_lo_ss, arg_hi_ss, arg_incr)

      mxdat = 1 + arg_hi_ss(X_AXIS,ARG1) - arg_lo_ss(X_AXIS,ARG1)


c* sort_dat
      CALL ef_set_work_array_dims (id, 1, 1, 1, 1, 1, mxdat, 512, 1, 1)

* sort_indx
      CALL ef_set_work_array_dims (id, 2, 1, 1, 1, 1, mxdat, 1, 1, 1)

*                                                                      ^
*                                                                      |
*                                            USER CONFIGURABLE PORTION |
* **********************************************************************

      RETURN
      END

*
* In this subroutine we compute the result
*
      SUBROUTINE sorti_str_compute(id, arg_1, result, sort_dat, sort_indx)

      INCLUDE 'EF_Util.cmn'
      INCLUDE 'EF_mem_subsc.cmn'

      REAL bad_flag(EF_MAX_ARGS), bad_flag_result
      REAL arg_1(2,mem1lox:mem1hix, mem1loy:mem1hiy, 
     .           mem1loz:mem1hiz, mem1lot:mem1hit)
      REAL result(memreslox:memreshix, memresloy:memreshiy,
     .            memresloz:memreshiz, memreslot:memreshit)

* After initialization, the 'res_' arrays contain indexing information 
* for the result axes.  The 'arg_' arrays will contain the indexing 
* information for each variable's axes. 

      INTEGER res_lo_ss(4), res_hi_ss(4), res_incr(4)
      INTEGER arg_lo_ss(4,EF_MAX_ARGS), arg_hi_ss(4,EF_MAX_ARGS),
     .     arg_incr(4,EF_MAX_ARGS)


***********************************************************************
*                                           USER CONFIGURABLE PORTION |
*                                                                     |
*                                                                     V

      INTEGER id
      INTEGER m, nsrt, nbad, slen
      INTEGER i, j, k, l
      INTEGER i1, j1, k1, l1

*  Dimension work arrays

      CHARACTER*512 sort_dat(wrk1lox:wrk1hix, wrk1loy:wrk1hiy,
     .               wrk1loz:wrk1hiz, wrk1lot:wrk1hit)
      REAL sort_indx(wrk2lox:wrk2hix, wrk2loy:wrk2hiy,
     .               wrk2loz:wrk2hiz, wrk2lot:wrk2hit)

      CALL ef_get_res_subscripts(id, res_lo_ss, res_hi_ss, res_incr)
      CALL ef_get_arg_subscripts(id, arg_lo_ss, arg_hi_ss, arg_incr)
      CALL ef_get_bad_flags(id, bad_flag, bad_flag_result)


      j1 = arg_lo_ss(Y_AXIS, ARG1)
      DO 600 j = res_lo_ss(Y_AXIS), res_hi_ss(Y_AXIS)

         k1 = arg_lo_ss(Z_AXIS, ARG1)
         DO 500 k = res_lo_ss(Z_AXIS), res_hi_ss(Z_AXIS)

            l1 = arg_lo_ss(T_AXIS, ARG1)
            DO 400 l = res_lo_ss(T_AXIS), res_hi_ss(T_AXIS)

               nsrt = 0
               nbad = 0

               i1 = arg_lo_ss(X_AXIS,ARG1)
               DO 100 i = res_lo_ss(X_AXIS), res_hi_ss(X_AXIS)

                  nsrt = nsrt + 1
                  CALL EF_GET_STRING_ARG_ELEMENT(id, ARG1, arg_1,
     .                       i1,j1,k1,l1, slen,sort_dat(nsrt,1,1,1))
                  IF (slen .EQ. 0) THEN
		     nsrt = nsrt - 1
		     nbad = nbad + 1
		  ELSE
                     sort_indx(nsrt,1,1,1) = i1
		  ENDIF

                  i1 = i1 + arg_incr(X_AXIS,ARG1)
 100           CONTINUE
      
* Sort based on sort_dat.  sort_indx goes along for the ride,
* elements moved when elements of sort_dat moved.

               IF (nsrt .GT. 1) CALL HEAP2_STR (sort_dat, 
     .                                   sort_indx, nsrt)

               DO 200 m = 1, nsrt
                  result(m,j,k,l) = sort_indx(m,1,1,1)
 200           CONTINUE

               DO 300 m = 1, nbad
                  m_indx = m + nsrt
                  result(m_indx,j,k,l) = bad_flag_result
 300           CONTINUE

               l1 = l1 + arg_incr(T_AXIS, ARG1)
 400        CONTINUE

            k1 = k1 + arg_incr(Z_AXIS, ARG1)
 500     CONTINUE

         j1 = j1 + arg_incr(Y_AXIS, ARG1)
 600  CONTINUE
*                                                                     ^
*                                                                     |
*                                           USER CONFIGURABLE PORTION |
***********************************************************************

      RETURN 
      END
