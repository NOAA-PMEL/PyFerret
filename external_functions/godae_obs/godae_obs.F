*
* godae_obs.F
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
* Steve Hankin
* Dac, 2000

* This function reads a GODAE (FNMOC) ship observations file (name passed
* in) and returns the variables in a multi-dimensional array NoxNv where
* No is the number of obs in the file and Nv is the number of variables
* read (list of variables as per source code)

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


      SUBROUTINE godae_obs_init (id)

      INCLUDE 'ferret_cmn/EF_Util.cmn'

      INTEGER id, arg

************************************************************************
*                                            USER CONFIGURABLE PORTION |
*                                                                      |
*                                                                      V

      CALL ef_set_desc (id,
     .   'Read an FNOC ship obs file - result in Nobs X 15 array' )

      CALL ef_set_num_args(id, 1)

      CALL ef_set_axis_inheritance(id,  ABSTRACT, ABSTRACT, 
     .                                  NORMAL,   NORMAL)

      CALL ef_set_piecemeal_ok(id, NO, NO, NO, NO)

      CALL ef_set_num_work_arrays(id, 0)


      arg = 1
      CALL ef_set_arg_type(id, arg, STRING_ARG)
      CALL ef_set_arg_name (id, arg, 'dtg FILENAME')
      CALL ef_set_arg_unit(id, arg, 'none')
      CALL ef_set_arg_desc(id, arg,
     .                  'FNMOC "dtg" file in the form YYYYMMDDHH')
      CALL ef_set_axis_influence (id, arg, NO, NO, NO, NO)

*                                                                      ^
*                                                                      |
*                                            USER CONFIGURABLE PORTION |
************************************************************************

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

      SUBROUTINE godae_obs_result_limits(id)

      INCLUDE 'ferret_cmn/EF_Util.cmn'

      INTEGER id

* **********************************************************************
*                                           USER CONFIGURABLE PORTION |
*                                                                     |
*                                                                     V

      INTEGER arg_lo_ss(4,EF_MAX_ARGS), arg_hi_ss(4,EF_MAX_ARGS),
     .     arg_incr(4,EF_MAX_ARGS)
      INTEGER OPEN_OCN_OBS
      CHARACTER*80 errmsg

      INCLUDE 'xgodae_obs.cmn'


      CALL ef_get_arg_string(id, 1, gobs_filename)

      IF (gobs_filename .EQ. " ") CALL ef_bail_out(id,
     .          'No filename specified')

      nobs = OPEN_OCN_OBS(gobs_filename,errmsg)
      IF (nobs .LE. 0) CALL ef_bail_out(id, errmsg)

* the same list of variables is read from every file
      nvars = 15

* indicate the number of variables to be returned on the X axis
      CALL ef_set_axis_limits(id, X_AXIS, 1, nobs)

* indicate the number of observations to be returned on the Y axis
      CALL ef_set_axis_limits(id, Y_AXIS, 1, nvars)

*                                                                     ^
*                                                                     |
*                                           USER CONFIGURABLE PORTION |
* **********************************************************************

      RETURN 
      END


*
* In this subroutine we compute the result
*
      SUBROUTINE godae_obs_compute (id, arg_1, result)

      INCLUDE 'ferret_cmn/EF_Util.cmn'
      INCLUDE 'ferret_cmn/EF_mem_subsc.cmn'

      INTEGER id
      REAL bad_flag(EF_MAX_ARGS), bad_flag_result
      REAL arg_1(mem1lox:mem1hix, mem1loy:mem1hiy, mem1loz:mem1hiz, 
     .     mem1lot:mem1hit)

      REAL result(memreslox:memreshix, memresloy:memreshiy, 
     .     memresloz:memreshiz, memreslot:memreshit)

* After initialization, the 'res_' arrays contain indexing information 
* for the result axes.  The 'arg_' arrays will contain the indexing 
* information for each variable's axes. 

      INTEGER res_lo_ss(4), res_hi_ss(4), res_incr(4)
      INTEGER arg_lo_ss(4,EF_MAX_ARGS), arg_hi_ss(4,EF_MAX_ARGS),
     .     arg_incr(4,EF_MAX_ARGS)
 

************************************************************************
*                                            USER CONFIGURABLE PORTION |
*                                                                      |
*                                                                      V
*  names, units, characteristics of variables to send to Vis5D utility functions.

        CHARACTER errmsg*80

        INCLUDE 'xgodae_obs.cmn'

C  Read data from the input file straight into Ferret
      CALL READ_OCN_OBS(gobs_filename, result, nvars, nobs, errmsg)
      IF ( errmsg.NE.' ' ) CALL ef_bail_out(id,
     .          'No filename specified')

*                                                                      ^
*                                                                      |
*                                            USER CONFIGURABLE PORTION |
************************************************************************

      RETURN 
      END


