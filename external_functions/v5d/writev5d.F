*
* writev5d.F
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
* June 30 1998
*
* This function takes up to 8 variables from ferret data sets
* and writes them in Vis5D format.  Send in variables to write in the
* first several parameter positions; fill the remaining positions with "0"
* and the name of the file to write in the 9th argument. 
* Requires variables to be on the same x-y grid, grids can vary in depth.
*
*  Use v5dcreate, v5dwrite, v5dclose from the v5d utility functions; these
*  are in binio.c and v5d.c
*
*  ACM  25-Mar-1999  put in filename as an argument.
*
*  ACM 8/May/99 Use REDUCED axes, call ef_set_work_array_lens 
*               function to set gwork
*
*  ACM 8/99 Read either a date and time or a timestep from the "datebuf"
*           that is returned from ef_get_axis_dates
*
*  ACM 8/00 Initialize variable lowlev
* 
*  ACM 11/01 Change testing of Time axis values to allow creating single-time
*            step files (later perhaps appended together)
*          -  single timestep and t>=0 use it
*          -  single timestep and t<0  use date=time=0 (some abstract axes...)
*          -  multiple timesteps and any t<0 bail out. V5d doesn't allow neg times
*
*  ACM 10/03 Pass the ibkwd flag to wrt_v5d_steps. Flip the data in Z only
*            if the Z axis is a depth axis.
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


      SUBROUTINE writev5d_init (id)

      INCLUDE 'ferret_cmn/EF_Util.cmn'

      INTEGER id, arg

************************************************************************
*                                            USER CONFIGURABLE PORTION |
*                                                                      |
*                                                                      V

      CALL ef_set_desc (id,
     .   'Write up to 8 variables to a Vis5D-formatted file  V5.1' )

      CALL ef_set_num_args(id, 9)
      CALL ef_set_axis_reduction (id, REDUCED, REDUCED, 
     .        REDUCED, REDUCED)

      CALL ef_set_axis_inheritance(id, IMPLIED_BY_ARGS, IMPLIED_BY_ARGS, 
     .        IMPLIED_BY_ARGS, IMPLIED_BY_ARGS)

      CALL ef_set_piecemeal_ok(id, NO, NO, NO, NO)

      CALL ef_set_num_work_arrays(id, 2)

*  All variables should be on the same grid (except for the vertical 
*  component). There is no result variable sent back to FERRET - a
*  Vis5D file is written.

      arg = 1
      CALL ef_set_arg_name (id, arg, 'V1')
      CALL ef_set_axis_influence (id, arg, YES, YES, YES, YES)

      arg = 2
      CALL ef_set_arg_name (id, arg, 'V2')
      CALL ef_set_axis_influence (id, arg, YES, YES, YES, YES)

      arg = 3
      CALL ef_set_arg_name (id, arg, 'V3')
      CALL ef_set_axis_influence (id, arg, YES, YES, YES, YES)

      arg = 4
      CALL ef_set_arg_name (id, arg, 'V4')
      CALL ef_set_axis_influence (id, arg, YES, YES, YES, YES)

      arg = 5
      CALL ef_set_arg_name (id, arg, 'V5')
      CALL ef_set_axis_influence (id, arg, YES, YES, YES, YES)

      arg = 6
      CALL ef_set_arg_name (id, arg, 'V6')
      CALL ef_set_axis_influence (id, arg, YES, YES, YES, YES)

      arg = 7
      CALL ef_set_arg_name (id, arg, 'V7')
      CALL ef_set_axis_influence (id, arg, YES, YES, YES, YES)

      arg = 8
      CALL ef_set_arg_name (id, arg, 'V8')
      CALL ef_set_axis_influence (id, arg, YES, YES, YES, YES)

      arg = 9
      CALL ef_set_arg_type(id, arg, STRING_ARG)
      CALL ef_set_arg_name (id, arg, 'FILENAME')
      CALL ef_set_arg_unit(id, arg, 'none')
      CALL ef_set_arg_desc(id, arg, 'Name of the Vis5D file to write')
      CALL ef_set_axis_influence (id, arg, NO, NO, NO, NO)

*                                                                      ^
*                                                                      |
*                                            USER CONFIGURABLE PORTION |
************************************************************************

      RETURN 
      END


*
* In this subroutine we request an amount of storage to be supplied
* by Ferret and passed as an additional argument.
*
      SUBROUTINE writev5d_work_size(id)

      INCLUDE 'ferret_cmn/EF_Util.cmn'
      INCLUDE 'ferret_cmn/EF_mem_subsc.cmn'

      INTEGER id

* **********************************************************************
*                                            USER CONFIGURABLE PORTION |
*                                                                      |
*                                                                      V

*
* Set the work array X/Y/Z/T dimensions
*
* ef_set_work_array_lens(id,array #,xlo,ylo,zlo,tlo,xhi,yhi,zhi,thi)
*
*  Include file for the Vis5D utility functions thad write the file.
*  We have a separate Ferret version that renames a parameter "MAXVARS.

      INCLUDE 'v5df_fer.h'

      INTEGER arg_lo_ss(4,EF_MAX_ARGS), arg_hi_ss(4,EF_MAX_ARGS),
     .     arg_incr(4,EF_MAX_ARGS)

      INTEGER max_row, min_col, max_col, min_lev, max_lev
      INTEGER array_num
      INTEGER nwork, nax
      INTEGER min_row, iarg, max_lev_all

*  Set work array size. gwork is a work array whose output dimensions
*  are reset in routine wrt_v5d_steps

      CALL ef_get_arg_subscripts(id, arg_lo_ss, arg_hi_ss, arg_incr)

*  Get max number of rows in all variables
      min_row = MAXROWS
      max_row = 1
      DO 111 iarg = 1, EF_MAX_ARGS
         IF (arg_lo_ss(X_AXIS,iarg) .NE. ef_unspecified_int4) 
     .     min_row = min(min_row, arg_lo_ss(X_AXIS,iarg))

         IF (arg_hi_ss(X_AXIS,iarg) .NE. ef_unspecified_int4) 
     .     max_row = max(max_row, arg_hi_ss(X_AXIS,iarg))

  111 CONTINUE
      IF (min_row .EQ. ef_unspecified_int4 .OR. 
     .    max_row .EQ. ef_unspecified_int4) THEN          
        min_row = 1
        max_row = 1
      ENDIF
      max_row = max_row - min_row + 1

*  Get max number of columns in all variables
      min_col = MAXCOLUMNS
      max_col = 1
      DO 222 iarg = 1, EF_MAX_ARGS
         IF (arg_lo_ss(Y_AXIS,iarg) .NE. ef_unspecified_int4) 
     .     min_col = min(min_col, arg_lo_ss(Y_AXIS,iarg))

         IF (arg_hi_ss(Y_AXIS,iarg) .NE. ef_unspecified_int4) 
     .     max_col = max(max_col, arg_hi_ss(Y_AXIS,iarg))

  222 CONTINUE
      IF (min_col .EQ. ef_unspecified_int4 .OR. 
     .    max_col .EQ. ef_unspecified_int4) THEN          
        min_col = 1
        max_col = 1
      ENDIF
      max_col = max_col - min_col + 1

*  Get total number of levels in all variables together

      min_lev = MAXLEVELS
      max_lev = 1
      max_lev_all = 0
      DO 333 iarg = 1, EF_MAX_ARGS
         IF (arg_lo_ss(Z_AXIS,iarg) .NE. ef_unspecified_int4) 
     .     min_lev = min(min_lev, arg_lo_ss(Z_AXIS,iarg))

         IF (arg_hi_ss(Z_AXIS,iarg) .NE. ef_unspecified_int4) THEN
           max_lev = max(max_lev, arg_hi_ss(Z_AXIS,iarg))
           max_lev_all = max_lev_all + max_lev
         ENDIF

  333 CONTINUE
      IF (min_lev .EQ. ef_unspecified_int4 .OR. 
     .    max_lev .EQ. ef_unspecified_int4) THEN          
        min_lev = 1
        max_lev = 1
        max_lev_all = 1
      ENDIF
      IF (max_lev_all .eq. 0) max_lev_all = 1

*  gwork

      nwork = max_row* max_col* max_lev_all

      array_num = 1
      CALL ef_set_work_array_dims(id, array_num, 1, 1, 1, 1, 
     .     NWORK, 1, 1, 1)

*  ax_vals

      nax = 4* max( max_row, max_col, max_lev_all)
      array_num = 2
      CALL ef_set_work_array_dims(id, array_num, 1, 1, 1, 1, 
     .     nax, 1, 1, 1)


*                                                                      ^
*                                                                      |
*                                            USER CONFIGURABLE PORTION |
* **********************************************************************

      RETURN 
      END

*
* In this subroutine we compute the result
*
      SUBROUTINE writev5d_compute (id, arg_1, arg_2, arg_3,
     .   arg_4, arg_5, arg_6, arg_7, arg_8, arg_9, result, 
     .   gwork, ax_vals)

      INCLUDE 'ferret_cmn/EF_Util.cmn'
      INCLUDE 'ferret_cmn/EF_mem_subsc.cmn'

      INTEGER id, arg
      REAL bad_flag(EF_MAX_ARGS), bad_flag_result
      REAL arg_1(mem1lox:mem1hix, mem1loy:mem1hiy, mem1loz:mem1hiz, 
     .     mem1lot:mem1hit)
      REAL arg_2(mem2lox:mem2hix, mem2loy:mem2hiy, mem2loz:mem2hiz, 
     .     mem2lot:mem2hit)
      REAL arg_3(mem3lox:mem3hix, mem3loy:mem3hiy, mem3loz:mem3hiz, 
     .     mem3lot:mem3hit)
      REAL arg_4(mem4lox:mem4hix, mem4loy:mem4hiy, mem4loz:mem4hiz, 
     .     mem4lot:mem4hit)
      REAL arg_5(mem5lox:mem5hix, mem5loy:mem5hiy, mem5loz:mem5hiz, 
     .     mem5lot:mem5hit)
      REAL arg_6(mem6lox:mem6hix, mem6loy:mem6hiy, mem6loz:mem6hiz, 
     .     mem6lot:mem6hit)
      REAL arg_7(mem7lox:mem7hix, mem7loy:mem7hiy, mem7loz:mem7hiz, 
     .     mem7lot:mem7hit)
      REAL arg_8(mem8lox:mem8hix, mem8loy:mem8hiy, mem8loz:mem8hiz, 
     .     mem8lot:mem8hit)
      REAL arg_9(mem9lox:mem9hix, mem9loy:mem9hiy, mem9loz:mem9hiz, 
     .     mem9lot:mem9hit)

      REAL result(memreslox:memreshix, memresloy:memreshiy, 
     .     memresloz:memreshiz, memreslot:memreshit)

      REAL gwork(wrk1lox:wrk1hix, wrk1loy:wrk1hiy,
     .               wrk1loz:wrk1hiz, wrk1lot:wrk1hit)

      REAL*8 ax_vals(wrk2lox:wrk2hix/2, wrk2loy:wrk2hiy,
     .               wrk2loz:wrk2hiz, wrk2lot:wrk2hit)

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
*  Include file for the Vis5D utility functions thad write the file.
*  We have a separate Ferret version that renames a parameter "MAXVARS.

      INCLUDE 'v5df_fer.h'

      CHARACTER v5d_filename*160
      CHARACTER arg_name*24, arg_title*128, arg_units*32

      CHARACTER ax_name(4)*16, ax_units(4)*16
      LOGICAL backward(4), modulo(4), regular(4)

      INTEGER i, m, n
      INTEGER numtimes, nlev, numvars, nr, nc, ier

      CHARACTER*255 err_msg

*  Set axis and output arrays.     MAX's are from v5df_fer.h

      REAL*8 xax(MAXCOLUMNS), yax(MAXROWS) 

      INTEGER MAXVARS_OUT
      PARAMETER (MAXVARS_OUT = EF_MAX_ARGS - 1)

      INTEGER timestamp(MAXTIMES), datestamp(MAXTIMES)
      INTEGER ndep(MAXVARS_OUT)

*  Initialize the variables to missing values

      DATA (datestamp(i),i=1,MAXTIMES) / MAXTIMES*IMISSING /
      DATA (timestamp(i),i=1,MAXTIMES) / MAXTIMES*IMISSING /

      INTEGER compress, projection, vertical

      INTEGER idx, idy
      REAL dx, dy, proj_args(4), vert_args(MAXLEVELS)
      INTEGER lowlev(MAX_V5_VARS)

*  Initialize the variable names

      CHARACTER*10 varname(MAXVARS_OUT)
      DATA varname/'var1', 'var2', 'var3', 'var4', 'var5', 'var6', 
     .          'var7', 'var8'/

*  Initialize lowlev
      DATA lowlev/MAX_V5_VARS*0/
 
*  Names of Vis5D utility functions.

      INTEGER v5dcreate
      INTEGER v5dsetlowlev
      INTEGER v5dclose

*  names, units, characteristics of variables to send to Vis5D utility functions.

      CHARACTER*24 zaxis_units(MAXVARS_OUT)
      CHARACTER var_units(MAXVARS_OUT)*32
      LOGICAL ibkwd(MAXVARS_OUT)

      CALL ef_get_res_subscripts (id, res_lo_ss, res_hi_ss, res_incr)
      CALL ef_get_arg_subscripts (id, arg_lo_ss, arg_hi_ss, arg_incr)
      CALL ef_get_bad_flags (id, bad_flag, bad_flag_result)

*  Get coordinate axes.  Note Ferret lists x axis in East longitude and
*  Vis5D assumes West longitude.

      CALL ef_get_coordinates (id, ARG1, X_AXIS, 
     .   arg_lo_ss(X_AXIS, ARG1), arg_hi_ss(X_AXIS, ARG1), xax)
      CALL ef_get_coordinates (id, ARG1, Y_AXIS, 
     .   arg_lo_ss(Y_AXIS, ARG1), arg_hi_ss(Y_AXIS, ARG1), yax)

*  Determine number of variables to write.

      numvars = 0

      DO 100 m = 1, MAXVARS_OUT
         idx = max(arg_incr(X_AXIS,m), 1)
         idy = max(arg_incr(Y_AXIS,m), 1)

         nc = (arg_hi_ss(X_AXIS, m) - arg_lo_ss(X_AXIS, m))/ idx
         nr = (arg_hi_ss(Y_AXIS, m) - arg_lo_ss(Y_AXIS, m))/ idy

         IF (nc .GT. 1  .AND.  nr .GT. 1) numvars = m
  100 CONTINUE

*  Check that the x and y axes for the grids are the same for all variables.
*  Set number of elements in each direction.  Make a NORMAL axis have 
*  one element.

      CALL checkxy (id, numvars, xax, yax, nr, nc, dx, dy, ax_vals, 
     .              err_msg, ier)
      IF (ier .EQ. 0) GO TO 999

C  Set projection arguments for the Vis5D file. 

      proj_args(1) = yax(nr)
      proj_args(3) = dy
      proj_args(2) = 360. - xax(1)
      proj_args(4) = dx

C  Get file name. variable names, and variable units from ferret internal 
C  definitions

      arg = 9
      CALL ef_get_arg_string(id, arg, v5d_filename)

      IF (v5d_filename .EQ. " ") v5d_filename = 'v5d_ferret_file.v5d'		

C  Get  variable names, and variable units from ferret internal definitions

      DO 50 n = 1, numvars

         CALL ef_get_arg_info (id, n, arg_name, arg_title, arg_units)
         CALL ef_get_axis_info (id,n, ax_name, ax_units, backward,
     .     modulo, regular)

         varname(n) = arg_name
         var_units(n) = arg_units
         zaxis_units(n) = ax_units(3)
         ibkwd(n) = backward(3) 

   50 CONTINUE 

C  Set time steps.

      CALL settimes (id, numvars, numtimes, timestamp, datestamp, 
     .               ax_vals, err_msg, ier)

      IF (ier .EQ. 0) GO TO 999

*  Set z steps. 

      CALL setzlevs (id, numvars, vertical, vert_args, zaxis_units,  
     .               ibkwd, ndep, nlev, lowlev, err_msg, ier)

      IF (ier .EQ. 0) THEN
         GO TO 999
      END IF

      compress = 1
      projection = 0

*  Create the Vis5D file
      
      ier = v5dcreate (v5d_filename, numtimes, numvars, nr, nc, 
     .             ndep, varname, timestamp, datestamp, compress, 
     .             projection, proj_args, vertical, vert_args)

      IF (ier .EQ. 0) THEN
        WRITE (err_msg,*) 
     . '(Above are v5dcreate error msgs.) Error during v5d file create.'
        GO TO 999
      END IF

*  Call v5dSetLowLev to specify vertical offset, in grid level index 
*  for each variable.

      ier = v5dsetlowlev (lowlev)

*  v5dsetlowlev returns 0, but the file still works.

c      IF (ier .EQ. 0) THEN
c        write(err_msg,*) ' Error during Vis5D set low levels.'
c        goto 999
c      END IF

*  Call routine to set variable units in Vis5D file.

      DO 300 m = 1, numvars
         CALL v5dsetunits (m, var_units(m))
  300 CONTINUE

*  Fill the data array and write to the Vis5D file for each timestep.

      CALL wrt_v5d_steps (id, nr, nc, ndep, nlev, numvars, ibkwd, arg_1, 
     .      arg_2, arg_3, arg_4, arg_5, arg_6, arg_7, arg_8,  
     .      result, gwork, err_msg, ier)
      IF (ier .EQ. 0) THEN
         GO TO 999
      END IF

*  Close the v5d file and exit

      ier = v5dclose()

      IF (ier .EQ. 0) THEN
         WRITE (err_msg,*) 
     . '(Above are v5dclose error msgs.) Error during v5d file close. '
         GO TO 999
      END IF

      RETURN

 999  CALL ef_bail_out (id, err_msg)

*                                                                      ^
*                                                                      |
*                                            USER CONFIGURABLE PORTION |
************************************************************************

      RETURN 
      END


************************************************************************

      SUBROUTINE wrt_v5d_steps (id, nr, nc, ndep, nlev, numvars, ibkwd,
     .     arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, arg_7, arg_8,  
     .     result, g, err_msg, ier)


*  Write an (x,y,z) grid for each timestep to the Vis5D file.
c
c   note that g(1,1,1) is the north west bottom corner
c   and g(nlat,nlon,ndep) is the south east top corner.
c
c  Order of g    3-D array of grid values; 
c    number of values = nlat*nlon*ndep(var)
c                ordered as data[row+nr*(col+nc*lev)] where row incr from
c                North to South, col increases from West to East, and lev
c                increases from bottom to top
c


      INCLUDE 'ferret_cmn/EF_Util.cmn'
      INCLUDE 'ferret_cmn/EF_mem_subsc.cmn'

      LOGICAL ibkwd(*)
      INTEGER id
      REAL bad_flag(EF_MAX_ARGS), bad_flag_result
      REAL arg_1(mem1lox:mem1hix, mem1loy:mem1hiy, mem1loz:mem1hiz, 
     .     mem1lot:mem1hit)
      REAL arg_2(mem2lox:mem2hix, mem2loy:mem2hiy, mem2loz:mem2hiz, 
     .     mem2lot:mem2hit)
      REAL arg_3(mem3lox:mem3hix, mem3loy:mem3hiy, mem3loz:mem3hiz, 
     .     mem3lot:mem3hit)
      REAL arg_4(mem4lox:mem4hix, mem4loy:mem4hiy, mem4loz:mem4hiz, 
     .     mem4lot:mem4hit)
      REAL arg_5(mem5lox:mem5hix, mem5loy:mem5hiy, mem5loz:mem5hiz, 
     .     mem5lot:mem5hit)
      REAL arg_6(mem6lox:mem6hix, mem6loy:mem6hiy, mem6loz:mem6hiz, 
     .     mem6lot:mem6hit)
      REAL arg_7(mem7lox:mem7hix, mem7loy:mem7hiy, mem7loz:mem7hiz, 
     .     mem7lot:mem7hit)
      REAL arg_8(mem8lox:mem8hix, mem8loy:mem8hiy, mem8loz:mem8hiz, 
     .     mem8lot:mem8hit)


      REAL result(memreslox:memreshix, memresloy:memreshiy,  
     .     memresloz:memreshiz,memreslot:memreshit)

* After initialization, the 'res_' arrays contain indexing information 
* for the result axes.  The 'arg_' arrays will contain the indexing 
* information for each variable's axes. 

      INTEGER res_lo_ss(4), res_hi_ss(4), res_incr(4)
      INTEGER arg_lo_ss(4,EF_MAX_ARGS), arg_hi_ss(4,EF_MAX_ARGS),
     .     arg_incr(4,EF_MAX_ARGS)


      INTEGER  l, m
      INTEGER i1, j1, k1, l1
      INTEGER i2, j2, k2, l2

      INCLUDE 'v5df_fer.h'

      INTEGER numvars, nr, nc, nl
      INTEGER ndep(*), nlev, index

      REAL baddat, argdat

      CHARACTER*(*) err_msg
      integer ier

*  Set output array.

      REAL*4 g(nr, nc, nlev)

      INTEGER v5dwrite

*  Get subscript limits.

      CALL ef_get_res_subscripts (id, res_lo_ss, res_hi_ss, res_incr)
      CALL ef_get_arg_subscripts (id, arg_lo_ss, arg_hi_ss, arg_incr)
      CALL ef_get_bad_flags (id, bad_flag, bad_flag_result)
  
*  Fill and write grids in space for each time step and each variable.
*  i1,j1,k1,l1 argument indices; ferret variables being written.
*  i2,j2,k2,l2 Vis5D grid indices 1:nc, 1:nr, 1:nl, 1:ntime

      DO 500 m = 1, numvars

         l2 = 1
         DO 400 l1 = arg_lo_ss(T_AXIS,ARG1),arg_hi_ss(T_AXIS,ARG1)

            i2 = 1
            DO 300 i1 = arg_lo_ss(X_AXIS,ARG1),arg_hi_ss(X_AXIS,ARG1)

               j2 = 1
               DO 200 j1 = arg_lo_ss(Y_AXIS,ARG1),arg_hi_ss(Y_AXIS,ARG1)

                  k1 = arg_lo_ss(Z_AXIS,m)      ! depth may vary w/ m

                  nl = ndep(m)
                  DO 100 k2 = 1, nl
                     index = k2
                     IF (ibkwd(m)) index = nl-k2+1
                     baddat = bad_flag(m)
 
                     IF (m .EQ. 1) argdat = arg_1(i1,j1,k1,l1) 
                     IF (m .EQ. 2) argdat = arg_2(i1,j1,k1,l1) 
                     IF (m .EQ. 3) argdat = arg_3(i1,j1,k1,l1) 
                     IF (m .EQ. 4) argdat = arg_4(i1,j1,k1,l1) 
                     IF (m .EQ. 5) argdat = arg_5(i1,j1,k1,l1) 
                     IF (m .EQ. 6) argdat = arg_6(i1,j1,k1,l1) 
                     IF (m .EQ. 7) argdat = arg_7(i1,j1,k1,l1) 
                     IF (m .EQ. 8) argdat = arg_8(i1,j1,k1,l1) 


                     IF (argdat .EQ. baddat) THEN
                        g(nr-j2+1,i2,index) = MISSING
                     ELSE 
                        g(nr-j2+1,i2,index) = argdat
                     END IF

                     k1 = k1 + arg_incr(Z_AXIS,m)	! depth may vary w/ m
 100              CONTINUE

                  j2 = j2 + 1
 200           CONTINUE

               i2 = i2 + 1
 300        CONTINUE
   
*  Write the time step to the Vis5D file.

            ier = v5dwrite (l2, m, g)		! l=time step, m=variable #

            IF (ier .EQ. 0) THEN
               WRITE (err_msg,*) ' Error during v5d file write. ',
     .                           ' Time step ', l
               GO TO 999
            END IF

            l2 = l2 + 1
 400     CONTINUE

 500  CONTINUE

      RETURN

 999  CONTINUE

      RETURN

      END

*                                                                      ^
*                                                                      |
************************************************************************

      SUBROUTINE checkxy (id, numvars, xax, yax, nr, nc, dx, dy, 
     .         ax_vals, err_msg, ier)

*  Check that there is at least one variable to write to the Vis5D file.
*  Check the x and y axes for the variables: they must agree with xax, yax
*  from arg_1, and not exceed the limits in the include file v5df.h
*  Set nr = # rows and nc = # columns.

      INCLUDE 'ferret_cmn/EF_Util.cmn'
      INCLUDE 'ferret_cmn/EF_mem_subsc.cmn'

      INCLUDE 'v5df_fer.h'

* After initialization, the 'arg_' arrays will contain the indexing 
* information for each variable's axes. 

      INTEGER id
      INTEGER arg_lo_ss(4,EF_MAX_ARGS), arg_hi_ss(4,EF_MAX_ARGS),
     .     arg_incr(4,EF_MAX_ARGS)

      INTEGER ier
      INTEGER i, j, m
      INTEGER numvars, nc, nr, ncm, nrm

      REAL*8 xax(*), yax(*), ax_vals(*)
      REAL   dx, dy

      CHARACTER*(*) err_msg

      CHARACTER ax_name(4)*16, ax_units(4)*16
      LOGICAL backward(4), modulo(4), regular(4)

      IF (numvars .LT. 1) THEN
         ier = 1
      ELSE
         ier = 1
         CALL ef_get_arg_subscripts (id, arg_lo_ss, arg_hi_ss, arg_incr)

         CALL ef_get_axis_info (id, 1, ax_name, ax_units, backward,
     .     modulo, regular)

*  Set number of elements in each direction.  Let a NORMAL axis have 
*  one element.

      IF (arg_hi_ss(X_AXIS,ARG1) .EQ. arg_lo_ss(X_AXIS,ARG1)  .OR.  
     .       arg_incr(X_AXIS,ARG1) .EQ. 0) THEN 
         dx = 1.
         nc = 1
      ELSE
         dx = abs(xax(2) - xax(1))
         nc = arg_hi_ss(X_AXIS,ARG1) - arg_lo_ss(X_AXIS,ARG1) + 1

*  Check that the x axis is equally-spaced.

         IF (.NOT. regular(1)) THEN
            WRITE (err_msg, *) 'Vis5D uses only equally spaced ',
     .            'X-axis.  Regrid the data in x.'
            GO TO 999
         END IF

      END IF

      IF (arg_hi_ss(Y_AXIS,ARG1) .EQ. arg_lo_ss(Y_AXIS,ARG1)  .OR.  
     .       arg_incr(Y_AXIS,ARG1) .EQ. 0) THEN 
         dy = 1.
         nr = 1
      ELSE
         dy = abs(yax(2) - yax(1))
         nr = arg_hi_ss(Y_AXIS,ARG1) - arg_lo_ss(Y_AXIS,ARG1) + 1

*  Check that the axis is equally-spaced.

         IF (.NOT. regular(2)) THEN
            WRITE (err_msg, *) 'Vis5D uses only equally spaced ',
     .            'Y-axis.  Regrid the data in y.'
            GO TO 999
         END IF

  
      END IF

C  Check number of columns and rows against Vis5D maxima.

         IF (nr .GT. MAXROWS) THEN
            WRITE (err_msg, *) nr, 'Too many rows. Vis5D Max is ', 
     .                  MAXROWS
            ier = 0
            GO TO 999
         ENDIF
         IF (nc .GT. MAXCOLUMNS) THEN
            WRITE (err_msg, *) nr, 'Too many columns. Vis5D Max is ',
     .                  MAXCOLUMNS
            ier = 0
            GO TO 999
         ENDIF

         IF (nr .LE. 1  .OR.  nc .LE. 1)  THEN
            WRITE (err_msg, *) 'The grid for all variables must',
     .         ' have two or more grid points in x and y directions '
             ier = 0
             GO TO 999
         ENDIF

*  Get coordinate axes for variables 2 through numvars.
*  Check with Vis5D max and compare with xax and yax.

         DO 400 m = 2, numvars
            ncm = 
     .        (arg_hi_ss(X_AXIS,m) - arg_lo_ss(X_AXIS,m) + 1)/ 
     .         arg_incr(X_AXIS,m)

            IF (ncm .GT. MAXCOLUMNS) ier = 0
            IF (ncm .NE. nc) ier = 0

            CALL ef_get_coordinates (id, m, X_AXIS, 
     .        arg_lo_ss(X_AXIS, m), arg_hi_ss(X_AXIS, m), ax_vals)
           

            DO 300 i = 1, nc
               IF (xax(i) .NE. ax_vals(i)) ier = 0
 300        CONTINUE

 400     CONTINUE  

         IF (ier .eq. 0) THEN
            WRITE (err_msg, *) 'Vis5D X axes for all variables must ',
     .           ' be the same.'
            go to 999
         ENDIF

         DO 600 m = 2, numvars
            nrm = 
     .        (arg_hi_ss(Y_AXIS,m) - arg_lo_ss(Y_AXIS,m) + 1)/ 
     .         arg_incr(Y_AXIS,m)

            IF (nrm .GT. MAXROWS) ier = 0
            IF (nrm .NE. nr) ier = 0

           CALL ef_get_coordinates (id, m, Y_AXIS, 
     .         arg_lo_ss(Y_AXIS, m), arg_hi_ss(Y_AXIS, m), ax_vals)

            DO 500 j = 1, nr
               IF (yax(j) .NE. ax_vals(j)) ier = 0
 500        CONTINUE

 600     CONTINUE  
         IF (ier .eq. 0) THEN
            WRITE (err_msg, *) 'Vis5D Y axes for all variables must ',
     .           ' be the same.'
            go to 999
         ENDIF

      END IF

      RETURN
  999 ier = 0
      END

*                                                                      ^
*                                                                      |
************************************************************************

      SUBROUTINE settimes (id, numvars, numtimes, timestamp, datestamp, 
     .                     ax_vals, err_msg, ier)

      INCLUDE 'ferret_cmn/EF_Util.cmn'
      INCLUDE 'ferret_cmn/EF_mem_subsc.cmn'

      INCLUDE 'v5df_fer.h'

* After initialization, the 'arg_' arrays will contain the indexing 
* information for each variable's axes. 


      INTEGER id, iarg, l
      INTEGER numtimes

      INTEGER arg_lo_ss(4,EF_MAX_ARGS), arg_hi_ss(4,EF_MAX_ARGS),
     .     arg_incr(4,EF_MAX_ARGS)

      INTEGER ier, m, nlm, icentury
      INTEGER numvars

      REAL*8 tax(MAXTIMES), ax_vals(*)
      INTEGER lasttime, thistime, lastdate, thisdate

      INTEGER iyear, ihour, iminute, isec
      CHARACTER*3 cmon
      INTEGER day_of_mon, day_of_year
      CHARACTER*20 datebuf(MAXTIMES)
      CHARACTER*(*) err_msg

      INTEGER timestamp(*), datestamp(*)

      ier = 1

*  Set time steps; check that there arent too many for Vis5D.

      CALL ef_get_arg_subscripts (id, arg_lo_ss, arg_hi_ss, arg_incr)

      IF (arg_hi_ss(T_AXIS,ARG1) .EQ. arg_lo_ss(T_AXIS,ARG1) .or.
     .      arg_incr(T_AXIS,ARG1) .EQ. 0) THEN 
         numtimes = 1
      ELSE
         numtimes = 
     .     (arg_hi_ss(T_AXIS,ARG1) - arg_lo_ss(T_AXIS,ARG1) + 1)/
     .     arg_incr(T_AXIS,ARG1)
         IF (numtimes .GT. MAXTIMES) THEN
            ier = 0
            WRITE (err_msg,*) ' Vis5D has a limit of ', MAXTIMES, 
     .        ' timesteps.', ' DATA region has', numtimes
            GO TO 999
         END IF
      END IF

*  Get time axis coordinates for first variable.

      CALL ef_get_coordinates (id, ARG1, T_AXIS, 
     .   arg_lo_ss(T_AXIS, ARG1), arg_hi_ss(T_AXIS, ARG1), tax)

*  Get time axis for remaining variables.  Check that thay are 
*  the same as for the first variable.  

      DO 200 m = 2, numvars
         IF (arg_hi_ss(T_AXIS,m) .EQ. arg_lo_ss(T_AXIS,m) .or.
     .      arg_incr(T_AXIS,m) .EQ. 0) THEN 
            nlm = 1
         ELSE
            nlm = 
     .        (arg_hi_ss(T_AXIS,m) - arg_lo_ss(T_AXIS,m) + 1)/ 
     .         arg_incr(T_AXIS,m)
         ENDIF

         IF (nlm .NE. numtimes) THEN
            ier = 0
            WRITE (err_msg,*) ' Vis5D: All variables must have same ',
     .                        '# time steps'
            GO TO 999
         END IF

         CALL ef_get_coordinates (id, m, T_AXIS, 
     .       arg_lo_ss(T_AXIS, m), arg_hi_ss(T_AXIS, m), ax_vals)

         IF (numtimes .GT. 1) THEN

            DO 100 l = 1, numtimes
               IF (tax(l) .NE. ax_vals(l)) GO TO 999
 100        CONTINUE
         ELSE
            ax_vals(1) = tax(1)
         ENDIF

 200  CONTINUE  

* If the time axis of grid is not a normal axis, convert to dates and
* write to timestamp and datestamp in the form that Vis5D expects.

      IF (numtimes .EQ. 1  .AND. tax(1) .LT. 0) THEN    ! some normal axes have -t.
         datestamp(1) = 0
         timestamp(1) = 0

      ELSE

*  If there are negative time-axis values, bail out.

         DO 101 l = 1, numtimes
            IF (tax(l) .LT. 0) THEN
               WRITE (err_msg,*) 
     .          ' Vis5D does not allow negative values on time axis.'
               GO TO 999
            ENDIF

101      CONTINUE

         iarg = 1
         CALL ef_get_axis_dates (id, iarg, tax, numtimes, datebuf)

*  Set up the datestamp and timestamp

         DO 300 l = 1, numtimes


*  If datebuf is in form "DD-MON-YYYY HH:MM:SS".  Read date and time; convert
*  to Vis5D time and date stamps.

            IF (datebuf(l)(3:3) .EQ. '-') THEN
               READ (datebuf(l), 110, err=900) day_of_mon, cmon, iyear
               CALL juldate (cmon, day_of_mon, iyear, day_of_year)

               READ (datebuf(l), 120, err=900) ihour, iminute, isec

* note:  Vis5D uses 2-digit years; years YY <=50 are considered to be 20YY,
*        years YY > 50 are considered to be 19YY.

               IF (iyear .GT. 0) THEN
                  icentury = iyear/ 100
                  icentury = icentury * 100
                  iyear = iyear - icentury
                  if (iyear .eq. 100) iyear = 0

               END IF
 
               datestamp(l) = 1000* iyear + day_of_year	
               timestamp(l) = isec + 100 * iminute + 10000* ihour

*  datebuf not in form "DD-MON-YYYY HH:MM:SS".  Try just reading the time as 
*  a number...

            ELSE
               datestamp(l) = 0
               READ (datebuf(l), *, err=900) timestamp(l)

            ENDIF	! IF (datebuf(l)(3:3) .EQ. '-')

  300    CONTINUE

      ENDIF		! IF (numtimes .EQ. 1) 

      lastdate = datestamp(1)
      lasttime = timestamp(1)

      DO 400 l = 2, numtimes

         thisdate = datestamp(l) 
         thistime = timestamp(l)

         IF ( (lastdate .GT. thisdate)  .OR. 
     .        ( (lastdate .EQ. thisdate) .AND.  
     .          (lasttime .GE. thistime) ) ) THEN

            WRITE (err_msg,*) 
     .     'Times must be strictly increasing ', datebuf(l-1),  
     .     ' ', datebuf(l) 
            GO TO 999
         ENDIF

         lasttime = thistime
         lastdate = thisdate

  400 CONTINUE


  110 FORMAT (i2, 1x, a3, 1x, i4)
  120 FORMAT (11x, 3(1x,i2))

      RETURN

*  Error reading the dates/times. 

  900 CONTINUE

      WRITE (err_msg,*) 
     .  'Error assigning dates/times to timestamp for Vis5D',  
     .  datebuf(l)

  999 ier = 0
      RETURN
      END

*                                                                      ^
*                                                                      |
************************************************************************

      SUBROUTINE juldate (cmon, day_of_mon, iyr, day_of_year)

*  Convert character month and day of month to day of year.

      INTEGER i, iyr
      CHARACTER*(*) cmon
      CHARACTER*3 months(12)
      INTEGER day_of_mon, day_of_year, imon, ndaymo(12)

      DATA months/'JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 
     .            'AUG', 'SEP', 'OCT', 'NOV', 'DEC'/
      DATA ndaymo/31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31/
 

      imon = 0
      DO 100 i = 1, 12
        IF (cmon .EQ. months(i)) imon = i
  100 CONTINUE

      IF (mod(iyr, 4) .EQ. 0) ndaymo(2) = 29

      day_of_year = day_of_mon
      DO 200 i = 1, imon - 1
         day_of_year = day_of_year + ndaymo(i)
  200 CONTINUE

      RETURN
      END
*                                                                      ^
*                                                                      |
************************************************************************
      SUBROUTINE setzlevs (id, numvars, vertical, vert_args, 
     .            zaxis_units, ibkwd, ndep, nlev, lowlev, err_msg, ier)

*  Set z steps.  Must be either the same for all variables or some variables 
*  can be surface data only.

      INCLUDE 'ferret_cmn/EF_Util.cmn'
      INCLUDE 'ferret_cmn/EF_mem_subsc.cmn'

      INTEGER id
      INTEGER arg_lo_ss(4,EF_MAX_ARGS), arg_hi_ss(4,EF_MAX_ARGS),
     .     arg_incr(4,EF_MAX_ARGS)

      INTEGER numvars, nlev, ndepmax, nltot, lv
      INTEGER m, l, ier


      INCLUDE 'v5df_fer.h'

      INTEGER ndep(*), vertical, vert(MAX_V5_VARS)
      INTEGER lowlev(*)
      REAL vert_args(MAXLEVELS), sort_args(MAXLEVELS*EF_MAX_ARGS)
      REAL*8 zax(MAXLEVELS)
      REAL all_levs(MAXLEVELS, EF_MAX_ARGS)

      LOGICAL seq, ibkwd(*)

      REAL dep_fac(EF_MAX_ARGS)

      CHARACTER*(*) zaxis_units(*), err_msg

*  Get axis subscripts and coordinates.

      CALL ef_get_arg_subscripts (id, arg_lo_ss, arg_hi_ss, arg_incr)

      nlev = 1
      ndepmax = 1
      DO 200 m = 1, numvars

         CALL ef_get_coordinates (id, m, Z_AXIS, 
     .   arg_lo_ss(Z_AXIS, m), arg_hi_ss(Z_AXIS, m), zax)

*  Get axis units and determine conversion to KM if possible.  If
*  units are other than cm, m, km, or mb then they will be Vis5D 
*  "generic units"; vertical = 0
* !ACM  ???  Leave cm or m alone and let them be generic units...
* !ACM       But, if some are CM, others in M, want to convert to same units...

         IF (zaxis_units(m) .EQ. 'cm'  .OR. 
     .         zaxis_units(m) .EQ. 'CM') THEN
            dep_fac(m) = 1./1000000.
            vert(m) = 1

         ELSE IF (zaxis_units(m) .EQ. 'm'  .OR.  zaxis_units(m) .EQ. 'M'  
     .            .OR.  zaxis_units(m) .EQ. 'METERS'   .OR.  
     .            zaxis_units(m) .EQ. 'meters') THEN
            dep_fac(m) = 1./1000.
            vert(m) = 1

         ELSE IF (zaxis_units(m) .EQ. 'km'  .OR. 
     .             zaxis_units(m) .EQ. 'KM') THEN
            dep_fac(m) = 1.
            vert(m) = 1

         ELSE IF (zaxis_units(m) .EQ. 'mb'  .OR.  
     .             zaxis_units(m) .EQ. 'MB') THEN
            dep_fac(m) = 1.
            vert(m) = 3

         ELSE 
            dep_fac(m) = 1.
            vert(m) = 0
         END IF

*  !ACM Here we could check and see if all the vertical axes are the same, and
*  set vert_args more simply if so. 

*  vertical = 0 for equally spaced levels in generic units 
*  vertical = 1 for equally spaced levels in km
*  vertical = 2 for unequally spaced levels in km
*  vertical = 3 for unequally spaced levels in mb

*  Get delta-z and determine if the axis is equally spaced.

         IF (arg_hi_ss(Z_AXIS,m) .EQ. arg_lo_ss(Z_AXIS,m)) THEN 
            ndep(m) = 1 
            vert(m) = 0

*  Put DATA for a var w/normal axis at the surface.
            all_levs(1,m) = 0.	
         ELSE

            vertical = 2

            ndep(m) = 
     .        (arg_hi_ss(Z_AXIS,m) - arg_lo_ss(Z_AXIS,m) + 1)/ 
     .         arg_incr(Z_AXIS,m)

            DO 100 l = 1, ndep(m)
               IF (ibkwd(m)) THEN
                  all_levs(ndep(m)-l+1, m) = -1.D0* zax(l)* dep_fac(m) 
               ELSE
                  all_levs(l,m) = zax(l)* dep_fac(m) 
               END IF
  100       CONTINUE


        END IF

         ndepmax = max(ndepmax, ndep(m))
  200 CONTINUE

!  !!!  Vis5D allows us to start some variables at one level and others
!       somewhere else.  doesn't have a structure to allow skipping some
!       levels in any variable. 

!ACM  Because of the result grid being set as inherited from the first 
!     variable, need to have any variables with depth levels come first
!     in the calling sequence to writev5d.F ??  Test for this...

*  Sort the union of all levels into the array vert_args.

      IF (ndepmax .GT. 1) THEN
         nltot = 0
         DO 400 m = 1, numvars
            DO 300 l = 1, ndep(m)
               nltot = nltot + 1
               sort_args(nltot) = all_levs(l, m)
  300       CONTINUE
  400    CONTINUE

         CALL shellsort1 (sort_args, nltot)

*  Remove duplicates.

         nlev = 1
         vert_args(1) = sort_args(1) 
         DO 500 l = 2, nltot 
            IF (sort_args(l) .NE. sort_args(nlev) ) THEN
               nlev = nlev + 1
               sort_args(nlev) = sort_args(l)
               vert_args(nlev) = sort_args(l)
            END IF
  500    CONTINUE

         if (nlev .GT. MAXLEVELS) THEN
            WRITE (err_msg,*) 'Too many levels, all variables ',
     .           'together. Max is ', MAXLEVELS
            ier = 0
            goto 999
         ENDIF

*  Compare the set of levels for each variable with vert_args.  If the 
*  variable's levels are in vert_args with the sequence intact, then set 
*  lowlev appropriately.

         DO 800 m = 1, numvars

*  Find the first element of vert_args that matches the low level of this var.

            seq = .false.
            DO 700 l = 1, nlev
               IF (all_levs(1, m) .EQ. vert_args(l) ) THEN
                  lowlev(m) = l - 1
                  seq = .true.

                  DO 600 lv = 1, ndep(m)
                     IF (all_levs(lv, m)  .NE.  vert_args(l+lv-1) )
     .                   seq = .false.
  600             CONTINUE

               END IF
  700       CONTINUE

            IF (.NOT. seq) THEN
               WRITE (err_msg,*) ' IF levels for different variables ', 
     .             'intersect they must be in sequence with no ', 
     .             'skips.  Levels for argument ', m, 
     .             ' not sequential'
               GO TO 999
            END IF

  800    CONTINUE

      ELSE
         vertical = 0
         vert_args(2) = 1.

      END IF	! ndepmax .GT. 1

      ier = 1

      RETURN

  999 ier = 0

      RETURN
      END

      SUBROUTINE shellsort1 (SDAT, N)


*from http://www.cs.mcgill.ca/~ratzer/progs15_6.html  23-APR-1998
* Changed data to REAL. 
* Removed NCOUNTS, NCOMP, NSWAP.

      REAL     SDAT(*)
      INTEGER  M, N, I, J

      M=N
      DO WHILE (M .GT. 1)
         M=(M+2)/3
         DO I=M+1,N
            DO J=I,M+1,-M
               IF (SDAT(J-M) .GE. SDAT(J)) THEN
                 CALL swapr (SDAT,J,J-M)
               END IF
            END DO 
         END DO 
      END DO
      RETURN
      END !SUBROUTINE SHELL
!

      SUBROUTINE swapr (SDAT,K,L)
      REAL SDAT(*), S
      INTEGER K,L
      S=SDAT(K)
      SDAT(K)=SDAT(L)
      SDAT(L)=S
      RETURN
      END !SUROUTINE SWAP


