c*
c* add_9.F
c*
c* Jonathan Callahan
c* Jan 15th 1998
c*
c* This function adds nine argments.
c*


c*
c* This subroutine tells Ferret that this is a
c* Fortran function.  You should not change this 
c* subroutine.
c*
      subroutine add_9_lang(language)

      include 'ferret_cmn/EF_Util.cmn'

      integer language

      language = EF_F

      return 
      end


c*
c* In this subroutine we provide information about
c* the function.  The user configurable information 
c* consists of the following:
c*
c* version            Ferret internal EF version number
c*
c* descr              Text description of the function
c*
c* num_args           Required number of arguments
c*
c* has_vari_args      Does this function have a variable number of arguments
c*                       ( YES, NO )
c*                       Only use 'NO' for now.
c*
c* axis_will_be       Type of axis offor the result
c*                       ( CUSTOM, IMPLIED_BY_ARGS, NORMAL, ABSTRACT )
c*                       CUSTOM          - not currently supported
c*                       IMPLIED_BY_ARGS - same axis as the incoming argument
c*                       NORMAL          - the result is normal to this axis
c*                       ABSTRACT        - not currently supported
c*
c* piecemeal_ok       For memory optimization:
c*                       axes where calculation may be performed piecemeal
c*                       ( YES, NO )


      subroutine add_9_get_info(version, descr, num_args, 
     .     has_vari_args, axis_will_be, piecemeal_ok)

      include 'ferret_cmn/EF_Util.cmn'

      real version
      character descr*128
      integer num_args
      integer has_vari_args
      integer axis_will_be(4)
      integer piecemeal_ok(4)


      version = EF_VERSION

c***********************************************************************
c*                                           USER CONFIGURABLE PORTION |
c*                                                                     |
c*                                                                     V

      descr = 'adds nine arguments'
      num_args = 9
      has_vari_args = NO
      axis_will_be(1) = IMPLIED_BY_ARGS
      axis_will_be(2) = IMPLIED_BY_ARGS
      axis_will_be(3) = IMPLIED_BY_ARGS
      axis_will_be(4) = IMPLIED_BY_ARGS
      piecemeal_ok(1) = YES 
      piecemeal_ok(2) = YES
      piecemeal_ok(3) = YES
      piecemeal_ok(4) = YES

c*                                                                     ^
c*                                                                     |
c*                                           USER CONFIGURABLE PORTION |
c***********************************************************************

      return 
      end


c*
c* In this subroutine we provide information about each
c* argument the function will accept.   The user configurable
c* information consists of the following:
c*
c* name               Text name for an argument
c*
c* units              Text units for an argument
c*
c* desc               Text description of an argument
c*
c* axis_implied_from  Should this argument's axes be merged into the result grid?
c*                       ( YES, NO )
c*
c* axis_extend_lo     How much does Ferret need to extend arg limits relative to result 
c* axis_extend_lo        ( YES, NO )
c*

      subroutine add_9_get_arg_info(arg, axis_implied_from, 
     .     axis_extend_lo, axis_extend_hi, name, units, desc)

      include 'ferret_cmn/EF_Util.cmn'

      integer arg
      integer axis_implied_from(4)
      integer axis_extend_lo(4)
      integer axis_extend_hi(4)
      character name*40
      character units*40
      character desc*128

c***********************************************************************
c*                                           USER CONFIGURABLE PORTION |
c*                                                                     |
c*                                                                     V

      if ( arg .eq. 1 ) then

         axis_implied_from(1) = YES
         axis_implied_from(2) = YES
         axis_implied_from(3) = YES
         axis_implied_from(4) = YES
         axis_extend_lo(1) = 0
         axis_extend_lo(2) = 0
         axis_extend_lo(3) = 0
         axis_extend_lo(4) = 0
         axis_extend_hi(1) = 0
         axis_extend_hi(2) = 0
         axis_extend_hi(3) = 0
         axis_extend_hi(4) = 0
         name = ' '
         units = ' '
         desc = ' '

      else if ( arg .eq. 2 ) then

         axis_implied_from(1) = YES
         axis_implied_from(2) = YES
         axis_implied_from(3) = YES
         axis_implied_from(4) = YES
         axis_extend_lo(1) = 0
         axis_extend_lo(2) = 0
         axis_extend_lo(3) = 0
         axis_extend_lo(4) = 0
         axis_extend_hi(1) = 0
         axis_extend_hi(2) = 0
         axis_extend_hi(3) = 0
         axis_extend_hi(4) = 0
         name = ' '
         units = ' '
         desc = ' '

      else if ( arg .eq. 3 ) then

         axis_implied_from(1) = YES
         axis_implied_from(2) = YES
         axis_implied_from(3) = YES
         axis_implied_from(4) = YES
         axis_extend_lo(1) = 0
         axis_extend_lo(2) = 0
         axis_extend_lo(3) = 0
         axis_extend_lo(4) = 0
         axis_extend_hi(1) = 0
         axis_extend_hi(2) = 0
         axis_extend_hi(3) = 0
         axis_extend_hi(4) = 0
         name = ' '
         units = ' '
         desc = ' '

      else if ( arg .eq. 4 ) then

         axis_implied_from(1) = YES
         axis_implied_from(2) = YES
         axis_implied_from(3) = YES
         axis_implied_from(4) = YES
         axis_extend_lo(1) = 0
         axis_extend_lo(2) = 0
         axis_extend_lo(3) = 0
         axis_extend_lo(4) = 0
         axis_extend_hi(1) = 0
         axis_extend_hi(2) = 0
         axis_extend_hi(3) = 0
         axis_extend_hi(4) = 0
         name = ' '
         units = ' '
         desc = ' '

      else if ( arg .eq. 5 ) then

         axis_implied_from(1) = YES
         axis_implied_from(2) = YES
         axis_implied_from(3) = YES
         axis_implied_from(4) = YES
         axis_extend_lo(1) = 0
         axis_extend_lo(2) = 0
         axis_extend_lo(3) = 0
         axis_extend_lo(4) = 0
         axis_extend_hi(1) = 0
         axis_extend_hi(2) = 0
         axis_extend_hi(3) = 0
         axis_extend_hi(4) = 0
         name = ' '
         units = ' '
         desc = ' '

      else if ( arg .eq. 6 ) then

         axis_implied_from(1) = YES
         axis_implied_from(2) = YES
         axis_implied_from(3) = YES
         axis_implied_from(4) = YES
         axis_extend_lo(1) = 0
         axis_extend_lo(2) = 0
         axis_extend_lo(3) = 0
         axis_extend_lo(4) = 0
         axis_extend_hi(1) = 0
         axis_extend_hi(2) = 0
         axis_extend_hi(3) = 0
         axis_extend_hi(4) = 0
         name = ' '
         units = ' '
         desc = ' '

      else if ( arg .eq. 7 ) then

         axis_implied_from(1) = YES
         axis_implied_from(2) = YES
         axis_implied_from(3) = YES
         axis_implied_from(4) = YES
         axis_extend_lo(1) = 0
         axis_extend_lo(2) = 0
         axis_extend_lo(3) = 0
         axis_extend_lo(4) = 0
         axis_extend_hi(1) = 0
         axis_extend_hi(2) = 0
         axis_extend_hi(3) = 0
         axis_extend_hi(4) = 0
         name = ' '
         units = ' '
         desc = ' '

      else if ( arg .eq. 8 ) then

         axis_implied_from(1) = YES
         axis_implied_from(2) = YES
         axis_implied_from(3) = YES
         axis_implied_from(4) = YES
         axis_extend_lo(1) = 0
         axis_extend_lo(2) = 0
         axis_extend_lo(3) = 0
         axis_extend_lo(4) = 0
         axis_extend_hi(1) = 0
         axis_extend_hi(2) = 0
         axis_extend_hi(3) = 0
         axis_extend_hi(4) = 0
         name = ' '
         units = ' '
         desc = ' '

      else if ( arg .eq. 9 ) then

         axis_implied_from(1) = YES
         axis_implied_from(2) = YES
         axis_implied_from(3) = YES
         axis_implied_from(4) = YES
         axis_extend_lo(1) = 0
         axis_extend_lo(2) = 0
         axis_extend_lo(3) = 0
         axis_extend_lo(4) = 0
         axis_extend_hi(1) = 0
         axis_extend_hi(2) = 0
         axis_extend_hi(3) = 0
         axis_extend_hi(4) = 0
         name = ' '
         units = ' '
         desc = ' '

      end if

c*                                                                     ^
c*                                                                     |
c*                                           USER CONFIGURABLE PORTION |
c***********************************************************************

      return 
      end



c*
c* In this subroutine we compute the result
c*
      subroutine add_9_compute(context_list, bad_flag,
     .     bad_flag_result, arg1, arg2, arg3, arg4, arg5,
     .     arg6, arg7, arg8, arg9, result)

      include 'ferret_cmn/EF_Util.cmn'
      include 'ferret_cmn/xmem_subsc.cmn'

      integer context_list(EF_MAX_ARGS)
      real bad_flag(EF_MAX_ARGS+1), bad_flag_result
      real arg1(m1lox:m1hix, m1loy:m1hiy, m1loz:m1hiz, m1lot:m1hit)
      real arg2(m2lox:m2hix, m2loy:m2hiy, m2loz:m2hiz, m2lot:m2hit)
      real arg3(m3lox:m3hix, m3loy:m3hiy, m3loz:m3hiz, m3lot:m3hit)
      real arg4(m4lox:m4hix, m4loy:m4hiy, m4loz:m4hiz, m4lot:m4hit)
      real arg5(m5lox:m5hix, m5loy:m5hiy, m5loz:m5hiz, m5lot:m5hit)
      real arg6(m6lox:m6hix, m6loy:m6hiy, m6loz:m6hiz, m6lot:m6hit)
      real arg7(m7lox:m7hix, m7loy:m7hiy, m7loz:m7hiz, m7lot:m7hit)
      real arg8(m8lox:m8hix, m8loy:m8hiy, m8loz:m8hiz, m8lot:m8hit)
      real arg9(m9lox:m9hix, m9loy:m9hiy, m9loz:m9hiz, m9lot:m9hit)
      real result(m10lox:m10hix, m10loy:m10hiy, m10loz:m10hiz, 
     .     m10lot:m10hit)

c* After initialization, the following arrays will contain the context
c* information for each variable for each axis.  The context refers to
c* the currently selected region for a variable.  The "ss" arrays contain
c* the lo and hi index values.  The "ww" arrays contain the 'world coordinate'
c* values.  The delta array contains the 'deltas' for axes with regular
c* spacing.

      integer context_lo_ss(4,EF_MAX_ARGS), 
     .     context_hi_ss(4,EF_MAX_ARGS),
     .     context_incr(4,EF_MAX_ARGS)
      real    context_lo_ww(4,EF_MAX_ARGS), 
     .     context_hi_ww(4,EF_MAX_ARGS),
     .     context_delta(4,EF_MAX_ARGS)

c***********************************************************************
c*                                           USER CONFIGURABLE PORTION |
c*                                                                     |
c*                                                                     V

      integer i1, j1, k1, l2
      integer i2, j2, k2, l2
      integer i3, j3, k3, l3
      integer i4, j4, k4, l4
      integer i5, j5, k5, l5
      integer i6, j6, k6, l6
      integer i7, j7, k7, l7
      integer i8, j8, k8, l8
      integer i9, j9, k9, l9

      call ef_init_context_vals(context_list, context_lo_ss,
     .     context_hi_ss, context_incr, context_lo_ww, context_hi_ww,
     .     context_delta)
      
      i2 = context_lo_ss(1,2)
      i3 = context_lo_ss(1,3)
      i4 = context_lo_ss(1,4)
      i5 = context_lo_ss(1,5)
      i6 = context_lo_ss(1,6)
      i7 = context_lo_ss(1,7)
      i8 = context_lo_ss(1,8)
      i9 = context_lo_ss(1,9)
      do 400 i1=m1lox, m1hix

         j2 = context_lo_ss(2,2)
         j3 = context_lo_ss(2,3)
         j4 = context_lo_ss(2,4)
         j5 = context_lo_ss(2,5)
         j6 = context_lo_ss(2,6)
         j7 = context_lo_ss(2,7)
         j8 = context_lo_ss(2,8)
         j9 = context_lo_ss(2,9)
         do 300 j1=m1loy, m1hiy

            k2 = context_lo_ss(3,2)
            k3 = context_lo_ss(3,3)
            k4 = context_lo_ss(3,4)
            k5 = context_lo_ss(3,5)
            k6 = context_lo_ss(3,6)
            k7 = context_lo_ss(3,7)
            k8 = context_lo_ss(3,8)
            k9 = context_lo_ss(3,9)
            do 200 k1=m1loz, m1hiz

               l2 = context_lo_ss(4,2)
               l3 = context_lo_ss(4,3)
               l4 = context_lo_ss(4,4)
               l5 = context_lo_ss(4,5)
               l6 = context_lo_ss(4,6)
               l7 = context_lo_ss(4,7)
               l8 = context_lo_ss(4,8)
               l9 = context_lo_ss(4,9)
               do 100 l1=m1lot, m1hit

                  if ( arg1(i1,j1,k1,l1) .eq. bad_flag(1) .or. 
     .                 arg2(i2,j2,k2,l2) .eq. bad_flag(2) .or. 
     .                 arg3(i3,j3,k3,l3) .eq. bad_flag(3) .or. 
     .                 arg4(i4,j4,k4,l4) .eq. bad_flag(4) .or. 
     .                 arg5(i5,j5,k5,l5) .eq. bad_flag(5) .or. 
     .                 arg6(i6,j6,k6,l6) .eq. bad_flag(6) .or.
     .                 arg7(i7,j7,k7,l7) .eq. bad_flag(7) .or.
     .                 arg8(i8,j8,k8,l8) .eq. bad_flag(8) .or.
     .                 arg9(i9,j9,k9,l9) .eq. bad_flag(9) ) then

                     result(i1,j1,k1,l1) = bad_flag_result

                  else

                     result(i1,j1,k1,l1) = arg1(i1,j1,k1,l1) + 
     .                  arg2(i2,j2,k2,l2) +
     .                  arg3(i3,j3,k3,l3) +
     .                  arg4(i4,j4,k4,l4) +
     .                  arg5(i5,j5,k5,l5) +
     .                  arg6(i6,j6,k6,l6) +
     .                  arg7(i7,j7,k7,l7) +
     .                  arg8(i8,j8,k8,l8) +
     .                  arg9(i9,j9,k9,l9)

                  end if

                  l2 = l2 + context_incr(4,2)
                  l3 = l3 + context_incr(4,3)
                  l4 = l4 + context_incr(4,4)
                  l5 = l5 + context_incr(4,5)
                  l6 = l6 + context_incr(4,6)
                  l7 = l7 + context_incr(4,7)
                  l8 = l8 + context_incr(4,8)
                  l9 = l9 + context_incr(4,9)
 100           continue

               k2 = k2 + context_incr(3,2)
               k3 = k3 + context_incr(3,3)
               k4 = k4 + context_incr(3,4)
               k5 = k5 + context_incr(3,5)
               k6 = k6 + context_incr(3,6)
               k7 = k7 + context_incr(3,7)
               k8 = k8 + context_incr(3,8)
               k9 = k9 + context_incr(3,9)
 200        continue

            j2 = j2 + context_incr(2,2)
            j3 = j3 + context_incr(2,3)
            j4 = j4 + context_incr(2,4)
            j5 = j5 + context_incr(2,5)
            j6 = j6 + context_incr(2,6)
            j7 = j7 + context_incr(2,7)
            j8 = j8 + context_incr(2,8)
            j9 = j9 + context_incr(2,9)
 300     continue

         i2 = i2 + context_incr(1,2)
         i3 = i3 + context_incr(1,3)
         i4 = i4 + context_incr(1,4)
         i5 = i5 + context_incr(1,5)
         i6 = i6 + context_incr(1,6)
         i7 = i7 + context_incr(1,7)
         i8 = i8 + context_incr(1,8)
         i9 = i9 + context_incr(1,9)
 400  continue
         
c*                                                                     ^
c*                                                                     |
c*                                           USER CONFIGURABLE PORTION |
c***********************************************************************

      return 
      end



