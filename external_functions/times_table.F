c*
c* times_table.F
c*
c* Jonathan Callahan
c* Jan 28th 1998
c*
c* This function averages the data along T and then returns a 
c* multiplication table of the result on an abstract L axis.
c*


c*
c* This subroutine tells Ferret that this is a
c* Fortran function.  You should not change this 
c* subroutine.
c*
      subroutine times_table_lang(language)

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
c*                       ABSTRACT        - an axis which only has index values
c*
c* piecemeal_ok       For memory optimization:
c*                       axes where calculation may be performed piecemeal
c*                       ( YES, NO )


      subroutine times_table_get_info(version, descr, num_args, 
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

      descr = 'creates a times table'
      num_args = 2
      has_vari_args = NO
      axis_will_be(1) = IMPLIED_BY_ARGS
      axis_will_be(2) = IMPLIED_BY_ARGS
      axis_will_be(3) = IMPLIED_BY_ARGS
      axis_will_be(4) = ABSTRACT
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

      subroutine times_table_get_arg_info(arg, axis_implied_from, 
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
         axis_implied_from(4) = NO
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
         desc = 'argument to be multiplied'

      else if ( arg .eq. 2 ) then

         axis_implied_from(1) = NO
         axis_implied_from(2) = NO
         axis_implied_from(3) = NO
         axis_implied_from(4) = NO
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
         desc = 'length of the table'

      end if

c*                                                                     ^
c*                                                                     |
c*                                           USER CONFIGURABLE PORTION |
c***********************************************************************

      return 
      end



c*
c* In this subroutine we provide information about the lo and hi
c* limits associated with each abstract axis.   The user configurable
c* information consists of the following:
c*
c* loss               lo subscript for an axis
c*
c* hiss               hi subscript for an axis
c*

      subroutine times_table_get_axis_abstract(memory, mr_list,
     .     cx_list, axis, loss, hiss)

      real    memory
      integer mr_list, cx_list
      integer axis, loss, hiss

c***********************************************************************
c*                                           USER CONFIGURABLE PORTION |
c*                                                                     |
c*                                                                     V

      common / times_table / mylot, myhit
      integer mylot, myhit

      real value
      integer ivalue

      call ef_get_one_val(2, memory, mr_list, cx_list, value)
      
      ivalue = value

      if ( axis .eq. 4 ) then
         mylot = 1
         myhit = ivalue
         loss = 1
         hiss = ivalue
      else
         write(6,*) 'ERROR in get_axis_abstract: asking' 
         write(6,29) axis
 29       format('for non-abstract axis ', i4)
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
      subroutine times_table_compute(context_list, bad_flag,
     .     bad_flag_result, arg1, length, result)

      include 'ferret_cmn/EF_Util.cmn'
      include 'ferret_cmn/xmem_subsc.cmn'

      integer context_list(EF_MAX_ARGS)
      real bad_flag(EF_MAX_ARGS+1), bad_flag_result
      real arg1(m1lox:m1hix, m1loy:m1hiy, m1loz:m1hiz, m1lot:m1hit)
      real length(m2lox:m2hix, m2loy:m2hiy, m2loz:m2hiz, m2lot:m2hit)
      real result(m10lox:m10hix, m10loy:m10hiy, m10loz:m10hiz, 
     .     m10lot:m10hit)

c* After initialization, the following arrays will contain the context
c* information for each variable for each axis.  The context refers to
c* the currently selected region for a variable.  The "ss" arrays contain
c* the lo and hi index values.  The "ww" arrays contain the 'world coordinate'
c* values.  The delta array contains the 'deltas' for axes with regular
c* spacing.

      integer context_lo_ss(4,EF_MAX_ARGS), 
     .     context_hi_ss(4,EF_MAX_ARGS)
      real    context_lo_ww(4,EF_MAX_ARGS), 
     .     context_hi_ww(4,EF_MAX_ARGS),
     .     context_delta(4,EF_MAX_ARGS)

c***********************************************************************
c*                                           USER CONFIGURABLE PORTION |
c*                                                                     |
c*                                                                     V

      common / times_table / mylot, myhit
      integer mylot, myhit

      integer i, j, k, l, l2

      call ef_init_context_vals(context_list, context_lo_ss,
     .     context_hi_ss, context_lo_ww, context_hi_ww, context_delta)
      

      if ( context_lo_ss(4,1) .ne. context_hi_ss(4,1) ) then
         write(6,*) 'EF ERROR: Improper range on T axis of argument.'
      endif

      do 500 l2=mylot, myhit

         do 400 i=context_lo_ss(1,1), context_hi_ss(1,1)
            do 300 j=context_lo_ss(2,1), context_hi_ss(2,1)
               do 200 k=context_lo_ss(3,1), context_hi_ss(3,1)

                  do 100 l=context_lo_ss(4,1), context_hi_ss(4,1)

                     if ( arg1(i,j,k,l) .eq. bad_flag(1) ) then

                        result(i,j,k,l2) = bad_flag_result

                     else

                        result(i,j,k,l2) = l2 * arg1(i,j,k,l)

                     end if

 100              continue

 200           continue
 300        continue
 400     continue

 500  continue
         
c*                                                                     ^
c*                                                                     |
c*                                           USER CONFIGURABLE PORTION |
c***********************************************************************

      return 
      end
