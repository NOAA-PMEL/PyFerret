c*
c* ef_get_one_val_sub_sub.F
c*
c* Jonathan Callahan
c* Dec 17th 1997
c*
c*
c*
c* This subroutine extracts a single value from a memory block.
c*
c* Currently, no checking is done.
c*
c JC_TODO: This is not general!!! Fix it!!!!!!!

      subroutine ef_get_one_val_sub_sub(data, 
     .     lox, hix, loy, hiy, loz, hiz, lot, hit, value, status)

      integer status, lox, hix, loy, hiy, loz, hiz, lot, hit
      real data(lox:hix, loy:hiy, loz:hiz, lot:hit)
      real value

      value = data(lox, loy, loz, lot)

      return
      end
