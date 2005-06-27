

C======================================================================

      subroutine buffer_fillpoly (xarg,yarg,npts,index)

#ifdef unix
        include         'xinterrupt.cmn' ! 1/89 *sh*
        include         'ppl_in_ferret.cmn'
        include         'shade_vars.cmn'
#else
        INCLUDE         'TMAP_PPLV11INC:XINTERRUPT.CMN' ! 1/89 *sh*
        INCLUDE         'TMAP_PPLV11INC:PPL_IN_FERRET.CMN'
        INCLUDE         'tmap_pplv11inc:tmap_shade_vars.inc'
#endif

*     J Davison 2.4.99

*     BUFFER_FILLPOLY() called when entire grid cell is a single color; when
*     color changes construct new polygon from individual cells and fill
*     it in FLUSH_BUFFERED_FILLPOLY().

      real xarg(*), yarg(*)
      integer npts, index, i

      buffer_empty = .false.
      num_buffered = num_buffered + 1

      if (npts .eq. 3 .and. num_buffered .eq. 1)
     .     upper_first = upperflg

      do 10 i = 1, npts
         xbuff(i,num_buffered) = xarg(i)         
         ybuff(i,num_buffered) = yarg(i)
 10   end do

      ilast = index + 1

      return
      end
