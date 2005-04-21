
      subroutine flush_buffered_fillpoly
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
*     BUFFER_FILLPOLY() called when entire grid cell is a single color; when
*     color changes construct new polygon from individual cells and fill
*     it in FLUSH_BUFFERED_FILLPOLY().
*
*     J Davison 2.4.99
* V581 ACM 3/05 removed to its own file from arfill

        include         'xinterrupt.cmn' ! 1/89 *sh*
        include         'ppl_in_ferret.cmn'
        include         'shade_vars.cmn'
	include 	'ARFILL.INC'
        include         'xcurvilinear.cmn'

      real xarg(2*buffer_max+2), yarg(2*buffer_max+2)
      integer m, ndx, neo, noe
      logical even

      if (sqfflg) then
*        ********************* RECTANGLES ********************
         if (curvilinear) then

*           FIRST POINT IS LOWER LEFT CORNER OF FIRST RECTANGLE
            xarg(1) = xbuff(1,1)
            yarg(1) = ybuff(1,1)
           
*           FOLLOWED BY LOWER RIGHT CORNER OF ALL RECTANGLES
            do 10 m = 1, num_buffered
               xarg(m+1) = xbuff(2,m)
               yarg(m+1) = ybuff(2,m)
 10         end do
            
*           PICK THE UPPER RIGHT CORNER OF THE LAST RECTANGLE
            xarg(num_buffered+2) = xbuff(3,num_buffered)
            yarg(num_buffered+2) = ybuff(3,num_buffered)
 
*           THEN RETURN ALONG UPPER LEFT CORNER OF EACH RECTANGLE
            ndx = 2*num_buffered + 3

            do 20 m = num_buffered, 1, -1
               xarg(ndx-m) = xbuff(4,m)
               yarg(ndx-m) = ybuff(4,m)
 20         end do

            call fillpoly (xarg, yarg, 2*num_buffered+2, ilast-1)
          else
            xarg(1) = xbuff(1,1)
            yarg(1) = ybuff(1,1)

            xarg(2) = xbuff(2,num_buffered)
            yarg(2) = ybuff(2,num_buffered)

            xarg(3) = xbuff(3,num_buffered)
            yarg(3) = ybuff(3,num_buffered)

            xarg(4) = xbuff(4,1)
            yarg(4) = ybuff(4,1)

            call fillpoly (xarg, yarg, 4, ilast-1)
         end if
      else
*        ******************* TRIANGLES *******************
         
         if (num_buffered .eq. 1) then
            
            xarg(1) = xbuff(1,1)
            yarg(1) = ybuff(1,1)

            xarg(2) = xbuff(2,1)
            yarg(2) = ybuff(2,1)

            xarg(3) = xbuff(3,1)
            yarg(3) = ybuff(3,1)

            call fillpoly (xarg, yarg, 3, ilast-1)
            goto 1000
         end if
            
         if (mod(num_buffered,2) .eq. 0) then
            even = .true.
         else
            even = .false.
         end if
         
         if (curvilinear) then

*           INDICES DEPEND ON WHETHER THERE ARE AN EVEN OR ODD # OF PTS
            if (even) then
               neo = num_buffered
               noe = num_buffered - 1
            else
               neo = num_buffered - 1
               noe = num_buffered
            end if

*           FIRST POINT IS LEFT CORNER OF FIRST TRIANGLE
            xarg(1) = xbuff(1,1)
            yarg(1) = ybuff(1,1)
               
*           FOLLOWED BY RIGHT CORNER OF ALL TRIANGLES
            do 110 m = 2, neo, 2
               xarg(m/2+1) = xbuff(3,m)
               yarg(m/2+1) = ybuff(3,m)
 110        end do
               
*           PICK THE RIGHT CORNER OF THE LAST TRIANGLE
            xarg(neo/2+2) = xbuff(2,num_buffered)
            yarg(neo/2+2) = ybuff(2,num_buffered)
               
*           THEN RETURN ALONG LEFT CORNER OF EACH TRIANGLE
            ndx = 2*num_buffered + 5
            
            do 120 m = noe, 1, -2
               xarg((ndx-m)/2) = xbuff(3,m)
               yarg((ndx-m)/2) = ybuff(3,m)
 120        end do

            call fillpoly (xarg, yarg, num_buffered+2, ilast-1)
         else
            if (upper_first .and. even) then
               xarg(1) = xbuff(1,1)
               yarg(1) = ybuff(1,1)
               
               xarg(2) = xbuff(3,num_buffered)
               yarg(2) = ybuff(3,num_buffered)
               
               xarg(3) = xbuff(2,num_buffered)
               yarg(3) = ybuff(2,num_buffered)
               
               xarg(4) = xbuff(3,1)
               yarg(4) = ybuff(3,1)

               call fillpoly (xarg, yarg, 4, ilast-1)
            else if (upper_first .and. .not.even) then

               xarg(1) = xbuff(1,1)
               yarg(1) = ybuff(1,1)
               
               xarg(2) = xbuff(1,num_buffered)
               yarg(2) = ybuff(1,num_buffered)
               
               xarg(3) = xbuff(2,num_buffered)
               yarg(3) = ybuff(2,num_buffered)
               
               xarg(4) = xbuff(3,1)
               yarg(4) = ybuff(3,1)

               call fillpoly (xarg, yarg, 4, ilast-1)
            else if (.not.upper_first .and. even) then

               xarg(1) = xbuff(1,1)
               yarg(1) = ybuff(1,1)
               
               xarg(2) = xbuff(1,num_buffered)
               yarg(2) = ybuff(1,num_buffered)
               
               xarg(3) = xbuff(2,num_buffered)
               yarg(3) = ybuff(2,num_buffered)
               
               xarg(4) = xbuff(2,1)
               yarg(4) = ybuff(2,1)

               call fillpoly (xarg, yarg, 4, ilast-1)
            else if (.not.upper_first .and. .not.even) then

               xarg(1) = xbuff(1,1)
               yarg(1) = ybuff(1,1)
               
               xarg(2) = xbuff(3,num_buffered)
               yarg(2) = ybuff(3,num_buffered)
               
               xarg(3) = xbuff(2,num_buffered)
               yarg(3) = ybuff(2,num_buffered)
               
               xarg(4) = xbuff(2,1)
               yarg(4) = ybuff(2,1)

               call fillpoly (xarg, yarg, 4, ilast-1)
            end if
         end if
      end if
      
 1000 buffer_empty = .true.
      num_buffered = 0
      
      return
      end
