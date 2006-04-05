
! curv_to_rect.F
! From Matt Harrison hinterp.F
! to call only the spherical interpolation modules, and
! recast in F77-compileable format.
! ACM 3/2004
!
!   
!   -- Check use of and document issue of cell bounds 
!      vs cell centers.
!
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


* acm 12/2004 Compute the sum of weights in routine apply_mapping; 
*             sum of weights used when there is good data contributing 
*             to the destination point.
*             Remove use of mask_dst variable; never used.
* acm  3/2005 The curvilinear data may be a subset.  If it was saved
*             with SAVE/HEADING=ENHANCED, then the lo and hi subscripts
*             in the dataset correspond to the subset, and we can use
*             a map pre-computed using the entire dataset for this regridding.
*             Pass the subscripts of the input curvilinear data to the 
*             apply_mapping routine (if not a subset these are 1:max_ss).


! In this SUBROUTINE we provide information about
! the function.  The user configurable information 
! consists of the following:
!
! descr              Text description of the function
!
! num_args           Required number of arguments
!
! axis_inheritance   Type of axis for the result
!                       ( CUSTOM, IMPLIED_BY_ARGS, NORMAL, ABSTRACT )
!                       CUSTOM          - user defined axis
!                       IMPLIED_BY_ARGS - same axis as the incoming argument
!                       NORMAL          - the result is normal to this axis
!                       ABSTRACT        - an axis which only has index values
!
! piecemeal_ok       For memory optimization:
!                       axes where calculation may be performed piecemeal
!                       ( YES, NO )
! 
!
! For each argument we provide the following information:
!
! name               Text name for an argument
!
! unit               Text units for an argument
!
! desc               Text description of an argument
!
! axis_influence     Are this argument's axes the same as the result grid?
!                       ( YES, NO )
!
! axis_extend       How much does Ferret need to extend arg limits relative to result 
!


      SUBROUTINE curv_to_rect_init(id)
        
      INCLUDE 'EF_Util.cmn'

      INTEGER   id, arg
      CHARACTER*100  descr

!**********************************************************************
!                                           USER CONFIGURABLE PORTION |
!                                                                     |
!                                                                     V

      WRITE (descr, 10)
   10 FORMAT
     .('Apply mapping to regrid from curvilinear to rectangular grid')
      CALL ef_set_desc(id, descr)

      CALL ef_set_num_args(id, 2)

      CALL ef_set_axis_inheritance(id, IMPLIED_BY_ARGS,  
     .     IMPLIED_BY_ARGS, IMPLIED_BY_ARGS, IMPLIED_BY_ARGS)

      arg = 1
      CALL ef_set_arg_name(id, arg, 'V')
      CALL ef_set_arg_desc(id, arg, 
     .  'Variable to regrid, on curvilinear grid V(x,y,z,t)')
      CALL ef_set_axis_influence(id, arg, NO, NO, YES, YES)

      arg = 2
      CALL ef_set_arg_name(id, arg, 'mapping')
      WRITE (descr, 30)
   30 FORMAT ('mapping computed by curv_to_rect_MAP')
      CALL ef_set_arg_desc(id, arg, descr)
      CALL ef_set_axis_influence(id, arg, YES, YES, NO, NO)

!                                                                     ^
!                                                                     |
!                                           USER CONFIGURABLE PORTION |
!***********************************************************************

      RETURN 
      END  ! SUBROUTINE curv_to_rect_init


!
! In this SUBROUTINE we compute the result
!
      SUBROUTINE curv_to_rect_compute(id, arg_1, arg_2, result)

      IMPLICIT none
      
      INCLUDE 'EF_Util.cmn'
      INCLUDE 'EF_mem_subsc.cmn'
      
      INTEGER  id

      REAL  bad_flag(EF_MAX_ARGS), bad_flag_result

      REAL  arg_1(mem1lox:mem1hix, mem1loy:mem1hiy, mem1loz:mem1hiz,
     .       mem1lot:mem1hit)
      REAL  arg_2(mem2lox:mem2hix, mem2loy:mem2hiy, mem2loz:mem2hiz,
     .       mem2lot:mem2hit)

      REAL  result(memreslox:memreshix, memresloy:memreshiy, 
     .      memresloz:memreshiz, memreslot:memreshit)

! After initialization, the 'res_' arrays contain indexing information 
! for the result axes.  The 'arg_' arrays will contain the indexing 
! information for each variable's axes. 

      INTEGER  res_lo_ss(4), res_hi_ss(4), res_incr(4)
      INTEGER  arg_lo_ss(4,EF_MAX_ARGS), arg_hi_ss(4,EF_MAX_ARGS),
     .         arg_incr(4,EF_MAX_ARGS)
      
!**********************************************************************
!                                           USER CONFIGURABLE PORTION |
!                                                                     |
!                                                                     V

      INTEGER  i, j, k, l, i1, j1, k1, l1, i2, j2, k2, l2, ih, jh,
     .         num_nbrs, nlon_src, nlat_src, nlon_dst, nlat_dst

      num_nbrs = 4
      
      CALL ef_get_res_subscripts(id, res_lo_ss, res_hi_ss, res_incr)
      CALL ef_get_arg_subscripts(id, arg_lo_ss, arg_hi_ss, arg_incr)
      CALL ef_get_bad_flags(id,bad_flag,bad_flag_result)

      nlon_src = arg_hi_ss(X_AXIS,ARG1) - arg_lo_ss(X_AXIS,ARG1) + 1
      nlat_src = arg_hi_ss(Y_AXIS,ARG1) - arg_lo_ss(Y_AXIS,ARG1) + 1

      nlon_dst = arg_hi_ss(X_AXIS,ARG2) - arg_lo_ss(X_AXIS,ARG2) + 1
      nlat_dst = arg_hi_ss(Y_AXIS,ARG2) - arg_lo_ss(Y_AXIS,ARG2) + 1

      i1 = arg_lo_ss(X_AXIS,ARG1)
      j1 = arg_lo_ss(Y_AXIS,ARG1)
      ih = arg_hi_ss(X_AXIS,ARG1)
      jh = arg_hi_ss(Y_AXIS,ARG1)

      i2 = arg_lo_ss(X_AXIS,ARG2)
      j2 = arg_lo_ss(Y_AXIS,ARG2)
      k2 = arg_lo_ss(Z_AXIS,ARG2)
      l2 = arg_lo_ss(T_AXIS,ARG2)

      i = res_lo_ss(X_AXIS)
      j = res_lo_ss(Y_AXIS)

      l1 = arg_lo_ss(T_AXIS,ARG1)
      DO l = res_lo_ss(T_AXIS),res_hi_ss(T_AXIS)

         k1 = arg_lo_ss(Z_AXIS,ARG1)
         DO k = res_lo_ss(Z_AXIS), res_hi_ss(Z_AXIS)

            CALL apply_mapping ( arg_1(i1,j1,k1,l1), 
     .        i1, ih, j1, jh,
     .        arg_2(i2,j2,k2,l2), arg_2(i2,j2,k2,l2+1),
     .        arg_2(i2,j2,k2,l2+2), result(i,j,k,l), nlon_src,
     .        nlat_src, nlon_dst, nlat_dst, num_nbrs, 
     .        bad_flag(ARG1), bad_flag_result )

            k1 = k1 + arg_incr(Z_AXIS,ARG1)
         ENDDO

         l1 = l1 + arg_incr(T_AXIS,ARG1)
      ENDDO


!                                                                     ^
!                                                                     |
!                                           USER CONFIGURABLE PORTION |
!**********************************************************************

      RETURN 
      END  ! SUBROUTINE curv_to_rect_compute

!#######################################################################
            
       subroutine apply_mapping (data_in, ilo, ihi, jlo, jhi,
     .                           wt, i_lon, j_lat, data_out, 
     .                           nlon_src, nlat_src, nlon_dst, nlat_dst,
     .                           num_nbrs, bad_flag_in, bad_flag_out)

! This is the last set of loops from horiz_interp_sphe, applying wt
! to the data to compute the regridded result.

! acm 7/2004 Add variable COUNT to set data to missing if there is nothing added
! to the sum in a given output grid cell.  This had been done in the original code
! by seeing if the value is a "land" value, but it is not general enough for our
! purposes.

! acm 12/2004 Compute the sum of weights here; weights only used when there
!             is good data contributing to the destination point.

       integer ilo, ihi, jlo, jhi
       real data_in(ilo:ihi, jlo:jhi)     ! input field
       real data_out(nlon_dst, nlat_dst)  ! output
       real wt(nlon_dst, nlat_dst, num_nbrs)  

       real i_lon(nlon_dst,nlat_dst,num_nbrs)
       real j_lat(nlon_dst,nlat_dst,num_nbrs) 

       real bad_flag_in, bad_flag_out

       integer nlon_src, nlat_src, nlon_dst, nlat_dst, 
     .         num_nbrs

!--- some local variables ----------------------------------------
       integer nlon_in, nlat_in, nlon_out, nlat_out, num_neighbors,
     .               m, n, k, i, j
       real sum
       logical okij
!-----------------------------------------------------------------

! parameters
      real epsln, large
cc      parameter (epsln=1.e-10, large=1.e20)
      parameter (epsln=4.e-7, large=1.e20)

         nlon_in  = nlon_src
         nlat_in  = nlat_src
         nlon_out = nlon_dst
         nlat_out = nlat_dst   
         num_neighbors = num_nbrs

         do m=1,nlon_out
         do n=1,nlat_out

            data_out(m,n) = 0.0
            sum=0.0
            do k=1, num_neighbors
               i = i_lon(m,n,k)
               j = j_lat(m,n,k)

               okij = .TRUE.
               IF (i .LT. ilo  .OR. i .GT. ihi) THEN
                  okij = .FALSE.
               ENDIF
               IF (j .LT. jlo  .OR.  j .GT. jhi) THEN
                  okij = .FALSE.
               ENDIF


               if (i.GT.0 .and. j.GT.0 .AND. okij) THEN
                  IF (data_in(i,j) .NE. bad_flag_in ) THEN
                     data_out(m,n) = data_out(m,n)+
     .                                  data_in(i,j)*wt(m,n,k)
                     sum = sum + wt(m,n,k)
                  endif
               else
                  data_out(m,n) = bad_flag_out
                  GO TO 33
               endif
            enddo

            IF (sum .GT. epsln) THEN
               data_out(m,n) = data_out(m,n)/sum
            ELSE
               data_out(m,n) = bad_flag_out
            ENDIF
  33        continue

         enddo
         enddo
      

         return
       end ! subroutine horiz_interp_sphe
      
      
