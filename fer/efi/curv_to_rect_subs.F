
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

       IMPLICIT NONE
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
      
      
