c     
c     efcn_set_work_array_dims.F
c     
c     Jonathan Callahan
c     Sep 22nd 1998
c     

c     
c     This subroutine is called by efcn_compute_ inside EF_ExternalUtil.c. 
c     It sets all of the dimensioning information for the work spaces in 
c     ef_mem_subsc.cmn.
c     

      SUBROUTINE EFCN_SET_WORK_ARRAY_DIMS(iarray,i,j,k,l)

      INCLUDE 'EF_mem_subsc.cmn'

      INTEGER iarray, i, j, k, l

      IF (iarray .EQ. 1) THEN
         wrk1lox = 1
         wrk1loy = 1
         wrk1loz = 1
         wrk1lot = 1
         wrk1hix = i
         wrk1hiy = j
         wrk1hiz = k
         wrk1hit = l

      ELSE IF (iarray .EQ. 2) THEN
         wrk2lox = 1
         wrk2loy = 1
         wrk2loz = 1
         wrk2lot = 1
         wrk2hix = i
         wrk2hiy = j
         wrk2hiz = k
         wrk2hit = l

      ELSE IF (iarray .EQ. 3) THEN
         wrk3lox = 1
         wrk3loy = 1
         wrk3loz = 1
         wrk3lot = 1
         wrk3hix = i
         wrk3hiy = j
         wrk3hiz = k
         wrk3hit = l

      ELSE IF (iarray .EQ. 4) THEN
         wrk4lox = 1
         wrk4loy = 1
         wrk4loz = 1
         wrk4lot = 1
         wrk4hix = i
         wrk4hiy = j
         wrk4hiz = k
         wrk4hit = l

      ELSE IF (iarray .EQ. 5) THEN
         wrk5lox = 1
         wrk5loy = 1
         wrk5loz = 1
         wrk5lot = 1
         wrk5hix = i
         wrk5hiy = j
         wrk5hiz = k
         wrk5hit = l

      ELSE IF (iarray .EQ. 6) THEN
         wrk6lox = 1
         wrk6loy = 1
         wrk6loz = 1
         wrk6lot = 1
         wrk6hix = i
         wrk6hiy = j
         wrk6hiz = k
         wrk6hit = l

      ELSE IF (iarray .EQ. 7) THEN
         wrk7lox = 1
         wrk7loy = 1
         wrk7loz = 1
         wrk7lot = 1
         wrk7hix = i
         wrk7hiy = j
         wrk7hiz = k
         wrk7hit = l

      ELSE IF (iarray .EQ. 8) THEN
         wrk8lox = 1
         wrk8loy = 1
         wrk8loz = 1
         wrk8lot = 1
         wrk8hix = i
         wrk8hiy = j
         wrk8hiz = k
         wrk8hit = l

      ELSE IF (iarray .EQ. 9) THEN
         wrk9lox = 1
         wrk9loy = 1
         wrk9loz = 1
         wrk9lot = 1
         wrk9hix = i
         wrk9hiy = j
         wrk9hiz = k
         wrk9hit = l

      ELSE IF (iarray .EQ. 10) THEN
         wrk10lox = 1
         wrk10loy = 1
         wrk10loz = 1
         wrk10lot = 1
         wrk10hix = i
         wrk10hiy = j
         wrk10hiz = k
         wrk10hit = l

      ELSE IF (iarray .EQ. 11) THEN
         wrk11lox = 1
         wrk11loy = 1
         wrk11loz = 1
         wrk11lot = 1
         wrk11hix = i
         wrk11hiy = j
         wrk11hiz = k
         wrk11hit = l

      ELSE IF (iarray .EQ. 12) THEN
         wrk12lox = 1
         wrk12loy = 1
         wrk12loz = 1
         wrk12lot = 1
         wrk12hix = i
         wrk12hiy = j
         wrk12hiz = k
         wrk12hit = l

      END IF


      RETURN 
      END
