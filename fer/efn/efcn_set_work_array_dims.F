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

      SUBROUTINE EFCN_SET_WORK_ARRAY_DIMS(iarray,xlo,ylo,zlo,tlo,
     .           xhi,yhi,zhi,thi)

      INCLUDE 'EF_mem_subsc.cmn'

      INTEGER iarray,xlo,ylo,zlo,tlo,xhi,yhi,zhi,thi

      IF (iarray .EQ. 1) THEN
         wrk1lox = xlo
         wrk1loy = ylo
         wrk1loz = zlo
         wrk1lot = tlo
         wrk1hix = xhi
         wrk1hiy = yhi
         wrk1hiz = zhi
         wrk1hit = thi

      ELSE IF (iarray .EQ. 2) THEN
         wrk2lox = xlo
         wrk2loy = ylo
         wrk2loz = zlo
         wrk2lot = tlo
         wrk2hix = xhi
         wrk2hiy = yhi
         wrk2hiz = zhi
         wrk2hit = thi

      ELSE IF (iarray .EQ. 3) THEN
         wrk3lox = xlo
         wrk3loy = ylo
         wrk3loz = zlo
         wrk3lot = tlo
         wrk3hix = xhi
         wrk3hiy = yhi
         wrk3hiz = zhi
         wrk3hit = thi

      ELSE IF (iarray .EQ. 4) THEN
         wrk4lox = xlo
         wrk4loy = ylo
         wrk4loz = zlo
         wrk4lot = tlo
         wrk4hix = xhi
         wrk4hiy = yhi
         wrk4hiz = zhi
         wrk4hit = thi

      ELSE IF (iarray .EQ. 5) THEN
         wrk5lox = xlo
         wrk5loy = ylo
         wrk5loz = zlo
         wrk5lot = tlo
         wrk5hix = xhi
         wrk5hiy = yhi
         wrk5hiz = zhi
         wrk5hit = thi

      ELSE IF (iarray .EQ. 6) THEN
         wrk6lox = xlo
         wrk6loy = ylo
         wrk6loz = zlo
         wrk6lot = tlo
         wrk6hix = xhi
         wrk6hiy = yhi
         wrk6hiz = zhi
         wrk6hit = thi

      ELSE IF (iarray .EQ. 7) THEN
         wrk7lox = xlo
         wrk7loy = ylo
         wrk7loz = zlo
         wrk7lot = tlo
         wrk7hix = xhi
         wrk7hiy = yhi
         wrk7hiz = zhi
         wrk7hit = thi

      ELSE IF (iarray .EQ. 8) THEN
         wrk8lox = xlo
         wrk8loy = ylo
         wrk8loz = zlo
         wrk8lot = tlo
         wrk8hix = xhi
         wrk8hiy = yhi
         wrk8hiz = zhi
         wrk8hit = thi

      ELSE IF (iarray .EQ. 9) THEN
         wrk9lox = xlo
         wrk9loy = ylo
         wrk9loz = zlo
         wrk9lot = tlo
         wrk9hix = xhi
         wrk9hiy = yhi
         wrk9hiz = zhi
         wrk9hit = thi

      ELSE IF (iarray .EQ. 10) THEN
         wrk10lox = xlo
         wrk10loy = ylo
         wrk10loz = zlo
         wrk10lot = tlo
         wrk10hix = xhi
         wrk10hiy = yhi
         wrk10hiz = zhi
         wrk10hit = thi

      ELSE IF (iarray .EQ. 11) THEN
         wrk11lox = xlo
         wrk11loy = ylo
         wrk11loz = zlo
         wrk11lot = tlo
         wrk11hix = xhi
         wrk11hiy = yhi
         wrk11hiz = zhi
         wrk11hit = thi

      ELSE IF (iarray .EQ. 12) THEN
         wrk12lox = xlo
         wrk12loy = ylo
         wrk12loz = zlo
         wrk12lot = tlo
         wrk12hix = xhi
         wrk12hiy = yhi
         wrk12hiz = zhi
         wrk12hit = thi

      END IF


      RETURN 
      END
