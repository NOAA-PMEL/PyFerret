c     
c     efcn_copy_array_dims.F
c     
c     Jonathan Callahan
c     Sep 22nd 1998
c     

c     
c     This subroutine is called by efcn_compute_ inside EF_ExternalUtil.c. 
c     It copies all of the dimensioning information from xmem_subsc.cmn to
c     the equivalent slots in ef_mem_subsc.cmn.
c     

      SUBROUTINE EFCN_COPY_ARRAY_DIMS

      INCLUDE 'ferret_cmn/xmem_subsc.cmn'
      INCLUDE 'ferret_cmn/EF_mem_subsc.cmn'

      mem1lox = m1lox
      mem1loy = m1loy
      mem1loz = m1loz
      mem1lot = m1lot
      mem1hix = m1hix
      mem1hiy = m1hiy
      mem1hiz = m1hiz
      mem1hit = m1hit

      mem2lox = m2lox
      mem2loy = m2loy
      mem2loz = m2loz
      mem2lot = m2lot
      mem2hix = m2hix
      mem2hiy = m2hiy
      mem2hiz = m2hiz
      mem2hit = m2hit

      mem3lox = m3lox
      mem3loy = m3loy
      mem3loz = m3loz
      mem3lot = m3lot
      mem3hix = m3hix
      mem3hiy = m3hiy
      mem3hiz = m3hiz
      mem3hit = m3hit

      mem4lox = m4lox
      mem4loy = m4loy
      mem4loz = m4loz
      mem4lot = m4lot
      mem4hix = m4hix
      mem4hiy = m4hiy
      mem4hiz = m4hiz
      mem4hit = m4hit

      mem5lox = m5lox
      mem5loy = m5loy
      mem5loz = m5loz
      mem5lot = m5lot
      mem5hix = m5hix
      mem5hiy = m5hiy
      mem5hiz = m5hiz
      mem5hit = m5hit

      mem6lox = m6lox
      mem6loy = m6loy
      mem6loz = m6loz
      mem6lot = m6lot
      mem6hix = m6hix
      mem6hiy = m6hiy
      mem6hiz = m6hiz
      mem6hit = m6hit

      mem7lox = m7lox
      mem7loy = m7loy
      mem7loz = m7loz
      mem7lot = m7lot
      mem7hix = m7hix
      mem7hiy = m7hiy
      mem7hiz = m7hiz
      mem7hit = m7hit

      mem8lox = m8lox
      mem8loy = m8loy
      mem8loz = m8loz
      mem8lot = m8lot
      mem8hix = m8hix
      mem8hiy = m8hiy
      mem8hiz = m8hiz
      mem8hit = m8hit

      mem9lox = m9lox
      mem9loy = m9loy
      mem9loz = m9loz
      mem9lot = m9lot
      mem9hix = m9hix
      mem9hiy = m9hiy
      mem9hiz = m9hiz
      mem9hit = m9hit

      memreslox = m10lox
      memresloy = m10loy
      memresloz = m10loz
      memreslot = m10lot
      memreshix = m10hix
      memreshiy = m10hiy
      memreshiz = m10hiz
      memreshit = m10hit

      RETURN 
      END
