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

      IMPLICIT NONE
      INCLUDE 'tmap_dims.parm'
      INCLUDE 'xmem_subsc.cmn'
      INCLUDE 'EF_mem_subsc.cmn'

      mem1lox = m1lox
      mem1loy = m1loy
      mem1loz = m1loz
      mem1lot = m1lot
      mem1loe = m1loe
      mem1lof = m1lof

      mem1hix = m1hix
      mem1hiy = m1hiy
      mem1hiz = m1hiz
      mem1hit = m1hit
      mem1hie = m1hie
      mem1hif = m1hif

      mem2lox = m2lox
      mem2loy = m2loy
      mem2loz = m2loz
      mem2lot = m2lot
      mem2loe = m2loe
      mem2lof = m2lof

      mem2hix = m2hix
      mem2hiy = m2hiy
      mem2hiz = m2hiz
      mem2hit = m2hit
      mem2hie = m2hie
      mem2hif = m2hif

      mem3lox = m3lox
      mem3loy = m3loy
      mem3loz = m3loz
      mem3lot = m3lot
      mem3loe = m3loe
      mem3lof = m3lof

      mem3hix = m3hix
      mem3hiy = m3hiy
      mem3hiz = m3hiz
      mem3hit = m3hit
      mem3hie = m3hie
      mem3hif = m3hif

      mem4lox = m4lox
      mem4loy = m4loy
      mem4loz = m4loz
      mem4lot = m4lot
      mem4loe = m4loe
      mem4lof = m4lof

      mem4hix = m4hix
      mem4hiy = m4hiy
      mem4hiz = m4hiz
      mem4hit = m4hit
      mem4hie = m4hie
      mem4hif = m4hif

      mem5lox = m5lox
      mem5loy = m5loy
      mem5loz = m5loz
      mem5lot = m5lot
      mem5loe = m5loe
      mem5lof = m5lof

      mem5hix = m5hix
      mem5hiy = m5hiy
      mem5hiz = m5hiz
      mem5hit = m5hit
      mem5hie = m5hie
      mem5hif = m5hif

      mem6lox = m6lox
      mem6loy = m6loy
      mem6loz = m6loz
      mem6lot = m6lot
      mem6loe = m6loe
      mem6lof = m6lof

      mem6hix = m6hix
      mem6hiy = m6hiy
      mem6hiz = m6hiz
      mem6hit = m6hit
      mem6hie = m6hie
      mem6hif = m6hif

      mem7lox = m7lox
      mem7loy = m7loy
      mem7loz = m7loz
      mem7lot = m7lot
      mem7loe = m7loe
      mem7lof = m7lof

      mem7hix = m7hix
      mem7hiy = m7hiy
      mem7hiz = m7hiz
      mem7hit = m7hit
      mem7hie = m7hie
      mem7hif = m7hif

      mem8lox = m8lox
      mem8loy = m8loy
      mem8loz = m8loz
      mem8lot = m8lot
      mem8loe = m8loe
      mem8lof = m8lof

      mem8hix = m8hix
      mem8hiy = m8hiy
      mem8hiz = m8hiz
      mem8hit = m8hit
      mem8hie = m8hie
      mem8hif = m8hif

      mem9lox = m9lox
      mem9loy = m9loy
      mem9loz = m9loz
      mem9lot = m9lot
      mem9loe = m9loe
      mem9lof = m9lof

      mem9hix = m9hix
      mem9hiy = m9hiy
      mem9hiz = m9hiz
      mem9hit = m9hit
      mem9hie = m9hie
      mem9hif = m9hif

      memreslox = m10lox
      memresloy = m10loy
      memresloz = m10loz
      memreslot = m10lot
      memresloe = m10loe
      memreslof = m10lof

      memreshix = m10hix
      memreshiy = m10hiy
      memreshiz = m10hiz
      memreshit = m10hit
      memreshie = m10hie
      memreshif = m10hif

      RETURN 
      END
