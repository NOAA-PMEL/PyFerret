c     
c     efcn_get_custom_axis.F
c     
c     Jonathan Callahan
c     Feb 19th 1998
c     
c     

c     
c     This subroutine allows an external function to create
c     a custom axis.
c     

      SUBROUTINE EFCN_GET_CUSTOM_AXIS(id, axis, dlo, dhi, ddel,
     .     unit, modulo_logical)

      INCLUDE 'ferret.parm'
      INCLUDE 'errmsg.parm'
      INCLUDE 'EF_Util.cmn'

      INTEGER id, axis, modulo
      REAL    lo, hi, del
      REAL*8  dlo, dhi, ddel
      CHARACTER*(*) unit
      LOGICAL modulo_logical

c     internal parameter declarations:
      INTEGER      slen
      PARAMETER  ( slen = ef_max_name_length )

#ifdef sun
      BYTE      fhol(slen)      ! c-type Hollerith string buffer
#else 
      INTEGER*1 fhol(slen)      ! c-type Hollerith string buffer
#endif
      
c     call the external functions library to get the value
      
      call efcn_get_custom_axis_sub( id, axis, lo, hi, del, 
     .     fhol, modulo )
      
      modulo_logical = .FALSE.
      IF ( modulo .EQ. 1 ) THEN
         modulo_logical = .TRUE.
      ENDIF

c     convert to a Fortran string
      call TM_CTOF_STRNG( fhol, unit, slen)

c     convert to double
      dlo = lo
      dhi = hi
      ddel = del
      
      RETURN 
      END
