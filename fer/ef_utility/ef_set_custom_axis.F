c     
c     ef_set_custom_axis.F
c     
c     Jonathan Callahan
c     Feb 19th 1998
c     
c     

c     
c     This subroutine allows an external function to create
c     a custom axis.
c     

      SUBROUTINE EF_SET_CUSTOM_AXIS(id, axis, lo, hi, del,
     .     unit, modulo)

      INCLUDE 'ferret.parm'
      INCLUDE 'errmsg.parm'
      INCLUDE 'EF_Util.cmn'

      INTEGER id, axis, modulo
      REAL    lo, hi, del
      CHARACTER*(*) unit

c     internal parameter declarations:
      INTEGER      slen
      PARAMETER  ( slen = ef_max_description )

#ifdef sun
      BYTE      fhol(slen)      ! c-type Hollerith string buffer
#else 
      INTEGER*1 fhol(slen)      ! c-type Hollerith string buffer
#endif
      
      INTEGER TM_LENSTR1, llen
      
c     true string length
c     convert to a null-terminated Hollerith
c     call the external functions library to set the value
      
      llen = TM_LENSTR1( unit )
      CALL TM_FTOC_STRNG( unit(:llen), fhol, slen )
      call ef_set_custom_axis_sub( id, axis, lo, hi, del, 
     .     fhol, modulo )
      
      
      RETURN 
      END
