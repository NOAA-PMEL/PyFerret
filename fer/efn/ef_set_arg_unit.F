c     
c     ef_set_arg_unit.F
c     
c     Jonathan Callahan
c     Feb 19th 1998
c     
c     

c     
c     This subroutine is called by code inside EF_ExternalUtil.c. It will
c     initialize the context limits the EF will use.  
c     

      SUBROUTINE EF_SET_ARG_UNIT(id, arg, text)

      INCLUDE 'ferret_cmn/ferret.parm'
      INCLUDE 'ferret_cmn/errmsg.parm'
      INCLUDE 'ferret_cmn/EF_Util.cmn'

      INTEGER id, arg
      CHARACTER*(*) text

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
      
      llen = TM_LENSTR1( text )
      CALL TM_FTOC_STRNG( text(:llen), fhol, slen )
      call ef_set_arg_unit_sub( id, arg, fhol )
      
      RETURN 
      END
