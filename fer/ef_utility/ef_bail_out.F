c     
c     ef_bail_out.F
c     
c     Jonathan Callahan
c     Feb 19th 1998
c     
c     

c     
c     This subroutine bails out of an external function with
c     the error message passed in.
c     

      SUBROUTINE EF_BAIL_OUT(id, text)

      INCLUDE 'ferret_cmn/ferret.parm'
      INCLUDE 'ferret_cmn/errmsg.parm'
      INCLUDE 'ferret_cmn/EF_Util.cmn'

      INTEGER id
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
c     call the external functions library routine to see if it is external
      
      llen = TM_LENSTR1( text )
      CALL TM_FTOC_STRNG( text(:llen), fhol, slen )
      call ef_err_bail_out( id, fhol )
      
      RETURN 
      END
