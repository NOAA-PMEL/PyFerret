c     
c     ef_bail_out.F
c     
c     Jonathan Callahan
c     Feb 19th 1998
c
c *acm* 11/05 V600  -- New symbol FER_LAST_ERROR contains all the text normally
c                       written to std out, put error text in common to save it
c             
c     
c     

c     
c     This subroutine bails out of an external function with
c     the error message passed in.
c     

      SUBROUTINE EF_BAIL_OUT(id, text)

      INCLUDE 'ferret.parm'
      INCLUDE 'errmsg.parm'
      INCLUDE 'EF_Util.parm'	
      INCLUDE 'xerrmsg_text.cmn'


      INTEGER id
      CHARACTER*(*) text

c     internal parameter declarations:
      INTEGER      slen
      PARAMETER  ( slen = ef_max_description )
      CHARACTER*40 GCF_NAME  ! EF_MAX_NAME_LENGTH is 40 

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

! Put name and text, and their lengths into common for error outputs.
      efname = GCF_NAME (id)
      len_efname = TM_LENSTR1( efname )
      ef_errtext = text
      len_eftext = llen

      CALL ef_err_bail_out( id, fhol )
      
      RETURN 
      END
