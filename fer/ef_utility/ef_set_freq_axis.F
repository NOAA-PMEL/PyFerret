c     
c     ef_set_freq_axis.F
c     
c     Ansley Manke
c     June 2003
c     
c     

c     
c     This subroutine allows an external function to create
c     a frequency axis from the # of points and boxsize.
c     (based on ef_set_custom_axis)
c     

      SUBROUTINE EF_SET_FREQ_AXIS(id, axis, npts, box,
     .     unit, modulo)

      INCLUDE 'ferret.parm'
      INCLUDE 'errmsg.parm'
      INCLUDE 'EF_Util.parm'

      INTEGER id, axis, npts, modulo
      REAL    box
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
      CALL ef_set_freq_axis_sub( id, axis, npts, box, 
     .     fhol, modulo )
      
      
      RETURN 
      END
