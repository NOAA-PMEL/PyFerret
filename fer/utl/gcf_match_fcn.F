      INTEGER FUNCTION GCF_MATCH_FCN( template )

*     return the ID number of the first grid changing function whose name
*     matches the template

*     V450: 12/96
*     V500: 09/98 *JC* added external function matching

*     calling argument declarations
      CHARACTER template*(*)

*     internal variable declarations
      LOGICAL  MATCH_TEMPLATE
      
#ifdef unix
      include 'ferret_cmn/ferret.parm'
      include 'ferret_cmn/xgrid_chg_fcns.cmn'
#else 
      INCLUDE 'FERRET_CMN:FERRET.PARM'
      INCLUDE 'FERRET_CMN:XGRID_CHG_FCNS.CMN'
#endif

*     added 09/98
*     
      INCLUDE 'ferret_cmn/EF_Util.cmn'

c     internal parameter declarations:
      INTEGER      slen, EFCN_MATCH_TEMPLATE, EFCN_SCAN
      PARAMETER  ( slen = ef_max_description )
      
#ifdef sun
      BYTE      fhol(slen)      ! c-type Hollerith string buffer
#else 
      INTEGER*1 fhol(slen)      ! c-type Hollerith string buffer
#endif
*     
*     endof 09/98
      
      INTEGER TM_LENSTR1, llen, i


      GCF_MATCH_FCN = ATOM_NOT_FOUND

*     check the internal grid changing functions first
      DO 100 i = 1, gfcn_num_internal
         IF ( MATCH_TEMPLATE(gfcn_name(i),template) ) THEN
            GCF_MATCH_FCN = i
            RETURN
         ENDIF
 100  CONTINUE

*     no match. so check external functions

      IF ( efcn_scan(gfcn_num_internal) .NE. 0 ) THEN

c     true string length
c     convert to a null-terminated Hollerith
c     call the external functions library routine to see if it is external
      
        llen = TM_LENSTR1( template )
        CALL TM_FTOC_STRNG( template(:llen), fhol, slen )

        GCF_MATCH_FCN = EFCN_MATCH_TEMPLATE(fhol)

      ENDIF

*     no luck -- return value indicating failure
      IF ( GCF_MATCH_FCN .EQ. ATOM_NOT_FOUND ) THEN
         GCF_MATCH_FCN = unspecified_int4
      ENDIF

      RETURN

      END
