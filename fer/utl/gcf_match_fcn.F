	INTEGER FUNCTION GCF_MATCH_FCN( template )

*
*
*  This software was developed by the Thermal Modeling and Analysis
*  Project(TMAP) of the National Oceanographic and Atmospheric
*  Administration's (NOAA) Pacific Marine Environmental Lab(PMEL),
*  hereafter referred to as NOAA/PMEL/TMAP.
*
*  Access and use of this software shall impose the following
*  obligations and understandings on the user. The user is granted the
*  right, without any fee or cost, to use, copy, modify, alter, enhance
*  and distribute this software, and any derivative works thereof, and
*  its supporting documentation for any purpose whatsoever, provided
*  that this entire notice appears in all copies of the software,
*  derivative works and supporting documentation.  Further, the user
*  agrees to credit NOAA/PMEL/TMAP in any publications that result from
*  the use of this software or in any product that includes this
*  software. The names TMAP, NOAA and/or PMEL, however, may not be used
*  in any advertising or publicity to endorse or promote any products
*  or commercial entity unless specific written permission is obtained
*  from NOAA/PMEL/TMAP. The user also understands that NOAA/PMEL/TMAP
*  is not obligated to provide the user with any support, consulting,
*  training or assistance of any kind with regard to the use, operation
*  and performance of this software nor to provide the user with any
*  updates, revisions, new versions or "bug fixes".
*
*  THIS SOFTWARE IS PROVIDED BY NOAA/PMEL/TMAP "AS IS" AND ANY EXPRESS
*  OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
*  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
*  ARE DISCLAIMED. IN NO EVENT SHALL NOAA/PMEL/TMAP BE LIABLE FOR ANY SPECIAL,
*  INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER
*  RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF
*  CONTRACT, NEGLIGENCE OR OTHER TORTUOUS ACTION, ARISING OUT OF OR IN
*  CONNECTION WITH THE ACCESS, USE OR PERFORMANCE OF THIS SOFTWARE. 
*
*
*     return the ID number of the first grid changing function whose name
*     matches the template

*     V450: 12/96
*     V500: 09/98 *JC* added external function matching

*     calling argument declarations
      CHARACTER template*(*)

*     internal variable declarations
      LOGICAL  MATCH_TEMPLATE
      
#ifdef unix
      include 'ferret.parm'
      include 'xgrid_chg_fcns.cmn'
#else 
      INCLUDE 'FERRET_CMN:FERRET.PARM'
      INCLUDE 'FERRET_CMN:XGRID_CHG_FCNS.CMN'
#endif

*     added 09/98
*     
      INCLUDE 'EF_Util.cmn'

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

