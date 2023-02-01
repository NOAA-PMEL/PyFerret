c     
c     ef_set_alt_fcn_name.F
c     
c     Ansley Manke
c     Nov 2008
c     
c     
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
c     
c     This subroutine is called by external function init routines.
c     it sets a name for another version of a function (e.g. xcat_str 
c     for a string version of xcat) which takes different argument types
c     and may have a different result type. E.g. this lets the user 
c     reference XCAT with string arguments and Ferret will run XCAT_STR

      SUBROUTINE EF_SET_ALT_FCN_NAME(id, text)

      IMPLICIT NONE
      INCLUDE 'tmap_dims.parm'
      INCLUDE 'ferret.parm'
      INCLUDE 'errmsg.parm'
      INCLUDE 'EF_Util.parm'

      INTEGER id
      CHARACTER*(*) text

c     internal parameter declarations:
      INTEGER      slen
      PARAMETER  ( slen = ef_max_name_length )

      INTEGER*1 fhol(slen)      ! c-type Hollerith string buffer
      INTEGER TM_LENSTR1, llen
      
c     true string length
c     convert to a null-terminated Hollerith
c     call the external functions library routine to see if it is external
      
      llen = TM_LENSTR1( text )
      CALL TM_FTOC_STRNG( text(:llen), fhol, slen )
      CALL EF_SET_ALT_FCN_NAME_SUB( id, fhol )
      
      RETURN 
      END
