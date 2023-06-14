c     
c     ef_bail_out.F
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

      IMPLICIT NONE
      INCLUDE 'tmap_dims.parm'
      INCLUDE 'ferret.parm'
      INCLUDE 'errmsg.parm'
      INCLUDE 'EF_Util.parm'
      INCLUDE 'xerrmsg_text.cmn'


      INTEGER id
      CHARACTER*(*) text

c     internal parameter declarations:
      INTEGER      slen
      PARAMETER  ( slen = ef_max_description )
      CHARACTER*(EF_MAX_NAME_LENGTH) GCF_NAME

      INTEGER*1 fhol(slen)      ! c-type Hollerith string buffer
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

      CALL EF_ERR_BAIL_OUT( id, fhol )

      RETURN 
      END
