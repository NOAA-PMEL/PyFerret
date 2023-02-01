	INTEGER FUNCTION GCF_FIND_FCN( name )

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
* return the ID number of a grid changing function based on its name 

* V450: 12/96
*       97.07.16 *jc* adding code to allow call to efcn_get_id()
*       *acm* 3/12 6D Ferret (common uses nferdims in tmap_dims.parm)

        IMPLICIT NONE
* calling argument declarations
	CHARACTER name*(*)

* internal variable declarations
	INTEGER         STR_MATCH, llen, TM_LENSTR1
	INTEGER efcn_scan, efcn_get_id, efcn_already_have_internals

* internal parameter declarations:
        INTEGER      slen
        PARAMETER  ( slen = 40 )
        INTEGER*1 fhol(slen)    ! c-type Hollerith string buffer
 
	include	'tmap_dims.parm'
	include 'ferret.parm'
	include 'xgrid_chg_fcns.cmn'

* check the internal grid changing functions first

	GCF_FIND_FCN = STR_MATCH ( name,
     .                               gfcn_name, gfcn_num_internal )

        IF ( GCF_FIND_FCN .NE. atom_not_found ) THEN
	   RETURN
	ENDIF


* no match. so check external functions
	IF ( efcn_scan(gfcn_num_internal) .EQ. 0 ) THEN

	   GCF_FIND_FCN = unspecified_int4
	   RETURN

	ELSE

*       true string length
*       convert to a null-terminated Hollerith
*       call the external functions library routine to see if it is external

	   llen = TM_LENSTR1( name )
	   CALL TM_FTOC_STRNG( name(:llen), fhol, slen )
	   GCF_FIND_FCN = efcn_get_id( fhol )

	   IF (GCF_FIND_FCN .EQ. ATOM_NOT_FOUND ) THEN
	      GCF_FIND_FCN = unspecified_int4
	   ELSE
*       chances are we're going to want to do something with this function
	      if ( efcn_already_have_internals(gcf_find_fcn) .eq. 0 ) then
		 call efcn_gather_info(gcf_find_fcn)
	      endif
	   ENDIF

	ENDIF	   
	
	RETURN

	END
 
