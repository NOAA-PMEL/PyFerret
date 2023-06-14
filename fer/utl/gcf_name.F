	CHARACTER*(*) FUNCTION GCF_NAME( ifcn )

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
* return the name of a grid changing function from its ID number

* V450: 1/97
*       97.07.16 *jc* adding code to allow call to efcn_get_name()
* V500: 3/99 *jc* convert external function names to upper case
*       *acm* 3/12 6D Ferret (common uses nferdims in tmap_dims.parm)

        IMPLICIT NONE
* calling argument declarations
	INTEGER ifcn

* internal variable declarations
        INTEGER STR_UPCASE, status
	CHARACTER text*40

* internal parameter declarations:
        INTEGER      slen
        PARAMETER  ( slen = 40 )
        INTEGER*1 fhol(slen)    ! c-type Hollerith string buffer
 
	include	'tmap_dims.parm'
	include 'ferret.parm'
	include 'errmsg.parm'
	include 'xgrid_chg_fcns.cmn'

* check the internal grid changing functions first
	IF (ifcn .LT. 0 ) THEN

	   CALL ERRMSG( ferr_internal, status, 'gcf_name', *5000)

	ELSEIF (ifcn .LE. gfcn_num_internal) THEN

	   GCF_NAME = gfcn_name(ifcn)

	ELSE

	   CALL EFCN_GET_NAME(ifcn, fhol)
	   CALL TM_CTOF_STRNG( fhol, text, slen)
           status =  STR_UPCASE(gcf_name,text)

	ENDIF	   
	
* success
	RETURN

* error exit
 5000	GCF_NAME = 'ILLEGAL NAME'
	RETURN

	END
