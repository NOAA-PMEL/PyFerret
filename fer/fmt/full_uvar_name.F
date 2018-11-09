	CHARACTER*(*) FUNCTION FULL_UVAR_NAME(uvar, len)

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
* append to the given user variable name the data set information
*	var[D=dset_name]

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program

* V402:
* V420:  4/96 - document "(/D default)" on pdset_irrelevant variables
*       *acm* 3/12 cleanup ifdefs and unnecessary include files
* V698 2/16 *acm*  For ticket 1786: if mode_upcase is canceled write axis name
*                  using original upper/lowercase spelling
* V698  2/16 *acm* For ticket 2352: variables defined with LET/D are
*                  added to that dataset in the attribute structure
* V720 *acm* 5/17 Implement specific calls in place of CD_GET_VAR_INFO

        IMPLICIT NONE
* calling argument declarations:
	INTEGER	uvar

* local variable declarations:

	INTEGER TM_LENSTR1, len, len2
	INTEGER dset, varid, status

	include 'tmap_dims.parm'
	include 'xdset_info.cmn_text'
	external xdset_info_data
	include 'ferret.parm'
	include	'xvariables.cmn'
	include	'xprog_state.cmn'
	include 'errmsg.parm'

* variable name and definition
*	len = TM_LENSTR1(uvar_name_code(uvar))
        CALL string_array_get_strlen1(uvar_name_code_head,uvar,len)
	FULL_UVAR_NAME = uvar_name_code(uvar)(:len)

* Get the original upper/lowercase spelling if mode upcase is canceled.
	IF (.NOT.mode_upcase_output) THEN  ! get original variable name
	   dset = pdset_uvars
	   CALL CD_GET_VAR_ID (dset, uvar_name_code(uvar), varid, status)
	   IF (status .NE. ferr_ok) THEN
              dset = uvar_dset(uvar)
	      CALL CD_GET_VAR_ID (dset, uvar_name_code(uvar), varid, status)
	   ENDIF
	   IF (status .EQ. ferr_ok) 
     .      CALL CD_GET_ID_VARNAME (dset, varid, FULL_UVAR_NAME, status)
	ENDIF

	IF (uvar_dset(uvar) .GT. pdset_irrelevant) THEN
	   len2 = TM_LENSTR1(ds_name(uvar_dset(uvar)))
	   FULL_UVAR_NAME = FULL_UVAR_NAME(:len)//'[D='//
     .			ds_name(uvar_dset(uvar))(:len2)//']'
	   len = len + len2 + 4
	ELSEIF (uvar_dset(uvar) .EQ. pdset_irrelevant) THEN
	   FULL_UVAR_NAME = FULL_UVAR_NAME(:len)//' (/D default)'
	   len = len + 13
	ENDIF
	IF (uvar_remote(uvar)) THEN
	   FULL_UVAR_NAME = FULL_UVAR_NAME(:len)//' (/REMOTE)'
	   len = len + 10
	ENDIF

	RETURN
	END


