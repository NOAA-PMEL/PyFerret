	CHARACTER*(*) FUNCTION FULL_UVAR_NAME_XML(uvar, len)

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
* For xml output give the full URL.

* programmer - Ansley Manke
*             from FULL_UVAR_NAME

* V 6.66

* calling argument declarations:
	INTEGER	uvar

* local variable declarations:
	INTEGER TM_LENSTR1, len, len2

	include 'tmap_dims.parm'
	include 'xdset_info.cmn_text'
	external xdset_info_data
	include 'ferret.parm'
	include	'xvariables.cmn'

* variable name and definition
*	len = TM_LENSTR1(uvar_name_code(uvar))
        CALL string_array_get_strlen1(uvar_name_code_head,uvar,len)
	FULL_UVAR_NAME_XML = uvar_name_code(uvar)(:len)
	IF (uvar_dset(uvar) .GT. pdset_irrelevant) THEN
	   len2 = TM_LENSTR1(ds_des_name(uvar_dset(uvar)))
	   FULL_UVAR_NAME_XML = FULL_UVAR_NAME_XML(:len)//'[D='//
     .			ds_des_name(uvar_dset(uvar))(:len2)//']'
	   len = len + len2 + 4
	ELSEIF (uvar_dset(uvar) .EQ. pdset_irrelevant) THEN
	   FULL_UVAR_NAME_XML = FULL_UVAR_NAME_XML(:len)//
     .			' (/D default)'
	   len = len + 13
	ENDIF

	RETURN
	END


