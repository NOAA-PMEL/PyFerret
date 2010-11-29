	INTEGER FUNCTION FVAR_NUMBER( dset_num, name_code )

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
* identify the given name_code by finding it among the given data set variables

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 - 3/18/87
* revision 0.1 - 8/29/87 - added vms_str_success

#ifdef unix
	include 'tmap_dims.parm'
	include 'xdset_info.cmn_text'
	external xdset_info_data
	include 'ferret.parm'
	include 'gfdl_vms.parm'
#else
	INCLUDE 'TMAP_FORMAT:TMAP_DIMS.PARM'
	INCLUDE 'TMAP_FORMAT:XDSET_INFO.CMN'
	INCLUDE 'FERRET_CMN:FERRET.PARM'
	INCLUDE 'FERRET_CMN:GFDL_VMS.PARM'
#endif

* calling argument declarations:
	INTEGER		dset_num
	CHARACTER*(*)	name_code

* internal variable declarations:
	INTEGER	        fvar, j, num_indices 

* loop through all file variables
*	DO 100 fvar = 1, maxvars
*	   IF ( ds_var_setnum( fvar ) .NE. dset_num ) GOTO 100
*	   vax_code = STR_CASE_BLIND_COMPARE( name_code, ds_var_code(fvar) )

* found the name code ?
*	   IF ( vax_code .EQ. vms_str_success ) THEN
*               FVAR_NUMBER = fvar
*	       RETURN
*	   ENDIF
* 100	CONTINUE

        CALL string_array_find_caseblind(ds_var_code_head,
     .                                   name_code,
     .                                   LEN(name_code),
     .                                   string_array_result1(1),
     .                                   maxvars,
     .                                   num_indices)
        DO 100 j = 1, num_indices
           fvar = string_array_result1(j)
           IF ( ds_var_setnum( fvar ) .NE. dset_num ) GOTO 100
	   FVAR_NUMBER = fvar
	   RETURN
 100	CONTINUE

* not found
	FVAR_NUMBER = var_name_nonexist
	RETURN

	END
