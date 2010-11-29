	INTEGER FUNCTION UVAR_NUMBER( name_code )

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
* find the number of the given user-defined variable

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 - 2/19/87
* revision 0.1 - 8/29/87 - added vms_str_success
* Ultrix/RISC port - 2/91 - removed VMS runtime library reference
*			    (STR_CASE_BLIND_COMPARE is in the TMAP library)

#ifdef unix
	include 'ferret.parm'
	include 'errmsg.parm'
	include 'gfdl_vms.parm'
	include 'xvariables.cmn'
#else
	INCLUDE 'FERRET_CMN:FERRET.PARM'
	INCLUDE 'FERRET_CMN:ERRMSG.PARM'
	INCLUDE 'FERRET_CMN:GFDL_VMS.PARM'
	INCLUDE 'FERRET_CMN:XVARIABLES.CMN'
#endif

* calling argument declarations:
	CHARACTER*(*)	name_code

* internal variable declarations:
	INTEGER uvar, j, num_indices


* locate the requested variable in the uvar variables
*	DO 100 uvar = 1, max_uvar
*	   vax_code = STR_CASE_BLIND_COMPARE( name_code, 
*     .                       uvar_name_code(uvar) )
*	   IF ( vax_code .EQ. vms_str_success
*     .	  .AND. uvar_num_items( uvar ) .NE. uvar_deleted ) THEN
        CALL string_array_find_caseblind(uvar_name_code_head,
     .                                   name_code,
     .                                   LEN(name_code),
     .                                   string_array_result(1),
     .                                   max_uvar,
     .                                   num_indices)
        DO 100 j = 1, num_indices
           uvar = string_array_result(j)
           IF(uvar_num_items(uvar) .NE. uvar_deleted) THEN
	      UVAR_NUMBER = uvar
	      RETURN
	   ENDIF
 100	CONTINUE

* not found
	UVAR_NUMBER = unspecified_int4
	RETURN
	END
