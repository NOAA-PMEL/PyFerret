	SUBROUTINE DSG_ONE_INSTANCE_VAL ( mr, itype, ifeature,value, 
     .                                    vstr, maxlen, strlen )

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
* Return a value from the full range of an instance (feature) variable
* that has been loaded 

* NOAA/PMEL, Seattle, WA - SDIG
* V75+ 4/19 *acm* 

	include 'tmap_dims.parm'
	include 'ferret.parm'
	include 'xvariables.cmn'

* calling argument declarations:
	INTEGER cx, mr, itype, ifeature, maxlen, strlen
	REAL	value
	CHARACTER*(*) vstr

* local declarations	
	INTEGER GET_OFFSET_C_STRING_LEN, iwd

* ... feature value

	IF ( itype .EQ. ptype_string ) THEN

	   vstr = '                '

	   iwd = ifeature - 1

	   strlen = GET_OFFSET_C_STRING_LEN(mr_c_pointer(mr),
     .					 iwd )
	   strlen = MIN(strlen, maxlen)  ! fixed length of vstr...
	   CALL GET_OFFSET_C_STRING(mr_c_pointer(mr),
     .				   iwd,
     .				   vstr,
     .				   strlen    ) 

	ELSE

	   value = memry(mr)%ptr(ifeature)
	ENDIF


	RETURN
	END
