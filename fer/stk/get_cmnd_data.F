	SUBROUTINE GET_CMND_DATA ( memory, dflt_cx, status )

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
* Expression(s) are in user-variable (uvar) structures.
* Implied expression is left from last command

* a sample command may look like:
*	prompt> CONTOUR/Z=50 TEMP[X=160E:160W]
* the "Z=50"		is a "command  qualifier" or "command  context modifier"
* the "X=160E:160W"	is a "variable qualifier" or "variable context modifier"
* contexts are transferred, modified and used in the following manner:
*   i)	skeletal context ( without grid-related info.) is specified in the call
*	Usually it is based on SET REGION,DTA_SET,etc. ( cx_last )
*  ii)	this context is modified by the command qualifiers given to produce a 
*	new skeletal context to be used for the evaluation of the current 
*	command ( cx_cmnd )
*  iii)	as the expression is evaluated each variable, as it is encountered,
*	begins with cx_cmnd and modifies it by the variable
*	qualifiers to produce the context at which the variable is to be
*	evaluated.

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program

* V230:  7/27/92 - split into two separate routines, GET_PROT_CMND_DATA and
*                  and UUNPROTECT_CMND_DATA so that memory management can be
*                  used during command (e.g. LIST command needs to create
*                  permutations)
* V312: 5/94 - array "memory" as a calling argument

#ifdef unix
	include 'ferret_cmn/errmsg.parm'
#else
	INCLUDE 'FERRET_CMN:ERRMSG.PARM'
#endif

* calling argument declarations
	INTEGER	dflt_cx, status
	REAL	memory(*)

* local variable declarations
* ...

        CALL GET_PROT_CMND_DATA( memory, dflt_cx, status )
        IF ( status .NE. ferr_ok ) RETURN

        CALL UNPROTECT_CMND_DATA

	RETURN
	END



