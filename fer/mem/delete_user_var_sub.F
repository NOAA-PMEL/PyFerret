	SUBROUTINE DELETE_USER_VAR_SUB ( uvar )

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
* delete a user-defined variable (7/97) ON THE ASSUMPTION THAT IT HAS
* NO CHILDREN (children are handled by DELETE_USER_VAR)
* this involves not only eliminating the variable from the list of user var
* definitions but also eliminating in memory any stored grids that depend on
* this variable AND eliminating from memory any stored grids of other user vars
* whose definitions depend on this variable
*  eg DEFINE VARIABLE TEST1 = SALT / 1000
*     DEFINE VARIABLE TEST2 = TEST1^2
*     DEFINE VARIABLE TEST3 = TEST2^2
* if DELETE_USER_VAR is called to delete TEST1 it must also eliminate all stored
* grids of TEST2 and TEST3

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
*
* revision 0.0 - 2/23/87
* Ultrix/RISC port - 2/91 - eliminated dependency on VAX STR$ routine
*     10/22/92 - increased uvar name length

* V420 9/95 - When user variable is deleted release any claim on dynamic grids
* V450 7/97 - renamed to DELETE_USER_VAR_SUB from DELETE_USER_VAR

#ifdef unix
	include 'tmap_dims.parm'
        include 'xtm_grid.cmn_text'
	include 'ferret.parm'
	include 'errmsg.parm'
	include 'interp_stack.parm'
	include 'xprog_state.cmn'
	include 'xvariables.cmn'
	include 'xrisc.cmn'
#else
	INCLUDE 'TMAP_FORMAT:TMAP_DIMS.PARM'
        INCLUDE 'TMAP_FORMAT:XTM_GRID.CMN_TEXT'
	INCLUDE 'FERRET_CMN:FERRET.PARM'
	INCLUDE 'FERRET_CMN:ERRMSG.PARM'
	INCLUDE 'FERRET_CMN:INTERP_STACK.PARM'
	INCLUDE 'FERRET_CMN:XPROG_STATE.CMN'
	INCLUDE 'FERRET_CMN:XVARIABLES.CMN'
	INCLUDE 'FERRET_CMN:XRISC.CMN'
#endif

* calling argument declarations:
	INTEGER	uvar

* internal variable declarations:
	INTEGER		STR_UPCASE,
     .			du_list( max_uvar ), ndu, kdu, uv, vax_code, ind,
     .			i, mr
	CHARACTER*24	check_name

* procedure:
*	generate a list of all user variables depending on the named uvar
*	then eliminate all refs. to all vars in the list

* variables:
*	du_list	- list of user variables that must be cleared from memory
*	ndu	- number of variables in list
*	kdu	- variable in du_list for which dependencies are being checked

* initialize list of variables to be cleared from memory
	du_list( 1 ) = uvar
	ndu = 1
	kdu = 1

* look through the stored uvar definitions for dependencies
* note: this search depends on uvar_name_code being upper case
*	use risc_buff as temporary storage for an upvase version
*	(unnecessary as of 2/91 but possibly lower case in future)
 100	check_name = uvar_name_code( du_list( kdu ) )	! upper case
	DO 200 uv = 1, max_uvar
	   IF ( uvar_num_items( uv ) .EQ. uvar_deleted ) GOTO 200
	   vax_code = STR_UPCASE( risc_buff, uvar_text( uv ) )
	   ind = INDEX( check_name, risc_buff )
	   IF ( ind .EQ. 0 ) GOTO 200

* make sure this user variable is not already in the list
	   DO 110 i = 1, ndu
	      IF ( du_list( i ) .EQ. uv ) GOTO 200
 110	   CONTINUE

* add another dependent variable to the list
	   ndu = ndu + 1
	   du_list( ndu ) = uv

 200	CONTINUE

* done checking for dependencies on this.  More to check ?
	kdu = kdu + 1
	IF ( kdu .LE. ndu ) GOTO 100

* eliminate all memory grids for variables in the list
	DO 300 i = 1, ndu
 300	CALL PURGE_USER_VAR( du_list( i ) )

* if this user variable is associated with any dynamic grids release 'em
	CALL DEALLO_UVAR_GRIDS(uvar)

* eliminate the named variable from the list of uvars
	uvar_num_items( uvar ) = uvar_deleted
	      
	RETURN
	END
