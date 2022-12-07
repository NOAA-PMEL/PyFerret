	SUBROUTINE DELETE_USER_VAR_SUB ( uvar, dset )

*
*
*  This software was developed by the Thermal Modeling and Analysis
*  Project(TMAP) of the National Oceanographic and Atmospheric
*  Administration''s (NOAA) Pacific Marine Environmental Lab(PMEL),
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

* ==> ***NOTE** The process of locating dependencies is a crude, inefficient
* hunt based upon scanning the text of uvar definitions to locate dependent
* variables.  Leveraging on the uvarGrid LIST structures that each ncvar has
* (see NCF_Util.h), a "descendent" list could be managed for each variable
* -- e.g. in IS_UVAR_GRID -- as the grid of a variable is being determined, it
* can add itself to a list maintained by each of children (because the child's
* grid must already exist fore the current (parent) grid to be determined


* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
*
* revision 0.0 - 2/23/87
* Ultrix/RISC port - 2/91 - eliminated dependency on VAX STR$ routine
*     10/22/92 - increased uvar name length

* V420 9/95 - When user variable is deleted release any claim on dynamic grids
* V450 7/97 - renamed to DELETE_USER_VAR_SUB from DELETE_USER_VAR
* V510 2/00 *sh* - with the advent of "child variables" created by grid
*	changing variables deleting a user variable must also remove the name
*	of the (child) variable
* V600  4/06 *acm* - remove deleted variable defs from the linked-list structure
*                    that does attribute handling
* V600  4/06 *acm* - additional declarations found by Solaris compiler
* V602 12/06 *acm* - When removeing var from the user-variables dataset in
*                    linked-list structure, no need to check for  EX# expressions
*        2/12 *kms* Add E and F dimensions (use nferdims in tmap_dims.parm)
* V695+ 5/15 *sh* - free cached c array, if any
* V695  7/15 *acm - reinitialize value of uvar_aux_stat to zero
* V698  2/16 *acm* For ticket 2352: variables defined with LET/D are
*                  added to that dataset in the attribute structure.
* V699 6/16 *sh* treat EX# variables just like other variables
* V751 *acm*11/19  Ticket 1957: Deleting variable from linked-list structure
*                  don't try to look in a dataset 
* 11/2022 *acm* Code cleanup: Modernize DO-loop syntax. 

        IMPLICIT NONE
        include 'tmap_dims.parm'
        include 'xtm_grid.cmn_text'
	include 'ferret.parm'
	include 'errmsg.parm'
	include 'interp_stack.parm'
	include 'xprog_state.cmn'
	include 'xvariables.cmn'
	include 'xrisc.cmn'

* calling argument declarations:
	INTEGER	uvar, dset

* internal variable declarations:
	INTEGER		STR_UPCASE, TM_LENSTR, TM_LENSTR1,du_list(max_uvar), 
     .			ndu, kdu, uv, vax_code,ind, i, j, num_indices, iset, 
     .			varid, slen, flen, status
	CHARACTER*24	check_name

      PARAMETER (flen=128)      ! size of variable name string

#ifdef sun
      BYTE      fhol(flen)      ! c-type Hollerith string buffer
#else
      INTEGER*1 fhol(flen)      ! c-type Hollerith string buffer
#endif 


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

        CALL deleted_list_get_undel(uvar_num_items_head,
     .                              deleted_list_result(1),
     .                              max_uvar, num_indices )
        DO 200 j = 1, num_indices
           uv = deleted_list_result(j)
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
	DO i = 1, ndu
 	   CALL PURGE_USER_VAR( du_list( i ) )
 	ENDDO

* if this user variable is associated with any dynamic grids release them
	CALL DEALLO_UVAR_GRIDS(uvar)

* if there is a c cache of the full range variable, free it
* note: this addition remained untested at the time it was entered
	IF (uvar_c_cache_ptr(uvar) .NE. 0.0) CALL FREE_CACHED_FULL_ARRAY
     .						(uvar_c_cache_ptr(uvar))

* eliminate the named variable from the list of uvars
*	uvar_num_items( uvar ) = uvar_deleted
        CALL deleted_list_modify(uvar_num_items_head,uvar,uvar_deleted)

* Remove from linked-list structure, the user-variables dataset
!        slen = TM_LENSTR( uvar_name_code(uvar)  )
!        IF (slen .GT. 0 .AND. uvar_name_code(uvar)(1:3) .NE. 'EX#') THEN
* 6/16 -- treat 'EX#' just like other variables
	   iset = dset
	   CALL CD_GET_VAR_ID (iset, uvar_name_code(uvar), varid, status)
	   IF (status .NE. ferr_ok) THEN
	      iset = uvar_dset(uvar)
	      IF (iset .EQ. unspecified_int4) iset = pdset_uvars

	      CALL CD_GET_VAR_ID (iset, uvar_name_code(uvar), varid, status)
	   ENDIF

! ?? *sh* what is the state of things here if status .NE. ferr_ok ??

! Case: two user-defined variables w/ same name and not associated with datsets, 
! as in bn_letd.jnl
!     LET   a = GLOBAL_PREEMPT ! global pre-empt
!     LET/D a = GLOBAL_DEFAULT ! global default 
!  yes? show var
!   Created by DEFINE VARIABLE:
!   >>> Definitions that replace any file variable of same name:
!       A = GLOBAL_PREEMPT
!   >>> Definitions used if no file variable of this name exists:
!       A (/D default) = GLOBAL_DEFAULT
!
! This distinction is not in the linked-list structure. Only one is stored there. 
! but both are listed in the ferret-side data structures.  In a CANCEL VAR/ALL, 
! once one of these has been deleted the other no longer lives in the linked-list.
 
	   IF (status .EQ. ferr_ok) THEN
              slen = TM_LENSTR1( uvar_name_code(uvar)  )
              CALL TM_FTOC_STRNG( uvar_name_code(uvar)(1:slen), fhol, flen )
              CALL NCF_DELETE_VAR (iset, fhol )
           ENDIF
!	ENDIF

*	uvar_name_code( uvar ) = " "
        CALL string_array_modify(uvar_name_code_head, uvar, " ", 1 )

	uvar_text     ( uvar ) = " "

	IF (uvar_remote (uvar)) THEN
	   rvar_dset (uvar) = 0
	   rvar_varid (uvar) = 0
	   rvar_on_server (uvar) = .FALSE.
	   rvar_uvar (uvar) = 0
	   uvar_remote (uvar) = .FALSE.
	ENDIF
	
* Reinitialize any aux_uvar settings 
        DO i = 1, nferdims
 	   uvar_aux_stat(uvar,i) = paux_stat_init
 	ENDDO

      
	RETURN
	END
