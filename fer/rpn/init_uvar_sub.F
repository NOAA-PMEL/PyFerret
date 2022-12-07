	SUBROUTINE INIT_UVAR_SUB ( name, text, title, units, dset, bad,
     .                             its_remote, implct_defn, uvar, parent_uvar,
     .			           at_type, at_id, at_start, at_end,
     .			           natom, txstart, txend, child, 
     .			           varname, status )

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
* Fill in a user variable structure with a new expression
* Expression should arrive with the atoms identified in at_type but
* still in infix order. Since multiple expressions may exist within
* the same text string (text) this routine has txstart and txend
* to indicate the offsets for the particular expression in question

* V450:  12/96 - mostly borrowed from the previous INIT_UVAR
* *kob*  6/97 - rpn_order was erroneously declared as array of size 
*               max_exp, changed it to maxatoms
* V450:  7/97 - save the value of parent_uvar in uvar_parent
* V491:  12/97 - add check to ensure natom_rpn returned from RPN isn't
*                greater than max_uvar_items  *kob*
* V495: 7/7/98 - *sh* variable names in single quotes - don't upcase them
*     : 9/30/98 - NAG F90 (linux) received error on above change.  
*                 need to put single  tick between double quotes  *kob*
*               - also needed a definition for HEXDF variable used
* V500:  1/99 *sh* - don't upcase double quoted strings, either
*        3/99 *sh* - bug fix: pass the full error text to RPN, since
*			at_start, at_end are relative to full text buffer
*	12/99 *sh* - initialize uvar_lo/hi*
* V510: 5/00 *sh* - added "bad" (missing value flag) to arguments
* 1/03 *kob* g77 port - requires use of risc_buff for concats
*            bug fix - reference to uvar_text was missing uvar reference 
* V600  4/06 *acm* - additional declarations found by Solaris compiler     
* V601 11/06 *acm* - add user variables always to dataset pdset_uvars. Previously
*                    variables defined with LET/D= were added to that dataset.
* V602  2/07 *acm* Fix bug 1492, changing attributes of coordinate variables 
* V62   2/09 *acm* Pass the original upper/lowercase spelling of user-defined variable
*                  names to cd_add_var; name to be saved in the attribute structure, for 
*                  use when writing out user-defined variables when MODE UPCASE_OUTPUT 
*                  is cancelled.
* V63  *acm* 10/09 Changes for gfortran build
* V671 7/11  *acm* Ticket 1868 IFV - masking expressions with ifValid, treating 
*		   zero as valid. 
*       *acm* 3/12 Add E and F dimensions (use nferdims in tmap_dims.parm)
* V693+ *sh* 12/14 add support for _DQ_ and _SQ_ as qutation delimiters
*      *acm*  2/15 Prevent run-time errors in string length for _DQ_ handling
* V697 12/15 *acm* Ticket 2337 Increase length of cmnd buffer to 20480. If long
*                  var definition, just truncate it in uvar_text(uvar)
* V697  2/16 *acm* DO NOT Increase length of cmnd buffer to 20480
*                  but keep the other changes at that time, handling long 
*                  defintions
* V698  2/16 *acm* For ticket 2352: variables defined with LET/D are
*                  added to that dataset in the attribute structure.
* V699  5/16 *sh* pass uvar to CD_ADD_VAR encoded as negative for a LET/D uvar
*                 include EX# and child vars in the LIST structures
* V710 *acm* 10/16 Ticket 2422: remove unneeded call to DEALLO_UVAR_GRIDS. 
*                  Also clean up long-unused lines that were commented out.
* 11/2022 *acm* Code cleanup: Modernize DO-loop syntax. 

        IMPLICIT NONE
	include 'tmap_dims.parm'
	include 'ferret.parm'	
	include 'errmsg.parm'	
	include 'rpn.parm'
	include 'xvariables.cmn'
	include 'xrisc_buff.cmn'

* calling argument declarations:
	LOGICAL		implct_defn, child, its_remote
	CHARACTER*(*)	name, text, title, units, varname
	INTEGER		dset, uvar, natom, txstart, txend, status,
     .			parent_uvar, 
     .			at_type  (maxatoms), at_id  (maxatoms),
     .			at_start (maxatoms), at_end (maxatoms)
	REAL		bad

* internal variable declarations:
	LOGICAL		inquote, toolong
	INTEGER	        TM_LENSTR1, TM_LENSTR, STR_SAME, i, idim,
     .                  natom_rpn, rpn_order(maxatoms), i2, quote_type,
     .                  dset_to_add, slen, iat, varid, type, coordvar
	CHARACTER	buff*255, c*1
! 6/16 -- declared but never used: CHARACTER attname*128

* internal parameter declarations
	INTEGER		p_single_quote, p_double_quote, p__DQ_, p__SQ_
	PARAMETER     ( p_single_quote = 1,
     .			p_double_quote = 2,
     .			p__DQ_         = 3,
     .			p__SQ_         = 4 )

* kob 9/30/98 This needed for 7/7/98 code addition
#ifdef FORTRAN_90
      INTEGER HEXDF
      DATA HEXDF/Z'DF'/
#endif


* initialize

* algebraic expression too long ?
	toolong =  ( txend-txstart .GE. uvar_text_len ) 
	IF (toolong) txend = uvar_text_len - 3

* convert to Reverse Polish (stack evaluation) order
	CALL RPN( text,
     .		  at_type, at_id, at_start, at_end,
     .            natom, natom_rpn, rpn_order,
     .            status, buff)
	IF ( status .NE. ferr_ok ) GOTO 5440
* kob 12/97 - natom_rpn cannot be greater than max_uvar_items
	IF (natom_rpn .GE. max_uvar_items) GOTO 5500

* copy name code and text into slot and change as needed to upper case
*	vax_code = STR_UPCASE( uvar_name_code( uvar ), name )
cc        CALL string_array_modify_upcase(uvar_name_code_head, uvar, 
cc     .                                  name, LEN(name))
        CALL string_array_modify_upcase(uvar_name_code_head, uvar, 
     .                                  name, LEN(name))

* (7/98) - don't upcase portions of the string inside single quotes
* (1/99) - same deal for double quotes
!	vax_code = STR_UPCASE( uvar_text     ( uvar ), text(txstart:txend) )
	uvar_text(uvar) = ' '
	i2 = 0
	inquote = .FALSE.
	i = txstart - 1

	DO WHILE (i .LT. txend)
	   i = i + 1
	   i2 = i2 + 1
	   c = text(i:i)
	   IF ( c .EQ. "'" ) THEN
* ... single quote -- start or end of string?
	      IF (inquote) THEN
	        IF (quote_type.EQ.p_single_quote) inquote = .FALSE.
	      ELSE
		inquote = .TRUE.
	        quote_type = p_single_quote
	      ENDIF
	   ELSEIF ( c .EQ. '"' ) THEN
* ... double quote -- start or end of string?
	      IF (inquote) THEN
	        IF (quote_type.EQ.p_double_quote) inquote = .FALSE.
	      ELSE
		inquote = .TRUE.
	        quote_type = p_double_quote
	      ENDIF
	   ELSEIF ( txend-txstart .GT. 7 .AND. i.LE.txend-3 ) THEN 
	      iat = 1
	      IF (text(i:i+3) .EQ. p_DQ_quote) THEN 
* ... found "_DQ_" -- start or end of string?
	        IF (inquote) THEN
	          IF (quote_type.EQ.p__DQ_) inquote = .FALSE.
	        ELSE
		  inquote = .TRUE.
	          quote_type = p__DQ_
	        ENDIF
	        uvar_text(uvar)(i2:i2+2) = text(i:i+2)
	        i  = i  + 3
	        i2 = i2 + 3
	      ELSEIF (text(i:i+3) .EQ. p_SQ_quote) THEN 
* ... found "_SQ_" -- start or end of string?
	        IF (inquote) THEN
	          IF (quote_type.EQ.p__SQ_) inquote = .FALSE.
	        ELSE
		  inquote = .TRUE.
	          quote_type = p__SQ_
	        ENDIF
	        uvar_text(uvar)(i2:i2+2) = text(i:i+2)
	        i  = i  + 3
	        i2 = i2 + 3
	      ENDIF
	   ENDIF
	   IF ( inquote .OR. (c .LT. 'a' .OR. c .GT. 'z' ) ) THEN
	      uvar_text(uvar)(i2:i2) = c   ! preserve case
	   ELSE
* ... convert lower case character to upper case
#if defined unix  && ! defined NEED_IAND
	      uvar_text(uvar)(i2:i2) = CHAR( AND('DF'X, ICHAR(c)) )
#else
# ifdef FORTRAN_90
	      uvar_text(uvar)(i2:i2) = CHAR( IAND(HEXDF, ICHAR(c)) )
# elif defined gfortran
	      uvar_text(uvar)(i2:i2) = CHAR( AND('DF'X, ICHAR(c)) )
#else
* VMS needs IAND
* bug fix - forgot (uvar) in var reference
	      uvar_text(uvar)(i2:i2) = CHAR( IAND('DF'X, ICHAR(c)) )
# endif
#endif
	   ENDIF

	ENDDO

	IF (toolong) uvar_text(uvar)(uvar_text_len-2:uvar_text_len) = '...'

* (7/98) end of quote-sensitive case changing loop

c* copy title, units, dset, and missing flag into buffers
	uvar_title( uvar ) = title
        uvar_units( uvar ) = units
        uvar_dset ( uvar ) = dset
        uvar_bad_data ( uvar ) = bad
        uvar_remote ( uvar ) = its_remote

* Add variable to the user-variable dataset in the linked list structure.
* The variable type is not known at this time - will set it later when the
* variable gets loaded. 

* 6/16 *sh* include EX# expressions and child variables in the LIST structures

       type = 0
       coordvar = 0
! 6/16       IF (name(1:3) .NE. 'EX#' .AND. .NOT.child) THEN
       varid = -1 * uvar   ! signals uvar (vs fvar) inside NCF_UTIL.c
       IF (dset .GT. pdset_irrelevant) THEN
	  dset_to_add = dset
       ELSE
	  dset_to_add = pdset_uvars  ! special dset for global uvars
       ENDIF

* if no title was provided, then use the LET definition as the title
       IF (title .EQ. " ") THEN
          buff =  uvar_text(uvar)
       ELSE
          buff = title
       ENDIF

       CALL CD_ADD_VAR( dset_to_add, varname, varid, buff, units, 
     .         type, coordvar, bad, status)

       IF ( status .NE. ferr_ok ) GOTO 5600

* !!! All the rest of the uvar_*(uvar) properties could become attributes of this uvar !!!

* save the ID of the parent variable (or zero if this is a parent, itself)
	uvar_parent( uvar ) = parent_uvar

* place the list of atoms into the uvar in RPN order.
	DO i = 1, natom_rpn
	   uvar_item_start(i,uvar) = at_start(rpn_order(i)) - txstart + 1
	   uvar_item_end  (i,uvar) = at_end  (rpn_order(i)) - txstart + 1
	   uvar_item_type (i,uvar) = at_type (rpn_order(i))
	ENDDO

        uvar_ifv(uvar) = .FALSE.
        IF ( STR_SAME( uvar_text(uvar)(1:3), 'IFV' ) .EQ. 0 )  
     .      uvar_ifv(uvar) = .TRUE.

* store number of items in definition
*	uvar_num_items( uvar ) = natom_rpn
        CALL deleted_list_modify(uvar_num_items_head, uvar,
     .                           natom_rpn )

* by default assume the definition requires an external data set specification
	uvar_need_dset( uvar ) = .TRUE.

* by default assume that all limits are irrelevant
	DO idim = 1, nferdims
	   uvar_lo_ss ( uvar, idim )  = unspecified_int4
	   uvar_hi_ss ( uvar, idim )  = unspecified_int4
	   uvar_lo_ww ( idim, uvar )  = unspecified_val8
	   uvar_hi_ww ( idim, uvar )  = unspecified_val8
	   uvar_by_ss ( idim, uvar )  = .FALSE.             !for bookkeeping
  	   uvar_given ( idim, uvar )  = uvlim_irrelevant
	ENDDO

* by default assume that there are no pseudo-variables involved
	DO idim = 1, nferdims
	   uvar_mod_pseudo(idim,uvar) = .FALSE.
 	ENDDO

* successful completion
	status = ferr_ok
	RETURN

* error exit
 5000	RETURN
 5440	risc_buff = text(txstart:txend)
	CALL ERRMSG( ferr_syntax, status,
     .                  risc_buff(:TM_LENSTR1(risc_buff))
     .                  //pCR//buff, *5000 )
* 12/97 *kob*
 5500   CALL ERRMSG( ferr_expr_too_complex, status,
     .                  text(txstart:txend), *5000 )
 5600   slen = TM_LENSTR1(name)
        WRITE (risc_buff,*) 'error initializing ', name(:slen)
        slen = TM_LENSTR1(risc_buff)
        CALL ERRMSG( ferr_syntax, status,
     .               risc_buff(:slen), *5000 )

	END
