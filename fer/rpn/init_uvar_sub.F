	SUBROUTINE INIT_UVAR_SUB ( name, text, title, units, dset, bad,
     .                             implct_defn, uvar, parent_uvar,
     .			           at_type, at_id, at_start, at_end,
     .			           natom, txstart, txend, status )

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
#ifdef unix
	include 'ferret.parm'	
	include 'errmsg.parm'	
	include 'rpn.parm'
	include 'xvariables.cmn'
	include 'xrisc_buff.cmn'
#else
	INCLUDE 'FERRET_CMN:FERRET.PARM'	
	INCLUDE 'FERRET_CMN:ERRMSG.PARM'	
	INCLUDE 'FERRET_CMN:RPN.PARM'
	INCLUDE 'FERRET_CMN:XVARIABLES.CMN'
#endif

* calling argument declarations:
	LOGICAL		implct_defn
	CHARACTER*(*)	name, text, title, units
	INTEGER		dset, uvar, natom, txstart, txend, status,
     .			parent_uvar,
     .			at_type  (maxatoms), at_id  (maxatoms),
     .			at_start (maxatoms), at_end (maxatoms)
	REAL		bad

* internal variable declarations:
	LOGICAL inquote
	INTEGER	        i, natom_rpn,
     .                  rpn_order(maxatoms), i2, quote_type,
     .                  TM_LENSTR1
	CHARACTER	err_string*255, c*1

* internal parameter declarations
	INTEGER		p_single_quote, p_double_quote
	PARAMETER     ( p_single_quote = 1,
     .			p_double_quote = 2 )

* kob 9/30/98 This needed for 7/7/98 code addition
#ifdef FORTRAN_90
      INTEGER HEXDF
      DATA HEXDF/Z'DF'/
#endif


* initialize

* algebraic expression too long ?
	IF ( txend-txstart .GE. uvar_text_len ) CALL ERRMSG
     .		( ferr_line_too_long, status, text(txend-14:txend), *5000 )

* convert to Reverse Polish (stack evaluation) order
	CALL RPN( text,
     .		  at_type, at_id, at_start, at_end,
     .            natom, natom_rpn, rpn_order,
     .            status, err_string)
	IF ( status .NE. ferr_ok ) GOTO 5440
* kob 12/97 - natom_rpn cannot be greater than max_uvar_items
	IF (natom_rpn .GE. max_uvar_items) GOTO 5500

* copy name code and text into slot and change as needed to upper case
*	vax_code = STR_UPCASE( uvar_name_code( uvar ), name )
        CALL string_array_modify_upcase(uvar_name_code_head, uvar, 
     .                                  name, LEN(name))

* (7/98) - don't upcase portions of the string inside single quotes
* (1/99) - same deal for double quotes
!	vax_code = STR_UPCASE( uvar_text     ( uvar ), text(txstart:txend) )
	uvar_text(uvar) = ' '
	i2 = 0
	inquote = .FALSE.
	DO 100 i = txstart, txend
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
# else
* VMS needs IAND
* bug fix - forgot (uvar) in var reference
	      uvar_text(uvar)(i2:12) = CHAR( IAND('DF'X, ICHAR(c)) )
# endif
#endif
	   ENDIF
 100	CONTINUE
* (7/98) end of quote-sensitive case changing loop

* copy title, units, dset, and missing flag into buffers
	uvar_title( uvar ) = title
        uvar_units( uvar ) = units
        uvar_dset ( uvar ) = dset
        uvar_bad_data ( uvar ) = bad

* save the ID of the parent variable (or zero if this is a parent, itself)
	uvar_parent( uvar ) = parent_uvar

* place the list of atoms into the uvar in RPN order.
	DO 300 i = 1, natom_rpn
	   uvar_item_start(i,uvar) = at_start(rpn_order(i)) - txstart + 1
	   uvar_item_end  (i,uvar) = at_end  (rpn_order(i)) - txstart + 1
	   uvar_item_type (i,uvar) = at_type (rpn_order(i))
 300	CONTINUE

!* add a terminator to the end of the list of RPN-ordered atoms
!	natom_rpn = natom_rpn + 1
!	uvar_item_type (natom_rpn,uvar) = alg_finished

* store number of items in definition
*	uvar_num_items( uvar ) = natom_rpn
        CALL deleted_list_modify(uvar_num_items_head, uvar,
     .                           natom_rpn )

!* 7/98 - remove surrounding single quotes from variable names
! no - keep 'em in to signal special name -- use 'em in MATCH_NAME
!	DO 310 i = 1, natom_rpn
!	   IF ( uvar_item_type (i,uvar) .EQ. alg_variable ) THEN
!	      is = uvar_item_start(i,uvar)
!	      ie = uvar_item_end(i,uvar)
!	      IF ( uvar_text(uvar)(is:is) .EQ.'\''
!	1	   .AND. uvar_text(uvar)(ie:ie).EQ.'\'') THEN
!		 uvar_item_start(i,uvar) = is + 1
!		 uvar_item_end  (i,uvar) = ie - 1
!	      ENDIF
!	   ENDIF
! 310	CONTINUE

* by default assume the definition requires an external data set specification
	uvar_need_dset( uvar ) = .TRUE.

* by default assume that all limits are irrelevant
	DO 320 i = 1, 4
	   uvar_lo_ss ( uvar, i )  = unspecified_int4
	   uvar_hi_ss ( uvar, i )  = unspecified_int4
	   uvar_lo_ww ( i, uvar )  = unspecified_val8
	   uvar_hi_ww ( i, uvar )  = unspecified_val8
	   uvar_by_ss ( i, uvar )  = .FALSE.             !for bookkeeping
  	   uvar_given ( i, uvar )  = uvlim_irrelevant
 320	CONTINUE

* by default assume that there are no pseudo-variables involved
	DO 330 i = 1, 4
 330	uvar_mod_pseudo(i,uvar) = .FALSE.

* set grid to unknown (to be determined at "run time") for all data sets
* and clear all references in memory to any associated dynamic grids
	CALL DEALLO_UVAR_GRIDS( uvar )	

* successful completion
	status = ferr_ok
	RETURN

* error exit
 5000	RETURN
 5440	risc_buff = text(txstart:txend)
	CALL ERRMSG( ferr_syntax, status,
     .                  risc_buff(:TM_LENSTR1(risc_buff))
     .                  //pCR//err_string, *5000 )
* 12/97 *kob*
 5500   CALL ERRMSG( ferr_expr_too_complex, status,
     .                  text(txstart:txend), *5000 )

	END
