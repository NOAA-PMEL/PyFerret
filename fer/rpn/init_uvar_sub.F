	SUBROUTINE INIT_UVAR_SUB ( name, text, title, units, dset,
     .                             implct_defn, uvar, parent_uvar,
     .			           at_type, at_id, at_start, at_end,
     .			           natom, txstart, txend, status )

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
#ifdef unix
	include 'ferret_cmn/ferret.parm'	
	include 'ferret_cmn/errmsg.parm'	
	include 'ferret_cmn/rpn.parm'
	include 'ferret_cmn/xvariables.cmn'
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

* internal variable declarations:
	INTEGER	STR_UPCASE, vax_code, i, natom_rpn,
     .                  rpn_order(maxatoms)
	CHARACTER	err_string*255

* initialize

* algebraic expression too long ?
	IF ( txend-txstart .GE. uvar_text_len ) CALL ERRMSG
     .		( ferr_line_too_long, status, text(txend-14:txend), *5000 )

* convert to Reverse Polish (stack evaluation) order
	CALL RPN( text(txstart:txend),
     .		  at_type, at_id, at_start, at_end,
     .            natom, natom_rpn, rpn_order,
     .            status, err_string)
	IF ( status .NE. ferr_ok ) GOTO 5440
* kob 12/97 - natom_rpn cannot be greater than max_uvar_items
	IF (natom_rpn .GE. max_uvar_items) GOTO 5500

* copy name code and text into slot and change to all upper case
	vax_code = STR_UPCASE( uvar_name_code( uvar ), name )
	vax_code = STR_UPCASE( uvar_text     ( uvar ), text(txstart:txend) )

* copy title, units, and dset into buffers
	uvar_title( uvar ) = title
        uvar_units( uvar ) = units
        uvar_dset ( uvar ) = dset

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
 310	uvar_num_items( uvar ) = natom_rpn

* by default assume the definition requires an external data set specification
	uvar_need_dset( uvar ) = .TRUE.

* by default assume that all limits are irrelevant
	DO 320 i = 1, 4
 320	uvar_given(i,uvar) = uvlim_irrelevant

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
 5440   CALL ERRMSG( ferr_syntax, status,
     .                  text(txstart:txend)//pCR//err_string, *5000 )
* 12/97 *kob*
 5500   CALL ERRMSG( ferr_expr_too_complex, status,
     .                  text(txstart:txend), *5000 )

	END
