	SUBROUTINE INIT_UVAR ( name, text, title, units, dset,
     .                         implct_defn, uvar,
     .			       at_type, at_id, at_start, at_end,
     .			       natom, txstart, txend, status )

* Compile a user variable structure into one or more user variables.
* Expression should arrive with the atoms identified in at_type but
* still in inifx order. Since multiple expressions may exist within
* the same text string (text) this routine has txstart and txend 
* to indicate the offsets to the particular expression in question

* The incoming expression may contain "grid-changing functions", for example
*	yes? LET vfft = TFFT(v) + A + B
* in which "TFFT" replaces the T axis in the course of computing an FFT.
* In such a case a "child variable" is created equivalent to 
*	yes? LET (013,001) = TFFT(v)
*	yes? LET vfft = (013,001) + A + B

* revision 0.0 - 2/19/87
* revision 0.1 - 9/22/87 - ferr_uvar_max --> *_prog_limit
* V200:  6/20/89 - re-worked user defined variable logic
*		 - syntax checking on variable names
*       11/27/89 - major re-write for RPN ordering
*        2/19/90 - corrected error messages
* Unix/RISC port - 7/10/91 - shorter "line too long" text string
*       10/22/92 - added units on uvars
* V402:  7/95 - Added "dset" argument for  LET/D=dset var = ... 
* V420:  12/95 - initialize uvar_mod_pseudo
* V450:  12/96 - much of the guts of the routine moved to INIT_UVAR_SUB
*		 Call RPN from within this routine instead of beforehand
*		 in order to simplify breaking out of grid-changing functions
*         5/97 - create separate child vars for expressions used as args to
*		 a grid-changing function
*         7/97 - bug fix for case of GCFCN(a,b)
* V491:   2/98 - fix to 5/97 changes: a constant is an expression, too

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
	INTEGER		uvar, natom, txstart, txend, status,
     .			at_type  (maxatoms), at_id  (maxatoms),
     .			at_start (maxatoms), at_end (maxatoms), dset

* internal variable declarations:
	INTEGER		tlen, iat, paren_level, atoms_lost, uvar_child,
     .			child_start, child_end, gc_arg_start
	CHARACTER	child_name*16

* initialize
	tlen  =  txend - txstart + 1

* find an unused uvar slot for the variable
	DO 100 uvar = 1, max_uvar
	   IF ( uvar_num_items( uvar ) .EQ. uvar_deleted ) GOTO 200
 100	CONTINUE
	GOTO 5100

* scan the atoms for grid-changing functions. These functions need to
* be compiled each as a separate variable definition.
* These variables will be invisible to the user of Ferret (except when
* using mode diagnostic). They have the special property that their result
* grid cannot necessarily be inferred from the component arguments.
 200	child_start = 0		! assume no grid changing fcn present
	DO 300 iat = 1, natom
	   IF ( at_type(iat).EQ. alg_grid_chg_fcn ) THEN
	      child_start = iat
	      paren_level = 0
	   ELSEIF ( child_start .GT. 0
     .	    .AND.   at_type(iat).EQ. alg_punctuation ) THEN
	      IF ( at_id(iat) .EQ. left_paren ) THEN
	         paren_level = paren_level + 1
	      ELSEIF ( at_id(iat) .EQ. right_paren ) THEN
	         paren_level = paren_level - 1
	         child_end = iat
* ... found the bounds of a grid changing function call ?
	         IF ( paren_level .EQ. 0 ) GOTO 500
	      ENDIF
	   ENDIF
 300	CONTINUE
	IF ( child_start .NE. 0 ) GOTO 5200	! syntax error

* no (more) child definitions. Go initialize the parent
	GOTO 1000

* initialize a "child variable"
*      -- a variable which consists only of a grid changing function
*	  and its parentheses-enclosed arguments.
* find an unused uvar slot for the child variable
 500	DO 600 uvar_child = 1, max_uvar
	   IF ( uvar_child .EQ. uvar ) GOTO 600
	   IF ( uvar_num_items( uvar_child ) .EQ. uvar_deleted ) GOTO 700
 600	CONTINUE
	GOTO 5900

* Grid-changing functions need to know the grids of each of their arguments
* This requires that each argument be "simple" -- expressions like a+b are not
* allowed. In this section, we scan the arguments of the grid-changing
* function and if we find a complex argument we create a separate child
* variable from it
* At this point we know there are no grid-changing functions used in the args.
* In this very simple parsing operation any arg with more than one "atom"
* will generate a separate child variable, including "gcfcn( (A) )"
 700	gc_arg_start = child_start + 2		! past first paren 
	paren_level = 1
	DO 750 iat = child_start+2, child_end
	   IF ( at_type(iat) .NE. alg_punctuation ) GOTO 750
	   IF ( at_id(iat) .EQ. left_paren ) THEN
	      paren_level = paren_level + 1
	   ELSEIF ( at_id(iat) .EQ. right_paren
     .	       .OR. at_id(iat) .EQ. comma ) THEN
	      IF ( at_id(iat) .EQ. right_paren )
     .				paren_level = paren_level - 1
	      IF ( paren_level .LE. 1 ) THEN
* ...(2/98) ".AND.": lone constant like a "complex" argument (an expression)
	         IF ( (iat .EQ. gc_arg_start+1)
     .          .AND. (at_type(gc_arg_start) .NE. alg_constant) ) THEN
* ... this argument is "simple" (a single atom) 
	            gc_arg_start = iat + 1	! look for next
	         ELSE
* ... make this (complex) argument into a seperate child variable
	            child_start = gc_arg_start
	            child_end = iat - 1
	            GOTO 800
	         ENDIF	
	      ENDIF
	   ELSE
	      CALL WARN('Unknown punctuation: INIT_UVAR')  ! never called (?!)
	   ENDIF
 750	CONTINUE

* There were no complex arguments in this grid-changing function
* If the envire definition is LET A = GCFCN(args) then no nned for child fcns
	IF ( child_start .EQ. 1
     . .AND. child_end   .EQ. natom ) GOTO 1000

* generate name of child variable
 800	CALL GCF_CHILD_VAR_NAME(uvar,at_start(child_start),child_name)


* initialize the child variable
	CALL INIT_UVAR_SUB (
     .			     child_name,
     .			     text, 'child of '//name, ' ', dset,
     .                       implct_defn, uvar_child, uvar,
     .			     at_type(child_start), at_id(child_start),
     .			     at_start(child_start), at_end(child_start),
     .			     child_end-child_start+1,
     .			     at_start(child_start), at_end(child_end),
     .			     status )
	IF ( status .NE. ferr_OK ) GOTO 5900

* consolidate the parent atom definitions so that the child appears as a
* single atom (a variable) in the definition
	atoms_lost = child_end - child_start
	at_end(child_start) = at_end(child_end)
	at_type(child_start) = alg_child_var	! alg_variable w/ diff. naming 
	at_id(child_start)   = unspecified_int4
	DO 900 iat = child_end+1, natom
	   child_start = child_start + 1
	   at_start(child_start) = at_start(iat)
	   at_end(child_start)   = at_end(iat)
	   at_type(child_start)  = at_type(iat)
	   at_id(child_start)    = at_id(iat)
 900	CONTINUE
	natom = natom - atoms_lost
	GOTO 200		! back for more children

*============================================================

* initialize the parent user variable, "uvar"
 1000	CALL INIT_UVAR_SUB ( name, text, title, units, dset,
     .                         implct_defn, uvar, 0,
     .			       at_type, at_id, at_start, at_end,
     .			       natom, txstart, txend, status )

	IF ( status .NE. ferr_OK ) GOTO 5900

* successful completion
	status = ferr_ok
	RETURN

* error exit
 5000	RETURN
 5100	CALL ERRMSG( ferr_prog_limit, status,
     .		'too many user-defined variables'//pCR//
     .		'cancel or redefine some variables', *5000 )
 5200	CALL ERRMSG( ferr_syntax, status,
     .		'unclosed parentheses: '//text(txstart:txend), *5000 )

* remove all child variables of the variable being initialized
 5900	CALL GCF_CHILD_VAR_NAME( uvar, 0, child_name )
	DO 5910 uvar_child = uvar, max_uvar
	   IF (uvar_name_code(uvar_child)(:6) .EQ. child_name(:6))
     .			uvar_num_items( uvar_child ) = uvar_deleted
 5910	CONTINUE
	GOTO 5000

	END
