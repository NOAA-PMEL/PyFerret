	LOGICAL FUNCTION OK_JUXTAPO (las_ptr,inf_ptr,atom_type,atom_id)

*	Determines whether juxtaposition of two atom types in the infix string
*	is allowed in the syntax of things.

*	J Davison TMAP/JISAO/PMEL/NOAA
*	10.22.87
* V200: 11/23/89 - replaced constants with parameter names and strings !!!
* 12/96 *kob* - Linux Port - had to move variable assignments off of
*               declaration line. used DATA rather than PARAMETER stmnt.
* V450: 12/96 - add checks for grid changing functions and child variables

#ifdef unix
	include		'ferret_cmn/ferret.parm'
	include		'ferret_cmn/rpn.parm'
	include		'ferret_cmn/xalgebra.cmn'
#else
	INCLUDE		'FERRET_CMN:FERRET.PARM'
	INCLUDE		'FERRET_CMN:RPN.PARM'
	INCLUDE		'FERRET_CMN:XALGEBRA.CMN'
#endif

	CHARACTER	punct*1
	integer		las_ptr,inf_ptr,atom_type(*),atom_id(*)
	integer		atom(2),ndx(2),i,val
	integer		kar_matrix(8,8)
	DATA	 	kar_matrix	/0,0,1,0,0,1,0,0,
     .					1,1,0,1,1,0,1,0,
     .					1,1,0,1,1,0,1,0,
     .					0,0,1,0,0,1,0,0,
     .					1,1,0,1,1,0,1,1,
     .					0,0,1,0,0,1,0,0,
     .					0,0,1,1,1,1,0,0,
     .					1,1,0,1,1,0,1,0/
********************************************************************************

	atom(1) = las_ptr
	atom(2) = inf_ptr

	do 100 i = 1,2

*	   LEFT_PAREN PLACED ON LEFT END OF INFIX STRING
	   if (atom(i) .eq. added_left_paren) then
	      ndx(i) = 5
	      goto 100
	   end if

*	   ALL OPERATORS EXCEPT NOT
	   if (atom_type(atom(i)) .eq. alg_operator) then
	      if (alg_op(atom_id(atom(i))) .ne. 'NOT' )then
		 ndx(i) = 1
	      else 
	         ndx(i) = 2
	      end if
	      goto 100
	   end if

*	   CONSTANT, VARIABLE, PSEUDO-VARIABLE OR OPERAND
	   if  (atom_type(atom(i)) .eq. alg_constant
     .	   .or.	atom_type(atom(i)) .eq. alg_variable
     .	   .or.	atom_type(atom(i)) .eq. alg_pseudo_var
     .	   .or.	atom_type(atom(i)) .eq. alg_child_var
     .	   .or. atom_type(atom(i)) .eq. alg_operand ) then
	      ndx(i) = 3
	      goto 100
	   end if

*	   COMMA,LEFT OR RIGHT PARENTHESIS
	   if (atom_type(atom(i)) .eq. alg_punctuation ) then
	      punct = alg_punct( atom_id(atom(i)) )
	      if ( punct .eq. ',' ) ndx(i) = 4
	      if ( punct .eq. '(' ) ndx(i) = 5
	      if ( punct .eq. ')' ) ndx(i) = 6
	      goto 100
	   end if

*	   WHERE,THEN OR ELSE
	   if (atom_type(atom(i)) .eq. alg_log_struct) then
	      ndx(i) = 7
	      goto 100
	   end if

*	   FUNCTION
	   if (atom_type(atom(i)) .eq. alg_function
     .	  .OR. atom_type(atom(i)) .eq. alg_grid_chg_fcn) then
	      ndx(i) = 8
	      goto 100
	   end if

*	   ALG_FINISHED (INTERPRETED AS A RIGHT PARENTHESIS)
	   if (atom_type(atom(i)) .eq. alg_finished) then
	      ndx(i) = 6
	      goto 100
	   end if

	   stop 'unknown atom type encountered in ok_juxtapo'
100	continue

	val = kar_matrix(ndx(1),ndx(2))	   

	if (val .eq. 0) then
	   ok_juxtapo = .false.
	else
	   ok_juxtapo = .true.
	end if

	return 
	end
