	SUBROUTINE INIT_UVAR ( name, text, title, units, dset, bad,
     .                         implct_defn, uvar,
     .			       at_type, at_id, at_start, at_end,
     .			       natom, txstart, txend, varname, 
     .			       its_remote, status )

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

* Note that the addition of grid-changing variables and child variables
* was introduced as a series of "hacks" that have made the code needlessly complex.
* Could be cleaned up ... (*sh*)


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
* V500:  2/99 *sh* - bug fix for GC child var parsing
*		GCFCN((a)) generated an error -- paren problem
* V510: 10/99 *sh* - bug fix: if, say, SST[Y=5s:5N:1] is an argument to a GC
*		function, this routine must create a child variable so that
*		the implicit grid is "owned" by someone.
*	11/99 *sh* - bug fix: if child var occurred in the second (or higher)
*			expression in a line the child var name was wrong
*	12/99 *sh* - reverted to this code following temporary commenting out
*		of 10/99 fix by Ansley to keep EFs running. Changes to
*		EF_GET_ARG_SUBSCRIPTS now permit 10/99 change to work
*	1/00 *sh* - bug fix: parent_uvar was always set to ultimate ancestor
*		 instead of actual parent. Created gcf stack
* V510: 5/00 *sh* - added "bad" (missing value flag) to arguments
* 1/03 *kob* g77 port - requires use of risc_buff for concats
* V62   2/09 *acm* Pass the original upper/lowercase spelling of user-defined variable
*                  names to init_uvar_sub; name to be saved in the attribute structure, for 
*                  use when writing out user-defined variables when MODE UPCASE_OUTPUT 
*                  is cancelled.
*       *acm* 3/12 6D Ferret (common uses nferdims in tmap_dims.parm)
* V694  *acm* 1214 Fixing ticket 2220: a variablename.attname argument to a function
*                  needs to be added as a child variable
* V699 *sh* 6/16 bail out code at 5900 needs to clean up child LIST structures
* V720: 4/17 *acm* Add direction-changing functions as a subset of grid-changing ones
*                  A new grid is defined for these at grid initialization time; it
*                  will be changed later.
* 11/2022 *acm* Code cleanup: Modernize DO-loop syntax. 

        IMPLICIT NONE
	include 'tmap_dims.parm'
	include 'ferret.parm'	
	include 'errmsg.parm'	
	include 'rpn.parm'
	include 'xvariables.cmn'
	include 'xrisc_buff.cmn'

* calling argument declarations:
	LOGICAL		implct_defn, its_remote
	CHARACTER*(*)	name, text, title, units, varname
	INTEGER		uvar, natom, txstart, txend, status,
     .			at_type  (maxatoms), at_id  (maxatoms),
     .			at_start (maxatoms), at_end (maxatoms), dset
	REAL		bad

* internal parameter declarations
	LOGICAL		itsa_gc
	INTEGER		max_gcf_stack
	PARAMETER     ( max_gcf_stack = 15 )
* internal variable declarations:
	LOGICAL		has_children, child, child_remote
	INTEGER		tlen, iat, paren_level, atoms_lost, uvar_child, gcfstk, 
     .			child_start, child_end, gc_arg_start, istart, iend,
     .			parent,
     .			gcf_start(0:max_gcf_stack), gcf_uvar(0:max_gcf_stack)
	CHARACTER	child_name*16

* initialize
	tlen  =  txend - txstart + 1
	has_children = .FALSE.

* find an unused uvar slot for the variable and initialize grid chg fcn stack

	DO uvar = 1, max_uvar
	   IF ( uvar_num_items( uvar ) .EQ. uvar_deleted ) THEN
	      gcfstk = 0	
	      gcf_start(0) = 0
	      gcf_uvar (0) = uvar ! assignment never used but what the heck ...
              CALL deleted_list_modify(uvar_num_items_head, uvar,
     .                uvar_on_hold)
	      GOTO 200
	   ENDIF
	ENDDO

	GOTO 5100

* scan the atoms for grid-changing functions. These functions need to
* be compiled each as a separate variable definition.
* These variables will be invisible to the user of Ferret (except when
* using mode diagnostic). They have the special property that their result
* grid cannot necessarily be inferred from the component arguments.
 200	paren_level = 0

	DO iat = gcf_start(gcfstk)+1, natom

	   IF ( at_type(iat).EQ.alg_grid_chg_fcn .OR.  at_type(iat).EQ.alg_dir_chg_fcn ) THEN
* ... find the variable number for this "child variable"
*     Note that for the simple non-nested case of LET v = GCF(arg, arg, ...)
*     no child variables are needed - so at the end of this routine we clean
*     up left-over "on_hold" uvars

	      DO uvar_child = 1, max_uvar
	         IF ( uvar_child .EQ. uvar ) CYCLE
	         IF ( uvar_num_items( uvar_child ) .EQ. uvar_deleted ) GOTO 260
	      ENDDO

	      GOTO 5900

 260	      gcfstk = gcfstk + 1 ! new stack level for the child
*	      uvar_num_items( uvar_child ) = uvar_on_hold
              CALL deleted_list_modify(uvar_num_items_head, uvar_child,
     .                                 uvar_on_hold)
	      gcf_uvar (gcfstk) = uvar_child
	      gcf_start(gcfstk) = iat
	      paren_level = 0
	   ELSEIF ( gcfstk .GT. 0
     .	    .AND.   at_type(iat).EQ. alg_punctuation ) THEN
	      IF ( at_id(iat) .EQ. left_paren ) THEN
	         paren_level = paren_level + 1
	      ELSEIF ( at_id(iat) .EQ. right_paren ) THEN
	         paren_level = paren_level - 1
	         child_end = iat
* ... found the bounds of a grid changing function call ?
	         IF ( paren_level .EQ. 0 ) GOTO 700
	      ENDIF
	   ENDIF

	ENDDO

	IF ( gcfstk .NE. 0 ) GOTO 5200	! syntax error

* no (more) child definitions. Go initialize the parent
	GOTO 1000

* Grid-changing functions need to know the grids of each of their arguments
* This requires that each argument be "simple" -- expressions like a+b are not
* allowed. In this section, we scan the arguments of the grid-changing
* function and if we find a complex argument or a simple argument that  may
* have an implicit grid, we create a separate child
* variable from it
* At this point we know there are no grid-changing functions used in the args.
* In this very simple parsing operation any arg with more than one "atom"
* will generate a separate child variable, including "gcfcn( (A) )"

 700	child_start = gcf_start(gcfstk)
	gc_arg_start = child_start + 2		! past first paren 
	paren_level = 1

	DO iat = child_start+2, child_end

	   IF ( at_type(iat) .NE. alg_punctuation ) CYCLE
	   IF     ( at_id(iat) .EQ. left_paren  ) THEN
	      paren_level = paren_level + 1
	   ELSEIF ( at_id(iat) .EQ. right_paren ) THEN
	      paren_level = paren_level - 1
	   ENDIF
* ... check for the end of an argument -- comma or final paren
	   IF (  (at_id(iat).EQ.comma .AND. paren_level.EQ.1)
     .	    .OR. paren_level .EQ. 0   ) THEN
* ...(2/98) treat lone constant like a "complex" argument (an expression)
* ...(10/99)also treat sst[y=5s:5n:1] as a "complex" argument (has impl grid)
	      istart = at_start (gc_arg_start)
	      iend   = at_end   (gc_arg_start)

	      IF ( at_type(gc_arg_start) .EQ. alg_attrib_val) THEN
* ... make a varname.attname argument into a separate child variable
	         child_start = gc_arg_start
	         child_end = iat - 1
	         GOTO 800
              ENDIF

	      IF ( (at_type(gc_arg_start) .EQ. alg_child_var)
     .	     .OR.  ( (iat .EQ. gc_arg_start+1)
     .         .AND. (at_type(gc_arg_start) .NE. alg_constant)
     .	       .AND. (INDEX(text(istart:iend),'[').EQ.0)    )   ) THEN
* ... this argument is "simple" (a single atom) 
	         gc_arg_start = iat + 1	! look for next
	      ELSE
* ... make this (complex) argument into a separate child variable
	         child_start = gc_arg_start
	         child_end = iat - 1
	         GOTO 800
	      ENDIF	
	   ENDIF

	ENDDO

* There were no complex arguments in this grid-changing function
* If the entire definition is LET A = GCFCN(args) then no need for child fcns
	IF ( gcfstk .EQ. 1
     . .AND. .NOT.has_children
     . .AND. child_start .LE. 1
     . .AND. child_end   .EQ. natom ) GOTO 1000

* generate name of child variable -- points to *ancestor*, not parent
 800	CALL GCF_CHILD_VAR_NAME( uvar,
     .				 at_start(child_start)-(at_start(1)-1),
     .				 child_name )
	itsa_gc = at_type(child_start) .EQ. alg_grid_chg_fcn .OR. 
     .            at_type(child_start) .EQ. alg_dir_chg_fcn

	IF ( itsa_gc ) THEN
	   uvar_child = gcf_uvar(gcfstk)
	   parent = gcf_uvar(gcfstk-1)
	ELSE

	   DO uvar_child = 1, max_uvar
	      IF ( uvar_child .EQ. uvar ) CYCLE
	      IF ( uvar_num_items( uvar_child ) .EQ. uvar_deleted ) THEN
	         parent = gcf_uvar(gcfstk)
	         GOTO 860
	      ENDIF
	   ENDDO

	   GOTO 5900
	ENDIF

* initialize the child variable  its_remote is set .FALSE. for child variables. (?)
 860	child = .TRUE.
        child_remote = .FALSE.
        CALL INIT_UVAR_SUB (
     .			     child_name,
     .			     text, ' ', ' ', dset, bad,
     .                       child_remote, implct_defn, uvar_child,parent,
     .			     at_type(child_start), at_id(child_start),
     .			     at_start(child_start), at_end(child_start),
     .			     child_end-child_start+1,
     .			     at_start(child_start), at_end(child_end),
     .			     child, child_name, status )
	IF ( status .NE. ferr_OK ) GOTO 5900

* ... if the child we just created was a GC then pop the gc stack
	IF ( itsa_gc ) gcfstk = gcfstk - 1

* consolidate the parent atom definitions so that the child appears as a
* single atom (a variable) in the definition
	atoms_lost = child_end - child_start
	at_end(child_start) = at_end(child_end)
	at_type(child_start) = alg_child_var	! alg_variable w/ diff. naming 
	at_id(child_start)   = unspecified_int4

	DO iat = child_end+1, natom
	   child_start = child_start + 1
	   at_start(child_start) = at_start(iat)
	   at_end(child_start)   = at_end(iat)
	   at_type(child_start)  = at_type(iat)
	   at_id(child_start)    = at_id(iat)
	ENDDO

	natom = natom - atoms_lost
	has_children = .TRUE.
	GOTO 200		! back for more children

*============================================================

* initialize the parent user variable, "uvar"
 1000	child = .FALSE.
        CALL INIT_UVAR_SUB ( name, text, title, units, dset, bad,
     .                         its_remote, implct_defn, uvar, 0,
     .			       at_type, at_id, at_start, at_end,
     .			       natom, txstart, txend, child, varname,
     .			       status )

	IF ( status .NE. ferr_OK ) GOTO 5900

* clean up left-over on-hold variables

	DO uvar_child = 1, max_uvar
	   IF (uvar_num_items(uvar_child) .EQ. uvar_on_hold)
     .	      CALL deleted_list_modify(uvar_num_items_head, uvar_child,
     .                                  uvar_deleted )
	ENDDO

* successful completion
	status = ferr_ok
	RETURN

* error exit
* ... remove any holds left on variable slots
 5000	DO  uvar = 1, max_uvar
	   IF  ( uvar_num_items(uvar) .EQ. uvar_on_hold )
     .	   CALL deleted_list_modify(uvar_num_items_head, uvar,
     .                              uvar_deleted )
     	ENDDO
	RETURN

 5100	CALL ERRMSG( ferr_prog_limit, status,
     .		'too many user-defined variables'//pCR//
     .		'cancel or redefine some variables', *5000 )
 5200	risc_buff = text(txstart:txend)
	CALL ERRMSG( ferr_syntax, status,
     .		'unclosed parentheses: '//risc_buff, *5000 )

* remove all child variables of the variable being initialized
 5900	CALL GCF_CHILD_VAR_NAME( uvar, 0, child_name )
	DO uvar_child = uvar, max_uvar
	   IF ( uvar_name_code(uvar_child)(7:10)
     .     .EQ. child_name(7:10) ) THEN
	      CALL DELETE_USER_VAR(uvar_child, dset)
	   ENDIF
	ENDDO
	GOTO 5000

	END
