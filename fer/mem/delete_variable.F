	SUBROUTINE DELETE_VARIABLE ( mr )

* delete a memory resident variable releaseing its space in the memory
* variable table and bulk memory

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 - 4/1/86
* revision 0.1 - 1/29/87 - change to leave illegal limits in mr if deleted
*			   in diagnostic mode and to accomodate mr_protected
*			   as declared INTEGER instead of LOGICAL
*		     and   to eliminate references to tvar_grid
* revision 0.2 - 9/30/87 - added mr_perm_prot
* revision 0.3 - 3/10/88 - added cat_named_var
* revision 1.0 - 7/21/88 - deletion involves removing the variable from its
*			   hash pile in the memory variable table and returning
*			   it to the free list
* revision 1.1 -10/12/88 - for named variables possibly call DELETE_NVAR
* V200:  5/25/89 - reordered lo/hi_ss arguments
* 		 - deleted variables flagged in mr_protected and mr_del_ links
*		 - nvar's variables eliminated
*		 - temporary variables represented via mr_protected
* V320: 12/29/94 - use SPLIT_LIST to redirect tty output for GUI

#ifdef unix
	include		'tmap_format/tmap_dims.parm'
	include		'ferret_cmn/ferret.parm'
	include 	'ferret_cmn/xvariables.cmn'
	include		'ferret_cmn/xprog_state.cmn'
#else
	INCLUDE		'TMAP_FORMAT:TMAP_DIMS.PARM'
	INCLUDE		'FERRET_CMN:FERRET.PARM'
	INCLUDE 	'FERRET_CMN:XVARIABLES.CMN'
	INCLUDE		'FERRET_CMN:XPROG_STATE.CMN'
#endif

* calling argument declarations:
	INTEGER		mr

* internal variable declarations:
	INTEGER		i, idim, prev, next
	CHARACTER*4	VAR_CODE

* diagnostic "delete" message
	IF (mode_diagnostic) THEN
	   CALL DIAGNOSTIC_OUT( '-DELETE', mr, point_to_mr )
	   DO 100 idim = 1, 4
	      mr_lo_ss( mr, idim ) = unspecified_int4
 100	   CONTINUE
	   IF     ( mr_protected( mr ) .EQ. mr_not_protected	) THEN
	      CONTINUE
	   ELSEIF ( mr_protected( mr ) .EQ. mr_temporary	) THEN
	      CONTINUE
	   ELSEIF ( mr_protected( mr ) .EQ. mr_table_entry_only	) THEN
	      CONTINUE
	   ELSEIF ( mr_protected( mr ) .EQ. mr_perm_protected	) THEN
	      CALL SPLIT_LIST(pttmode_ops, ttout_lun,'PERMANENT', 9)
	   ELSEIF ( mr_protected( mr ) .EQ. mr_in_progress	) THEN
	      CONTINUE
	   ELSE
	      CALL WARN( 'CORRUPTION! mr_protected ????' )
	   ENDIF
 3100	   FORMAT (1X,A,' VARIABLE DELETED !!')
	ENDIF

* remove variable slot from hash pile doubly linked list
	prev = mv_blink( mr )
	next = mv_flink( mr )
	mv_flink( prev ) = next
	mv_blink( next ) = prev

* return the slot to the memory variable free list
	mv_flink( mr ) = mv_flink( 0 )	
	mv_blink( mr ) = unspecified_int4		! temp diag
	mv_flink( 0 ) = mr

* remove the variable from the deletion chain if it's there
	IF ( mr_protected(mr) .EQ. mr_not_protected ) THEN
	   prev = mr_del_blink(mr)
	   next = mr_del_flink(mr)
	   mr_del_flink(prev) = next
	   mr_del_blink(next) = prev
	   mr_del_flink(mr)   = unspecified_int4	! temp diag
	   mr_del_blink(mr)   = unspecified_int4	! temp diag
	ENDIF

* release bulk memory that the variable used
	IF ( mr_protected( mr ) .NE. mr_table_entry_only )
     .		CALL FREE_MEMORY( mr_blk1(mr), mr_nblks(mr) )

* set protection to indicate deleted
	mr_protected( mr ) = mr_deleted

	RETURN
	END

