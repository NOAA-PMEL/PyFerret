	SUBROUTINE CREATE_PERMUTATION( memory,
     .				cx, msrc, perm, mdst, status )

* create a memory-resident variable from input data, msrc, using region limits
* from context cx permuted into the order specified by array perm
* To optimize speed pre-test to see if the input data is already suitably
* organized
* -- flag the result as temporary

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* V200:  9/1/89
* V312: 5/94 - array "memory" as a calling argument

#ifdef unix
	include 'ferret_cmn/ferret.parm'
	include 'ferret_cmn/errmsg.parm'
	include 'ferret_cmn/xvariables.cmn'
	include 'ferret_cmn/xcontext.cmn'
#else
	INCLUDE 'FERRET_CMN:FERRET.PARM'
	INCLUDE 'FERRET_CMN:ERRMSG.PARM'
	INCLUDE 'FERRET_CMN:XVARIABLES.CMN'
	INCLUDE 'FERRET_CMN:XCONTEXT.CMN'
#endif

* calling argument declarations:
	INTEGER	cx, msrc, perm(4), mdst, status
	REAL	memory( mem_blk_size, max_mem_blks )

* internal variable declarations:
	LOGICAL	table_entry_only
	INTEGER	CX_DIM_LEN, MR_DIM_LEN, nsrc, ndst, i, idim, dim, prm,
     .		src_len(4), dst_len(4), src_ax(4), dst_ax(4)

* initialize
	table_entry_only = .FALSE.

* test if any action is necessary
* ... it is unnecessary if all dimensions with length greater than 1 are
* the same length and in the same order in the source data as in the requested
* permutation
* (Note: msrc regions always span cx regions)
* Compute src_len and dst_len - the non-unity axis lengths in the source and
* destination data
	nsrc = 0
	DO 200 idim = 1, 4
	   dim = mr_hi_ss(msrc,idim) - mr_lo_ss(msrc,idim)
	   IF ( dim .GT. 0 ) THEN
	      nsrc = nsrc + 1
	      src_ax (nsrc) = idim
	      src_len(nsrc) = dim
	   ENDIF
 200	CONTINUE
	ndst = 0
	DO 210 idim = 1, 4
	   prm = perm(idim)
	   dim = cx_hi_ss(cx,prm) - cx_lo_ss(cx,prm)
	   IF ( dim .GT. 0 ) THEN
	      ndst = ndst + 1
	      dst_ax (ndst) = prm
	      dst_len(ndst) = dim
	   ENDIF
 210	CONTINUE

* must "extract" data if the source data is a larger object than the request
* must permute if the axis ordering is different
	IF ( nsrc .NE. ndst ) GOTO 500
	DO 250 idim = 1, nsrc
	   IF ( dst_len(idim) .NE. src_len(idim)
     .	  .OR.  dst_ax (idim) .NE. src_ax (idim) ) GOTO 500
 250	CONTINUE

* no actual permutation of values is necessary
* see if a dummy mr table entry is needed to represent the permutation
	DO 300 idim = 1, 4
	   IF ( perm(idim) .NE. idim ) THEN
	      table_entry_only = .TRUE.
	      GOTO 500
	   ENDIF
 300	CONTINUE

* dimensions havn't been permuted nor are context region sizes < stored data
	status = ferr_ok
	mdst = msrc
	RETURN

* need a dummy entry

* create a permuted context
 500	DO 510 idim = 1,4
	   prm = perm(idim)
	   cx_lo_ss ( cx_buff, idim )   = cx_lo_ss ( cx, prm )
	   cx_hi_ss ( cx_buff, idim )   = cx_hi_ss ( cx, prm )
	   cx_lo_ww ( idim, cx_buff )   = cx_lo_ww ( prm, cx )
	   cx_hi_ww ( idim, cx_buff )   = cx_hi_ww ( prm, cx )
	   cx_trans ( idim, cx_buff )   = cx_trans ( prm, cx )
	   cx_by_ss ( idim, cx_buff )   = cx_by_ss ( prm, cx )
	   cx_given ( idim, cx_buff )   = cx_given ( prm, cx )
	   cx_delta ( idim, cx_buff )   = cx_delta ( prm, cx )
	   cx_trans_arg(idim, cx_buff)  = cx_trans_arg(prm, cx)
	   cx_regrid_trans(idim,cx_buff)=cx_regrid_trans(prm,cx)
 510	CONTINUE

	cx_data_set( cx_buff ) = cx_data_set( cx )
	cx_dset_gvn( cx_buff ) = cx_dset_gvn( cx )
	cx_category( cx_buff ) = cx_category( cx )
	cx_variable( cx_buff ) = cx_variable( cx )
	cx_grid    ( cx_buff ) = cx_grid    ( cx )
	cx_ovsiz   ( cx_buff ) = cx_ovsiz   ( cx )
	cx_bad_data( cx_buff ) = cx_bad_data( cx )

	cx_unstand_grid( cx_buff ) = cx_unstand_grid( cx )

* create a memory space for the result
	CALL CREATE_TEMP_MEM_VAR( cx_buff, mdst, status )
	IF ( status .NE. ferr_ok ) RETURN

* permute or kludge around permuting the data
	IF ( table_entry_only ) THEN
* ... point to the original data using permuted limits
	   CALL FREE_MEMORY( mr_blk1(mdst), mr_nblks(mdst) )
	   mr_blk1     (mdst) = mr_blk1(msrc)
	   mr_nblks    (mdst) = mr_nblks(msrc)
	   mr_protected(mdst) = mr_table_entry_only
	ELSE
* ... permute the data
	   CALL PERMUTE( cx, memory(1, mr_blk1(msrc)), msrc,
     .			     memory(1, mr_blk1(mdst)), mdst, perm )
	ENDIF

	RETURN

* error exits
! 5000	RETURN
! 5100	CALL ERRMSG( ferr_invalid_command, status,
!     .				'axis order not possible', *5000 )
	END
