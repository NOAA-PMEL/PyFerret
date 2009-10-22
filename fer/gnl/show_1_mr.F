	SUBROUTINE SHOW_1_MR( mr )

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
* display a description of variable MR

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* V200:  6/12/89 - extracted from SHOW_MEM_VARS
*	 2/27/90 - changed VAR_CODE to *8
*	 3/30/90 - allow 6 digit subscripts
* Risc/UNIX port 4/91 *sh* cannot use functions in WRITE list
* V230:  11/4/92 - expanded variable name length
* V420: 11/95 - indicate un-cached variables
* V500 *kob* 3/99- up VAR_CODE and vname decl's to 64 chars
* V552 *acm* 4/03- up VAR_CODE and vname decl's from 64 to 128 chars
* V63  *acm* 10/09 Changes for gfortran build

	include	'tmap_dims.parm'
	include	'xtm_grid.cmn_text'
	external xgt_grid_data
	include	'xdset_info.cmn_text'
	external xdset_info_data
	include	'ferret.parm'
	include 'xprog_state.cmn'
	include 'xvariables.cmn'
	include 'xtext_info.cmn'
	include 'xalgebra.cmn'
	include 'xrisc.cmn'          ! 12/94 SPLIT_LIST buffer

* calling argument declarations:
	INTEGER	mr

* V500 *kob* 3/99- up VAR_CODE and vname decl's to 64 chars
* V552 *acm* 4/03- up VAR_CODE and vname decl's from 64 to 128 chars
* local variable declarations:
	LOGICAL	ITS_UN_CACHED
	INTEGER TM_LENSTR1, MR_DIM_LEN,
     .		idim, ndim, mrlen, bk1len, nbklen, dset
	CHARACTER VAR_CODE*128, ALG_TRANS_CODE*3, ARG_STRING*12, TM_FMT*8,
     .		  protection*4, shape*4, dname*14, gname*12, vname*128,
     .		  ww_lo(4)*20, ww_hi(4)*20, ss_lo(4)*6, ss_hi(4)*6,
     .		  mrbuf*4, bk1buf*4, 
     .		  nbkbuf*4, cache_stat*8

* get shape
	shape = ' '
	ndim = 0
	DO 170 idim = 1, 4
	   IF ( MR_DIM_LEN( idim, mr ) .GT. 1 ) THEN
	      ndim = ndim + 1
	      shape(ndim:ndim) = ww_dim_name(idim)
	   ENDIF
 170	CONTINUE

* get formatted world positions
	DO 180 idim = 1, 4
	   IF ( mr_lo_ww(idim,mr) .EQ. unspecified_val8 ) THEN
	      ww_lo(idim) = ' ...'
	      ww_hi(idim) = ' ...'
	   ELSE
	      CALL TRANSLATE_TO_WORLD
     .	           ( mr_lo_ww(idim,mr),idim,mr_grid(mr),
     .		     ax_dec_pt(idim), ww_lo(idim) )
	      CALL TRANSLATE_TO_WORLD
     .	           ( mr_hi_ww(idim,mr),idim,mr_grid(mr),
     .		     ax_dec_pt(idim), ww_hi(idim) )
	   ENDIF
	   IF ( mr_lo_ss(mr,idim) .EQ. unspecified_int4 ) THEN
	      ss_lo(idim) = ' ...'
	      ss_hi(idim) = ' ...'
	   ELSEIF ( mr_hi_ss(mr,idim) .GT. 9999 ) THEN
	      WRITE ( ss_lo(idim), '(I6)' ) mr_lo_ss(mr,idim)
	      WRITE ( ss_hi(idim), '(I6)' ) mr_hi_ss(mr,idim)
	   ELSE
	      WRITE ( ss_lo(idim), '(I4)' ) mr_lo_ss(mr,idim)
	      WRITE ( ss_hi(idim), '(I4)' ) mr_hi_ss(mr,idim)
	   ENDIF
 180	CONTINUE

* write a 4 line summary of this memory variable
	IF ( mr_protected(mr) .EQ. mr_not_protected ) THEN
	   protection = ' '
	ELSEIF ( mr_protected(mr) .EQ. mr_perm_protected ) THEN
	   protection = 'PERM'
	ELSE
	   protection = '*ERR'
	ENDIF
	IF ( ITS_UN_CACHED(mr) ) THEN
	   cache_stat = 'UNCACHED'
	ELSE
	   cache_stat = ' '
	ENDIF
	vname = VAR_CODE( mr_category(mr),mr_variable(mr) )
	dset = mr_data_set(mr)
	IF ( dset .EQ. pdset_irrelevant ) THEN
	   dname = ' '
	ELSE
	   dname = 'D='//ds_name(dset)
	ENDIF
	IF ( mr_unstand_grid(mr) ) THEN
	   gname = grid_name(mr_grid(mr))
	   gname = ',G='//gname(:TM_LENSTR1(gname))//']'
	ELSE
	   gname = ']'
	ENDIF
	mrbuf  = TM_FMT( FLOAT(mr), 0, 4, mrlen )
	nbkbuf = TM_FMT( FLOAT(mr_nblks(mr)),0,4,nbklen )
	bk1buf = TM_FMT( FLOAT(mr_blk1(mr)),0,4,bk1len )
	risc_buff = ' '
	WRITE ( risc_buff, 3010 )	vname(:TM_LENSTR1(vname)),
     .					dname(:TM_LENSTR1(dname)),
     .					gname(:TM_LENSTR1(gname)),
     .					shape,
     .					mrbuf(:mrlen),
     .					bk1buf(:bk1len),
     .					nbkbuf(:nbklen),
     .					protection,
     .					cache_stat

	CALL SPLIT_LIST(pttmode_explct, ttout_lun, risc_buff, 0)
	risc_buff = ' '
	WRITE ( risc_buff, 3020 )
     .	  ( ss_lo(idim), ww_lo(idim), idim=1,4 )
	CALL SPLIT_LIST(pttmode_explct, ttout_lun, risc_buff, 0)
	risc_buff = ' '
	WRITE ( risc_buff, 3020 )
     .	  ( ss_hi(idim), ww_hi(idim), idim=1,4 )
	CALL SPLIT_LIST(pttmode_explct, ttout_lun, risc_buff, 0)
	risc_buff = ' '

#ifdef unix
* RISC compiler bug: cannot use character functions in WRITE statement
* ... use ww_lo and ss_lo as temporary buffers
        DO 200 idim = 1, 4
           ss_lo(idim) = ALG_TRANS_CODE( mr_trans(idim,mr) )
           ww_lo(idim) = ARG_STRING( mr_trans_arg(idim,mr) )
 200    CONTINUE
	WRITE ( risc_buff, 3030 ) (ss_lo(idim), ww_lo(idim), idim = 1,4)
#else
	WRITE ( risc_buff, 3030 ) 
     .		( ALG_TRANS_CODE( mr_trans(idim,mr) ),
     .		  ARG_STRING( mr_trans_arg(idim,mr) ), idim = 1,4 )
#endif
	CALL SPLIT_LIST(pttmode_explct, ttout_lun, risc_buff, 0)

 3010	FORMAT(T2,A,'[',2A,T36,A4,'  mr:',A,'  blk1:',A,
     .'  nblk:',A,1X,A4,1x,A8)
 3020	FORMAT(T2,3(A6,'/',A12),A6,'/',A12)
 3030	FORMAT(T2,4(4X,A3,A12))
	END
