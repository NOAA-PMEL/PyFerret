	BLOCK DATA XALGEBRA_DATA

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
* data initialization of algebraic operators and functions used by program GFDL

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 - 7/9/86
* revision 1.0 - 12/9/86 - *sh*
*		added functions and pseudo-variables
* revision 1.1 - 6/22/87 - added SHF and AAV
* revision 2.0 - 9/22/87 - eliminated geog* --> into XCONTEXT
* revision 2.1 - 6/23/88 - added FAV - filling missing data with average
* revision 2.2 - 1/3/89  - added regridding transformations
* V200:  6/6/89-8/3/89 - added XY integrating, binomial smoothing, IGNORE0, EXP
*	11/22/89 - reworked expression syntax for in-fix notation
*	 1/12/90 - added info about the arguments of transformations
*	 3/19/90 - added ATAN,ATAN2,ASIN,ACOS
*	 3/29/90 - added RANDU,RANDN
* for Unix/RISC port 9/91 - added @SHN,@SPZ,@SWL
*                         - "integ'd" --> "integ" to avoid "'" in PPLUS
* V230: 1/3/92 - added @SUM, @WEQ, @RSUM, @VAR, @NBD, @NGD
*              - fleshed out trans_title for use with SHOW TRANSFORMATIONS
*      3/23/92 - added @FLN and @FNR
*       7/2/92 - changed @FLN and @FNR defaults to 1 (edge) from 3 (window)
*       8/5/92 - added new functions RHO and THETA
*      8/22/92 - added @ASN regridding
*     10/22/92 - RHO --> RHO_UN, THETA --> THETA_UN
* V300: 9/2/93 - reordered transforms for SHOW TRANS and QUERY TRANS
* V320:  11/94 - added DAYS1900 function
* V420:  10/95 - added regridding transforms: min,max,var,ngd,nbd,npt,sum 
*	 11/95 - added modulo regridding: "@MOD"
* V430:   7/96 - incorporate 4D transformations: AVE, VAR, DIN, NDG, NBD
* V450:   7/97 - added function descriptions
* V455    9/97 - added transforms @CDA,CDB,CIA,CIB
* V490   10/97 - added regrid transform @XACT (removed unimplemented g=@NRS)
*	       - changed title for g=@ASN from "1 to 1" to "index assn"
* V491    4/98 - *kob* modified documentation order for function days1900
*                      to correct year-month-day
* V500   10/98 - *jc* added @RDC
*         1/00 - *acm* Change alg_fcn_descr to "since 1900" (was 1990)
*         1/00 - *acm* Corrected the order of argument listing for THETA_FO to 
*                      match how the function is actually called.
* V530   1/01 - *sh* - added XBOXLO, XBOXHI and family ...
*	 3/01 - *sh* - adde g=ax@nrst
* V533   6/01 - *sh* - added transformation @EVNT
* 2/03 *kob* - g77 port - g77 will not allow intrinsic functions in PARAMETER
*                         statements.  use character constant instead
* V554 1/04  *acm* - additions for /RANGE repeat loop control; alg_cvar, countervar_name
* V603 5/07  *acm* - added MODNBD modulo regridding statistic
* V66+ 4/10  *acm* - change pressure to reference pressure for RHO_UN as 
*                    suggested by James Orr and Billy Kessler
* V6.7 1/11  *acm* - fix ticket 1788, initializing alg_trans_*
* V671  6/11 *acm*  New 2D linear xy regridding, fixing ticket 1862
* V671 7/11  *acm* Ticket 1868 IFV - masking expressions with ifValid, treating 
*		   zero as valid. 
*      2/12  *kms* Add M, N, E, F, EBOX, FBOX, EBOXLO, FBOXLO, EBOXHI, and FBOXHI
*                  pseudo-variables
*       *acm* 4/12 Add new MODE FOURDEE to allow E,F,M,N,etc as user-vars
* V690 12/13 *sh* added auxiliary regridding transforms
* v695+ 3/15 *sh* added pauxrgrd_fmrc for T-F orthoganalization
* V695 *acm* 10/15 Fix #2319: @STD as a multi-D transform
* V696 *acm* 11/15 Added @BIN and @NBIN regridding transforms
* V696 *acm8 11/15 Clarify descriptions for @BIN, @NBIN, @AVE, @NGD, @SUM regridding
* V720: 6/17 *acm* Ticket 2346: new @IIN regridding transform

        IMPLICIT NONE
	include	'ferret.parm'
	include	'xalgebra.cmn'

	CHARACTER*1	rubout
	PARAMETER     ( rubout = CHAR(127) )
	INTEGER	i

* single character operators:
	DATA alg_op(  1 ) / '+' /,
     .	     alg_op(  2 ) / '-' /,
     .	     alg_op(  3 ) / '*' /,
     .	     alg_op(  4 ) / '/' /,
     .	     alg_op(  5 ) / '^' /,
     .	     alg_op(  6 ) / 'AND' /,
     .	     alg_op(  7 ) / 'OR' /,
     .	     alg_op(  8 ) / 'GT' /,
     .	     alg_op(  9 ) / 'GE' /,
     .	     alg_op( 10 ) / 'LT' /,
     .	     alg_op( 11 ) / 'LE' /,
     .	     alg_op( 12 ) / 'EQ' /,
     .	     alg_op( 13 ) / 'NE' /,
     .	     alg_op( 14 ) / '%' /,	! undefined
     .	     alg_op( 15 ) / '%' /,	! undefined
     .	     alg_op( 16 ) / '%' /,	! undefined
     .	     alg_op( 17 ) / '%' /,	! undefined
     .	     alg_op( 18 ) / '%' /	! undefined

* functions:
	DATA alg_fcn		(  1 ) / 'EXP'      /,
     .	     alg_num_fcn_args	(  1 ) / 1 	    /,
     .	     alg_fcn_descr	(  1 ) / 'exponential e(X)' 	    /,
     .	     alg_fcn_arg_ptr    (  1 ) / 1	    /,
     .	     alg_arg_name     (  1 ) / 'X'	    /,
     .	     alg_arg_units    (  1 ) / ' '	    /,
     .	     alg_arg_descr    (  1 ) / ' '	    /,
     .	     alg_fcn		(  2 ) / 'LOG'      /,
     .	     alg_num_fcn_args	(  2 ) / 1 	    /
     .	     alg_fcn_descr	(  2 ) / 'base 10 log(X)' 	    /,
     .	     alg_fcn_arg_ptr    (  2 ) / 2	    /,
     .	     alg_arg_name     (  2 ) / 'X'	    /,
     .	     alg_arg_units    (  2 ) / ' '	    /,
     .	     alg_arg_descr    (  2 ) / ' '	    /,
     .	     alg_fcn		(  3 ) / 'MAX'      /,
     .	     alg_num_fcn_args	(  3 ) / 2	    /,
     .	     alg_fcn_descr	(  3 ) /
     .			'point-by-point greater of A and B' 	    /,
     .	     alg_fcn_arg_ptr    (  3 ) / 3	    /,
     .	     alg_arg_name     (  3 ) / 'A'	    /,
     .	     alg_arg_units    (  3 ) / ' '	    /,
     .	     alg_arg_descr    (  3 ) / ' '	    /,
     .	     alg_arg_name     (  4 ) / 'B'	    /,
     .	     alg_arg_units    (  4 ) / ' '	    /,
     .	     alg_arg_descr    (  4 ) / ' '	    /,
     .       alg_fcn		(  4 ) / 'MIN'      /,
     .	     alg_num_fcn_args 	(  4 ) / 2	    /,
     .	     alg_fcn_descr	(  4 ) /
     .			'point-by-point lesser of A and B' 	    /,
     .	     alg_fcn_arg_ptr    (  4 ) / 5	    /,
     .	     alg_arg_name     (  5 ) / 'A'	    /,
     .	     alg_arg_units    (  5 ) / ' '	    /,
     .	     alg_arg_descr    (  5 ) / ' '	    /,
     .	     alg_arg_name     (  6 ) / 'B'	    /,
     .	     alg_arg_units    (  6 ) / ' '	    /,
     .	     alg_arg_descr    (  6 ) / ' '	    /,
     .       alg_fcn		(  5 ) / 'INT'      /,
     .	     alg_num_fcn_args	(  5 ) / 1	    /,
     .	     alg_fcn_descr	(  5 ) / 'truncate to integer' 	    /,
     .	     alg_fcn_arg_ptr    (  5 ) / 7	    /,
     .	     alg_arg_name     (  7 ) / 'X'	    /,
     .	     alg_arg_units    (  7 ) / ' '	    /,
     .	     alg_arg_descr    (  7 ) / ' '	    /,
     .       alg_fcn		(  6 ) / 'ABS'      /,
     .	     alg_num_fcn_args	(  6 ) / 1	    /,
     .	     alg_fcn_descr	(  6 ) / 'absolute value' 	    /,
     .	     alg_fcn_arg_ptr    (  6 ) / 8	    /,
     .	     alg_arg_name     (  8 ) / 'X'	    /,
     .	     alg_arg_units    (  8 ) / ' '	    /,
     .	     alg_arg_descr    (  8 ) / ' '	    /,
     .       alg_fcn		(  7 ) / 'SIN'      /,
     .	     alg_num_fcn_args	(  7 ) / 1 	    /,
     .	     alg_fcn_descr	(  7 ) / ' ' 	    /,
     .	     alg_fcn_arg_ptr    (  7 ) / 9	    /,
     .	     alg_arg_name     (  9 ) / 'theta'	    /,
     .	     alg_arg_units    (  9 ) / 'radians'	    /,
     .	     alg_arg_descr    (  9 ) / 'angle'	    /,
     .       alg_fcn		(  8 ) / 'COS'      /,
     .	     alg_num_fcn_args	(  8 ) / 1 	    /,
     .	     alg_fcn_descr	(  8 ) / ' ' 	    /,
     .	     alg_fcn_arg_ptr    (  8 ) / 10	    /,
     .	     alg_arg_name     ( 10 ) / 'theta'	    /,
     .	     alg_arg_units    ( 10 ) / 'radians'	    /,
     .	     alg_arg_descr    ( 10 ) / 'angle'	    /,
     .       alg_fcn		(  9 ) / 'TAN'      /,
     .	     alg_num_fcn_args	(  9 ) / 1 	    /,
     .	     alg_fcn_descr	(  9 ) / ' ' 	    /,
     .	     alg_fcn_arg_ptr    (  9 ) / 11	    /,
     .	     alg_arg_name     ( 11 ) / 'theta'	    /,
     .	     alg_arg_units    ( 11 ) / 'radians'	    /,
     .	     alg_arg_descr    ( 11 ) / 'angle'	    /,
     .       alg_fcn		( 10 ) / 'LN'       /,
     .	     alg_num_fcn_args	( 10 ) / 1 	    /,
     .	     alg_fcn_descr	( 10 ) / 'natural logarithm(X)'     /,
     .	     alg_fcn_arg_ptr    ( 10 ) / 12	    /,
     .	     alg_arg_name     ( 12 ) / 'X'	    /,
     .	     alg_arg_units    ( 12 ) / ' '	    /,
     .	     alg_arg_descr    ( 12 ) / ' '	    /
	DATA
     .       alg_fcn		( 11 ) / 'MOD'      /,
     .	     alg_num_fcn_args	( 11 ) / 2 	    /,
     .	     alg_fcn_descr	( 11 ) / 'modulo A using base B'  /,
     .	     alg_fcn_arg_ptr    ( 11 ) / 13	    /,
     .	     alg_arg_name     ( 13 ) / 'A'	    /,
     .	     alg_arg_units    ( 13 ) / ' '	    /,
     .	     alg_arg_descr    ( 13 ) / ' '	    /,
     .	     alg_arg_name     ( 14 ) / 'B'	    /,
     .	     alg_arg_units    ( 14 ) / ' '	    /,
     .	     alg_arg_descr    ( 14 ) / ' '	    /,
     .       alg_fcn		( 12 ) / 'MISSING'  /,
     .	     alg_num_fcn_args	( 12 ) / 2 	    /,
     .	     alg_fcn_descr	( 12 ) /
     .				'substitute B where A is missing' /,
     .	     alg_fcn_arg_ptr    ( 12 ) / 15	    /,
     .	     alg_arg_name     ( 15 ) / 'A'	    /,
     .	     alg_arg_units    ( 15 ) / ' '	    /,
     .	     alg_arg_descr    ( 15 ) / ' '	    /,
     .	     alg_arg_name     ( 16 ) / 'B'	    /,
     .	     alg_arg_units    ( 16 ) / ' '	    /,
     .	     alg_arg_descr    ( 16 ) / ' '	    /,
     .       alg_fcn		( 13 ) / 'IGNORE0'  /,
     .	     alg_num_fcn_args	( 13 ) / 1 	    /,
     .	     alg_fcn_descr	( 13 )
     .				/ 'substitute missing value for zeros' /,
     .	     alg_fcn_arg_ptr    ( 13 ) / 17	    /,
     .	     alg_arg_name     ( 17 ) / 'X'	    /,
     .	     alg_arg_units    ( 17 ) / ' '	    /,
     .	     alg_arg_descr    ( 17 ) / ' '	    /,
     .       alg_fcn		( 14 ) / 'ATAN'	    /,
     .	     alg_num_fcn_args	( 14 ) / 1 	    /,
     .	     alg_fcn_descr	( 14 ) / 'arctan(X) in radians'    /,
     .	     alg_fcn_arg_ptr    ( 14 ) / 18	    /,
     .	     alg_arg_name     ( 18 ) / 'X'	    /,
     .	     alg_arg_units    ( 18 ) / ' '	    /,
     .	     alg_arg_descr    ( 18 ) / ' '	    /,
     .       alg_fcn		( 15 ) / 'ATAN2'    /,
     .	     alg_num_fcn_args	( 15 ) / 2 	    /,
     .	     alg_fcn_descr	( 15 ) /
     .			'arctan(A/B) in radians,  -pi<result<=pi'   /,
     .	     alg_fcn_arg_ptr    ( 15 ) / 19	    /,
     .	     alg_arg_name     ( 19 ) / 'A'	    /,
     .	     alg_arg_units    ( 19 ) / ' '	    /,
     .	     alg_arg_descr    ( 19 ) /
     .		'if A>0 result>0, if A=0 B determines result'	    /,
     .	     alg_arg_name     ( 20 ) / 'B'	    /,
     .	     alg_arg_units    ( 20 ) / ' '	    /,
     .	     alg_arg_descr    ( 20 ) /
     .	'if A=0&B>0 res=0, if A=0&B<0 res=pi, if B=0 ABS(res)=pi/2' /,
     .       alg_fcn		( 16 ) / 'ASIN'     /,
     .	     alg_num_fcn_args	( 16 ) / 1 	    /,
     .	     alg_fcn_descr	( 16 ) / 'arcsin(X) in radians'     /,
     .	     alg_fcn_arg_ptr    ( 16 ) / 21	    /,
     .	     alg_arg_name     ( 21 ) / 'X'	    /,
     .	     alg_arg_units    ( 21 ) / ' '	    /,
     .	     alg_arg_descr    ( 21 ) /
     .			'ABS(X) must be less than or equal to 1'    /,
     .       alg_fcn		( 17 ) / 'ACOS'     /,
     .	     alg_num_fcn_args	( 17 ) / 1 	    /
     .	     alg_fcn_descr	( 17 ) / 'arccos(X) in radians'     /,
     .	     alg_fcn_arg_ptr    ( 17 ) / 22	    /,
     .	     alg_arg_name     ( 22 ) / 'X'	    /,
     .	     alg_arg_units    ( 22 ) / ' '	    /,
     .	     alg_arg_descr    ( 22 ) /
     .			'ABS(X) must be less than or equal to 1'    /,
     .       alg_fcn		( 18 ) / 'RANDU'    /,
     .	     alg_num_fcn_args	( 18 ) / 1 	    /
     .	     alg_fcn_descr	( 18 )
     .		/ 'random uniform [0,1] seeded from 1st value of A' /,
     .	     alg_fcn_arg_ptr    ( 18 ) / 23	    /,
     .	     alg_arg_name     ( 23 ) / 'A'	    /,
     .	     alg_arg_units    ( 23 ) / ' '	    /,
     .	     alg_arg_descr    ( 23 ) /
     .			'field of random values will have shape of A'/,
     .       alg_fcn		( 19 ) / 'RANDN'    /,
     .	     alg_num_fcn_args	( 19 ) / 1 	    /,
     .	     alg_fcn_descr	( 19 )
     .			/ 'random normal seeded from 1st value of A' /,
     .	     alg_fcn_arg_ptr    ( 19 ) / 24	    /,
     .	     alg_arg_name     ( 24 ) / 'A'	    /,
     .	     alg_arg_units    ( 24 ) / ' '	    /,
     .	     alg_arg_descr    ( 24 ) /
     .			'field of random values will have shape of A'/
	DATA
     .       alg_fcn		( 20 ) / 'RHO_UN'   /,
     .	     alg_num_fcn_args	( 20 ) / 3 	    /,
     .	     alg_fcn_descr	( 20 ) /
     .		'UNESCO state equation (density) for ocean H2O (kg/m^3)'/,
     .	     alg_fcn_arg_ptr    ( 20 ) / 25	    /,
     .	     alg_arg_name     ( 25 ) / 'salt'	    /,
     .	     alg_arg_units    ( 25 ) / 'psu'	    /,
     .	     alg_arg_descr    ( 25 ) / 'salinity'   /,
     .	     alg_arg_name     ( 26 ) / 'temp'	    /,
     .	     alg_arg_units    ( 26 ) / 'deg. C'	    /,
     .	     alg_arg_descr    ( 26 ) / 'temperature'/,
     .	     alg_arg_name     ( 27 ) / 'p'	    /,
     .	     alg_arg_units    ( 27 ) / 'decibars'   /,
     .	     alg_arg_descr    ( 27 ) / 'reference pressure'   /,
     .       alg_fcn		( 21 ) / 'THETA_FO' /,
     .	     alg_num_fcn_args	( 21 ) / 4 	    /
     .	     alg_fcn_descr	( 21 ) /
     .			'Fofonoff (1977) potential temperature'/,
     .	     alg_fcn_arg_ptr    ( 21 ) / 28	    /,
     .	     alg_arg_name     ( 28 ) / 'salt'	    /,
     .	     alg_arg_units    ( 28 ) / 'psu'	    /,
     .	     alg_arg_descr    ( 28 ) / 'salinity'   /,
     .	     alg_arg_name     ( 29 ) / 'temp'	    /,
     .	     alg_arg_units    ( 29 ) / 'deg. C'	    /,
     .	     alg_arg_descr    ( 29 ) / 'temperature'/,
     .	     alg_arg_name     ( 30 ) / 'p'	    /,
     .	     alg_arg_units    ( 30 ) / 'decibars'   /,
     .	     alg_arg_descr    ( 30 ) / 'pressure'   /,
     .	     alg_arg_name     ( 31 ) / 'ref'	    /,
     .	     alg_arg_units    ( 31 ) / 'decibars'   /,
     .	     alg_arg_descr    ( 31 ) / 'reference pressure' /,
     .       alg_fcn		( 22 ) / 'DAYS1900'  /,
     .	     alg_num_fcn_args	( 22 ) / 3 	    /
     .	     alg_fcn_descr	( 22 ) 
     .	     / 'days elapsed since 1-Jan-1900 (standard calendar)'/,
     .	     alg_fcn_arg_ptr    ( 22 ) / 32	    /,
     .	     alg_arg_name     ( 32 ) / 'year'	    /,
     .	     alg_arg_units    ( 32 ) / ' '	    /,
     .	     alg_arg_descr    ( 32 ) / ' '	    /,
     .	     alg_arg_name     ( 33 ) / 'month'	    /,
     .	     alg_arg_units    ( 33 ) / ' '	    /,
     .	     alg_arg_descr    ( 33 ) / ' '	    /,
     .	     alg_arg_name     ( 34 ) / 'day'	    /,
     .	     alg_arg_units    ( 34 ) / ' '	    /,
     .	     alg_arg_descr    ( 34 ) / ' '	    /,
     .       alg_fcn		( 23 ) / 'RANDU2'    /,
     .	     alg_num_fcn_args	( 23 ) / 2 	    /
     .	     alg_fcn_descr	( 23 )
     .		/ 'random uniform [0,1), Alternate algorithm' /,
     .	     alg_fcn_arg_ptr    ( 23 ) / 35	    /,
     .	     alg_arg_name     ( 35 ) / 'A'	    /,
     .	     alg_arg_units    ( 35 ) / ' '	    /,
     .	     alg_arg_descr    ( 35 ) /
     .			'field of random values will have shape of A'/,
     .	     alg_arg_name     ( 36 ) / 'ISEED'	    /,
     .	     alg_arg_units    ( 36 ) / ' '	    /,
     .	     alg_arg_descr    ( 36 ) / 
     .         '-1=sys clock, 0=continue w/ previous seed, N>0 user-defined seed'/,
     .       alg_fcn		( 24 ) / 'RANDN2'    /,
     .	     alg_num_fcn_args	( 24 ) / 2 	    /,
     .	     alg_fcn_descr	( 24 )
     .			/ 'random normal, Alternate algorithm' /,
     .	     alg_fcn_arg_ptr    ( 24 ) / 37	    /,
     .	     alg_arg_name     ( 37 ) / 'A'	    /,
     .	     alg_arg_units    ( 37 ) / ' '	    /,
     .	     alg_arg_descr    ( 37 ) /
     .			'field of random values will have shape of A'/,
     .	     alg_arg_name     ( 38 ) / 'ISEED'	    /,
     .	     alg_arg_units    ( 38 ) / ' '	    /,
     .	     alg_arg_descr    ( 38 ) /
     .         '-1=sys clock, 0=continue w/ previous seed, N>0 user-defined seed'/
	DATA (alg_fcn(i),i=23,num_functions)/ 3*unspecified_name4 /  

* logical structures:
	DATA alg_struct(  1 ) / 'IF' /,		! formerly "WHERE"
     .	     alg_struct(  2 ) / 'THEN' /,
     .	     alg_struct(  3 ) / 'ELSE' /

* logical if-valid structures:
	DATA alg_vstruct(  1 ) / 'IFV' /,	
     .	     alg_vstruct(  2 ) / 'THEN' /,
     .	     alg_vstruct(  3 ) / 'ELSE' /

* punctuation:
	DATA alg_punct(  1 ) / ',' /,
     .	     alg_punct(  2 ) / '(' /,
     .	     alg_punct(  3 ) / ')' /

* pseudo_variables:
* Keep the classic 4D variables together, then the
* ones for 6D. Lets us set mode fourdee to use M,N etc as
* user variables.

        DATA alg_pvar_head     / 0 /
        DATA alg_pvar_padding  / 0 /
	DATA alg_pvar	(  1 ) / 'I       ' /,
     .       alg_pvar	(  2 ) / 'J       ' /,
     .       alg_pvar	(  3 ) / 'K       ' /,
     .       alg_pvar	(  4 ) / 'L       ' /,
     .       alg_pvar	(  5 ) / '_M      ' /,
     .       alg_pvar	(  6 ) / '_N      ' /,
     .       alg_pvar	(  7 ) / 'X       ' /,
     .       alg_pvar	(  8 ) / 'Y       ' /,
     .       alg_pvar	(  9 ) / 'Z       ' /,
     .       alg_pvar	( 10 ) / 'T       ' /,
     .       alg_pvar	( 11 ) / '_E      ' /,
     .       alg_pvar	( 12 ) / '_F      ' /
     .       alg_pvar	( 13 ) / 'XBOX    ' /,
     .       alg_pvar	( 14 ) / 'YBOX    ' /,
     .       alg_pvar	( 15 ) / 'ZBOX    ' /,
     .       alg_pvar	( 16 ) / 'TBOX    ' /,
     .       alg_pvar	( 17 ) / 'EBOX    ' /,
     .       alg_pvar	( 18 ) / 'FBOX    ' /,
     .       alg_pvar	( 19 ) / 'XBOXLO  ' /,
     .       alg_pvar	( 20 ) / 'YBOXLO  ' /,
     .       alg_pvar	( 21 ) / 'ZBOXLO  ' /,
     .       alg_pvar	( 22 ) / 'TBOXLO  ' /,
     .       alg_pvar	( 23 ) / 'EBOXLO  ' /,
     .       alg_pvar	( 24 ) / 'FBOXLO  ' /,
     .       alg_pvar	( 25 ) / 'XBOXHI  ' /,
     .       alg_pvar	( 26 ) / 'YBOXHI  ' /,
     .       alg_pvar	( 27 ) / 'ZBOXHI  ' /,
     .       alg_pvar	( 28 ) / 'TBOXHI  ' /,
     .       alg_pvar	( 29 ) / 'EBOXHI  ' /,
     .       alg_pvar	( 30 ) / 'FBOXHI  ' /
! allow _I,_J,_K,_L,_X,_Y,_Z,_T which will be translated to just I,J,...Z,T.
	DATA alg_pvar	( 31 ) / '_I      ' /,
     .       alg_pvar	( 32 ) / '_J      ' /,
     .       alg_pvar	( 33 ) / '_K      ' /,
     .       alg_pvar	( 34 ) / '_L      ' /,
     .       alg_pvar	( 35 ) / '_M      ' /,
     .       alg_pvar	( 36 ) / '_N      ' /,
     .       alg_pvar	( 37 ) / '_X      ' /,
     .       alg_pvar	( 38 ) / '_Y      ' /,
     .       alg_pvar	( 39 ) / '_Z      ' /,
     .       alg_pvar	( 40 ) / '_T      ' /
        DATA alg_cvar        /60*unspecified_name4/
        DATA countervar_name_head / 0 /
        DATA countervar_name_padding / 0 /
        DATA countervar_name /60*unspecified_name4/

	DATA
     .		alg_trans_text ( 1 )	/ 'AV4' 		/, 
     .		alg_trans_title( 1 )	/ '4D ave'		/,
     .		alg_trans_num  ( 1 )	/ trans_4d_ave		/,
     .		alg_trans_its_int ( 1)	/ .FALSE.		/,
     .		alg_trans_has_arg ( 1)	/ .FALSE.		/,
     .		alg_trans_dflt_arg( 1)	/ bad_val4		/,	
     .		alg_trans_text ( 2 )	/ 'IN4' 		/, 
     .		alg_trans_title( 2 )	/ '4D integ.'		/,
     .		alg_trans_num  ( 2 )	/ trans_4d_int_def	/,
     .		alg_trans_its_int ( 2 )	/ .FALSE.		/,
     .		alg_trans_has_arg ( 2 )	/ .FALSE.		/,
     .		alg_trans_dflt_arg( 2 )	/ bad_val4		/,
     .		alg_trans_text ( 3 )	/ 'ITP' 		/, 
     .		alg_trans_title( 3 )	/ 'interpolated'	/,
     .		alg_trans_num  ( 3 )	/ trans_interpolate	/,
     .		alg_trans_its_int ( 3 )	/ .FALSE.		/,
     .		alg_trans_has_arg ( 3 )	/ .FALSE.		/,
     .		alg_trans_dflt_arg( 3 )	/ bad_val4		/,
     .		alg_trans_text ( 4  )	/ 'AVE' 		/, 
     .          alg_trans_title( 4  )	/ 'averaged'      	/,
     .		alg_trans_num  ( 4  )	/ trans_average		/,
     .		alg_trans_its_int ( 4 )	/ .FALSE.		/,
     .		alg_trans_has_arg ( 4 )	/ .FALSE.		/,
     .		alg_trans_dflt_arg( 4 )	/ bad_val4		/,
     .		alg_trans_text ( 5 )	/ 'VAR'  		/, 
     .		alg_trans_title( 5 )	/ 'variance'    	/,
     .		alg_trans_num  ( 5 )	/ trans_variance  	/,
     .		alg_trans_its_int ( 5)	/ .FALSE.		/,
     .		alg_trans_has_arg ( 5)	/ .FALSE.		/,
     .		alg_trans_dflt_arg( 5)	/ bad_val4		/,
     .		alg_trans_text ( 6 )	/ 'SUM'  		/, 
     .		alg_trans_title( 6 )	/ 'summed'              /,
     .          alg_trans_num  ( 6 )	/ trans_sum	        /,
     .		alg_trans_its_int ( 6)	/ .FALSE.		/,
     .		alg_trans_has_arg ( 6)	/ .FALSE.		/,
     .		alg_trans_dflt_arg( 6)	/ bad_val4		/,
     .		alg_trans_text ( 7 )	/ 'RSU'  		/, 
     .		alg_trans_title( 7 )	/ 'running sum'  	/,
     .		alg_trans_num  ( 7 )	/ trans_run_sum  	/,
     .		alg_trans_its_int ( 7)	/ .FALSE.		/,
     .		alg_trans_has_arg ( 7)	/ .FALSE.		/,
     .		alg_trans_dflt_arg( 7)	/ bad_val4		/,
     .		alg_trans_text ( 8 )	/ 'SHF' 		/, 
     .		alg_trans_title( 8 )	/ 'shifted'		/,
     .		alg_trans_num  ( 8 )	/ trans_shift		/,
     .		alg_trans_its_int ( 8)	/ .TRUE.		/,
     .		alg_trans_has_arg ( 8)	/ .TRUE.		/,
     .		alg_trans_dflt_arg( 8)	/ 1.			/,
     .		alg_trans_text ( 9  )	/ 'MIN' 		/, 
     .		alg_trans_title( 9  )	/ 'minimum'		/,
     .		alg_trans_num  ( 9  )	/ trans_min		/,
     .		alg_trans_its_int ( 9)	/ .FALSE.		/,
     .		alg_trans_has_arg ( 9)	/ .FALSE.		/,
     .		alg_trans_dflt_arg( 9)	/ bad_val4		/,
     .		alg_trans_text ( 10  )	/ 'MAX' 		/, 
     .		alg_trans_title( 10  )	/ 'maximum'		/,
     .		alg_trans_num  ( 10  )	/ trans_max		/,
     .		alg_trans_its_int ( 10)	/ .FALSE.		/,
     .		alg_trans_has_arg ( 10)	/ .FALSE.		/,
     .		alg_trans_dflt_arg( 10)	/ bad_val4		/
	DATA
     .		alg_trans_text ( 11 )	/ ' '			/, ! for EXAM
     .		alg_trans_title( 11 )	/ ' '			/,
     .		alg_trans_num  ( 11 )	/ trans_no_transform	/,
     .		alg_trans_its_int ( 11)	/ .FALSE.		/,
     .		alg_trans_has_arg ( 11)	/ .FALSE.		/,
     .		alg_trans_dflt_arg( 11)	/ bad_val4		/,
     .		alg_trans_text ( 12 )	/ 'NUL'			/, ! for @NUL
     .		alg_trans_title( 12 )	/ ' '			/,
     .		alg_trans_num  ( 12 )	/ trans_no_transform	/,
     .		alg_trans_its_int ( 12)	/ .FALSE.		/,
     .		alg_trans_has_arg ( 12)	/ .FALSE.		/,
     .		alg_trans_dflt_arg( 12)	/ bad_val4		/,
     .		alg_trans_text ( 13  )	/ 'DDC' 		/, 
     .		alg_trans_title( 13 )	/ 'centered derivative'	/,
     .		alg_trans_num  ( 13 )	/ trans_deriv_cntr	/,
     .		alg_trans_its_int ( 13)	/ .FALSE.		/,
     .		alg_trans_has_arg ( 13)	/ .FALSE.		/,
     .		alg_trans_dflt_arg( 13)	/ bad_val4		/,
     .		alg_trans_text ( 14 )	/ 'DDF' 		/, 
     .		alg_trans_title( 14 )	/ 'forward derivative'	/,
     .		alg_trans_num  ( 14 )	/ trans_deriv_fwrd	/,
     .		alg_trans_its_int ( 14)	/ .FALSE.		/,
     .		alg_trans_has_arg ( 14)	/ .FALSE.		/,
     .		alg_trans_dflt_arg( 14)	/ bad_val4		/,
     .		alg_trans_text ( 15 )	/ 'DDB' 		/, 
     .		alg_trans_title( 15 )	/ 'backwards derivative'/,
     .		alg_trans_num  ( 15 )	/ trans_deriv_bkwd	/,
     .		alg_trans_its_int ( 15)	/ .FALSE.		/,
     .		alg_trans_has_arg ( 15)	/ .FALSE.		/,
     .		alg_trans_dflt_arg( 15)	/ bad_val4		/,
     .		alg_trans_text ( 16 )	/ 'DIN' 		/, 
     .		alg_trans_title( 16 )	/ 'integrated'		/,
     .		alg_trans_num  ( 16 )	/ trans_integ_def	/,
     .		alg_trans_its_int ( 16)	/ .FALSE.		/,
     .		alg_trans_has_arg ( 16)	/ .FALSE.		/,
     .		alg_trans_dflt_arg( 16)	/ bad_val4		/,
     .		alg_trans_text ( 17 )	/ 'IIN' 		/, 
     .		alg_trans_title( 17 )	/ 'indef. integ.'	/,
     .		alg_trans_num  ( 17 )	/ trans_integ_indef	/,
     .		alg_trans_its_int ( 17)	/ .FALSE.		/,
     .		alg_trans_has_arg ( 17)	/ .FALSE.		/,
     .		alg_trans_dflt_arg( 17)	/ bad_val4		/,
     .		alg_trans_text ( 18 )	/ 'SBX' 		/, 
     .		alg_trans_title( 18 )	/ 'box smoothed'	/,
     .		alg_trans_num  ( 18 )	/ trans_smth_box	/,
     .		alg_trans_its_int ( 18)	/ .TRUE.		/,
     .		alg_trans_has_arg ( 18)	/ .TRUE.		/,
     .		alg_trans_dflt_arg( 18)	/ 3.			/,
     .		alg_trans_text ( 19 )	/ 'SBN' 		/, 
     .		alg_trans_title( 19 )	/ 'binomial smoothed'	/,
     .		alg_trans_num  ( 19 )	/ trans_smth_binml	/,
     .		alg_trans_its_int ( 19)	/ .TRUE.		/,
     .		alg_trans_has_arg ( 19)	/ .TRUE.		/,
     .		alg_trans_dflt_arg( 19)	/ 3.			/,
     .		alg_trans_text ( 20 )	/ 'SWL' 		/, 
     .		alg_trans_title( 20 )	/ 'Welch smoothed'	/,
     .		alg_trans_num  ( 20 )	/ trans_smth_welch	/,
     .		alg_trans_its_int ( 20)	/ .TRUE.		/,
     .		alg_trans_has_arg ( 20)	/ .TRUE.		/,
     .		alg_trans_dflt_arg( 20)	/ 3.			/
	DATA
     .		alg_trans_text ( 21 )	/ 'SHN' 		/, 
     .		alg_trans_title( 21 )	/ 'Hanning smoothed'	/,
     .		alg_trans_num  ( 21 )	/ trans_smth_hanng	/,
     .		alg_trans_its_int ( 21)	/ .TRUE.		/,
     .		alg_trans_has_arg ( 21)	/ .TRUE.		/,
     .		alg_trans_dflt_arg( 21)	/ 3.			/,
     .		alg_trans_text ( 22 )	/ 'SPZ' 		/, 
     .		alg_trans_title( 22 )	/ 'Parzen smoothed'	/,
     .		alg_trans_num  ( 22 )	/ trans_smth_parzn	/,
     .		alg_trans_its_int ( 22)	/ .TRUE.		/,
     .		alg_trans_has_arg ( 22)	/ .TRUE.		/,
     .		alg_trans_dflt_arg( 22)	/ 3.			/,
     .		alg_trans_text ( 23 )	/ 'FAV' 		/, 
     .		alg_trans_title( 23 )	/ 'ave-filled'		/,
     .		alg_trans_num  ( 23 )	/ trans_fill_ave	/,
     .		alg_trans_its_int ( 23)	/ .TRUE.		/,
     .		alg_trans_has_arg ( 23)	/ .TRUE.		/,
     .		alg_trans_dflt_arg( 23)	/ 3			/,
     .		alg_trans_text ( 24 )	/ 'FLN' 		/, 
     .		alg_trans_title( 24 )	/ 'linear-filled'	/,
     .		alg_trans_num  ( 24 )	/ trans_fill_interp	/,
     .		alg_trans_its_int ( 24)	/ .TRUE.		/,
     .		alg_trans_has_arg ( 24)	/ .TRUE.		/,
     .		alg_trans_dflt_arg( 24)	/ 1.			/,
     .		alg_trans_text ( 25 )	/ 'FNR' 		/, 
     .		alg_trans_title( 25 )	/ 'nearest-filled'	/,
     .		alg_trans_num  ( 25 )	/ trans_fill_near	/,
     .		alg_trans_its_int ( 25)	/ .TRUE.		/,
     .		alg_trans_has_arg ( 25)	/ .TRUE.		/,
     .		alg_trans_dflt_arg( 25)	/ 1.			/,
     .		alg_trans_text ( 26 )	/ 'NGD'  		/, 
     .		alg_trans_title( 26 )	/ 'number of valid'  	/,
     .		alg_trans_num  ( 26 )	/ trans_good_pt  	/,
     .		alg_trans_its_int ( 26)	/ .FALSE.		/,
     .		alg_trans_has_arg ( 26)	/ .FALSE.		/,
     .		alg_trans_dflt_arg( 26)	/ bad_val4		/,
     .		alg_trans_text ( 27 )	/ 'NBD'  		/, 
     .		alg_trans_title( 27 )	/ 'number flagged bad' 	/,
     .		alg_trans_num  ( 27 )	/ trans_bad_pt  	/,
     .		alg_trans_its_int ( 27)	/ .FALSE.		/,
     .		alg_trans_has_arg ( 27)	/ .FALSE.		/,
     .		alg_trans_dflt_arg( 27)	/ bad_val4		/,
     .		alg_trans_text ( 28 )	/ 'LOC'			/, 
     .		alg_trans_title( 28 )	/ 'location'		/,
     .		alg_trans_num  ( 28 )	/ trans_locate		/,
     .		alg_trans_its_int ( 28)	/ .FALSE.		/,
     .		alg_trans_has_arg ( 28)	/ .TRUE.		/,
     .		alg_trans_dflt_arg( 28)	/ 0.0			/,
     .		alg_trans_text ( 29 )	/ 'WEQ'  		/, 
     .		alg_trans_title( 29 )	/ 'weighted equal'	/,
     .		alg_trans_num  ( 29 )	/ trans_wgted_eq	/,
     .		alg_trans_its_int ( 29)	/ .FALSE.		/,
     .		alg_trans_has_arg ( 29)	/ .TRUE.		/,
     .          alg_trans_dflt_arg( 29)	/ 0.0                   /,
     .		alg_trans_text ( 30 )	/ 'VA4' 		/,
     .		alg_trans_title( 30 )	/ '4D variance'		/,
     .		alg_trans_num  ( 30 )	/ trans_4d_var		/,
     .		alg_trans_its_int ( 30)	/ .FALSE.		/,
     .		alg_trans_has_arg ( 30)	/ .FALSE.		/,
     .		alg_trans_dflt_arg( 30)	/ bad_val4		/
	DATA
     .		alg_trans_text ( 31 )	/ 'NG4' 		/,
     .		alg_trans_title( 31 )	/ '4D # valid'		/,
     .		alg_trans_num  ( 31 )	/ trans_4d_good_pt	/,
     .		alg_trans_its_int ( 31)	/ .FALSE.		/,
     .		alg_trans_has_arg ( 31)	/ .FALSE.		/,
     .		alg_trans_dflt_arg( 31)	/ bad_val4		/,
     .		alg_trans_text ( 32 )	/ 'NB4' 		/,
     .		alg_trans_title( 32 )	/ '4D # bad'		/,
     .		alg_trans_num  ( 32 )	/ trans_4d_bad_pt	/,
     .		alg_trans_its_int ( 32)	/ .FALSE.		/,
     .		alg_trans_has_arg ( 32)	/ .FALSE.		/,
     .		alg_trans_dflt_arg( 32)	/ bad_val4		/
	DATA
     .		alg_trans_text ( 33 )	/ 'CDA' 		/,
     .		alg_trans_title( 33 )	/ 'closest dist above'  /,
     .		alg_trans_num  ( 33 )	/ trans_nrst_dist_abv	/,
     .		alg_trans_its_int ( 33)	/ .TRUE.		/,
     .		alg_trans_has_arg ( 33)	/ .TRUE.		/,
     .		alg_trans_dflt_arg( 33)	/ 1			/,
     .		alg_trans_text ( 34 )	/ 'CDB' 		/,
     .		alg_trans_title( 34 )	/ 'closest dist below'  /,
     .		alg_trans_num  ( 34 )	/ trans_nrst_dist_blw	/,
     .		alg_trans_its_int ( 34)	/ .TRUE.		/,
     .		alg_trans_has_arg ( 34)	/ .TRUE.		/,
     .		alg_trans_dflt_arg( 34)	/ 1			/,
     .		alg_trans_text ( 35 )	/ 'CIA' 		/,
     .		alg_trans_title( 35 )	/ 'closest index above' /,
     .		alg_trans_num  ( 35 )	/ trans_nrst_indx_abv	/,
     .		alg_trans_its_int ( 35)	/ .TRUE.		/,
     .		alg_trans_has_arg ( 35)	/ .TRUE.		/,
     .		alg_trans_dflt_arg( 35)	/ 1			/,
     .		alg_trans_text ( 36 )	/ 'CIB' 		/,
     .		alg_trans_title( 36 )	/ 'closest index below' /,
     .		alg_trans_num  ( 36 )	/ trans_nrst_indx_blw	/,
     .		alg_trans_its_int ( 36)	/ .TRUE.		/,
     .		alg_trans_has_arg ( 36)	/ .TRUE.		/,
     .		alg_trans_dflt_arg( 36)	/ 1			/,
     .		alg_trans_text ( 37 )	/ 'EVN' 		/,
     .		alg_trans_title( 37 )	/ 'event mask'		/,
     .		alg_trans_num  ( 37 )	/ trans_event_mask	/,
     .		alg_trans_its_int ( 37)	/ .FALSE.		/,
     .		alg_trans_has_arg ( 37)	/ .TRUE.		/,
     .		alg_trans_dflt_arg( 37)	/ 0.0			/,
     .		alg_trans_text ( 38 )	/ 'MED' 		/,
     .		alg_trans_title( 38 )	/ 'median smoothed'     /,
     .		alg_trans_num  ( 38 )	/ trans_smth_median	/,
     .		alg_trans_its_int ( 38)	/ .TRUE.		/,
     .		alg_trans_has_arg ( 38)	/ .TRUE.		/,
     .		alg_trans_dflt_arg( 38)	/ 3		/,
     .		alg_trans_text ( 39 )	/ 'SMX' 		/,
     .		alg_trans_title( 39 )	/ 'maximum smoothed'     /,
     .		alg_trans_num  ( 39 )	/ trans_smth_max	/,
     .		alg_trans_its_int ( 39)	/ .TRUE.		/,
     .		alg_trans_has_arg ( 39)	/ .TRUE.		/,
     .		alg_trans_dflt_arg( 39)	/ 3		/,
     .		alg_trans_text ( 40 )	/ 'SMN' 		/,
     .		alg_trans_title( 40 )	/ 'minimum smoothed'     /,
     .		alg_trans_num  ( 40 )	/ trans_smth_min	/,
     .		alg_trans_its_int ( 40)	/ .TRUE.		/,
     .		alg_trans_has_arg ( 40)	/ .TRUE.		/,
     .		alg_trans_dflt_arg( 40)	/ 3		        /,
     .		alg_trans_text ( 41 )	/ 'SM4'  		/, 
     .		alg_trans_title( 41 )	/ '4D summed'           /,
     .          alg_trans_num  ( 41 )	/ trans_4d_sum	        /,
     .		alg_trans_its_int ( 41)	/ .FALSE.		/,
     .		alg_trans_has_arg ( 41)	/ .FALSE.		/,
     .		alg_trans_dflt_arg( 41)/ bad_val4		/,
     .		alg_trans_text ( 42 )	/ 'STD'  		/, 
     .		alg_trans_title( 42 )	/ 'standard deviation'  /,
     .		alg_trans_num  ( 42 )	/ trans_std_dev  	/,
     .		alg_trans_text ( 43 )	/ 'WGT'  		/, 
     .		alg_trans_title( 43 )	/ 'weights for avg,int'  /,
     .		alg_trans_num  ( 43 )	/ trans_avg_wgt  	/,
     .		alg_trans_text ( 44 )	/ 'WG4'  		/, 
     .		alg_trans_title( 44 )	/ '4D weights, avg,int'  /,
     .		alg_trans_num  ( 44 )	/ trans_4d_avg_wgt  	/,
     .		alg_trans_text ( 45 )	/ 'SPX' 		/, 
     .		alg_trans_title( 45 )	/ 'box smooth w/missing'/,
     .		alg_trans_num  ( 45 )	/ trans_smth_box_p	/,
     .		alg_trans_its_int ( 45)	/ .TRUE.		/,
     .		alg_trans_has_arg ( 45)	/ .TRUE.		/,
     .		alg_trans_dflt_arg( 45)	/ 3.			/,
     .		alg_trans_text ( 46 )	/ 'SD4'  		/, 
     .		alg_trans_title( 46 )	/ '4D stddev'           /,
     .		alg_trans_num  ( 46 )	/ trans_4d_std  	/

        DATA    (alg_trans_num(i),i = 47, num_transforms)
     .            /14*unspecified_int4 /

	DATA
     .		alg_regrid_title( prgrd_exact )		/ ' '	/,
     .		alg_regrid_code ( prgrd_exact )		/ 'XAK'	/,	
     .		alg_regrid_vizb ( prgrd_exact )		/ .FALSE.	/,
     .		alg_regrid_title( prgrd_linear_interp )	/ 'lin. interp.'/,
     .		alg_regrid_code ( prgrd_linear_interp )	/ 'LIN'	/,
     .		alg_regrid_vizb ( prgrd_linear_interp )	/ .TRUE.	/,
     .		alg_regrid_title( prgrd_average )	/ 'area-wgtd avgd'/,
     .		alg_regrid_code ( prgrd_average )	/ 'AVE'	/,
     .		alg_regrid_vizb ( prgrd_average )	/ .TRUE. 	/,
     .		alg_regrid_title( prgrd_xy_ave )	/ 'area avgd'	/,
     .		alg_regrid_code ( prgrd_xy_ave )	/ 'AAV'	/,
     .		alg_regrid_vizb ( prgrd_xy_ave )	/ .FALSE.	/,
     .		alg_regrid_title( prgrd_xy_lin )	/ 'area intrp.' /,
     .		alg_regrid_code ( prgrd_xy_lin )	/ 'LIN'	/,
     .		alg_regrid_vizb ( prgrd_xy_lin )	/ .FALSE.	/,
     .		alg_regrid_title( prgrd_xact_pts )	/ 'exact match'	/,
     .		alg_regrid_code ( prgrd_xact_pts )	/ 'XACT'	/
     .		alg_regrid_vizb ( prgrd_xact_pts )	/ .TRUE.	/,
     .		alg_regrid_title( prgrd_associate )	/ 'index assn'	/,
     .		alg_regrid_code ( prgrd_associate )	/ 'ASN'	/,
     .		alg_regrid_vizb ( prgrd_associate )	/ .TRUE.	/,
     .		alg_regrid_title( prgrd_variance )	/ 'variance'	/,
     .		alg_regrid_code ( prgrd_variance )	/ 'VAR'	/,
     .		alg_regrid_vizb ( prgrd_variance )	/ .TRUE.	/,
     .		alg_regrid_title( prgrd_min )		/ 'minimum'	/,
     .		alg_regrid_code ( prgrd_min )		/ 'MIN'	/,
     .		alg_regrid_vizb(  prgrd_min )		/ .TRUE.	/,
     .		alg_regrid_title( prgrd_max )		/ 'maximum'	/,
     .		alg_regrid_code ( prgrd_max )		/ 'MAX'	/,
     .		alg_regrid_vizb(  prgrd_max )		/ .TRUE.	/,
     .		alg_regrid_title( prgrd_sum )		/ 'area-wgtd sum'/,
     .		alg_regrid_code ( prgrd_sum )		/ 'SUM'	/,
     .		alg_regrid_vizb(  prgrd_sum )		/ .TRUE.	/,
     .		alg_regrid_title( prgrd_good_pt )	/ '#gd pts, area wgtd' /,
     .		alg_regrid_code ( prgrd_good_pt )	/ 'NGD'	/,
     .		alg_regrid_vizb ( prgrd_good_pt )	/ .TRUE.	/,
     .		alg_regrid_title( prgrd_modulo )	/ 'modulo ave'	/,
     .		alg_regrid_code ( prgrd_modulo )	/ 'MOD'	/
     .		alg_regrid_vizb ( prgrd_modulo )	/ .TRUE.	/,
     .		alg_regrid_title( prgrd_modvar )	/ 'modulo var'	/,
     .		alg_regrid_code ( prgrd_modvar )	/ 'MODVAR'	/
     .		alg_regrid_vizb ( prgrd_modvar )	/ .TRUE.	/,
     .		alg_regrid_title( prgrd_modngd )	/ '# gd mod pts'/,
     .		alg_regrid_code ( prgrd_modngd )	/ 'MODNGD'	/
     .		alg_regrid_vizb ( prgrd_modngd )	/ .TRUE.	/,
     .		alg_regrid_title( prgrd_modnbd )	/ '#bad mod pts'/,
     .		alg_regrid_code ( prgrd_modnbd )	/ 'MODNBD'	/
     .		alg_regrid_vizb ( prgrd_modnbd )	/ .TRUE.	/,
     .		alg_regrid_title( prgrd_modsum )	/ 'modulo sum'	/,
     .		alg_regrid_code ( prgrd_modsum )	/ 'MODSUM'	/
     .		alg_regrid_vizb ( prgrd_modsum )	/ .TRUE.	/,
     .		alg_regrid_title( prgrd_modmin )	/ 'modulo min'	/,
     .		alg_regrid_code ( prgrd_modmin )	/ 'MODMIN'	/
     .		alg_regrid_vizb ( prgrd_modmin )	/ .TRUE.	/,
     .		alg_regrid_title( prgrd_modmax )	/ 'modulo max'	/,
     .		alg_regrid_code ( prgrd_modmax )	/ 'MODMAX'	/
     .		alg_regrid_vizb ( prgrd_modmax )	/ .TRUE.	/,
     .		alg_regrid_title( prgrd_nearest )	/ 'nearst coord'/,
     .		alg_regrid_code ( prgrd_nearest )	/ 'NRST'	/
     .		alg_regrid_vizb ( prgrd_nearest )	/ .TRUE.	/,
     .		alg_regrid_title( prgrd_bin )		/ 'unwgtd avg'  /,
     .		alg_regrid_code ( prgrd_bin )		/ 'BIN'		/
     .		alg_regrid_vizb ( prgrd_bin )		/ .TRUE.	/,
     .		alg_regrid_title( prgrd_nbin )		/ '#good in bin'/,
     .		alg_regrid_code ( prgrd_nbin )		/ 'NBIN'	/
     .		alg_regrid_vizb ( prgrd_nbin )		/ .TRUE.	/,
     .		alg_regrid_title( prgrd_iin )		/ '#indef integral'/,
     .		alg_regrid_code ( prgrd_iin )		/ 'IIN'	/
     .		alg_regrid_vizb ( prgrd_iin )		/ .TRUE.	/,

     .		(alg_regrid_title(i),i=prgrd_iin+1,max_regrid_trans)
     .				/7*unspecified_name4/,
     .		(alg_regrid_code (i),i=prgrd_iin+1,max_regrid_trans)
     .				/7*unspecified_name4/,
     .		(alg_regrid_vizb (i),i=prgrd_iin+1,max_regrid_trans)
     .				/7*.FALSE./


	DATA
     .		alg_regrid_title(pauxrgrd_linear)	/ 'lin. interp.'/,
     .		alg_regrid_code (pauxrgrd_linear)	/ 'LIN'	        /,
     .		alg_regrid_vizb (pauxrgrd_linear)	/ .TRUE.	/,
     .		alg_regrid_title(pauxrgrd_average)	/ 'box avgd'    /,
     .		alg_regrid_code (pauxrgrd_average)	/ 'AVE'	        /,
     .		alg_regrid_vizb (pauxrgrd_average)	/ .FALSE.	/,
     .		alg_regrid_title(pauxrgrd_pwlave )	
     .						/ 'pc-wise linear ave'  /,
     .		alg_regrid_code (pauxrgrd_pwlave )	/ 'PLAVE'      /,
     .		alg_regrid_vizb (pauxrgrd_pwlave )	/ .TRUE.	/,
     .		alg_regrid_title(pauxrgrd_thick_ave)
     .						/ 'box ave from thkness'/,
     .		alg_regrid_code (pauxrgrd_thick_ave)/ unspecified_name4 /,
     .		alg_regrid_vizb (pauxrgrd_thick_ave)	/ .FALSE.	/,
     .		alg_regrid_title(pauxrgrd_fmrc)
     .						     / 'reorg T-F array'/,
     .		alg_regrid_code (pauxrgrd_fmrc)		/ 'FMRC'/,
     .		alg_regrid_vizb (pauxrgrd_fmrc)		/ .FALSE.	/,

     .		(alg_regrid_title(i),
     .				i=pauxrgrd_fmrc+1,max_auxregrid_trans)
     .				/1*unspecified_name4/,
     .		(alg_regrid_code (i),
     .				i=pauxrgrd_fmrc+1,max_auxregrid_trans)
     .				/1*unspecified_name4/,
     .		(alg_regrid_vizb (i),
     .				i=pauxrgrd_fmrc+1,max_auxregrid_trans)
     .				/1*.FALSE./

* translation table for substituting 1D transformations with 4D transforms
	DATA	num_4d_trans/ 8 /
	DATA	alg_1d_to_4d(1) /trans_average/,
     .		alg_4d_trans(1) /trans_4d_ave/,
     .		alg_1d_to_4d(2) /trans_integ_def/,
     .		alg_4d_trans(2) /trans_4d_int_def/,
     .		alg_1d_to_4d(3) /trans_variance/,
     .		alg_4d_trans(3) /trans_4d_var/,
     .		alg_1d_to_4d(4) /trans_good_pt/,
     .		alg_4d_trans(4) /trans_4d_good_pt/,
     .		alg_1d_to_4d(5) /trans_bad_pt/,
     .		alg_4d_trans(5) /trans_4d_bad_pt/,
     .		alg_1d_to_4d(6) /trans_sum/,
     .		alg_4d_trans(6) /trans_4d_sum/,
     .		alg_1d_to_4d(7) /trans_avg_wgt/,
     .		alg_4d_trans(7) /trans_4d_avg_wgt/
     .		alg_1d_to_4d(8) /trans_std_dev/,
     .		alg_4d_trans(8) /trans_4d_std/

	END

