      SUBROUTINE STATE ( temp, mtemp, salt, msalt, rho, mrho )

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
* compute density "referenced to surface" from equation of state
* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 - 3/10/86	extracted from SUBROUTINE HDIFFUS
*			with major modifications to convert i-k oriented
*			calculations to i-j oriented calculations
* revision 1.0 - 1/8/87 - STATED --> STATE with STATED added as ENTRY
* V200:  7/25/89 - 4D symmetrical version
*	10/11/89 - modified array declarations using XMEM_SUBSC.CMN

#ifdef unix
	include 'ferret_cmn/ferret.parm'
	include 'ferret_cmn/gfdl.parm'
	include	'ferret_cmn/xvariables.cmn'
	include	'ferret_cmn/xmem_subsc.cmn'
	include 'ferret_cmn/xonedim.cmn'
	include 'ferret_cmn/xcontext.cmn'
#else
	INCLUDE 'FERRET_CMN:FERRET.PARM'
	INCLUDE 'FERRET_CMN:GFDL.PARM'
	INCLUDE	'FERRET_CMN:XVARIABLES.CMN'
	INCLUDE	'FERRET_CMN:XMEM_SUBSC.CMN'
	INCLUDE 'FERRET_CMN:XONEDIM.CMN'
	INCLUDE 'FERRET_CMN:XCONTEXT.CMN'
#endif

* calling argument declarations:
	INTEGER	mtemp, msalt, mrho
* subscript ranges from memory variable table ...
	REAL    temp( m1lox:m1hix,m1loy:m1hiy,m1loz:m1hiz,m1lot:m1hit ),
     .		salt( m2lox:m2hix,m2loy:m2hiy,m2loz:m2hiz,m2lot:m2hit ),
     .		 rho( m3lox:m3hix,m3loy:m3hiy,m3loz:m3hiz,m3lot:m3hit )

* internal variable declarations:
	LOGICAL	use_k
	INTEGER	n, i, j, k, l, m
	REAL	bad_temp, bad_salt, bad_rho, tq, sq
	REAL	TO(KMP1),SO(KMP1),C(KM,9)			! 2672 mod

C     POLYNOMIAL COEFFS GO IN HERE					! 2674
C									! 2675
      DATA TO/13.4993058,13.4979166,13.4965263,13.4951349,13.4937423,	! 2676
     *13.4923486,13.4909537,13.4895577,13.4881606,13.4867624,13.4851881,! 2677
     *13.4832619,13.4809829,13.4783495,13.4751843,13.4713080,13.4663623,! 2678
     * 8.4694918, 8.4614842, 8.4482321, 5.9391351, 4.4212668, 2.8993113,! 2679
     * 2.8539693, 2.7957242, 2.7277752, 2.6540454,0./			! 2680
      DATA SO/-.00225,-.00225,-.00225,-.00225,-.00225,-.00225,-.00225,	! 2681
     *-.00225,-.00225,-.00225,-.00225,-.00225,-.00225,-.00225,-.00225,	! 2682
     *-.00225,-.00225, .00015, .00015, .00015,-.00010,-.00010,-.00025,	! 2683
     *-.00025,-.00020,-.00020,-.00020,0./				! 2684
      DATA (C( 1,N),N=1,9)/						! 2685
     1         -.2017283E-03, .7710055E+00,-.4918875E-05,-.2008622E-02,	! 2686
     1          .4495770E+00, .3656158E-07, .4728884E-02, .3770116E-04,	! 2687
     2          .6548569E+01/						! 2688
      DATA (C( 2,N),N=1,9)/						! 2689
     1         -.2019152E-03, .7709387E+00,-.4915766E-05,-.2007672E-02,	! 2690
     1          .4496149E+00, .3652747E-07, .4725372E-02, .3768380E-04,	! 2691
     2          .6548196E+01/						! 2692
      DATA (C( 3,N),N=1,9)/						! 2693
     1         -.2021021E-03, .7708719E+00,-.4912657E-05,-.2006722E-02,	! 2694
     1          .4496524E+00, .3649331E-07, .4721864E-02, .3766645E-04,	! 2695
     2          .6547823E+01/						! 2696
      DATA (C( 4,N),N=1,9)/						! 2697
     1         -.2022889E-03, .7708051E+00,-.4909550E-05,-.2005772E-02,	! 2698
     1          .4496897E+00, .3645911E-07, .4718360E-02, .3764911E-04,	! 2699
     2          .6547449E+01/						! 2700
      DATA (C( 5,N),N=1,9)/						! 2701
     1         -.2024757E-03, .7707383E+00,-.4906444E-05,-.2004821E-02,	! 2702
     1          .4497267E+00, .3642486E-07, .4714860E-02, .3763177E-04,	! 2703
     2          .6547076E+01/						! 2704
      DATA (C( 6,N),N=1,9)/						! 2705
     1         -.2026624E-03, .7706715E+00,-.4903338E-05,-.2003869E-02,	! 2706
     1          .4497634E+00, .3639055E-07, .4711363E-02, .3761443E-04,	! 2707
     2          .6546703E+01/						! 2708
      DATA (C( 7,N),N=1,9)/						! 2709
     1         -.2028490E-03, .7706047E+00,-.4900234E-05,-.2002918E-02,	! 2710
     1          .4497997E+00, .3635621E-07, .4707871E-02, .3759710E-04,	! 2711
     2          .6546329E+01/						! 2712
      DATA (C( 8,N),N=1,9)/						! 2713
     1         -.2030356E-03, .7705380E+00,-.4897130E-05,-.2001966E-02,	! 2714
     1          .4498358E+00, .3632181E-07, .4704383E-02, .3757977E-04,	! 2715
     2          .6545956E+01/						! 2716
      DATA (C( 9,N),N=1,9)/						! 2717
     1         -.2032221E-03, .7704712E+00,-.4894028E-05,-.2001014E-02,	! 2718
     1          .4498716E+00, .3628736E-07, .4700898E-02, .3756245E-04,	! 2719
     2          .6545583E+01/						! 2720
      DATA (C(10,N),N=1,9)/						! 2721
     1         -.2034085E-03, .7704045E+00,-.4890926E-05,-.2000061E-02,	! 2722
     1          .4499071E+00, .3625287E-07, .4697418E-02, .3754514E-04,	! 2723
     2          .6545209E+01/						! 2724
      DATA (C(11,N),N=1,9)/						! 2725
     1         -.2036182E-03, .7703294E+00,-.4887438E-05,-.1998989E-02,	! 2726
     1          .4499466E+00, .3621401E-07, .4693507E-02, .3752566E-04,	! 2727
     2          .6544789E+01/						! 2728
      DATA (C(12,N),N=1,9)/						! 2729
     1         -.2038743E-03, .7702377E+00,-.4883176E-05,-.1997678E-02,	! 2730
     1          .4499945E+00, .3616642E-07, .4688734E-02, .3750187E-04,	! 2731
     2          .6544276E+01/						! 2732
      DATA (C(13,N),N=1,9)/						! 2733
     1         -.2041769E-03, .7701293E+00,-.4878142E-05,-.1996128E-02,	! 2734
     1          .4500503E+00, .3611007E-07, .4683103E-02, .3747376E-04,	! 2735
     2          .6543669E+01/						! 2736
      DATA (C(14,N),N=1,9)/						! 2737
     1         -.2045258E-03, .7700043E+00,-.4872336E-05,-.1994338E-02,	! 2738
     1          .4501137E+00, .3604489E-07, .4676619E-02, .3744135E-04,	! 2739
     2          .6542968E+01/						! 2740
      DATA (C(15,N),N=1,9)/						! 2741
     1         -.2049441E-03, .7698544E+00,-.4865374E-05,-.1992189E-02,	! 2742
     1          .4501885E+00, .3596645E-07, .4668857E-02, .3740247E-04,	! 2743
     2          .6542128E+01/						! 2744
      DATA (C(16,N),N=1,9)/						! 2745
     1         -.2054550E-03, .7696712E+00,-.4856871E-05,-.1989560E-02,	! 2746
     1          .4502778E+00, .3587025E-07, .4659397E-02, .3735500E-04,	! 2747
     2          .6541100E+01/						! 2748
      DATA (C(17,N),N=1,9)/						! 2749
     1         -.2061045E-03, .7694382E+00,-.4846060E-05,-.1986211E-02,	! 2750
     1          .4503884E+00, .3574728E-07, .4647401E-02, .3729463E-04,	! 2751
     2          .6539793E+01/						! 2752
      DATA (C(18,N),N=1,9)/						! 2753
     1         -.1629093E-03, .7811063E+00,-.5282466E-05,-.2324422E-02,	! 2754
     1          .7177684E+00, .4946791E-07, .5152206E-02, .3861644E-04,	! 2755
     2          .6613048E+01/						! 2756
      DATA (C(19,N),N=1,9)/						! 2757
     1         -.1645366E-03, .7805728E+00,-.5255967E-05,-.2315935E-02,	! 2758
     1          .7082405E+00, .4927405E-07, .5132079E-02, .3850946E-04,	! 2759
     2          .6610456E+01/						! 2760
      DATA (C(20,N),N=1,9)/						! 2761
     1         -.1671881E-03, .7797018E+00,-.5212693E-05,-.2302030E-02,	! 2762
     1          .6921882E+00, .4895038E-07, .5099609E-02, .3833500E-04,	! 2763
     2          .6606212E+01/						! 2764
      DATA (C(21,N),N=1,9)/						! 2765
     1         -.1447954E-03, .7840290E+00,-.5507283E-05,-.2469817E-02,	! 2766
     1          .5823823E+00, .5786840E-07, .6147739E-02, .3922164E-04,	! 2767
     2          .6637359E+01/						! 2768
      DATA (C(22,N),N=1,9)/						! 2769
     1         -.1356332E-03, .7854813E+00,-.5641448E-05,-.2546893E-02,	! 2770
     1          .3319438E+00, .6384623E-07, .6990543E-02, .3953988E-04,	! 2771
     2          .6652626E+01/						! 2772
      DATA (C(23,N),N=1,9)/						! 2773
     1         -.1289633E-03, .7859899E+00,-.5750134E-05,-.2609364E-02,	! 2774
     1         -.2227097E+00, .7043575E-07, .7961671E-02, .3998952E-04,	! 2775
     2          .6664665E+01/						! 2776
      DATA (C(24,N),N=1,9)/						! 2777
     1         -.1422700E-03, .7819175E+00,-.5540261E-05,-.2539449E-02,	! 2778
     1         -.1417770E+01, .6943923E-07, .7901023E-02, .3952927E-04,	! 2779
     2          .6646847E+01/						! 2780
      DATA (C(25,N),N=1,9)/						! 2781
     1         -.1576122E-03, .7772468E+00,-.5293956E-05,-.2457059E-02,	! 2782
     1         -.3896984E+01, .6802500E-07, .7840945E-02, .3896220E-04,	! 2783
     2          .6626305E+01/						! 2784
      DATA (C(26,N),N=1,9)/						! 2785
     1         -.1734919E-03, .7722991E+00,-.5034400E-05,-.2370115E-02,	! 2786
     1         -.5751618E+01, .6623761E-07, .7797404E-02, .3842858E-04,	! 2787
     2          .6604176E+01/						! 2788
      DATA (C(27,N),N=1,9)/						! 2789
     1         -.1889668E-03, .7674361E+00,-.4777494E-05,-.2283773E-02,	! 2790
     1         -.7821981E+01, .6415805E-07, .7773554E-02, .3792553E-04,	! 2791
     2          .6582285E+01/						! 2792

* for entry STATE include compression effects ( depth passed throuugh "m" )
	use_k = .TRUE.
	GOTO 100

* for entry STATED REFERENCE DENSITY TO SURFACE ( depth m = 1 always )
	ENTRY STATED ( temp, mtemp, salt, msalt, rho, mrho )
	use_k = .FALSE.	
	m=1								! 2811

* flag for bad/missing data
 100	bad_temp = mr_bad_data( mtemp )
	bad_salt = mr_bad_data( msalt )
	bad_rho  = mr_bad_data( mrho )

	DO 1000 l = mr_lo_s4(mrho), mr_hi_s4(mrho)
	DO 1000 k = mr_lo_s3(mrho), mr_hi_s3(mrho)
	IF ( use_k ) m = k

	DO 120 j = mr_lo_s2(mrho), mr_hi_s2(mrho)
	DO 120 i = mr_lo_s1(mrho), mr_hi_s1(mrho)		! 2813 mod

	IF ( temp(i,j,k,l) .EQ. bad_temp
     .	.OR. salt(i,j,k,l) .EQ. bad_salt ) THEN
	   rho(i,j,k,l) = bad_rho
	   GOTO 120
	ENDIF

      TQ = temp(i,j,k,l) - TO(M)				! 2814 mod
      SQ = salt(i,j,k,l) - SO(M)				! 2815 mod
      RHO(i,j,k,l)=( C(M,1) + (C(M,4) + C(M,7)* SQ ) * SQ	! 2816 mod
     1+( C(M,3) + C(M,8)*SQ + C(M,6)*TQ ) * TQ )		! 2817 mod
     2*TQ + ( C(M,2) + (C(M,5) + C(M,9)				! 2818 mod
     3*SQ)*SQ)*SQ						! 2819 mod
 120  CONTINUE								! 2820
 1000	CONTINUE

      RETURN								! 2821
	END

