	SUBROUTINE GFDL_TERMS

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

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 - 3/7/86	extracted from SUBROUTINE HDIFFUS
* revision 0.1 - 1/16/87 - added terms needed for momentum eqn
* revision 0.2 - 7/18/88 - Am entered from data sets - not computed here

#ifdef unix
	include 'ferret_cmn/gfdl.parm'
	include 'ferret_cmn/xonedim.cmn'
#else
	INCLUDE 'FERRET_CMN:GFDL.PARM'
	INCLUDE 'FERRET_CMN:XONEDIM.CMN'
#endif

* internal variable declarations:
	INTEGER	i,j,k
	REAL	ph1deg, radius, radian, sumdy, dym

* begin executable code:
* initialize - compute basic grid coordinates ( dxt and dyt in degrees )
	CALL MESH( dxt, dyt )

* for COMMON/XEDDY
!	am = 2.E7
	ah = 2.E7
	fkpm = .0134
	fkph = .00134
	dtuv = 3600.	! 1 hour
        c2dtuv=2.0*dtuv	! 2028
	nts30 = 365.25*86400. / (12.*dtuv) + .1
	dtts = 3600.
	C2DTTS = 2.0*DTTS	! seconds to 2 hour timestep leapfrogs	! 2027
	cmonthly = c2dtts * (730./2.)	! convert seconds to months

C									!  937
       PH1DEG=-30.0+dyt(1)					! 1019 mod
C									! 1022
C   DEFINE PHYSICAL CONSTANTS						! 1023
C									! 1024
       OMEGA=3.1415927/43082.						! 1025
       RADIUS=6370.E5							! 1026
       RADIAN=57.29578							! 1027
       GRAV=980.6							! 1028

      BAH=FKPH								! 1045
      BAM=FKPM								! 1046

*   --- Z axis definitions ---
      DO 100 K=1,KM							! 1059
      C2DZ(K)=2.0*DZ(K)							! 1060
      DZ2R(K)=1.0/C2DZ(K)						! 1061
 100  CONTINUE								! 1062
      DZZ(1)=0.5*DZ(1)							! 1063
      ZDZ(1)=DZ(1)							! 1064
      DO 110 K=2,km							! 1065
      DZZ(K)=0.5*(DZ(K-1)+DZ(K))					! 1066
      ZDZ(K)=ZDZ(K-1)+DZ(K)						! 1067
 110  CONTINUE								! 1068
      DZZ(KM+1)=0.5*DZ(KM)						! 1069
!      DZZ2R(KMP1)=.5/DZZ(KMP1)						! 1070
      ZDZZ(1)=DZZ(1)							! 1071
      DO 120 K=1,km							! 1072
!      DZZ2R(K)=.5/DZZ(K)						! 1073
      ZDZZ(K+1)=ZDZZ(K)+DZZ(K+1)					! 1074
      EEH(K)=1./(DZ(K)*DZZ(K))						! 1075
      FFH(K)=1./(DZ(K)*DZZ(K+1))					! 1076
      EEM(K)=1./(DZ(K)*DZZ(K))						! 1077
      FFM(K)=1./(DZ(K)*DZZ(K+1))					! 1078
!      KAR(K)=K								! 1079
 120  CONTINUE								! 1080

* convert dxt,dyt from degrees to cm
!      DXT=DXT*RADIUS/RADIAN						! 1090
!      DYT=DYT*RADIUS/RADIAN						! 1091
	DO 124 i = 1,imt
 124	DXT(i) = dxt(i) * radius/radian
	DO 125 j = 1,jmt
 125	DYT(j) = dyt(j) * radius/radian

*   --- Y axis definitions ---
       PHI(1)=PH1DEG/RADIAN						! 1092
      PHIT(1)=PHI(1)-.5*DYT(1)/RADIUS					! 1093
      SUMDY=PHI(1)							! 1094
      DYU(JMT)=DYT(JMT)							! 1095
      DO 130 J=1,JMT							! 1096
      IF(J.NE.JMT) DYU(J)=.5*(DYT(J)+DYT(J+1))				! 1097
      DYTR(J)=1./DYT(J)							! 1098
!      DYT2R(J)=.5/DYT(J)						! 1099
!      DYT4R(J)=.25/DYT(J)						! 1100
      DYUR(J)=1./DYU(J)							! 1101
      DYU2R(J)=.5/DYU(J)						! 1102
      DYU4R(J)=.25/DYU(J)						! 1103
      IF(J.NE.JMT) SUMDY=SUMDY+DYT(J+1)/RADIUS				! 1104
      IF(J.NE.JMT) PHI(J+1)=SUMDY					! 1105
      IF(J.NE.1) PHIT(J)=.5*(PHI(J-1)+PHI(J))				! 1106
      CST(J)=COS(PHIT(J))						! 1107
      CS (J)=COS(PHI (J))						! 1108
      SINE(J)=SIN(PHI(J))						! 1109
      CSTR(J)=1.0/CST(J)						! 1110
      CSR(J)=1.0/CS(J)							! 1111
      TNG(J)=SINE(J)/CS(J)						! 1112
 130  CONTINUE								! 1113

*   --- X axis definitions ---
      DXU(IMT)=DXT(IMT)					! 1114 mod
!      IF(LCYC.EQ.1) DXU(IMT)=.5*(DXT(2)+DXT(3))		! 1115 mod
      DXU(IMT)=.5*(DXT(2)+DXT(3))		! 1115 mod 2/93
      DO 140  I=1,IMT							! 1116
      IF(I.NE.IMT) DXU(I)=.5*(DXT(I)+DXT(I+1))		! 1117 mod
      DXTR(I)=1./DXT(I)					! 1118 mod
!      DXT2R(I)=.5/DXT(I)					! 1119 mod
      DXT4R(I)=.25/DXT(I)					! 1120 mod
      DXUR(I)=1./DXU(I)							! 1121
      DXU2R(I)=.5/DXU(I)						! 1122
!      DXU4R(I)=.25/DXU(I)						! 1123
  140 CONTINUE								! 1124

* compute actual coordinate positions for x and y in degrees
* **** NOTE: the yv coordinates computed here DO NOT agree with the NMC
* (and other) wind point coordinates, although the errors are small.  According
* to the COX documentation the t,s grid points should lie at the midpoints of
* the u,v points.  This was reversed in the SUBROUTINE GRIDS (by Ron Pacanowski)
* which generated the NMC wind coordinates.  It has been corrected here.
	xv(1) = 131.0				! from NMCWIN.DAT file
	xt(1) = 130.5
	DO 150 i = 2, imt
	   xv(i) = xv( i-1 ) + 0.5 * ( dxu(i-1) + dxu(i) ) * radian/radius
	   xt(i) = xt( i-1 ) + 0.5 * ( dxt(i-1) + dxt(i) ) * radian/radius
 150	CONTINUE
	DO 160 j = 1, jmt
	   yv(j) = phi (j) * radian
	   yt(j) = phit(j) * radian
 160	CONTINUE

C									! 1137
C   PRINT SOME THINGS							! 1138
C									! 1139
!      PRINT 9701							! 1140
! 9701 FORMAT(50H0 GRID BOX THICKNESS  'DZ'                        )	! 1141
!      PRINT 970, DZ							! 1142
!      PRINT 9702							! 1143
! 9702 FORMAT(50H0 GRID POINT SEPARATION  'DZZ'                    )	! 1144
!      PRINT 970, DZZ							! 1145
!      PRINT 9703							! 1146
! 9703 FORMAT(50H0 DEPTH OF BOX BOTTOM  'ZDZ'                      )	! 1147
!      PRINT 970, ZDZ							! 1148
!      PRINT 9704							! 1149
! 9704 FORMAT(50H0 DEPTH OF GRID POINT  'ZDZZ'                     )	! 1150
!      PRINT 970, ZDZZ							! 1151
!      PRINT 9705							! 1152
! 9705 FORMAT(50H0 LATITUDE OF T,S POINTS (RADIANS)  'PHIT'        )	! 1153
!      PRINT 970, PHIT							! 1154
!      PRINT 9706							! 1155
! 9706 FORMAT(50H0 LATITUDE OF U,V POINTS (RADIANS)  'PHI'         )	! 1156
!      PRINT 970, PHI							! 1157
!      PRINT 9707							! 1158
! 9707 FORMAT(50H0 COSINE OF T,S LATITUDE  'CST'                   )	! 1159
!      PRINT 970, CST							! 1160
!      PRINT 9708							! 1161
! 9708 FORMAT(50H0 COSINE OF U,V LATITUDE  'CS'                    )	! 1162
!      PRINT 970, CS							! 1163
!      PRINT 9709							! 1164
! 9709 FORMAT(50H0 SINE OF U,V LATITUDE  'SINE'                    )	! 1165
!      PRINT 970, SINE							! 1166
 970  FORMAT(1X,10E13.5)						! 1167
!C									! 1168
!C  NEWTONIAN DAMPING COEFFICIENTS POLEWARD OF 30 DEGREES NORTH &	! 1169
!C  20 DEGREES SOUTH.							! 1170
!C									! 1171
!      RGAMMA=1./(GAMMA*86400.)						! 1172
!      DO 1200 J=1,NPS							! 1173
! 1200 RGS(J)=RGAMMA*.5*(1.+COS(3.14159*(J-2.)/(NPS-1.)))		! 1174
!      DO 1201 J=1,NPN							! 1175
! 1201 RGN(J)=RGAMMA*.5*(1.+COS(3.14159*(J-2.)/(NPN-1.)))		! 1176
!C									! 1177
!C     VARIABLE HORIZONTAL DIFFUSION FACTORS...ALLOW UP TO 25.*AM	! 1178
!C     AND 25.*AH FOR THE COARSE REGIONS				! 1179
!C									! 1180
!      XIV(*)=1.							! 1181
!      XIT(*)=1.							! 1182
	DO 1205 i = 1,imt
	xiv(i) = 1.0
 1205	xit(i) = 1.0
      DYM=1./DYT(50)							! 1183
      DO 1210 J=1,JMT							! 1184
1210  XJT(J)=3.166*DYT(J)*DYM-2.166					! 1185
      XJV(JMT)=XJT(JMT)							! 1186
      DO 1211 J=1,JMTM1							! 1187
1211  XJV(J)=.5*(XJT(J)+XJT(J+1))					! 1188

	RETURN
	END
