      BLOCK DATA BIBO5
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
C**    @(#)bibo5.f	1.1    3/10/88
C**
C**
C***********************************************************************
C**
C**                 PLOT+ Scientific Graphics System
C**
C***********************************************************************
C**
C**
C
CC  DEFINITION OF INDEXES USED FOR BIBO HEADER FORMAT
C
      INTEGER NPTS,MIN,MAX,MEAN,PROJ,MOOR
      INTEGER INST,DTYPE,STIME,SRATE,SLAT,SLON
      INTEGER SDEP,INDEP,IADEP,FILT,CUTLO,CUTHI
      INTEGER WIND,SIG,HEAD,SCODE,PCODE,MODEN 
      INTEGER SPTIME
      COMMON /BIBO/ NPTS,MIN,MAX,MEAN,PROJ,MOOR
      COMMON /BIBO/ INST,DTYPE,STIME,SRATE,SLAT,SLON
      COMMON /BIBO/ SDEP,INDEP,IADEP,FILT,CUTLO,CUTHI
      COMMON /BIBO/ WIND,SIG,HEAD,SCODE,PCODE,MODEN
      COMMON /BIBO/ SPTIME
      DATA NPTS,MIN,MAX,MEAN,PROJ,MOOR /1,2,3,4,5,6/
      DATA INST,DTYPE,STIME,SRATE,SLAT,SLON /8,10,11,14,15,16/
      DATA SDEP,INDEP,IADEP,FILT,CUTLO,CUTHI /17,18,19,20,21,22/
      DATA WIND,SIG,HEAD,SCODE,PCODE,MODEN /23,24,25,131,132,133/
      DATA SPTIME/136/
      END
      BLOCK DATA TBLES

      COMMON /TBLE/PTAB(8,4),DTTAB(8,29),FTTAB(8,9),WTTAB(8,11)
      COMMON /TBLE/STTAB(8,19),PCTAB(8,5),UNIT(4,15)

      COMMON /NTBLE/NPTAB,NDTTAB,NFTTAB,NWTTAB
      COMMON /NTBLE/NSTTAB,NPCTAB,NUNIT

      DATA PTAB
     *   /2.0,0.0,4HCUE-,4HII  ,4H    ,4H    ,4H    ,4H    ,
     1    3.0,0.0,4HSUPE,4HR CO,4HDE  ,4H    ,4H    ,4H    ,
     2    4.0,0.0,4HCODE,4H    ,4H    ,4H    ,4H    ,4H    ,
     3    6.0,0.0,4HSLEE,4HP   ,4H    ,4H    ,4H    ,4H    /
      DATA DTTAB
     1   /20.0,6.0,4H    ,4HATMO,4HSPHE,4HRIC ,4HPRES,4HSURE,
     2    12.0,2.0,4H    ,4H  MA,4HJOR ,4HAXIS,4H CUR,4HRENT,
     3    13.0,2.0,4H    ,4H  MI,4HNOR ,4HAXIS,4H CUR,4HRENT,
     4    14.0,2.0,4H    ,4H  AL,4HONGS,4HHORE,4H CUR,4HRENT,
     5    15.0,2.0,4H    ,4H CRO,4HSS-S,4HHORE,4H CUR,4HRENT,
     6    26.0,8.0,4H    ,4H    ,4H WIN,4HD U ,4HVELO,4HCITY,
     7    27.0,8.0,4H    ,4H    ,4H WIN,4HD V ,4HVELO,4HCITY,
     8    28.0,7.0,4H    ,4H    ,4H    ,4H    ,4H   T,4HAU X,
     9    29.0,7.0,4H    ,4H    ,4H    ,4H    ,4H   T,4HAU Y,
     *    30.0,9.0,4H    ,4H    ,4H    ,4H   S,4HEA L,4HEVEL,
     1    31.0,1.0,4H    ,4HDRY ,4HBULB,4H TEM,4HPERA,4HTURE,
     2    16.0,8.0,4H    ,4H    ,4H MAJ,4HOR A,4HXIS ,4HWIND,
     3    35.0,1.0,4H    ,4H    ,4H    ,4H TEM,4HPERA,4HTURE,
     4    36.0,2.0,4H    ,4H    ,4H    ,4H  U ,4HVELO,4HCITY,
     5    37.0,2.0,4H    ,4H    ,4H    ,4H  V ,4HVELO,4HCITY,
     6    38.0,3.0,4H    ,4H    ,4H    ,4H    ,4HPRES,4HSURE,
     7    17.0,8.0,4H    ,4H    ,4H MIN,4HOR A,4HXIS ,4HWIND,
     8    43.0,10.,4H    ,4H    ,4H    ,4HCOND,4HUCTI,4HVITY,
     9    18.0,7.0,4H  MA,4HJOR ,4HAXIS,4H WIN,4HD ST,4HRESS,
     *    49.0,8.0,4H    ,4H    ,4H ALO,4HNGSH,4HORE ,4HWIND,
     1    50.0,8.0,4H    ,4H    ,4HCROS,4HS-SH,4HORE ,4HWIND,
     2    51.0,7.0,4H  AL,4HONGS,4HHORE,4H WIN,4HD ST,4HRESS,
     3    52.0,7.0,4H CRO,4HSS-S,4HHORE,4H WIN,4HD ST,4HRESS,
     4    53.0,0.0,4H  WI,4HND S,4HTRES,4HS DI,4HVERG,4HENCE,
     5    54.0,0.0,4H    ,4H    ,4HWIND,4H STR,4HESS ,4HCURL,
     6    55.0,9.0,4H    ,4H  AD,4HJUST,4HED S,4HEA L,4HEVEL,
     7    19.0,7.0,4H  MI,4HNOR ,4HAXIS,4H WIN,4HD ST,4HRESS,
     8    58.0,12.,4H    ,4H    ,4H    ,4HHEAT,4H CON,4HTENT,
     9    59.0,13.,4H    ,4H    ,4H  DY,4HNAMI,4HC HE,4HIGHT/
      DATA FTTAB
     1   / 0.0,0.0,4H    ,4H    ,4H    ,4H    ,4H  -N,4HONE-,
     2    10.0,0.0,4H    ,4H    ,4H RUN,4HNING,4H AVE,4HRAGE,
     3    11.0,0.0,4H    ,4H    ,4H    ,4H  LO,4HW CO,4HSINE,
     4    12.0,0.0,4H    ,4H    ,4H    ,4HLOW ,4HGAUS,4HSIAN,
     5    20.0,0.0,4H    ,4H    ,4H    ,4H    ,4HDEME,4HANED,
     6    21.0,0.0,4H    ,4H    ,4H    ,4H HIG,4HH CO,4HSINE,
     7    22.0,0.0,4H    ,4H    ,4H RAW,4H - L,4HOW G,4HAUSS,
     8    23.0,0.0,4H    ,4H    ,4HRAW ,4H- LO,4HW CO,4HSINE,
     9    8*0./
      DATA WTTAB
     1   / 0.0,0.0,4H    ,4H    ,4H    ,4H    ,4H  -N,4HONE-,
     2     2.0,0.0,4H    ,4H    ,4H    ,4H    ,4H  CO,4HSINE,
     3     3.0,0.0,4H    ,4H    ,4H    ,4H COS,4HINE ,4H** 2,
     4     4.0,0.0,4H    ,4H    ,4H    ,4H COS,4HINE ,4H** 3,
     5     5.0,0.0,4H    ,4H    ,4H    ,4H COS,4HINE ,4H** 4,
     6     6.0,0.0,4H    ,4H    ,4H    ,4H EXP,4HONEN,4HTIAL,
     7     7.0,0.0,4H    ,4H    ,4H    ,4H    ,4HGAUS,4HSIAN,
     8     8.0,0.0,4H    ,4H    ,4H    ,4H    ,4H TRI,4HPLET,
     9     9.0,0.0,4H    ,4H    ,4H    ,4H    ,4H HAM,4HMING,
     9    10.0,0.0,4H    ,4H    ,4H    ,4H LAN,4HCOZ ,4H** 2,
     *    8*0./
      DATA STTAB
     1    /0.0,0.0,4H    ,4H    ,4H    ,4H    ,4H    ,4H    ,
     2     1.0,0.0,4H    ,4HFOUR,4HIER ,4HCOEF,4HFICI,4HENTS,
     3     2.0,0.0,4H    ,4H    ,4H    ,4HAUTO,4HSPEC,4HTRUM,
     4     3.0,0.0,4H    ,4H    ,4H    ,4H  RE,4HGULA,4HR KE,
     5     4.0,0.0,4H    ,4H    ,4H   N,4HORMA,4HLIZE,4HD KE,
     6     5.0,0.0,4H    ,4H    ,4H    ,4H   C,4HW RO,4HTARY,
     7     6.0,0.0,4H    ,4H    ,4H    ,4H  CC,4HW RO,4HTARY,
     8     7.0,0.0,4H AUT,4HOSPE,4HCTRU,4HM CO,4HNFID,4HENCE,
     9     8.0,0.0,4H    ,4HCROS,4HS-SP,4HECTR,4HAL P,4HHASE,
     *     9.0,0.0,4HCROS,4HS-SP,4HECTR,4HAL C,4HOHER,4HENCE,
     1    10.0,0.0,4H    ,4H    ,4HPHAS,4HE CO,4HNFID,4HENCE,
     2    11.0,0.0,4H    ,4HCOHE,4HRENC,4HE CO,4HNFID,4HENCE,
     3    12.0,0.0,4H    ,4H    ,4H    ,4H  CO,4HSPEC,4HTRUM,
     4    13.0,0.0,4H    ,4H    ,4H    ,4H  QU,4HADRA,4HTURE,
     5    14.0,0.0,4H   C,4HOSPE,4HCTRU,4HM CO,4HNFID,4HENCE,
     6    15.0,0.0,4H   Q,4HUADR,4HATUR,4HE CO,4HNFID,4HENCE,
     7    16.0,0.0,4H    ,4H    ,4H    ,4H   A,4HMPLI,4HTUDE,
     8    17.0,0.0,4H    ,4H    ,4H    ,4H    ,4H   P,4HHASE,
     9    8*0./
      DATA PCTAB
     1   / 0.0,0.0,4H    ,4H    ,4H    ,4H    ,4H    ,4H    ,
     2     1.0,0.0,4H EMP,4HIRIC,4HAL D,4HECOM,4HPOSI,4HTION,
     3    11.0,0.0,4H DYN,4HAMIC,4HAL D,4HECOM,4HPOSI,4HTION,
     4    21.0,0.0,4H    ,4HCOMP,4HLEX ,4HDEMO,4HDULA,4HTION,
     5    8*0.0/
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
*      DATA UNIT
*     1  /1H ,1H ,1H ,1H ,
*     2   4H(DEG,4HREES,4H C) ,4H    ,
*     3   4H(CM/,4HSEC),4H    ,4H    ,
*     4   4H(DEC,4HIBAR,4HS)  ,4H    ,
*     5   4H(DEG,4HREES,4H)   ,4H    ,
*     6   4H(CM*,4HDEG ,4HC/SE,4HC)  ,
*     7   4H(MIL,4HLIBA,4HRS) ,4H    ,
*     8   4H(DYN,4HES/C,4HM**2,4H)   ,
*     9   4H(MET,4HERS/,4HSEC),4H    ,
*     *   4H(CEN,4HTIME,4HTERS,4H)   ,
*     1   4H(MMH,4HOS/C,4HM)  ,4H    ,
*     2   4H(0/0,4H0)  ,4H    ,4H    ,
*     3   4H(JOU,4HLES/,4HM**2,4H)   ,
*    4   4H(DYN,4H MET,4HERS),4H    ,
*    5   4*0./

      DATA NPTAB/4/,NDTTAB/29/,NFTTAB/8/,NWTTAB/10/
      DATA NSTTAB/18/,NPCTAB/4/,NUNIT/14/

      END
