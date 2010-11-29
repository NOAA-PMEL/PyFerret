* 
*  pco2.F
* 
*  Andreas Schmittner (andreas@passagen.uni-kiel.de)
*  Mai 26th 2004
* 
*  Returns pCO2 of water
* 
* 
*


      SUBROUTINE pco2_init(id)

      INCLUDE 'ferret_cmn/EF_Util.cmn'

      INTEGER id, arg

* **********************************************************************
*                                            USER CONFIGURABLE PORTION |
*                                                                      |
*                                                                      V

      CALL ef_set_desc(id,
     .  'returns pCO2 (ppmv=uatm) using OCMIP routines' )

      CALL ef_set_num_args(id, 4)
      CALL ef_set_has_vari_args(id, NO)
      CALL ef_set_axis_inheritance(id, IMPLIED_BY_ARGS, 
     .     IMPLIED_BY_ARGS, IMPLIED_BY_ARGS, IMPLIED_BY_ARGS)
      CALL ef_set_piecemeal_ok(id, NO, NO, NO, NO)

      arg = 1
      CALL ef_set_arg_name(id, arg, 'TEMP')
      CALL ef_set_arg_unit(id, arg, 'deg C')
      CALL ef_set_arg_desc(id, arg, 'temperature')
      CALL ef_set_axis_influence(id, arg, YES, YES, YES, YES)

      arg = 2
      CALL ef_set_arg_name(id, arg, 'SALT')
      CALL ef_set_arg_unit(id, arg, 'su')
      CALL ef_set_arg_desc(id, arg, 'salinity')
      CALL ef_set_axis_influence(id, arg, YES, YES, YES, YES)

      arg = 3
      CALL ef_set_arg_name(id, arg, 'DIC')
      CALL ef_set_arg_unit(id, arg, 'mmol/m^3')
      CALL ef_set_arg_desc(id, arg, 'dissolved inorganic carbon')
      CALL ef_set_axis_influence(id, arg, YES, YES, YES, YES)

      arg = 4
      CALL ef_set_arg_name(id, arg, 'ALK')
      CALL ef_set_arg_unit(id, arg, 'mmol/m^3')
      CALL ef_set_arg_desc(id, arg, 'total alkalinity')
      CALL ef_set_axis_influence(id, arg, YES, YES, YES, YES)
*                                                                      ^
*                                                                      |
*                                            USER CONFIGURABLE PORTION |
* **********************************************************************

      RETURN 
      END


* 
*  In this subroutine we compute the result
* 
      SUBROUTINE pco2_compute(id, arg_1, arg_2, arg_3, arg_4, result)

      INCLUDE 'ferret_cmn/EF_Util.cmn'
      INCLUDE 'ferret_cmn/EF_mem_subsc.cmn'

      INTEGER id

      REAL bad_flag(EF_MAX_ARGS), bad_flag_result
      REAL arg_1(mem1lox:mem1hix, mem1loy:mem1hiy,
     .	   mem1loz:mem1hiz, mem1lot:mem1hit)
      REAL arg_2(mem2lox:mem2hix, mem2loy:mem2hiy,
     .     mem2loz:mem2hiz, mem2lot:mem2hit)
      REAL arg_3(mem3lox:mem3hix, mem3loy:mem3hiy,
     .     mem3loz:mem3hiz, mem3lot:mem3hit)
      REAL arg_4(mem4lox:mem4hix, mem4loy:mem4hiy,
     .     mem4loz:mem4hiz, mem4lot:mem4hit)
      REAL result(memreslox:memreshix, memresloy:memreshiy,
     .     memresloz:memreshiz, memreslot:memreshit)

* After initialization, the 'res_' arrays contain indexing information 
* for the result axes.  The 'arg_' arrays will contain the indexing 
* information for each variable's axes. 

      INTEGER res_lo_ss(4), res_hi_ss(4), res_incr(4)
      INTEGER arg_lo_ss(4,EF_MAX_ARGS), arg_hi_ss(4,EF_MAX_ARGS),
     .     arg_incr(4,EF_MAX_ARGS)


* **********************************************************************
*                                            USER CONFIGURABLE PORTION |
*                                                                      |
*                                                                      V

      INTEGER i,j,k,l
      INTEGER i1, j1, k1, l1
      INTEGER i2, j2, k2, l2
      INTEGER i3, j3, k3, l3
      INTEGER i4, j4, k4, l4
      real pt_in,sit_in,atmpres,phlo,phhi,co2ccn, ph, co2star, 
     .     co2starair, dco2star, pco2surf, dpco2, pco2atm

      CALL ef_get_res_subscripts(id, res_lo_ss, res_hi_ss, res_incr)
      CALL ef_get_arg_subscripts(id, arg_lo_ss, arg_hi_ss, arg_incr)
      CALL ef_get_bad_flags(id, bad_flag, bad_flag_result)

      co2ccn=280.
      pt_in=0.5125e-3           !mol/m^3
      atmpres=1.0               !atm
      sit_in=7.6875e-03         !mol/m^3
      phlo=6.
      phhi=9.

      i1 = arg_lo_ss(X_AXIS,ARG1)
      i2 = arg_lo_ss(X_AXIS,ARG2)
      i3 = arg_lo_ss(X_AXIS,ARG3)
      i4 = arg_lo_ss(X_AXIS,ARG4)
      DO 400 i=res_lo_ss(X_AXIS), res_hi_ss(X_AXIS)

         j1 = arg_lo_ss(Y_AXIS,ARG1)
         j2 = arg_lo_ss(Y_AXIS,ARG2)
         j3 = arg_lo_ss(Y_AXIS,ARG3)
         j4 = arg_lo_ss(Y_AXIS,ARG4)
         DO 300 j=res_lo_ss(Y_AXIS), res_hi_ss(Y_AXIS)

            k1 = arg_lo_ss(Z_AXIS,ARG1)
            k2 = arg_lo_ss(Z_AXIS,ARG2)
            k3 = arg_lo_ss(Z_AXIS,ARG3)
            k4 = arg_lo_ss(Z_AXIS,ARG4)
            DO 200 k=res_lo_ss(Z_AXIS), res_hi_ss(Z_AXIS)

            l1 = arg_lo_ss(T_AXIS,ARG1)
            l2 = arg_lo_ss(T_AXIS,ARG2)
            l3 = arg_lo_ss(T_AXIS,ARG3)
            l4 = arg_lo_ss(T_AXIS,ARG4)
            DO 100 l=res_lo_ss(T_AXIS), res_hi_ss(T_AXIS)


                  IF ( arg_1(i1,j1,k1,l1) .EQ. bad_flag(1) .OR. 
     .                 arg_2(i2,j2,k2,l2) .EQ. bad_flag(2) .OR.
     .                 arg_3(i3,j3,k3,l3) .EQ. bad_flag(3) .OR.
     .                 arg_4(i4,j4,k4,l4) .EQ. bad_flag(4) ) THEN

                     result(i,j,k,l) = bad_flag_result

                  ELSE

                     call co2calc(arg_1(i1,j1,k1,l1)
     &                    ,arg_2(i2,j2,k2,l2),arg_3(i3,j3,k3,l3)
     &                    ,arg_4(i4,j4,k4,l4),pt_in,sit_in
     &                    ,phlo,phhi,ph,co2ccn,atmpres,co2star
     &                    ,co2starair,dco2star,pCO2surf,dpco2
     &                    ,pCO2atm)
                     result(i,j,k,l) = pCO2surf

                  END IF


                  l1 = l1 + arg_incr(T_AXIS,ARG1)
                  l2 = l2 + arg_incr(T_AXIS,ARG2)
                  l3 = l3 + arg_incr(T_AXIS,ARG3)
                  l4 = l4 + arg_incr(T_AXIS,ARG4)
 100           CONTINUE

               k1 = k1 + arg_incr(Z_AXIS,ARG1)
               k2 = k2 + arg_incr(Z_AXIS,ARG2)
               k3 = k3 + arg_incr(Z_AXIS,ARG3)
               k4 = k4 + arg_incr(Z_AXIS,ARG4)
 200        CONTINUE

            j1 = j1 + arg_incr(Y_AXIS,ARG1)
            j2 = j2 + arg_incr(Y_AXIS,ARG2)
            j3 = j3 + arg_incr(Y_AXIS,ARG3)
            j4 = j4 + arg_incr(Y_AXIS,ARG4)
 300     CONTINUE

         i1 = i1 + arg_incr(X_AXIS,ARG1)
         i2 = i2 + arg_incr(X_AXIS,ARG2)
         i3 = i3 + arg_incr(X_AXIS,ARG3)
         i4 = i4 + arg_incr(X_AXIS,ARG4)
 400  CONTINUE
      
         
*                                                                      ^
*                                                                      |
*                                            USER CONFIGURABLE PORTION |
* **********************************************************************

      RETURN 
      END

c_ ---------------------------------------------------------------------
c_ RCS lines preceded by "c_ "
c_ ---------------------------------------------------------------------
c_
c_ $Source$ 
c_ $Revision$
c_ $Date$   ;  $State$
c_ $Author$ ;  $Locker$
c_
c_ ---------------------------------------------------------------------
c_ $Log$
c_ Revision 1.1  2004/06/01 17:38:06  ansley
c_ add pco2 function from Andreas Schmittner
c_
c_ Revision 1.8  1999/07/16 11:40:33  orr
c_ Modifications by Keith Lindsay to fix inconsistency with common block
c_ "species" (not the same in "ta_iter_1.f").
c_ Also comment lines changed/added by J. Orr.
c_
c_ Revision 1.7  1999/04/26  13:04:54  orr
c_ Modified USAGE comment, to include new arguments
c_
c_ Revision 1.6  1999/04/14 12:55:52  orr
c_ Changed units for input arguments for tracers:
c_ formerly in mol/metric ton (T); now in mol/m^3.
c_ Used 1024.5 kg/m^3 as a constant conversion factor.
c_ Modelers can now pass tracers on a per volume basis, as carried in models.
c_
c_ Revision 1.5  1999/04/06 16:57:37  orr
c_ Changed calc of dpCO2 to account for diff atm pressure
c_
c_ Revision 1.4  1999/04/06 13:17:58  orr
c_ Added 2 output arguments: pCO2surf and dpCO2
c_ (see section 4 of Biotic HOWTO)
c_
c_ Revision 1.3  1999/04/05 15:59:11  orr
c_ Cleaned up comments regarding units
c_
c_ Revision 1.2  1999/04/04 02:35:00  orr
c_ Changed units for input and output tracers:
c_ previously in umol/kg; now in mol/T
c_ Can also pass input and output in mol/m^3 with little error.
c_
c_ Revision 1.1  1999/04/03  21:59:56  orr
c_ Initial revision
c_
c_ ---------------------------------------------------------------------
c_ 
      subroutine co2calc(t,s,dic_in,ta_in,pt_in,sit_in
     &                  ,phlo,phhi,ph,xtco2in,atmpres,co2star
     &                  ,co2starair,dco2star,pCO2surf,dpCO2
     &                  ,pCO2atm)
C
C-------------------------------------------------------------------------
C SUBROUTINE CO2CALC
C
C PURPOSE
C       Calculate delta co2* from total alkalinity and total CO2 at
C temperature (t), salinity (s) and "atmpres" atmosphere total pressure. 
C
C USAGE
C       call co2calc(t,s,dic_in,ta_in,pt_in,sit_in
C    &                  ,phlo,phhi,ph,xco2_in,atmpres
C    &                  ,co2star,dco2star,pCO2surf,dpco2)
C
C INPUT
C       dic_in = total inorganic carbon (mol/m^3) 
C                where 1 T = 1 metric ton = 1000 kg
C       ta_in  = total alkalinity (eq/m^3) 
C       pt_in  = inorganic phosphate (mol/m^3) 
C       sit_in = inorganic silicate (mol/m^3) 
C       t      = temperature (degrees C)
C       s      = salinity (PSU)
C       phlo   = lower limit of pH range
C       phhi   = upper limit of pH range
C       xco2_in=atmospheric mole fraction CO2 in dry air (ppmv) 
C       atmpres= atmospheric pressure in atmospheres (1 atm==1013.25mbar)
C
C       Note: arguments dic_in, ta_in, pt_in, sit_in, and xco2_in are 
C             used to initialize variables dic, ta, pt, sit, and xco2.
C             * Variables dic, ta, pt, and sit are in the common block 
C               "species".
C             * Variable xco2 is a local variable.
C             * Variables with "_in" suffix have different units 
C               than those without.

C OUTPUT
C       co2star  = CO2*water (mol/m^3)
C       dco2star = delta CO2 (mol/m^3)
c       pco2surf = oceanic pCO2 (ppmv)
c       dpco2    = Delta pCO2, i.e, pCO2ocn - pCO2atm (ppmv)
C
C IMPORTANT: Some words about units - (JCO, 4/4/1999)
c     - Models carry tracers in mol/m^3 (on a per volume basis)
c     - Conversely, this routine, which was written by observationalists 
c       (C. Sabine and R. Key), passes input arguments in umol/kg  
c       (i.e., on a per mass basis)
c     - I have changed things slightly so that input arguments are in mol/m^3,
c     - Thus, all input concentrations (dic_in, ta_in, pt_in, and st_in) 
c       should be given in mol/m^3; output arguments "co2star" and "dco2star"  
c       are likewise in mol/m^3.

C FILES and PROGRAMS NEEDED
C       drtsafe
C       ta_iter_1
C
C--------------------------------------------------------------------------
C
        include "species.h"
        real invtk,is,is2,xtco2in
c        real k0,k1,k2,kkw,kb,ks,kf,k1p,k2p,k3p,ksi
        real drtsafe
        real t, s, dic_in, ta_in, pt_in, sit_in, phlo, phhi, ph, 
     .       atmpres, co2star, co2starair, dco2star, pco2surf, dpco2, 
     .       pco2atm, permil, permeg, xtco2, tk, tk100, tk1002, dlogtk, 
     .       sqrtis, s2, sqrts, s15, scl, ff, x1, x2, xacc, htotal, 
     .       htotal2

        external ta_iter_1
C

c       ---------------------------------------------------------------------
C       Change units from the input of mol/m^3 -> mol/kg:
c       (1 mol/m^3)  x (1 m^3/1024.5 kg)
c       where the ocean's mean surface density is 1024.5 kg/m^3
c       Note: mol/kg are actually what the body of this routine uses 
c       for calculations.  
c       ---------------------------------------------------------------------
        permil = 1.0 / 1024.5
c       To convert input in mol/m^3 -> mol/kg 
        pt=pt_in*permil
        sit=sit_in*permil
        ta=ta_in*permil
        dic=dic_in*permil

c       ---------------------------------------------------------------------
C       Change units from uatm to atm. That is, atm is what the body of 
c       this routine uses for calculations.
c       ---------------------------------------------------------------------
        permeg=1.e-6
c       To convert input in uatm -> atm
        xtco2=xtco2in*permeg
c       ---------------------------------------------------------------------
C
C*************************************************************************
C Calculate all constants needed to convert between various measured
C carbon species. References for each equation are noted in the code. 
C Once calculated, the constants are
C stored and passed in the common block "const". The original version of this
C code was based on the code by Dickson in Version 2 of "Handbook of Methods
C for the Analysis of the Various Parameters of the Carbon Dioxide System
C in Seawater", DOE, 1994 (SOP No. 3, p25-26). 
C
C Derive simple terms used more than once
C
        tk = 273.15 + t
        tk100 = tk/100.0
        tk1002=tk100*tk100
        invtk=1.0/tk
        dlogtk=log(tk)
        is=19.924*s/(1000.-1.005*s)
        is2=is*is
        sqrtis=sqrt(is)
        s2=s*s
        sqrts=sqrt(s)
        s15=s**1.5
        scl=s/1.80655
C
C f = k0(1-pH2O)*correction term for non-ideality
C
C Weiss & Price (1980, Mar. Chem., 8, 347-359; Eq 13 with table 6 values)
C
        ff = exp(-162.8301 + 218.2968/tk100  +
     &          90.9241*log(tk100) - 1.47696*tk1002 +
     &          s * (.025695 - .025225*tk100 + 
     &          0.0049867*tk1002))
C
C K0 from Weiss 1974
C
        k0 = exp(93.4517/tk100 - 60.2409 + 23.3585 * log(tk100) +
     &          s * (.023517 - 0.023656 * tk100 + 0.0047036 * tk1002))

C
C kk1 = [H][HCO3]/[H2CO3]
C k2 = [H][CO3]/[HCO3]
C
C Millero p.664 (1995) using Mehrbach et al. data on seawater scale 
C
        kk1=10**(-1*(3670.7*invtk - 62.008 + 9.7944*dlogtk -
     &          0.0118 * s + 0.000116*s2))
C
        k2=10**(-1*(1394.7*invtk + 4.777 - 
     &          0.0184*s + 0.000118*s2))
C
C kb = [H][BO2]/[HBO2]
C
C Millero p.669 (1995) using data from Dickson (1990)
C
        kb=exp((-8966.90 - 2890.53*sqrts - 77.942*s +
     &          1.728*s15 - 0.0996*s2)*invtk +
     &          (148.0248 + 137.1942*sqrts + 1.62142*s) +
     &          (-24.4344 - 25.085*sqrts - 0.2474*s) *
     &          dlogtk + 0.053105*sqrts*tk)
C
C k1p = [H][H2PO4]/[H3PO4]
C
C DOE(1994) eq 7.2.20 with footnote using data from Millero (1974)
C
        k1p = exp(-4576.752*invtk + 115.525 - 18.453 * dlogtk +
     &          (-106.736*invtk + 0.69171) * sqrts +
     &          (-0.65643*invtk - 0.01844) * s)
C
C k2p = [H][HPO4]/[H2PO4]
C
C DOE(1994) eq 7.2.23 with footnote using data from Millero (1974))
C
        k2p = exp(-8814.715*invtk + 172.0883 - 27.927 * dlogtk +
     &          (-160.340*invtk + 1.3566) * sqrts +
     &          (0.37335*invtk - 0.05778) * s)
C
C------------------------------------------------------------------------
C k3p = [H][PO4]/[HPO4]
C
C DOE(1994) eq 7.2.26 with footnote using data from Millero (1974)
C
        k3p = exp(-3070.75*invtk - 18.141 + 
     &          (17.27039*invtk + 2.81197) *
     &          sqrts + (-44.99486*invtk - 0.09984) * s)
C
C------------------------------------------------------------------------
C ksi = [H][SiO(OH)3]/[Si(OH)4]
C
C Millero p.671 (1995) using data from Yao and Millero (1995)
C
        ksi = exp(-8904.2*invtk + 117.385 - 19.334 * dlogtk +
     &          (-458.79*invtk + 3.5913) * sqrtis +
     &          (188.74*invtk - 1.5998) * is +
     &          (-12.1652*invtk + 0.07871) * is2 +
     &          log(1.0-0.001005*s))
C
C------------------------------------------------------------------------
C kkw = [H][OH]
C
C Millero p.670 (1995) using composite data
C
        kkw = exp(-13847.26*invtk + 148.9652 - 23.6521 * dlogtk +
     &          (118.67*invtk - 5.977 + 1.0495 * dlogtk) *
     &          sqrts - 0.01615 * s)
C
C------------------------------------------------------------------------
C ks = [H][SO4]/[HSO4]
C
C Dickson (1990, J. chem. Thermodynamics 22, 113)
C
        ks=exp(-4276.1*invtk + 141.328 - 23.093*dlogtk +
     &          (-13856*invtk + 324.57 - 47.986*dlogtk) * sqrtis +
     &          (35474*invtk - 771.54 + 114.723*dlogtk) * is -
     &          2698*invtk*is**1.5 + 1776*invtk*is2 +
     &          log(1.0 - 0.001005*s))
C
C------------------------------------------------------------------------
C kf = [H][F]/[HF]
C
C Dickson and Riley (1979) -- change pH scale to total
C
        kf=exp(1590.2*invtk - 12.641 + 1.525*sqrtis +
     &          log(1.0 - 0.001005*s) + 
     &          log(1.0 + (0.1400/96.062)*(scl)/ks))
C
C------------------------------------------------------------------------
C Calculate concentrations for borate, sulfate, and fluoride
C
C Uppstrom (1974)
        bt = 0.000232 * scl/10.811
C Morris & Riley (1966)
        st = 0.14 * scl/96.062
C Riley (1965)
        ft = 0.000067 * scl/18.9984
c        write(*,*) 'co2calc'
c        write(*,*) k0,kk1,k2,kkw,kb,ks,kf,k1p,k2p,k3p,ksi
c        write(*,*) ff,htotal
c        write(*,*) bt,st,ft,sit,pt,ta

C*************************************************************************
C
C Calculate [H+] total when DIC and TA are known at T, S and 1 atm.
C The solution converges to err of xacc. The solution must be within
C the range x1 to x2.
C
C If DIC and TA are known then either a root finding or iterative method
C must be used to calculate htotal. In this case we use the Newton-Raphson
C "safe" method taken from "Numerical Recipes" (function "rtsafe.f" with
C error trapping removed).
C
C As currently set, this procedure iterates about 12 times. The x1 and x2
C values set below will accomodate ANY oceanographic values. If an initial
C guess of the pH is known, then the number of iterations can be reduced to
C about 5 by narrowing the gap between x1 and x2. It is recommended that
C the first few time steps be run with x1 and x2 set as below. After that,
C set x1 and x2 to the previous value of the pH +/- ~0.5. The current
C setting of xacc will result in co2star accurate to 3 significant figures
C (xx.y). Making xacc bigger will result in faster convergence also, but this
C is not recommended (xacc of 10**-9 drops precision to 2 significant figures).
C
C Parentheses added around negative exponents (Keith Lindsay)
C
        x1 = 10.0**(-phhi)
        x2 = 10.0**(-phlo)
        xacc = 1.e-10
        htotal=drtsafe(ta_iter_1,x1,x2,xacc)

c        write(*,*) 'htotal', htotal 
C
C Calculate [CO2*] as defined in DOE Methods Handbook 1994 Ver.2, 
C ORNL/CDIAC-74, Dickson and Goyet, eds. (Ch 2 p 10, Eq A.49)
C
        htotal2=htotal*htotal
        co2star = 1.0
        co2star=dic*htotal2/(htotal2 + kk1*htotal + kk1*k2)
        co2starair=xtco2*ff*atmpres
        dco2star=co2starair-co2star
        ph=-log10(htotal)

c        write(*,*) 'co2calc'
c        write(*,*) k12,k12p,k123p
c        write(*,*) k0,kk1,k2,kkw,kb,ks,kf,k1p,k2p,k3p,ksi
c        write(*,*) 'co2calc',ff,htotal
c        write(*,*) bt,st,ft,sit,pt,ta
c        write(*,*) 'CO2CALC' 
c        write(*,*) 'starair, star', co2starair, co2star
c        write(*,*) 'dic, dic_in', dic, dic_in
c        write(*,*) 'dco2star', dco2star
c        write(*,*) 'ph',ph

c
c       ---------------------------------------------------------------
cc      Add two output arguments for storing pCO2surf
cc      Should we be using K0 or ff for the solubility here?
c       ---------------------------------------------------------------
        pCO2surf = co2star / ff
        dpCO2    = pCO2surf - xtco2*atmpres
        pCO2atm = (pCO2surf - dpCO2) / permeg
c        dco2star = dpCO2*ff
C
C  Convert units of output arguments
c      Note: co2star and dco2star are calculated in mol/kg within this routine 
c      Thus Convert now from mol/kg -> mol/m^3
       co2star  = co2star / permil
       dco2star = dco2star / permil


c      Note: pCO2surf and dpCO2 are calculated in atm above. 
c      Thus convert now to uatm
       pCO2surf = pCO2surf / permeg
       dpCO2    = dpCO2 / permeg
c       dco2star = dco2star/ permeg
C
        return
        end

! source file: /home/andreas/models/UVic/2.6/npzd_co2/ctr/updates/ta_iter_1.F
c_ ---------------------------------------------------------------------
c_ RCS lines preceded by "c_ "
c_ ---------------------------------------------------------------------
c_
c_ $Source$
c_ $Revision$
c_ $Date$   ;  $State$
c_ $Author$ ;  $Locker$
c_
c_ ---------------------------------------------------------------------
c_ $Log$
c_ Revision 1.1  2004/06/01 17:38:06  ansley
c_ add pco2 function from Andreas Schmittner
c_
c_ Revision 1.2  1999/09/01 17:55:41  orr
c_ Fixed sign error in dfn/dx following remarks of C. Voelker (10/Aug/1999)
c_
c_ Revision 1.1  1999/04/03 22:00:42  orr
c_ Initial revision
c_
c_ ---------------------------------------------------------------------
c_

        subroutine ta_iter_1(x,fn,df)
      include "species.h"
        real k12,k12p,k123p
        real  x, fn, df, x2, x3, c, a, a2, da, b, b2, db
c        real k0,k1,k2,kw,kb,ks,kf,k1p,k2p,k3p,ksi
C
C This routine expresses TA as a function of DIC, htotal and constants.
C It also calculates the derivative of this function with respect to
C htotal. It is used in the iterative solution for htotal. In the call
C "x" is the input value for htotal, "fn" is the calculated value for TA
C and "df" is the value for dTA/dhtotal
C
c        write(*,*) 'ta_iter_1'
c        write(*,*) k0,k1,k2,kw,kb,ks,kf,k1p,k2p,k3p,ksi
c        write(*,*) ff,htotal
c        write(*,*) bt,st,ft,sit,pt,ta
c        write(*,*) x

	x2=x*x
	x3=x2*x
	k12 = kk1*k2
	k12p = k1p*k2p
	k123p = k12p*k3p
	c = 1.0 + st/ks
	a = x3 + k1p*x2 + k12p*x + k123p
	a2=a*a
	da = 3.0*x2 + 2.0*k1p*x + k12p
	b = x2 + kk1*x + k12
	b2=b*b
	db = 2.0*x + kk1

C
C	fn = hco3+co3+borate+oh+hpo4+2*po4+silicate+hfree+hso4+hf+h3po4-ta
C
	fn = kk1*x*dic/b +
     &	     2.0*dic*k12/b +
     &	     bt/(1.0 + x/kb) +
     &	     kkw/x +
     &	     pt*k12p*x/a +
     &	     2.0*pt*k123p/a +
     &	     sit/(1.0 + x/ksi) -
     &	     x/c -
     &	     st/(1.0 + ks/(x*c)) -
     &	     ft/(1.0 + kf/x) -
     &	     pt*x3/a -
     &	     ta

C
C	df = dfn/dx
C
	df = ((kk1*dic*b) - kk1*x*dic*db)/b2 -
     &	     2.0*dic*k12*db/b2 -
     &	     bt/kb/(1.0+x/kb)**2. -
     &	     kkw/x2 +
     &	     (pt*k12p*(a - x*da))/a2 -
     &	     2.0*pt*k123p*da/a2 -
     &	     sit/ksi/(1.0+x/ksi)**2. -
     &	     1.0/c +
     &	     st*(1.0 + ks/(x*c))**(-2.0)*(ks/(c*x2)) +
     &	     ft*(1.0 + kf/x)**(-2.)*kf/x2 -
     &	     pt*x2*(3.0*a-x*da)/a2

c        write(*,*) k12,k12p,k123p
c        write(*,*) fn, df

	return
	end


! source file: /home/andreas/models/UVic/2.6/npzd_co2/ctr/updates/drtsafe.F
c_ ---------------------------------------------------------------------
c_ RCS lines preceded by "c_ "
c_ ---------------------------------------------------------------------
c_
c_ $Source$
c_ $Revision$
c_ $Date$   ;  $State$
c_ $Author$ ;  $Locker$
c_
c_ ---------------------------------------------------------------------
c_ $Log$
c_ Revision 1.1  2004/06/01 17:38:06  ansley
c_ add pco2 function from Andreas Schmittner
c_
c_ Revision 1.1  1999/04/03 22:00:42  orr
c_ Initial revision
c_
c_ ---------------------------------------------------------------------
c_
       REAL FUNCTION DRTSAFE(FUNCD,X1,X2,XACC)
      include "species.h"

      real funcd, x1, x2, xacc, fl, df, fh, xl, xh, swap, dxold,
     .     dx, f, temp
      integer j, maxit
C
C	File taken from Numerical Recipes. Modified  R.M.Key 4/94
C
      MAXIT=100
      CALL FUNCD(X1,FL,DF)
      CALL FUNCD(X2,FH,DF)
      IF(FL .LT. 0.0) THEN
        XL=X1
        XH=X2
      ELSE
        XH=X1
        XL=X2
        SWAP=FL
        FL=FH
        FH=SWAP
      END IF
      DRTSAFE=.5*(X1+X2)
      DXOLD=ABS(X2-X1)
      DX=DXOLD
      CALL FUNCD(DRTSAFE,F,DF)
      DO 100, J=1,MAXIT
        IF(((DRTSAFE-XH)*DF-F)*((DRTSAFE-XL)*DF-F) .GE. 0.0 .OR.
     &	      ABS(2.0*F) .GT. ABS(DXOLD*DF)) THEN
          DXOLD=DX
          DX=0.5*(XH-XL)
          DRTSAFE=XL+DX
          IF(XL .EQ. DRTSAFE)RETURN
        ELSE
          DXOLD=DX
          DX=F/DF
          TEMP=DRTSAFE
          DRTSAFE=DRTSAFE-DX
          IF(TEMP .EQ. DRTSAFE)RETURN
	END IF
        IF(ABS(DX) .LT. XACC)RETURN
        CALL FUNCD(DRTSAFE,F,DF)
        IF(F .LT. 0.0) THEN
          XL=DRTSAFE
          FL=F
        ELSE
          XH=DRTSAFE
          FH=F
        END IF
  100  CONTINUE
      RETURN
      END
