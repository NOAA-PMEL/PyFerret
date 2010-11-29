c	rr_density.F
c
c     Rick Romea
c           Jan. 4, 2000
c
c  Returns density from in-situ temperature,
c  salinity, & depth using the UNESCO equation of state
c
c      call RR_unesco(temp,salinity,pressure,density)
c
c     input:
c       temp             in-situ temperature  [deg C]
c       salinity:        practical salinity units [approx. = ppt]
c       pressure:        bars
c     output:
c       density:         Kg/m^3
c
c***********************************************************************

      SUBROUTINE RR_unesco (t, s, p, rho)
      IMPLICIT NONE
      REAL   t,s,p,rho
      REAL*8 rw,rsto,xkw,xksto,xkstp 
      rw =     9.99842594d2 + 6.793952d-2*t - 9.095290d-3*t**2
     &        + 1.001685d-4*t**3 - 1.120083d-6*t**4 + 6.536332d-9*t**5
      rsto =   rw + (8.24493d-1 - 4.0899d-3*t + 7.6438d-5*t**2
     &        - 8.2467d-7*t**3 + 5.3875d-9*t**4) * s
     &       + (-5.72466d-3 + 1.0227d-4*t - 1.6546d-6*t**2) * s**1.5d0
     &       + 4.8314d-4 * s**2
      xkw =     1.965221d4 + 1.484206d2*t - 2.327105d0*t**2 +
     &         1.360477d-2*t**3 - 5.155288d-5*t**4
      xksto =   xkw + (5.46746d1 - 6.03459d-1*t + 1.09987d-2*t**2
     &        - 6.1670d-5*t**3) * s
     &       + (7.944d-2 + 1.6483d-2*t - 5.3009d-4*t**2) * s**1.5d0
      xkstp =   xksto + (3.239908d0 + 1.43713d-3*t + 1.16092d-4*t**2
     &        - 5.77905d-7*t**3) * p
     &       + (2.2838d-3 - 1.0981d-5*t - 1.6078d-6*t**2) * p * s
     &       + 1.91075d-4 * p * s**1.5d0
     &       + (8.50935d-5 - 6.12293d-6*t + 5.2787d-8*t**2) * p**2
     &       + (-9.9348d-7 + 2.0816d-8*t + 9.1697d-10*t**2) * p**2 * s
      rho = SNGL(rsto / (1.d0 - p/xkstp)) 
      END









