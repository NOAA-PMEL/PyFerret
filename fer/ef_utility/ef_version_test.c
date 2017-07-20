/* Make sure Python.h is included first */
#include <Python.h>
#include <stdlib.h>
#include <stdio.h>
#include "ferret.h"
#include "EF_Util.h"

/*
 * Test the EF version number of the Fortran EF against
 * the EF version number of the Ferret code.
 */
void FORTRAN(ef_version_test)(DFTYPE *version)
{
   int id;
   int int_version, ext_version;

   int_version = (int) (EF_VERSION * 100.0 + 0.5);
   ext_version = (int) (*version * 100.0 + 0.5);

   if ( ext_version != int_version ) {
      fprintf(stderr, "**ERROR version mismatch:\n"
                      "        External version [%4.2f] does not match \n"
                      "        Ferret version   [%4.2f].\n"
                      "        Please upgrade either Ferret or the\n"
                      "        External Function support files from\n"
                      "            http://tmap.pmel.noaa.gov/Ferret/\n\n",
                      *version, EF_VERSION);
      /* Do not know the id, so use -1 to skip getting the function name */
      id = -1;
      FORTRAN(ef_err_bail_out)(&id, "External function version number mismatch");
      /* The C function ef_err_bail_out should not return */
      abort();
   }
}

