#include <stdio.h>
#include <stdlib.h>
#include "binaryRead.h"

int FORTRAN(pl_read_var)(float *mem, int *lengths){
  return 0;
}

void FORTRAN(pl_get_error)(char *buf) {
}

int FORTRAN(pl_open)(char *script_name) {
  fprintf(stderr, "Attempt to use pl_open with Perl disabled\n");
  exit(1);
}

int FORTRAN(pl_close)() {
}

