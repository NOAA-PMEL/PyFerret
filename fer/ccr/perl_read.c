#ifdef USE_PERL
#include <EXTERN.h>
#include <perl.h>
#endif
#include <assert.h>
#include "binaryRead.h"
#include <stdio.h>

#ifdef USE_PERL
static char Errbuf[8192];
static PerlInterpreter *My_perl;

EXTERN_C void xs_init _((void));

EXTERN_C void boot_DynaLoader _((CV* cv));

EXTERN_C void
xs_init(void)
{
	char *file = __FILE__;
	dXSUB_SYS;

	/* DynaLoader is a special case */
	newXS("DynaLoader::boot_DynaLoader", boot_DynaLoader, file);
}

static void set_error(char *str, char *mess) {
  sprintf(Errbuf, str, mess);
}

/*
 * TODO -- error checking!
 */

get_data(float *val)
{
  char *result = 0;
  double dval;
  dSP;                            /* initialize stack pointer      */
  ENTER;                          /* everything created after here */
  SAVETMPS;                       /* ...is a temporary variable.   */
  PUSHMARK(SP);                   /* remember the stack pointer    */
  PUTBACK;                      /* make local stack pointer global */
  perl_call_pv("ferret_get_data", G_SCALAR | G_EVAL );
  SPAGAIN;                        /* refresh stack pointer         */
  result = SvPV(perl_get_sv("@", FALSE), PL_na);
  /* pop the return value from stack */
  dval = POPn;
  *val = dval;			/* Ferret needs a float */
  PUTBACK;
  FREETMPS;                       /* free that return value        */
  LEAVE;                       /* ...and the XPUSHed "mortal" args.*/
  assert(result != 0);
  if (result[0] != '\0'){
    set_error("%s", result);
    return 0;
  } else {
    return 1;
  }
}

int FORTRAN(pl_read_var)(float *mem, int *lengths){
  int i;
  int total = lengths[3] * lengths[2] * lengths[1] * lengths[0];
  float val = 0;
  for (i=0; i < total; i++){
    if (!get_data(&val)){
      return 0;
    }
    mem[i] = val;
  }
  return 1;
}

void FORTRAN(pl_get_error)(char *buf) {
  strcpy(buf, Errbuf);
  Errbuf[0] = '\0';
}

int FORTRAN(pl_open)(char *script_name) {
  char *my_argv[2];
  assert(My_perl == 0);
  my_argv[0] = "";
  my_argv[1] = script_name;

  My_perl = perl_alloc();
  if (!My_perl){
    set_error("%s","Couldn't allocate memory for Perl interpreter");
    return 0;
  }
  perl_construct( My_perl );
  if (perl_parse(My_perl, xs_init, 2, my_argv, (char **)NULL)){
    set_error("Couldn't parse script '%s'", script_name);
    return 0;
  }
  if (perl_run(My_perl)){
    set_error("Couldn't run script %s", script_name);
    return 0;
  }
    
  return 1;
}

int FORTRAN(pl_close)() {
  assert(My_perl != 0);
  perl_destruct(My_perl);
  perl_free(My_perl);
  My_perl = 0;
}

#ifdef DEBUG_PERL_READ
static int Lengths[] = {1,1,1,8};
static float Mem[8];

int main (int argc, char **argv, char **env)
{
  char errbuf[1024];
  if (!FORTRAN(pl_open)("getit.pl")){
    FORTRAN(pl_get_error)(errbuf);
    fprintf(stderr, "%s\n", errbuf);
    exit(1);
  }

  {
    int i=0;
    if (!FORTRAN(pl_read_var)(Mem, Lengths)){
      FORTRAN(pl_get_error)(errbuf);
      fprintf(stderr, "%s\n", errbuf);
      exit(1);
    }
    for(; i < 8; ++i){
      printf("%f\n", Mem[i]);
    }
  }
  FORTRAN(pl_close)();
}

#endif
#else /* ifdef USE_PERL */

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
#endif

