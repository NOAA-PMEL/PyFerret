#include <stdio.h>

void my_close_(void ** lun)
{
   FILE * fp;
   fp = (FILE*)(*lun);
   //   printf("\nclose fp=%p", fp);
   if(fp!=NULL){
      fclose(fp);
      (FILE*)(*lun)=NULL;
   }
}

