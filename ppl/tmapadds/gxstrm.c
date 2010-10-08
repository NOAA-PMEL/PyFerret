/* Included in the Ferret source repository only for reference */
/* Code is available at
   http://grads.sourcearchive.com/documentation/2.0.a7.1-3/main.html */


#include <stdio.h>
#include <malloc.h>
#include <math.h>
#include "gx.h"

void gxstrm (float *u, float *v, float *c, int is, int js,
   float uund, float vund, float cund, int flag, float *shdlvs,
   int *shdcls, int shdcnt, int den) {
float *up, *vp; 
float x,y,xx,yy,uv1,uv2,uv,vv1,vv2,vv,auv,avv,xsav,ysav,xold,yold;
int i,j,ii,jj,ipen,ii1,ij1,i2,j2,ipt,acnt,icol,scol,kk,dis;
int *it,siz,iacc,iisav,iscl,imn,imx,jmn,jmx,iz,jz,iss,jss,bflg;
float fact,rscl,xxsv,yysv,*cp,cv1,cv2,cv;

  scol = -9;
  icol = 1;

  /* Figure out the interval for the flag grid */

  i = is;
  if (js>i) i = js;
  iscl = 200/i;
  iscl = iscl + den - 5;
  if (iscl<1) iscl=1;
  if (iscl>10) iscl=10;
  fact = 0.5/((float)iscl);
  rscl = (float)iscl;
 
  /* Allocate memory for the flag grid */

  iss = is*iscl; jss = js*iscl;
  siz = iss*jss;
  it = (int *)malloc(sizeof(int) * siz);
  if (it==NULL) {
    printf ("Cannot allocate memory for streamline function\n");
    return;
  }
  for (i=0; i<siz; i++) *(it+i) = 0;

  /* Loop through flag grid to look for start of streamlines.  
     To start requires no streams drawn within surrounding 
     flag boxes.  */

  i2 = 0;
  j2 = 0;
  for (i=0; i<siz; i++) {
    dis = 2;
    if (den<5) dis = 3;
    if (den>5) dis = 1;
    imn = i2-dis; imx = i2+dis+1; 
    jmn = j2-dis; jmx = j2+dis+1;
    if (imn<0) imn = 0;
    if (imx>iss) imx = iss;
    if (jmn<0) jmn = 0;
    if (jmx>jss) jmx = jss;
    iacc = 0;
    for (jz=jmn; jz<jmx; jz++) {
      ipt = jz*iss+imn;
      for (iz=imn; iz<imx; iz++) {
        iacc = iacc + *(it+ipt);
        ipt++;
      }
    }
    if (iacc==0){
      x = ((float)i2)/rscl;
      y = ((float)j2)/rscl;
      xsav = x;
      ysav = y;
      gxconv (x+1.0,y+1.0,&xx,&yy,3);
      gxplot (xx,yy,3);
      xxsv = xx; yysv = yy;
      iisav = -999;
      iacc = 0;
      acnt = 0;
      bflg = 0;
      while (x>=0.0 && x<(float)(is-1) && y>=0.0 && y<(float)(js-1)) {
        ii = (int)x;
        jj = (int)y;
        xx = x - (float)ii;
        yy = y - (float)jj;
        up = u + jj*is+ii;      
        vp = v + jj*is+ii;
        if (*up==uund || *(up+1)==uund ||
            *(up+is)==uund || *(up+is+1)==uund) break;
        if (*vp==vund || *(vp+1)==vund ||
            *(vp+is)==vund || *(vp+is+1)==vund) break;
        if (flag) {
          cp = c + jj*is+ii;
          if (*cp==cund || *(cp+1)==cund ||
              *(cp+is)==cund || *(cp+is+1)==cund) icol = 15;
          else {
            cv1 = *cp + (*(cp+1)-*cp)*xx;
            cv2 = *(cp+is) + (*(cp+is+1)-*(cp+is))*xx;
            cv = cv1 + (cv2-cv1)*yy;
            icol = gxshdc(shdlvs,shdcls,shdcnt,cv);
          }
          if (icol!=scol && icol>-1) gxcolr(icol);
          scol = icol;
        }
        uv1 = *up + (*(up+1)-*up)*xx;
        uv2 = *(up+is) + (*(up+is+1)-*(up+is))*xx;
        uv = uv1 + (uv2-uv1)*yy;
        vv1 = *vp + (*(vp+1)-*vp)*xx;
        vv2 = *(vp+is) + (*(vp+is+1)-*(vp+is))*xx;
        vv = vv1 + (vv2-vv1)*yy;
        auv = fabs(uv); avv=fabs(vv);
        if (auv<0.1 && avv<0.1) break;
        if (auv>avv) {
          vv = vv*fact/auv;
          uv = uv*fact/auv;
        } else {
          uv = uv*fact/avv;
          vv = vv*fact/avv;
        }
        x = x + uv;
        y = y + vv;
        ii1 = (int)(x*rscl);
        ij1 = (int)(y*rscl);
        ii1 = ij1*iss + ii1;
        if (ii1<0 || ii1>=siz) break;
        if (*(it+ii1)==1) break;
        if (ii1!=iisav && iisav>-1) *(it+iisav) = 1;
        if (ii1==iisav) iacc++;
        else iacc = 0;
        if (iacc>10) break;
        iisav = ii1;
        gxconv (x+1.0,y+1.0,&xx,&yy,3);
        if (icol>-1) {
          if (bflg) {gxplot(xold,yold,3); bflg=0;}
          gxplot (xx,yy,2);
        } else bflg = 1;
        xold = xx;
        yold = yy;
        acnt++;
        if (acnt>20) {
          if (icol>-1) strmar (xxsv,yysv,xx,yy);
          acnt = 0;
        }
        xxsv = xx; yysv = yy;
      }
      bflg = 0;
      x = xsav; y = ysav;
      gxconv (x+1.0,y+1.0,&xx,&yy,3);
      gxplot (xx,yy,3);
      xxsv = xx;
      yysv = yy;
      iisav = -999;
      iacc = 0;
      acnt = 19;
      while (x>=0.0 && x<(float)(is-1) && y>=0.0 && y<(float)(js-1)) {
        ii = (int)x;
        jj = (int)y;
        xx = x - (float)ii;
        yy = y - (float)jj;
        up = u + jj*is+ii;      
        vp = v + jj*is+ii;
        if (*up==uund || *(up+1)==uund ||
            *(up+is)==uund || *(up+is+1)==uund) break;
        if (*vp==vund || *(vp+1)==vund ||
            *(vp+is)==vund || *(vp+is+1)==vund) break;
        if (flag) {
          cp = c + jj*is+ii;
          if (*cp==cund || *(cp+1)==cund ||
              *(cp+is)==cund || *(cp+is+1)==cund) icol = 15;
          else {
            cv1 = *cp + (*(cp+1)-*cp)*xx;
            cv2 = *(cp+is) + (*(cp+is+1)-*(cp+is))*xx;
            cv = cv1 + (cv2-cv1)*yy;
            icol = gxshdc(shdlvs,shdcls,shdcnt,cv);
          }
          if (icol!=scol && icol>-1) gxcolr(icol);
          scol = icol;
        }
        uv1 = *up + (*(up+1)-*up)*xx;
        uv2 = *(up+is) + (*(up+is+1)-*(up+is))*xx;
        uv = uv1 + (uv2-uv1)*yy;
        vv1 = *vp + (*(vp+1)-*vp)*xx;
        vv2 = *(vp+is) + (*(vp+is+1)-*(vp+is))*xx;
        vv = vv1 + (vv2-vv1)*yy;
        auv = fabs(uv); avv=fabs(vv);
        if (auv<0.1 && avv<0.1) break;
        if (auv>avv) {
          vv = vv*fact/auv;
          uv = uv*fact/auv;
        } else {
          uv = uv*fact/avv;
          vv = vv*fact/avv;
        }
        x = x - uv;
        y = y - vv;
        ii1 = (int)(x*rscl);
        ij1 = (int)(y*rscl);
        ii1 = ij1*iss + ii1;
        if (ii1<0 || ii1>=siz) break;
        if (*(it+ii1)==1) break;
        if (ii1!=iisav && iisav>-1) *(it+iisav) = 1;
        if (ii1==iisav) iacc++;
        else iacc = 0;
        if (iacc>10) break;
        iisav = ii1;
        gxconv (x+1.0,y+1.0,&xx,&yy,3);
        if (icol>-1) {
          if (bflg) {gxplot(xold,yold,3); bflg=0;}
          gxplot (xx,yy,2);
        } else bflg = 1;
        xold = xx;
        yold = yy;
        acnt++;
        if (acnt>20) {
          if (icol>-1) strmar(xx,yy,xxsv,yysv);
          acnt = 0;
        }
        xxsv = xx; yysv = yy;
      }
    }
    i2++;
    if (i2==iss) { i2 = 0; j2++; }
  }
  free (it);
}

static float a150 = 150.0*3.1416/180.0;

void strmar (float xx1, float yy1, float xx2, float yy2) {
float dir;
  dir = atan2(yy2-yy1,xx2-xx1);
  gxplot (xx2,yy2,3);
  gxplot (xx2+0.05*cos(dir+a150),yy2+0.05*sin(dir+a150),2);
  gxplot (xx2,yy2,3);
  gxplot (xx2+0.05*cos(dir-a150),yy2+0.05*sin(dir-a150),2);
  gxplot (xx2,yy2,3);
}

/* Given a shade value, return the relevent color */

int gxshdc (float *shdlvs, int *shdcls, int shdcnt, float val) {
int i;

  if (shdcnt==0) return(1);
  if (shdcnt==1) return(shdcls[0]);
  if (val<shdlvs[1]) return(shdcls[0]);
  for (i=1; i<shdcnt-1; i++) {
    if (val>=shdlvs[i] && val<shdlvs[i+1])
                 return(shdcls[i]); 
  }
  return(shdcls[shdcnt-1]);
}
