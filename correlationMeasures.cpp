/* Gigal: An open source system for classification learning from very large data
** Copyright (C) 2012 Geoffrey I Webb
**
** This program is free software: you can redistribute it and/or modify
** it under the terms of the GNU General Public License as published by
** the Free Software Foundation, either version 3 of the License, or
** (at your option) any later version.
** 
** This program is distributed in the hope that it will be useful,
** but WITHOUT ANY WARRANTY; without even the implied warranty of
** MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
** GNU General Public License for more details.
** 
** You should have received a copy of the GNU General Public License
** along with this program. If not, see <http://www.gnu.org/licenses/>.
**
** Please report any bugs to Geoff Webb <geoff.webb@monash.edu>
*/

#include <math.h>
#include <assert.h>


#include "ALGLIB_specialfunctions.h"
#include "utils.h"
#include "correlationMeasures.h"


/**
 *
 *             __
 *             \               P(x,y)
 *  MI(X,Y)=   /_  P(x,y)log------------
 *            x,y             P(x)P(y)
 *
 *
 */
void getMutualInformation(xyDist &dist, std::vector<float> &mi)
{
  mi.assign(dist.getNoCatAtts(), 0.0);

  const double totalCount = dist.count;

  for (CategoricalAttribute a = 0; a < dist.getNoCatAtts(); a++) {
    double m = 0.0;

    for (CatValue v = 0; v < dist.getNoValues(a); v++) {
      for (CatValue y = 0; y < dist.getNoClasses(); y++) {
        const InstanceCount avyCount = dist.getCount(a,v,y);
        if (avyCount) {
          m += (avyCount / totalCount) * log2(avyCount/((dist.getCount(a, v)/totalCount) 
                                       * dist.getClassCount(y)));
        }
      }
    }

    mi[a] = m;
  }
}

/*
 *                 __
 *                 \                    P(x1,x2|y)
 * CMI(X1,X2|Y)= = /_   P(x1,x2,y) log-------------
 *               x1,x2,y              P(x1|y)P(x2|y)
 *
 */
void getCondMutualInf(xxyDist &dist, crosstab<float> &cmi)
{
  const double totalCount = dist.xyCounts.count;

  for (CategoricalAttribute x1 = 1; x1 < dist.getNoCatAtts(); x1++) {
      for (CategoricalAttribute x2 = 0; x2 < x1; x2++) {
        float m = 0.0;
        for (CatValue v1 = 0; v1 < dist.getNoValues(x1); v1++) {
          for (CatValue v2 = 0; v2 < dist.getNoValues(x2); v2++) {
            for (CatValue y = 0; y < dist.getNoClasses(); y++) {
              const double x1x2y = dist.getCount(x1, v1, x2, v2, y);
              if (x1x2y) {
                //const unsigned int yCount = dist->xyCounts.getClassCount(y);
                //const unsigned int  x1y = dist->xyCounts.getCount(x1, v1, y);
                //const unsigned int  x2y = dist->xyCounts.getCount(x2, v2, y);
                m += (x1x2y/totalCount) * log2(dist.xyCounts.getClassCount(y) * x1x2y / 
                     (static_cast<double>(dist.xyCounts.getCount(x1, v1, y)) * 
                      dist.xyCounts.getCount(x2, v2, y)));
              }
            }
          }
        }

        assert(m >= -0.00000001); // CMI is always positive, but allow for some imprecision

      cmi[x1][x2] = m;
      cmi[x2][x1] = m;
    }
  }
}


void getrow(crosstab<InstanceCount> &xtab, unsigned int noClasses, unsigned int trow, std::vector<InstanceCount> &Crow){
  for (unsigned int k = 0; k < noClasses; k++) {
    Crow[k] = xtab[trow][k];
  }
}

void getcol(crosstab<InstanceCount> &xtab, unsigned int noClasses, unsigned int tcol, std::vector<InstanceCount> &Ccol){
  for (unsigned int k = 0; k < noClasses; k++) {
    Ccol[k] = xtab[k][tcol];
  }
}

unsigned long long int dotproduct(std::vector<InstanceCount> &Crow, std::vector<InstanceCount> &Ccol, unsigned int noClasses){
  unsigned long long int val = 0;
  fflush(stdout);
  for (unsigned int k = 0; k < noClasses; k++) {
    val += static_cast<unsigned long long>(Crow[k])*static_cast<unsigned long long>(Ccol[k]);
  }
  return val;
}

double calcMCC(crosstab<InstanceCount> &xtab){
  // Compute MCC for multi-class problems as in http://rk.kvl.dk/
    unsigned int noClasses = xtab[0].size();
    double MCC = 0.0;

    //Compute N, sum of all values
    double N = 0.0;
    for (unsigned int k = 0; k < noClasses; k++) {
      for (unsigned int l = 0; l < noClasses; l++) {
        N += xtab[k][l];
      }
    }

    //compute correlation coefficient
    double trace = 0.0;
    for (unsigned int k = 0; k < noClasses; k++) {
      trace += xtab[k][k];
    }
    //sum row col dot product
    unsigned long long int rowcol_sumprod=0;
    std::vector<InstanceCount> Crow (noClasses);
    std::vector<InstanceCount> Ccol (noClasses);
    for (unsigned int k = 0; k < noClasses; k++) {
      for (unsigned int l = 0; l < noClasses; l++) {
        getrow(xtab,noClasses,k,Crow);
        getcol(xtab,noClasses,l,Ccol);
        rowcol_sumprod += dotproduct(Crow, Ccol, noClasses);
      }
    }

    //sum over row dot products
    unsigned long long int rowrow_sumprod=0;
    std::vector<InstanceCount> Crowk (noClasses);
    std::vector<InstanceCount> Crowl (noClasses);
    for (unsigned int k = 0; k < noClasses; k++) {
      for (unsigned int l = 0; l < noClasses; l++) {
        getrow(xtab,noClasses,k,Crowk);
        getrow(xtab,noClasses,l,Crowl);
        rowrow_sumprod += dotproduct(Crowk, Crowl, noClasses);
      }
    }

    //sum over col dot products
    unsigned long long int colcol_sumprod=0;
    std::vector<InstanceCount> Ccolk (noClasses);
    std::vector<InstanceCount> Ccoll (noClasses);
    for (unsigned int k = 0; k < noClasses; k++) {
      for (unsigned int l = 0; l < noClasses; l++) {
        getcol(xtab,noClasses,k,Ccolk);
        getcol(xtab,noClasses,l,Ccoll);
        colcol_sumprod += dotproduct(Ccolk, Ccoll, noClasses);
      }
    }

    double cov_XY = N*trace - rowcol_sumprod;
    double cov_XX = N*N - rowrow_sumprod;
    double cov_YY = N*N - colcol_sumprod;
    double denominator = sqrt(cov_XX*cov_YY);

    if(denominator > 0){
      MCC = cov_XY / denominator;
    }
    else if (denominator == 0){
      MCC = 0;
    }
    else{
      printf("Error when calculating MCC2");
    }
    return MCC;
}
