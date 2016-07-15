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

#include "crosstab.h"
#include "xyDist.h"
#include "xxyDist.h"

#include "math.h"
/**
<!-- globalinfo-start -->
 * File that includes different correlation measures among variables.<br/>
 <!-- globalinfo-end -->
 *
 * @author Geoff Webb (geoff.webb@monash.edu)
 * @author Ana M. Martinez (anam.martinez@monash.edu)
 */


/**
 * Calculates the mutual information between the attributes in dist and the
 * class
 * 
 * MI(X;C) = H(X) - H(X|C)
 * 
 * @param dist  counts for the xy distributions.
 * @param[out] mi mutual information between the attributes and the class.
 */
void getMutualInformation(xyDist &dist, std::vector<float> &mi);


/**
 * Calculates the class conditional mutual information between the attributes in 
 * dist conditioned on the class
 * 
 * CMI(X;Y|C) = H(X|C) - H(X|Y,C)
 * 
 * @param dist  counts for the xy distributions.
 * @param[out] cmi class conditional mutual information between the attributes.
 */
void getCondMutualInf(xxyDist &dist, crosstab<float> &cmi);



/// Calculates the Matthew's Correlation Coefficient from a set of TP, FP, TN, FN counts
inline double calcBinaryMCC(const InstanceCount TP, const InstanceCount FP, const InstanceCount TN, const InstanceCount FN) {
  if (TP+TN == 0) return -1.0;
  if (FP+FN == 0) return 1.0;
  if ((TP+FN == 0) || (TN+FP == 0) || (TP+FP == 0) || (TN+FN == 0)) return 0.0;
  else return (static_cast<double>(TP)*TN-static_cast<double>(FP)*FN)
                    / sqrt(static_cast<double>(TP+FP)
                            * static_cast<double>(TP+FN)
                            * static_cast<double>(TN+FP)
                            * static_cast<double>(TN+FN));

}

/**
 * Calculates the Matthew's Correlation Coefficient given a confusion matrix (any number of classes) as in http://rk.kvl.dk/
 * 
 * @param xtab  confusion matrix.
 * @return the MCC
 */

double calcMCC(crosstab<InstanceCount> &xtab);


