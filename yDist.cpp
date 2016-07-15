/* Open source system for classification learning from very large data
** Copyright (C) 2012 Geoffrey I Webb
**
** Class for representing a class distribution.
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
#include "yDist.h"
#include "smoothing.h"
#include "utils.h"

yDist::yDist() {
}

yDist::yDist(InstanceStream &is)
{
  counts.assign(is.getNoClasses(), 0);
  total = 0;
}

yDist::~yDist(void)
{
}

void yDist::reset(const InstanceStream &is) {
  counts.assign(is.getNoClasses(), 0);
  total = 0;
}

void yDist::clear() {
  counts.assign(counts.size(), 0);
  total = 0;
}

void yDist::update(const instance &i) {
  counts[i.getClass()]++;
  total++;
}

double yDist::p(CatValue y) const {
  return mEstimate(counts[y], total, counts.size());
}

double yDist::rawP(CatValue y) const {
  return counts[y] / static_cast<double>(total);
}

//probability when using leave one out cross validation, the t value is discounted
double yDist::ploocv(CatValue y, CatValue t) const {
  if(y==t)
      return mEstimate(counts[y]-1, total-1, counts.size());
  else
      return mEstimate(counts[y], total-1, counts.size());
}

InstanceCount yDist::count(CatValue y) const {
  return counts[y];
}

