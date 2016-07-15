/* Open source system for classification learning from very large data
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
#include "xyDist.h"
#include "utils.h"

#include <memory.h>

xyDist::xyDist() {
}

xyDist::xyDist(InstanceStream *is)
{
  reset(is);

  instance inst(*is);

  while (!is->isAtEnd()) {
    if (is->advance(inst)) update(inst);
  }
}

xyDist::~xyDist(void)
{
}

void xyDist::reset(InstanceStream *is) {

  metaData_ = is->getMetaData();
  noOfClasses_ = is->getNoClasses();
  count = 0;

  counts_.resize(is->getNoCatAtts());

  for (CategoricalAttribute a = 0; a < is->getNoCatAtts(); a++) {
    counts_[a].assign(is->getNoValues(a)*noOfClasses_, 0);
  }

  classCounts.assign(noOfClasses_, 0);
}

void xyDist::update(const instance &inst) {
  count++;

  const CatValue y = inst.getClass();

  classCounts[y]++;

  for (CategoricalAttribute a = 0; a < metaData_->getNoCatAtts(); a++) {
    counts_[a][inst.getCatVal(a)*noOfClasses_+y]++;
  }
}

void xyDist::clear(){
  classCounts.clear();
  for (CategoricalAttribute a = 0; a < getNoAtts(); a++) {
    counts_[a].assign(metaData_->getNoValues(a)*noOfClasses_, 0);
  }
}
