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
#include "xxyDist.h"
#include "utils.h"
#include <assert.h>

xxyDist::xxyDist() {
}

xxyDist::xxyDist(InstanceStream& stream) : noOfClasses_(stream.getNoClasses()) {
  reset(stream);
  
  // pass through the stream updating the counts incrementally
  stream.rewind();

  instance i;

  while (stream.advance(i)) {
    update(i);
  }
}

xxyDist::~xxyDist(void)
{
}

void xxyDist::reset(InstanceStream& stream) {
  metaData_ = stream.getMetaData();

  noOfClasses_  = stream.getNoClasses();

  xyCounts.reset(&stream);

#if 0
  offset1.resize(stream.getNoCatAtts());
  offset2.resize(stream.getNoCatAtts());

  int next = 0;

  for (CategoricalAttribute a = 0; a < stream.getNoCatAtts(); a++) {
    offset1[a] = next;
    next += stream.getNoValues(a) * stream.getNoClasses();
  }

  next = 0;

  for (CategoricalAttribute a = 0; a < stream.getNoCatAtts(); a++) {
    offset2[a] = next;
    next += stream.getNoValues(a) * offset1[a];
  }

  countSize = next;
  count.assign(next, 0);
#endif

  count_.resize(stream.getNoCatAtts());

  for (CategoricalAttribute x1 = 1; x1 < stream.getNoCatAtts(); x1++) {
    count_[x1].resize(stream.getNoValues(x1) * x1);

    for (CatValue v1 = 0; v1 < stream.getNoValues(x1); ++v1) {
      for (CategoricalAttribute x2 = 0; x2 < x1; x2++) {
        count_[x1][v1*x1+x2].assign(stream.getNoValues(x2)*noOfClasses_, 0);
      }
    }
  }
}

void xxyDist::update(const instance& i) {
  xyCounts.update(i);

  const CatValue theClass = i.getClass();

  for (CategoricalAttribute x1 = 1; x1 < getNoCatAtts(); x1++) {
    const CatValue v1 = i.getCatVal(x1);
    XYSubDist xySubDist(getXYSubDist(x1, v1), noOfClasses_);

    for (CategoricalAttribute x2 = 0; x2 < x1; x2++) {
      const CatValue v2 = i.getCatVal(x2);

      xySubDist.incCount(x2, v2, theClass);

      assert(*ref(x1,v1,x2,v2,theClass) <= xyCounts.count);
    }
  }
}


void xxyDist::clear(){
  count_.clear();
  xyCounts.clear();
}
