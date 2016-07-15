/* Open source system for classification learning from very large data
** Class for an input stream of randomly sampled instances
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
#include "instanceStreamFilter.h"
#include <assert.h>

InstanceStreamFilter::InstanceStreamFilter()
{
}

InstanceStreamFilter::~InstanceStreamFilter(void)
{
}

void InstanceStreamFilter::setSource(InstanceStream &source) {
  source_ = &source;
  metaData_ = source.getMetaData();
}

void InstanceStreamFilter::rewind() {
  source_->rewind();
}

InstanceCount InstanceStreamFilter::size() {
  return source_->size();
}

/// advance, discarding the next instance in the stream.  Return true iff successful.
bool InstanceStreamFilter::advance() {
  return source_->advance();
}

/// advance to the next instance in the stream. Return true iff successful. @param inst the instance record to receive the new instance. 
bool InstanceStreamFilter::advance(instance &inst) {
  return source_->advance(inst);
}

/// true if we have advanced past the last instance
bool InstanceStreamFilter::isAtEnd() {
  return source_->isAtEnd();
}

