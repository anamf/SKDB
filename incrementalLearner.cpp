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
#include "incrementalLearner.h"
#include <assert.h>

IncrementalLearner::IncrementalLearner()
{
}

IncrementalLearner::~IncrementalLearner(void)
{
}

/// train the classifier from an instance stream
void IncrementalLearner::train(InstanceStream &is) {
  instance inst(is);
  
  testCapabilities(is);
  
  reset(is);

  while (!trainingIsFinished()) {
    initialisePass();
    is.rewind();
    while (is.advance(inst)) {
      train(inst);
    }
    finalisePass();
  }
}

