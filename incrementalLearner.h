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
#pragma once

#include <string>
#include <vector>

#include "learner.h"

/**
 <!-- globalinfo-start -->
 * Generic class for a learner/classifier.<br/>
 <!-- globalinfo-end -->
 * 
 * @author Geoff Webb (geoff.webb@monash.edu)
 */

class IncrementalLearner : public learner
{
public:
  IncrementalLearner();
  virtual ~IncrementalLearner(void);

  virtual void reset(InstanceStream &is) = 0;   ///< reset the learner prior to training
  virtual void initialisePass() = 0;            ///< must be called to initialise a pass through an instance stream before calling train(const instance). should not be used with train(InstanceStream)
  virtual void train(const instance &inst) = 0; ///< primary training method. train from a single instance. used in conjunction with initialisePass and finalisePass
  virtual void finalisePass() = 0;              ///< must be called to finalise a pass through an instance stream using train(const instance). should not be used with train(InstanceStream)
  virtual bool trainingIsFinished() = 0;        ///< true iff no more passes are required. updated by finalisePass()

  virtual void train(InstanceStream &is);       ///< train the classifier from an instance stream
};
